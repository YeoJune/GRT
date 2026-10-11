"""Current catalog selection and one short production-utils CPU workflow."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from grt.config import load_config, save_config
from grt.checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from grt.experiments import load_experiment, parse_overrides
from grt.utils.kaggle import WikiTextRun, file_hash

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('model', ['rmt', 'grt'])
@pytest.mark.parametrize('size', ['base', 'small', 'tiny'])
@pytest.mark.parametrize('budget', ['smoke', '20k', 'full'])
@pytest.mark.parametrize('tokenization', ['word', 'gpt2_bpe'])
def test_catalog_current_contract(model, size, budget, tokenization):
    cfg = load_experiment(task='wikitext103', model=model, size=size,
        tokenization=tokenization, budget=budget, runtime='kaggle')
    dimensions = {'base': (410, 16, 10, 2100), 'small': (256, 6, 8, 1024), 'tiny': (128, 3, 4, 512)}
    assert (cfg.model.d_model, cfg.model.alu.num_layers, cfg.model.alu.nhead, cfg.model.alu.d_ff) == dimensions[size]
    assert cfg.model.vocab_size == {'word': 267735, 'gpt2_bpe': 50257}[tokenization]
    assert cfg.training.max_steps == {'smoke': 200, '20k': 20000, 'full': 200000}[budget]
    assert cfg.training.batch_size * cfg.training.grad_accum_steps * 2 == cfg.training.global_batch_size == 60
    assert cfg.model.implementation == 'neox' and cfg.model.tie_word_embeddings
    assert cfg.model.segment_len == 150 and cfg.model.num_registers == 10
    assert cfg.training.bptt_segments == 3 and not cfg.training.curriculum
    if model == 'rmt':
        assert cfg.model.router is None and cfg.model.register is None
    else:
        assert cfg.model.router.pool_nhead == cfg.model.alu.nhead
        assert not cfg.model.register.conditional_read


def test_explicit_values_and_invalid_choices():
    cfg = load_experiment(task='wikitext103', steps=1234,
        overrides=parse_overrides(['evaluation.every_steps=100', 'wandb.enabled=false']))
    assert cfg.training.max_steps == 1234 and not cfg.training.curriculum
    assert cfg.evaluation.every_steps == 100 and not cfg.wandb.enabled
    with pytest.raises(ValueError, match='Unknown sizes'):
        load_experiment(task='wikitext103', size='typo')
    with pytest.raises(ValueError, match='Unknown keys'):
        load_experiment(task='wikitext103', overrides={'training': {'max_step': 1}})
    with pytest.raises(ValueError, match='Invalid type'):
        load_experiment(task='wikitext103', steps='200')


def test_calibrated_batch_is_preserved_on_existing_run(tmp_path):
    job = WikiTextRun(REPO, work_dir=tmp_path, run_id='existing', wandb=False)
    probe = job.run_dir / 'rmt_preflight'
    probe.mkdir(parents=True)
    report = dict(model='rmt', world_size=2, global_batch_size=60, previous_segments=3,
        target_tokens_per_update=9000,
        calibrated_override={'training': {'batch_size': 5, 'grad_accum_steps': 6}})
    (probe / 'benchmark.json').write_text(json.dumps(report))
    cfg = job.config('rmt')
    root = job.run_dir / 'rmt'
    root.mkdir()
    (root / 'experiment_config.json').write_text(json.dumps(asdict(cfg)))
    restored = WikiTextRun(REPO, work_dir=tmp_path, run_id='existing', wandb=False)
    assert restored.config('rmt').training.batch_size == 5
    assert restored.config('rmt').training.grad_accum_steps == 6
    restored.steps = 123
    with pytest.raises(ValueError, match='new run_id'):
        restored.config('rmt')


def test_utils_actual_preflight_and_two_updates(tmp_path, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')
    prepared = tmp_path / 'input'
    prepared.mkdir()
    manifest = dict(vocab_size=128, tokenization='word', token_counts={}, sha256={})
    for split, count in [('train', 40), ('validation', 9), ('test', 9)]:
        tokens = torch.arange(count, dtype=torch.long) % 128
        torch.save(tokens, prepared / (split + '.pt'))
        manifest['token_counts'][split] = count
        manifest['sha256'][split] = file_hash(prepared / (split + '.pt'))
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    job = WikiTextRun(REPO, size='tiny', budget='smoke', models=('rmt',), tokenization='word',
        work_dir=tmp_path / 'working', steps=2, wandb=False, device='cpu', world_size=1,
        benchmark_steps=1, overrides={
            'model': {'vocab_size': 128, 'd_model': 32, 'segment_len': 4, 'num_registers': 2,
                      'alu': {'num_layers': 1, 'nhead': 4, 'd_ff': 64}},
            'training': {'batch_size': 1, 'grad_accum_steps': 1, 'global_batch_size': 1, 'bptt_segments': 1},
            'evaluation': {'batch_size': 1, 'every_steps': 2},
            'wandb': {'every_steps': 1}})
    job.prepare_data(prepared_input=prepared)
    report = job.preflight()['rmt']
    assert report['global_batch_size'] == 1
    assert job.train()['rmt']['step'] == 2
    result = job.report(plot=False)['rmt']
    assert result['status'] == 'completed' and result['score_kind'] == 'autoregressive'
    rows = [json.loads(line) for line in (job.run_dir / 'rmt/metrics.jsonl').read_text().splitlines()]
    assert {row['step'] for row in rows} == {1, 2}
    assert 'total_compute_hours' in json.loads((job.run_dir / 'rmt_eta.json').read_text())
    assert sorted(p.name for p in job.run_dir.glob('*.yaml')) == ['rmt_config.yaml']
    job.steps = 3
    with pytest.raises(ValueError, match='new run_id'):
        job.config('rmt')


def test_current_only_config_and_checkpoint(tmp_path):
    cfg = load_experiment(task='remember', runtime='cpu', steps=2)
    path = tmp_path / 'resolved.yaml'
    save_config(cfg, path)
    assert asdict(load_config(path)) == asdict(cfg)
    with pytest.raises(FileNotFoundError):
        load_config(REPO / 'configs/base.yaml')
    with pytest.raises(TypeError):
        load_config([path, path])
    with pytest.raises(ValueError, match='Only the current NeoX'):
        load_experiment(task='remember', overrides={'model': {'implementation': 'legacy'}})
    path = tmp_path / 'old.pt'
    torch.save({'format': 'author_rmt/1'}, path)
    with pytest.raises(ValueError, match='Unsupported checkpoint'):
        load_checkpoint(path)
    torch.save({'format': 'memory_experiment/2'}, path)
    with pytest.raises(ValueError, match='Unsupported checkpoint'):
        load_checkpoint(path)
    torch.save({'format': CHECKPOINT_FORMAT}, path)
    assert load_checkpoint(path)['format'] == CHECKPOINT_FORMAT
