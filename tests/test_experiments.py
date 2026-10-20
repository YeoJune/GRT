"""Current catalog selection and one short production-utils CPU workflow."""
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from grt.config import load_config, save_config
from grt.checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from grt.experiments import load_experiment, parse_overrides
from grt.utils.kaggle import WikiTextRun, file_hash

REPO = Path(__file__).resolve().parents[1]


def test_notebook_selection_does_not_import_transformers_before_setup():
    code = ("import sys; from grt.utils.kaggle import WikiTextRun; "
            f"WikiTextRun({str(REPO)!r}, wandb=False); "
            "assert 'transformers' not in sys.modules; "
            "assert 'grt.models.neox' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check=True, cwd=REPO,
                   env={**os.environ, 'PYTHONPATH': str(REPO/'src')})


@pytest.mark.parametrize('model', ['rmt', 'grt'])
@pytest.mark.parametrize('size', ['base', 'small', 'tiny'])
@pytest.mark.parametrize('budget', ['smoke', '10k', '20k', '100k'])
@pytest.mark.parametrize('tokenization', ['word', 'gpt2_bpe'])
def test_catalog_current_contract(model, size, budget, tokenization):
    cfg = load_experiment(task='wikitext103', model=model, size=size,
        tokenization=tokenization, budget=budget, runtime='kaggle')
    dimensions = {'base': (410, 16, 10, 2100), 'small': (256, 6, 8, 1024), 'tiny': (128, 3, 4, 512)}
    assert (cfg.model.d_model, cfg.model.alu.num_layers, cfg.model.alu.nhead, cfg.model.alu.d_ff) == dimensions[size]
    assert cfg.model.vocab_size == {'word': 267735, 'gpt2_bpe': 50257}[tokenization]
    assert cfg.training.max_steps == {'smoke': 200, '10k': 10000, '20k': 20000, '100k': 100000}[budget]
    assert cfg.training.batch_size * cfg.training.grad_accum_steps * 2 == cfg.training.global_batch_size == 256
    assert cfg.training.auto_batch and cfg.training.memory_fraction == 0.8
    assert cfg.training.lr == 0.0005
    local_batch = cfg.training.batch_size * cfg.training.grad_accum_steps
    assert local_batch == 128 and local_batch & (local_batch - 1) == 0
    assert cfg.evaluation.every_steps == (20 if budget == 'smoke' else 1000)
    assert cfg.wandb.every_steps == 10
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
    with pytest.raises(ValueError, match='Unknown keys'):
        load_experiment(task='wikitext103', overrides={'model': {'register_read_source': 'fixed_state'}})


@pytest.mark.parametrize('budget,steps', [('20k', 20000), ('1m', 1000000),
    ('40000', 40000), (40000, 40000), ('30k', 30000), ('7', 7), ('002k', 2000)])
def test_free_budget_through_notebook_runner(tmp_path, budget, steps):
    job = WikiTextRun(REPO, budget=budget, work_dir=tmp_path, wandb=False)
    cfg = job.config('rmt')
    assert cfg.training.max_steps == steps and cfg.training.curriculum == []
    assert cfg.evaluation.every_steps == 1000 and cfg.wandb.every_steps == 10
    assert cfg.training.min_lr_ratio == .01


@pytest.mark.parametrize('budget', ['0', '0k', '0m', 0, -1, True, 20000., '',
    '1.5m', '20,000', '2e4', '40_000', '20kk', '1g', '-20k', '+20k', '20 k',
    '２０k', '20K', 'updates'])
def test_invalid_free_budget(budget):
    with pytest.raises(ValueError, match='Invalid budget'):
        load_experiment(task='wikitext103', budget=budget)


def test_free_budget_merge_order_and_remember_curriculum():
    cfg = load_experiment(task='remember', budget='40k')
    assert cfg.training.max_steps == 40000 and not cfg.training.curriculum
    cfg = load_experiment(task='wikitext103', budget='1m', runtime='cpu', steps=12,
        overrides={'training': {'max_steps': 14}, 'evaluation': {'every_steps': 7}})
    assert cfg.training.max_steps == 14 and cfg.evaluation.every_steps == 7


@pytest.mark.parametrize('ratio', [0., .01, .2, 1.])
def test_cosine_min_lr_config_roundtrip(tmp_path, ratio):
    cfg = load_experiment(task='wikitext103',
        overrides=parse_overrides([f'training.min_lr_ratio={ratio}']))
    save_config(cfg, tmp_path/'config.yaml')
    assert load_config(tmp_path/'config.yaml').training.min_lr_ratio == ratio


@pytest.mark.parametrize('ratio', [-.01, 1.01, float('nan'), float('inf'), '0.01', True])
def test_invalid_cosine_min_lr_ratio(ratio):
    with pytest.raises(ValueError):
        load_experiment(task='wikitext103', overrides={'training': {'min_lr_ratio': ratio}})


@pytest.mark.parametrize('ratio', [0., .05, .5, .99])
def test_warmup_config_roundtrip(tmp_path, ratio):
    cfg = load_experiment(task='wikitext103',
        overrides=parse_overrides([f'training.warmup_ratio={ratio}']))
    save_config(cfg, tmp_path/'config.yaml')
    assert load_config(tmp_path/'config.yaml').training.warmup_ratio == ratio


@pytest.mark.parametrize('ratio', [-.01, 1., 1.01, float('nan'), float('inf'), '0.05', True])
def test_invalid_warmup_ratio(ratio):
    with pytest.raises(ValueError):
        load_experiment(task='wikitext103', overrides={'training': {'warmup_ratio': ratio}})


def test_colab_shared_budget_and_secret_setup(tmp_path, monkeypatch):
    colab = WikiTextRun(REPO, runtime='colab', world_size=1, work_dir=tmp_path,
                       models=('rmt', 'grt'), device='cpu')
    kaggle = WikiTextRun(REPO, work_dir=tmp_path, models=('rmt', 'grt'))
    for model in colab.models:
        a, b = colab.config(model), kaggle.config(model)
        assert a.training.batch_size == b.training.batch_size == 32
        assert a.training.grad_accum_steps == 8 and b.training.grad_accum_steps == 4
        assert a.training.batch_size*a.training.grad_accum_steps == a.training.global_batch_size == 256
        assert a.training.lr == b.training.lr and a.training.max_steps == b.training.max_steps
        assert a.model == b.model and a.evaluation == b.evaluation and a.rtla == b.rtla
        assert a.data.options['path'].startswith(str(tmp_path))
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    monkeypatch.setenv('HF_HOME', str(tmp_path/'hf'))
    keys, calls = [], []
    def get(name):
        keys.append(name)
        return 'mock-colab-key'
    monkeypatch.setitem(sys.modules, 'google.colab', SimpleNamespace(userdata=SimpleNamespace(get=get)))
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(login=lambda **kwargs: calls.append(kwargs)))
    monkeypatch.setattr('grt.utils.kaggle.subprocess.run', lambda *args, **kwargs: None)
    colab.setup()
    assert keys == ['WANDB_API_KEY'] and calls == [{'key': 'mock-colab-key'}]


@pytest.mark.parametrize('runtime', ['kaggle', 'colab'])
@pytest.mark.parametrize('budget,steps', [('20k', 20000), ('1m', 1000000), ('40000', 40000)])
@pytest.mark.parametrize('warmup_ratio', [0., .05])
def test_selected_penalty_dropout_grid_through_actual_notebook(runtime, budget, steps, warmup_ratio, monkeypatch):
    """Current notebook exposes the chosen method while preserving all run budgets."""
    import ast
    notebook = json.loads((REPO/'notebooks'/f'{runtime}.ipynb').read_text())
    assert 'min_lr_ratio' not in json.dumps(notebook)
    source = ast.parse(''.join(notebook['cells'][3]['source']))
    warmup = next(node for node in source.body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'warmup_ratio')
    assert ast.literal_eval(warmup.value) == .05
    default_budget = next(node for node in source.body if isinstance(node, ast.Assign)
                          and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'budget')
    assert ast.literal_eval(default_budget.value) == '40k'
    monkeypatch.setattr(WikiTextRun, 'setup', lambda self: None)
    names = set()
    for memory in (8, 20, 75):
        for dropout in (0., .1):
            selection = copy.deepcopy(source)
            values = dict(num_registers=memory, write_dropout=dropout, budget=budget,
                          warmup_ratio=warmup_ratio)
            for node in selection.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    key = node.targets[0].id
                    if key in values:
                        node.value = ast.Constant(values[key])
            namespace = dict(Path=Path, WikiTextRun=WikiTextRun, repo=REPO)
            exec(compile(ast.fix_missing_locations(selection), '<selection>', 'exec'), namespace)
            job = namespace['job']; cfg = job.config('grt', calibrated=False)
            assert job.models == ('grt',) and job.test_at_end is False
            assert cfg.model.segment_len == 75 and cfg.model.num_registers == memory
            assert cfg.model.register.write_gate_bias_init == 0
            assert cfg.model.register.dropout_prob == dropout
            assert cfg.model.register.conditional_read is False
            assert cfg.model.register.write_logit_threshold == 10
            assert cfg.model.register.write_logit_coef == 1e-4
            assert cfg.model.d_model == 256 and cfg.model.alu.num_layers == 6
            assert cfg.model.router.pool_nhead == 8 and cfg.model.router.mlp_hidden == 512
            assert cfg.training.max_steps == steps and cfg.training.bptt_segments == 3
            assert cfg.training.min_lr_ratio == .01
            assert cfg.training.warmup_ratio == warmup_ratio
            assert cfg.training.batch_size*cfg.training.grad_accum_steps*job.world_size == 256
            assert cfg.training.lr == .0005 and cfg.training.mixed_precision == 'fp32'
            assert cfg.training.memory_fraction == .95 and cfg.rtla.enabled
            assert cfg.evaluation.batch_size == 10 and cfg.evaluation.every_steps == 1000
            names.add(job.run_dir)
    assert len(names) == 6


def test_calibrated_batch_is_preserved_on_existing_run(tmp_path):
    job = WikiTextRun(REPO, work_dir=tmp_path, run_id='existing', wandb=False,
                      overrides={'training': {'auto_batch': True}})
    probe = job.run_dir / 'rmt_preflight'
    probe.mkdir(parents=True)
    report = dict(model='rmt', world_size=2, global_batch_size=256, previous_segments=3,
        target_tokens_per_update=38400,
        batch_tuning=dict(enabled=True, applied=True, memory_fraction=0.8),
        calibrated_override={'training': {'batch_size': 64, 'grad_accum_steps': 2}})
    (probe / 'benchmark.json').write_text(json.dumps(report))
    cfg = job.config('rmt')
    root = job.run_dir / 'rmt'
    root.mkdir()
    (root / 'experiment_config.json').write_text(json.dumps(asdict(cfg)))
    restored = WikiTextRun(REPO, work_dir=tmp_path, run_id='existing', wandb=False,
                           overrides={'training': {'auto_batch': True}})
    assert restored.config('rmt').training.batch_size == 64
    assert restored.config('rmt').training.grad_accum_steps == 2
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
    job.report(plot=False, posthoc=True)
    with ZipFile(job.run_dir/'experiment_results.zip') as bundle:
        assert bundle.namelist() == ['experiment_report.json']
    rows = [json.loads(line) for line in (job.run_dir / 'rmt/metrics.jsonl').read_text().splitlines()]
    assert {row['step'] for row in rows} == {1, 2}
    assert rows[-1]['train/lr'] == pytest.approx(.0005*.01)
    assert 'total_compute_hours' in json.loads((job.run_dir / 'rmt_eta.json').read_text())
    assert sorted(p.name for p in job.run_dir.glob('*.yaml')) == ['rmt_config.yaml']
    job.steps = 3
    with pytest.raises(ValueError, match='new run_id'):
        job.config('rmt')


def test_training_report_bundles_real_posthoc_and_reuses_analysis(tmp_path, monkeypatch):
    """Run the notebook's training -> analysis -> one-file handoff on CPU."""
    prepared = tmp_path/'input'
    prepared.mkdir()
    manifest = dict(vocab_size=128, tokenization='word', token_counts={}, sha256={})
    for split, count in [('train', 80), ('validation', 160), ('test', 9)]:
        torch.save(torch.arange(count, dtype=torch.long)%128, prepared/f'{split}.pt')
        manifest['token_counts'][split] = count
        manifest['sha256'][split] = file_hash(prepared/f'{split}.pt')
    (prepared/'manifest.json').write_text(json.dumps(manifest))
    job = WikiTextRun(REPO, runtime='colab', size='tiny', models=('grt',),
        tokenization='word', work_dir=tmp_path/'working', steps=2, wandb=False,
        device='cpu', world_size=1, benchmark_steps=1, overrides={
            'model': {'vocab_size': 128, 'd_model': 32, 'segment_len': 4, 'num_registers': 3,
                      'alu': {'num_layers': 1, 'nhead': 4, 'd_ff': 64},
                      'router': {'mlp_hidden': 32}},
            'training': {'auto_batch': False, 'batch_size': 1, 'grad_accum_steps': 1,
                         'global_batch_size': 1, 'bptt_segments': 1},
            'evaluation': {'batch_size': 1, 'every_steps': 2},
            'rtla': {'enabled': True, 'batch_size': 2, 'segments': 2, 'warmup_segments': 1}})
    job.prepare_data(prepared_input=prepared)
    job.preflight()
    summary = job.train()['grt']
    checkpoint = job.run_dir/'grt/best.pt'
    before = file_hash(checkpoint)
    bundle_path = job.run_dir/'experiment_results.zip'
    with pytest.raises(subprocess.CalledProcessError):
        job.report(plot=False, posthoc=True, posthoc_options={'window_starts': (0, 1)})
    assert 'disjoint' in (job.run_dir/'grt_posthoc_console.txt').read_text()
    assert job.report_path.is_file() and not bundle_path.exists()
    options = dict(batch_size=2, segments=2, warmup_segments=1, horizon=2,
                   window_starts=(0, 6, 12), group_repeats=1, lesion_fractions=(.5,))
    downloads = []
    monkeypatch.setitem(sys.modules, 'google.colab',
                        SimpleNamespace(files=SimpleNamespace(download=downloads.append)))
    # A Kaggle kernel may retain a preinstalled Config after setup downgrades the
    # files on disk. Reproduce the missing rotary_pct in the parent interpreter.
    from grt.models import neox
    from grt.checkpoint import restore_model
    original_config = neox.GPTNeoXConfig
    def stale_config(**kwargs):
        config = original_config(**kwargs)
        del config.rotary_pct
        return config
    monkeypatch.setattr(neox, 'GPTNeoXConfig', stale_config)
    with pytest.raises(AttributeError, match='rotary_pct'):
        restore_model(load_checkpoint(checkpoint), 'cpu')
    results = job.report(plot=False, posthoc=True, posthoc_options=options, download=True)
    assert results['grt'] == summary and downloads == [str(bundle_path)]
    with ZipFile(bundle_path) as bundle:
        assert set(bundle.namelist()) == {'experiment_report.json', 'posthoc/grt/posthoc_report.json'}
        assert bundle.read('experiment_report.json') == job.report_path.read_bytes()
        training = json.loads(bundle.read('experiment_report.json'))
        posthoc = json.loads(bundle.read('posthoc/grt/posthoc_report.json'))
    assert posthoc['status'] == 'completed' and posthoc['checkpoint']['sha256'] == before
    assert posthoc['checkpoint']['step'] == summary['selected_step']
    assert posthoc['analysis_settings']['slots'] and posthoc['analysis_settings']['registers']
    assert len(posthoc['register_analysis']['windows']) == 3
    assert len(posthoc['summary']['slot_delta_nll']) == 3
    assert posthoc['training_config'] == training['experiments'][0]['config']
    assert file_hash(checkpoint) == before
    job.report(plot=False, posthoc=True, posthoc_options=options)
    with ZipFile(bundle_path) as bundle:
        repeated = json.loads(bundle.read('posthoc/grt/posthoc_report.json'))
    assert repeated['report_id'] == posthoc['report_id']
    with pytest.raises(ValueError, match='posthoc=True'):
        job.report(plot=False, posthoc_options=options)
    wrong_summary = {**summary, 'selected_step': summary['selected_step']+1}
    (job.run_dir/'grt/summary.json').write_text(json.dumps(wrong_summary))
    with pytest.raises(subprocess.CalledProcessError):
        job.report(plot=False, posthoc=True, posthoc_options=options)
    assert 'step differs' in (job.run_dir/'grt_posthoc_console.txt').read_text()
    assert file_hash(checkpoint) == before


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
