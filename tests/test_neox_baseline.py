"""Memory-free baseline, fair notebook settings and checkpoint result collection."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from grt.checkpoint import load_checkpoint, restore_model, seed_all
from grt.distributed import DistributedContext
from grt.experiments import load_experiment
from grt.models.factory import create_model
from grt.trainer import run
from grt.utils.kaggle import WikiTextRun, file_hash
from grt.wikitext import WikiTextAdapter

REPO = Path(__file__).resolve().parents[1]


def small_cpu(root, model='neox'):
    return load_experiment(task='wikitext103', model=model, size='small', runtime='cpu',
        tokenization='word', steps=4, output_dir=root, overrides={
            'model': {'vocab_size': 32, 'segment_len': 4, 'd_model': 32,
                      'alu': {'num_layers': 2, 'nhead': 4, 'd_ff': 64, 'dropout': 0.0}},
            'training': {'batch_size': 2, 'grad_accum_steps': 1, 'global_batch_size': 2},
            'evaluation': {'batch_size': 2, 'every_steps': 2, 'test_at_end': False}})


def prepared(path):
    path.mkdir()
    manifest = dict(vocab_size=32, tokenization='word', token_counts={}, sha256={})
    for split, count in [('train', 256), ('validation', 64), ('test', 64)]:
        tokens = torch.arange(count, dtype=torch.long) % 32
        torch.save(tokens, path/f'{split}.pt')
        manifest['token_counts'][split] = count
        manifest['sha256'][split] = file_hash(path/f'{split}.pt')
    (path/'manifest.json').write_text(json.dumps(manifest))


def test_neox_has_same_backbone_but_no_memory_or_cross_segment_context(tmp_path):
    cfg = small_cpu(tmp_path)
    seed_all(54)
    model = create_model(cfg).eval()
    seed_all(54)
    rmt = create_model(small_cpu(tmp_path, 'rmt')).eval()
    assert not hasattr(model, 'initial_memory') and not hasattr(model, 'router')
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, rmt.backbone.state_dict()[name], rtol=0, atol=0)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model(ids).logits
    changed = ids.clone()
    changed[:, :4] = 9
    torch.testing.assert_close(logits[:, 4:], model(changed).logits[:, 4:], rtol=0, atol=0)
    changed = ids.clone()
    changed[:, 2:4] = 10
    torch.testing.assert_close(logits[:, :2], model(changed).logits[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(logits[:, 4:], model(ids[:, 4:]).logits, rtol=0, atol=0)
    targets = (ids+1) % 32
    out = model(ids, target_ids=targets)
    assert out.memory_state.shape == (1, 0, 32)
    expected = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(out.loss, expected)
    out.loss.backward()
    assert model.backbone.embedding.weight.grad.abs().sum() > 0


def test_neox_train_resume_does_not_read_test_and_selects_validation(tmp_path, monkeypatch):
    data = tmp_path/'data'
    prepared(data)
    cfg = small_cpu(tmp_path/'run')
    cfg.data.options['path'] = str(data)
    calls = []
    dataset = WikiTextAdapter.dataset
    def read(adapter, config, split, pairs=None):
        calls.append(split)
        assert split != 'test'
        return dataset(adapter, config, split, pairs)
    monkeypatch.setattr(WikiTextAdapter, 'dataset', read)
    from grt import stream_training
    save = stream_training.atomic_save
    early = tmp_path/'step2.pt'
    def snapshot(state, path):
        save(state, path)
        if state['step'] == 2 and Path(path).name == 'last.pt':
            save(state, early)
    monkeypatch.setattr(stream_training, 'atomic_save', snapshot)
    summary = run(cfg, torch.device('cpu'))
    root = Path(cfg.run.output_dir)
    full = load_checkpoint(root/'last.pt')
    assert full['target_tokens_seen'] == 32
    assert summary['evaluation_split'] == 'validation'
    assert summary['selected_rtla'] == {}
    model, saved_cfg = restore_model(load_checkpoint(root/'best.pt'), 'cpu')
    expected = WikiTextAdapter().evaluate(model, saved_cfg, dataset(WikiTextAdapter(), cfg, 'validation'), 'cpu')
    assert summary['selected_validation']['loss'] == pytest.approx(expected['loss'])
    torch.save(load_checkpoint(early), root/'last.pt')
    run(cfg, torch.device('cpu'))
    resumed = load_checkpoint(root/'last.pt')
    for key, value in full['model'].items():
        torch.testing.assert_close(value, resumed['model'][key], rtol=0, atol=0)
    assert 'test' not in calls


def test_notebook_baselines_and_sequential_sweeps_preserve_attention(tmp_path):
    base = dict(repo_dir=REPO, work_dir=tmp_path, models=('neox','rmt','grt'),
                num_registers=40, router_hidden=128, wandb=False,
                overrides={'training': {'memory_fraction': .95}})
    job = WikiTextRun(**base)
    a, b, c = [job.config(name) for name in job.models]
    assert (a.model.num_registers, a.training.bptt_segments) == (0, 0)
    assert (b.model.num_registers, b.training.bptt_segments) == (40, 3)
    assert (c.model.num_registers, c.model.router.mlp_hidden) == (40, 128)
    assert c.model.router.pool_nhead == c.model.alu.nhead == 8 and c.model.d_model == 256
    assert all(not cfg.evaluation.test_at_end for cfg in (a, b, c))
    other = WikiTextRun(**{**base, 'router_hidden': 1024})
    assert other.run_dir != job.run_dir
    assert other.config('grt').model.router.pool_nhead == 8
    registers = WikiTextRun(**{**base, 'num_registers': 160})
    assert registers.run_dir != job.run_dir
    assert registers.config('grt').model.router.mlp_hidden == 128


def test_notebook_exports_one_portable_report_for_this_experiment(tmp_path, monkeypatch):
    data = tmp_path/'input'
    prepared(data)
    job = WikiTextRun(REPO, runtime='cpu', work_dir=tmp_path/'work', models=('neox','rmt','grt'),
        tokenization='word', device='cpu', world_size=1, steps=2, wandb=False,
        num_registers=2, router_hidden=32, benchmark_steps=1, overrides={
            'model': {'vocab_size': 32, 'segment_len': 4, 'd_model': 32,
                      'alu': {'num_layers': 1, 'nhead': 4, 'd_ff': 64}},
            'training': {'batch_size': 2, 'grad_accum_steps': 1, 'global_batch_size': 2},
            'evaluation': {'batch_size': 2, 'every_steps': 1},
            'rtla': {'enabled': True, 'batch_size': 2, 'segments': 2, 'warmup_segments': 1}})
    def launch(name, output, *, benchmark=False):
        cfg = job.config(name, calibrated=not benchmark)
        cfg.run.output_dir = str(output)
        if benchmark:
            from grt.benchmark import benchmark as measure
            measure(cfg, DistributedContext('cpu'), steps=1)
        else:
            run(cfg, torch.device('cpu'))
    monkeypatch.setattr(job, '_launch', launch)
    job.prepare_data(prepared_input=data)
    reports = job.preflight()
    assert reports['neox']['replay_tokens_per_stream'] == 4
    assert reports['rmt']['replay_tokens_per_stream'] == 16
    summaries = job.train()
    assert all(value['evaluation_split'] == 'validation' for value in summaries.values())
    path = job.export_report()
    report = json.loads(path.read_text())
    assert report['report_format'] == 'grt.experiment_report/1'
    assert report['environment'] == dict(runtime='cpu', world_size=1, device='cpu')
    assert len(report['experiments']) == 3
    grt = next(row for row in report['experiments'] if row['model'] == 'grt')
    assert grt['summary']['selected_validation'] == summaries['grt']['selected_validation']
    assert grt['summary']['selected_step'] == summaries['grt']['selected_step']
    assert grt['summary']['selected_rtla'] == summaries['grt']['selected_rtla']
    assert grt['config']['model']['router']['mlp_hidden'] == 32
    assert grt['metadata']['tokenizer']['sha256']
    assert grt['benchmark']['global_batch_size'] == 2
    assert grt['metrics'] and grt['rtla']
    assert report['dataset_manifest']['sha256']
    assert next(row for row in report['experiments'] if row['model'] == 'neox')['summary']['selected_rtla'] == {}
    job.report(plot=False)
    assert json.loads(path.read_text()) == report
    assert not (job.work_dir/'experiment_results.csv').exists()
    assert len(list(job.run_dir.glob('experiment_report*.json'))) == 1
    # A second experiment in another environment must not alter or accumulate the first.
    other = WikiTextRun(REPO, runtime='colab', world_size=1, device='cpu',
                        work_dir=job.work_dir, models=('grt',), num_registers=40, wandb=False)
    second = json.loads(other.export_report().read_text())
    assert second['report_id'] != report['report_id']
    assert len(second['experiments']) == 1 and second['experiments'][0]['status'] == 'no_summary'
    assert json.loads(path.read_text()) == report
    downloads = []
    monkeypatch.setitem(sys.modules, 'google.colab', SimpleNamespace(
        files=SimpleNamespace(download=downloads.append)))
    other.report(plot=False, download=True)
    assert downloads == [str(other.report_path)]
    assert json.loads((job.run_dir/'neox_eta.json').read_text())['final_test_hours'] == 0


def test_neox_rejects_registers_and_replay(tmp_path):
    for overrides in [{'model': {'num_registers': 10}}, {'training': {'bptt_segments': 3}}]:
        with pytest.raises(ValueError):
            load_experiment(task='wikitext103', model='neox', overrides=overrides)


def test_report_preserves_incomplete_run_and_marks_truncated_metrics(tmp_path):
    job = WikiTextRun(REPO, runtime='cpu', world_size=1, device='cpu',
                      work_dir=tmp_path, models=('neox',), wandb=False)
    root = job.run_dir/'neox'
    root.mkdir(parents=True)
    (root/'metrics.jsonl').write_text('{"step": 1, "train/loss": 2.5}\n{"step": 2')
    job.report(plot=False)
    report = json.loads(job.report_path.read_text())
    entry = report['experiments'][0]
    assert entry['status'] == 'no_summary' and entry['summary'] is None
    assert entry['metrics'] == [{'step': 1, 'train/loss': 2.5}]
    assert entry['metrics_parse_errors'][0]['line'] == 2


def test_neox_two_worker_cli_uses_empty_state_and_fixed_target_budget(tmp_path):
    from dataclasses import asdict
    import yaml
    from test_distributed import invoke
    data = tmp_path/'data'
    prepared(data)
    cfg = small_cpu(tmp_path/'run')
    cfg.data.options['path'] = str(data)
    cfg.training.batch_size = 1
    path = tmp_path/'neox.yaml'
    path.write_text(yaml.safe_dump(asdict(cfg)))
    invoke(path, cfg.run.output_dir)
    saved = load_checkpoint(Path(cfg.run.output_dir)/'last.pt')
    assert saved['world_size'] == 2 and saved['target_tokens_seen'] == 32
    assert len(saved['stream_by_rank']) == 2
    assert all(local['boundaries'][-1].shape == (1, 0, 32) for local in saved['stream_by_rank'])
    summary = json.loads((Path(cfg.run.output_dir)/'summary.json').read_text())
    assert summary['evaluation_split'] == 'validation'
