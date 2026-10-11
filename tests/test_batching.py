"""Batch-policy bounds and calibration lifecycle without GPU training."""
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from grt.config import TrainingConfig, save_config
from grt.experiments import load_experiment
from grt.utils.batching import select_batch, calibrate_batch, apply_batch_override, prepare_batch


@pytest.mark.parametrize('limit,expected', [(100, 30), (16, 15), (10, 10), (7, 6), (1, 1)])
def test_selection_grows_or_shrinks_with_fixed_effective_batch(limit, expected):
    training = TrainingConfig(batch_size=10, grad_accum_steps=3, global_batch_size=60, auto_batch=True)
    candidates = []
    def probe():
        candidates.append(training.batch_size)
        assert training.batch_size * training.grad_accum_steps * 2 == training.global_batch_size
        assert training.batch_size <= training.global_batch_size // 2
        return training.batch_size <= limit
    select_batch(training, torch.device('cuda'), probe)
    assert training.batch_size == expected and candidates[-1] == expected


@pytest.mark.parametrize('enabled,device', [(False, 'cuda'), (False, 'cpu'), (True, 'cpu')])
def test_off_and_cpu_do_not_probe_or_change_config(enabled, device):
    cfg = load_experiment(task='remember', runtime='cpu', steps=2,
                          overrides={'training': {'auto_batch': enabled}})
    original = asdict(cfg)
    ctx = SimpleNamespace(device=torch.device(device))
    def unexpected(*args):
        pytest.fail('Disabled/CPU policy must not allocate a calibration model')
    policy = calibrate_batch(cfg, ctx, unexpected, unexpected, unexpected)
    assert not policy['applied'] and asdict(cfg) == original


def test_no_fit_is_reported_without_changing_effective_batch():
    training = TrainingConfig(batch_size=10, grad_accum_steps=3, auto_batch=True)
    with pytest.raises(RuntimeError, match='No microbatch fits'):
        select_batch(training, torch.device('cuda'), lambda: False)
    assert training.batch_size * training.grad_accum_steps == 30


def test_calibration_handles_oom_and_remote_rank_limit(monkeypatch, tmp_path):
    cfg = load_experiment(task='remember', runtime='cpu', steps=2, output_dir=tmp_path,
        overrides={'training': {'auto_batch': True, 'batch_size': 10, 'grad_accum_steps': 3}})
    attempts = []
    class Model(torch.nn.Linear):
        def to(self, device):
            return self  # CPU arithmetic; CUDA memory/collectives are simulated below.
    monkeypatch.setattr('grt.utils.batching.create_model', lambda cfg: Model(1, 1))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(total_memory=1000))
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: cfg.training.batch_size * 50)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (1000, 1000))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    # The other GPU rejects micro15 even though this simulated GPU fits it.
    ctx = SimpleNamespace(device=torch.device('cuda'), world_size=2, wrap=lambda model: model,
        synchronize=lambda *args: nullcontext(), max=lambda value: max(value, cfg.training.batch_size == 15))
    def optimizer(model, cfg):
        return torch.optim.SGD(model.parameters(), lr=0.1)
    def update(model, wrapped, optimizer, distributed):
        if not distributed:
            attempts.append(cfg.training.batch_size)
        if cfg.training.batch_size == 30:
            raise torch.cuda.OutOfMemoryError('simulated')
        optimizer.zero_grad()
        wrapped(torch.ones(1, 1)).sum().backward()
        optimizer.step()
    policy = calibrate_batch(cfg, ctx, optimizer, update, lambda model: None)
    assert attempts == [30, 15, 10]
    assert cfg.training.batch_size == 10 and cfg.training.grad_accum_steps == 3
    assert policy['applied']


def test_override_rejects_budget_and_optimizer_changes(tmp_path):
    cfg = load_experiment(task='wikitext103', runtime='kaggle', output_dir=tmp_path)
    with pytest.raises(ValueError, match='effective batches'):
        apply_batch_override(cfg, {'training': {'batch_size': 30, 'grad_accum_steps': 2}}, 2)
    with pytest.raises(ValueError, match='only change'):
        apply_batch_override(cfg, {'training': {'lr': 0.1}}, 2)
    with pytest.raises(ValueError, match='effective batches'):
        apply_batch_override(cfg, {'training': {'batch_size': 30, 'grad_accum_steps': 1}}, 1)


def test_preflight_and_resume_reuse_without_remeasurement(tmp_path, monkeypatch):
    cfg = load_experiment(task='wikitext103', model='rmt', runtime='kaggle', budget='smoke',
        output_dir=tmp_path / 'rmt', overrides={'training': {'auto_batch': True}})
    ctx = SimpleNamespace(device=torch.device('cuda'), world_size=2)
    probe = tmp_path / 'rmt_preflight'
    probe.mkdir()
    save_config(cfg, probe / 'resolved_config.yaml')
    override = {'training': {'batch_size': 128, 'grad_accum_steps': 1}}
    (probe / 'benchmark.json').write_text(json.dumps(dict(world_size=2,
        batch_tuning={'applied': True}, calibrated_override=override)))
    def unexpected(*args):
        pytest.fail('Matching calibration must not be measured again')
    monkeypatch.setattr('grt.benchmark.benchmark', unexpected)
    resolved = prepare_batch(cfg, ctx)
    assert resolved.training.batch_size == 128 and resolved.training.grad_accum_steps == 1
    root = Path(cfg.run.output_dir)
    root.mkdir()
    (root / 'experiment_config.json').write_text(json.dumps(asdict(resolved)))
    assert asdict(prepare_batch(cfg, ctx)) == asdict(resolved)
    cfg.training.lr *= 2
    with pytest.raises(ValueError, match='conditions changed'):
        prepare_batch(cfg, ctx)


def test_policy_values_are_validated():
    for override in [{'auto_batch': 'true'}, {'memory_fraction': 0}, {'memory_fraction': 1}]:
        with pytest.raises(ValueError):
            load_experiment(task='remember', overrides={'training': override})


def test_fresh_cuda_training_prepares_batch_once(tmp_path, monkeypatch):
    cfg = load_experiment(task='wikitext103', runtime='kaggle', output_dir=tmp_path / 'rmt',
                          overrides={'training': {'auto_batch': True}})
    ctx = SimpleNamespace(device=torch.device('cuda'), world_size=2)
    calls = []
    def measure(requested, context):
        calls.append(Path(requested.run.output_dir))
        assert context is ctx
        return {'calibrated_override': {'training': {'batch_size': 128, 'grad_accum_steps': 1}}}
    monkeypatch.setattr('grt.benchmark.benchmark', measure)
    result = prepare_batch(cfg, ctx)
    assert calls == [tmp_path / 'rmt_preflight']
    assert cfg.training.batch_size == 32  # Requested config is preserved.
    assert result.training.batch_size * result.training.grad_accum_steps * ctx.world_size == 256


def test_remember_cpu_preflight_keeps_config_batch(tmp_path):
    from grt.benchmark import benchmark
    from grt.distributed import DistributedContext
    cfg = load_experiment(task='remember', model='grt', runtime='cpu', steps=2,
        output_dir=tmp_path, overrides={
            'model': {'d_model': 32, 'num_registers': 2,
                      'alu': {'num_layers': 1, 'nhead': 4, 'd_ff': 64},
                      'router': {'mlp_hidden': 32}},
            'training': {'batch_size': 2, 'grad_accum_steps': 2, 'global_batch_size': 4},
            'evaluation': {'batch_size': 2}})
    report = benchmark(cfg, DistributedContext('cpu'), steps=1)
    assert report['calibrated_override'] == {}
    assert not report['batch_tuning']['enabled']
    assert all((row['microbatch_per_gpu'], row['accumulation'], row['global_batch_size'])
               == (2, 2, 4) for row in report['stages'])
