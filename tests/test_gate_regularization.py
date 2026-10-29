"""CPU software checks only; these are not experimental gate measurements."""
import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from grt.config import validate_config
from grt.distributed import DistributedContext
from grt.logger import TrainingMetrics
from grt.models.factory import create_model
from grt.regularization import write_logit_regularization
from grt.utils.notebook import notebook_run
from grt.utils.kaggle import WikiTextRun
from conftest import tiny_config


def test_tail_penalty_weights_tokens_ignores_replay_and_has_unsaturated_gradient():
    old = torch.full((1, 2), 100., requires_grad=True)
    new = torch.tensor([[12., -20.]], requires_grad=True)
    tail = torch.tensor([[5., 6.]], requires_grad=True)
    valid = torch.tensor([[False, False, True, True, True]])
    task = torch.tensor(3., requires_grad=True)
    observations = [old, new, tail]
    total, metrics = write_logit_regularization(task, observations, valid, 2, 10., .1)
    # Two supervised tokens at mean penalty52; one short-tail token at penalty0.
    expected = torch.tensor(52. * 2 / 3)
    torch.testing.assert_close(metrics['train/gate_logit_penalty'], expected)
    torch.testing.assert_close(total, task + .1 * expected)
    total.backward()
    torch.testing.assert_close(old.grad, torch.zeros_like(old))
    torch.testing.assert_close(new.grad, torch.tensor([[.1 * 4 / 3, -.1 * 20 / 3]]))
    torch.testing.assert_close(tail.grad, torch.zeros_like(tail))
    assert metrics['train/gate_fraction_abs_over20'] == 0
    assert all(not value.requires_grad for value in metrics.values())
    # Duplicating identical registers preserves the slot-average objective.
    duplicated = [z.detach().repeat(1, 3) for z in observations]
    _, other = write_logit_regularization(task.detach(), duplicated, valid, 2, 10., .1)
    torch.testing.assert_close(other['train/gate_logit_penalty'], expected)


def test_model_uses_raw_logits_before_dropout_and_excludes_masked_history(tmp_path):
    cfg = tiny_config(tmp_path, 'grt')
    cfg.model.segment_len = 3
    r = cfg.model.register
    r.write_logit_coef, r.write_logit_threshold, r.dropout_prob = .1, 5., 1.
    model = create_model(cfg).train()
    with torch.no_grad():
        model.router.mlp_w[-1].weight.zero_()
        model.router.mlp_w[-1].bias.fill_(6)
    calls = []
    def historical_logits(module, args, output):
        calls.append(None)
        return output + (94 if len(calls) <= 2 else 0)
    handle = model.router.mlp_w[-1].register_forward_hook(historical_logits)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    targets = ids.clone()
    targets[:, :6] = -100
    out = model(ids, target_ids=targets)
    handle.remove()
    assert len(calls) == 3
    assert out.training_metrics['train/gate_logit_penalty'] == 1
    assert out.training_metrics['train/gate_fraction_abs_over20'] == 0
    torch.testing.assert_close(out.memory_state, torch.zeros_like(out.memory_state))
    torch.testing.assert_close(out.loss, out.training_metrics['train/loss'] + .1)
    out.loss.backward()
    # With all writes dropped, task CE has no gate path; raw-logit penalty still learns.
    torch.testing.assert_close(model.router.mlp_w[-1].bias.grad,
                               torch.full_like(model.router.mlp_w[-1].bias, .2 / cfg.model.num_registers))


def test_off_is_exact_and_eval_loss_does_not_include_penalty(tmp_path):
    cfg = tiny_config(tmp_path, 'grt')
    cfg.model.register.dropout_prob = .2
    cfg.model.register.write_logit_threshold = 5
    cfg.model.register.write_gate_bias_init = 12
    model = create_model(cfg).train()
    enabled = copy.deepcopy(model)
    enabled.cfg = copy.deepcopy(model.cfg)
    enabled.cfg.register.write_logit_coef = .01
    ids = torch.arange(12).reshape(2, 6)
    rng = torch.get_rng_state()
    off = model(ids, target_ids=ids)
    off_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    on = enabled(ids, target_ids=ids)
    assert torch.equal(torch.get_rng_state(), off_rng)
    assert off.training_metrics is None
    torch.testing.assert_close(off.memory_state, on.memory_state, rtol=0, atol=0)
    torch.testing.assert_close(off.loss, on.training_metrics['train/loss'], rtol=0, atol=0)
    assert on.loss > off.loss
    assert set(model.state_dict()) == set(enabled.state_dict())
    model.eval(); enabled.eval()
    off = model(ids, target_ids=ids)
    on = enabled(ids, target_ids=ids)
    assert on.training_metrics is None
    torch.testing.assert_close(off.loss, on.loss, rtol=0, atol=0)
    torch.testing.assert_close(off.memory_state, on.memory_state, rtol=0, atol=0)


def test_microbatch_gradients_and_metrics_match_full_stream_batch(tmp_path):
    cfg = tiny_config(tmp_path, 'grt')
    cfg.model.register.write_logit_coef = .01
    cfg.model.register.write_logit_threshold = 5
    cfg.model.register.write_gate_bias_init = 12
    # Keep this numerical comparison independent of earlier tests' RNG use.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(54)
        full = create_model(cfg).train()
    split = copy.deepcopy(full)
    ids = torch.arange(24).reshape(4, 6)
    targets = ids.clone(); targets[:, :4] = -100
    expected = full(ids, target_ids=targets)
    expected.loss.backward()
    accumulator = TrainingMetrics()
    for start, stop in ((0, 1), (1, 4)):
        out = split(ids[start:stop], target_ids=targets[start:stop])
        weight = (stop - start) / 4
        (out.loss * weight).backward()
        accumulator.update(out, weight)
    for a, b in zip(full.parameters(), split.parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)
    actual = accumulator.reduce(DistributedContext('cpu'))
    for key, value in expected.training_metrics.items():
        assert actual[key] == pytest.approx(value.item(), rel=2e-6, abs=2e-7)


@pytest.mark.parametrize('key,value', [('write_logit_threshold', 0),
    ('write_logit_threshold', float('inf')), ('write_logit_coef', -.1),
    ('write_logit_coef', float('nan'))])
def test_invalid_gate_penalty_settings(tmp_path, key, value):
    cfg = tiny_config(tmp_path, 'grt')
    setattr(cfg.model.register, key, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('runtime', ['colab', 'kaggle'])
def test_notebook_penalty_conditions_are_isolated_and_exportable(tmp_path, monkeypatch, runtime):
    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo / 'notebooks' / f'{runtime}.ipynb').read_text())
    code = ast.parse(''.join(notebook['cells'][3]['source']))
    monkeypatch.setattr(WikiTextRun, 'setup', lambda self: None)
    paths = set()
    for threshold, coef in ((10., 0.), (10., 1e-4), (5., 1e-4), (10., 1e-3)):
        selection = copy.deepcopy(code)
        values = dict(write_logit_threshold=threshold, write_logit_coef=coef)
        for node in selection.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in values:
                    node.value = ast.Constant(values[node.targets[0].id])
        namespace = dict(Path=Path, notebook_run=notebook_run, repo=repo, session_started=None)
        exec(compile(ast.fix_missing_locations(selection), '<selection>', 'exec'), namespace)
        job = namespace['job']; cfg = job.config('grt', calibrated=False)
        assert cfg.model.register.write_logit_threshold == threshold
        assert cfg.model.register.write_logit_coef == coef
        assert (cfg.model.segment_len, cfg.model.num_registers) == (75, 20)
        assert cfg.model.register.write_gate_bias_init == cfg.model.register.dropout_prob == 0
        assert cfg.training.max_steps == 40000 and cfg.training.model_seed == 54
        assert cfg.training.warmup_ratio == .05 and cfg.training.min_lr_ratio == .01
        assert cfg.training.batch_size * cfg.training.grad_accum_steps * job.world_size == 256
        assert cfg.training.mixed_precision == 'fp32'
        paths.add(job.run_dir)
    assert len(paths) == 4


def test_stream_logs_task_and_penalty_and_exactly_resumes(tmp_path, monkeypatch):
    from test_neox import stream_config
    from grt import stream_training
    from grt.trainer import run
    cfg = stream_config(tmp_path)
    cfg.model.register.write_logit_coef = .01
    cfg.model.register.write_logit_threshold = .1
    cfg.model.register.write_gate_bias_init = 6
    cfg.training.batch_size = 2; cfg.training.grad_accum_steps = 2
    cfg.training.global_batch_size = 4
    save = stream_training.atomic_save
    early = tmp_path / 'early.pt'
    def snapshot(state, path):
        save(state, path)
        if state['step'] == 2 and Path(path).name == 'last.pt':
            save(state, early)
    monkeypatch.setattr(stream_training, 'atomic_save', snapshot)
    result = run(cfg, torch.device('cpu'))
    root = Path(cfg.run.output_dir)
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    for row in rows:
        assert row['train/total_loss'] == pytest.approx(row['train/loss'] + row['train/gate_regularization_loss'])
        assert row['train/gate_regularization_loss'] > 0
    full = torch.load(root / 'last.pt', weights_only=False)
    torch.save(torch.load(early, weights_only=False), root / 'last.pt')
    resumed = run(cfg, torch.device('cpu'))
    actual = torch.load(root / 'last.pt', weights_only=False)
    for key in full['model']:
        assert torch.equal(full['model'][key], actual['model'][key])
    assert result['selected_validation'] == resumed['selected_validation']


@pytest.mark.parametrize('source', ['independent', 'separate_head', 'candidate'])
def test_two_worker_stream_matches_serial_with_active_penalty(tmp_path, source):
    from dataclasses import asdict
    import yaml
    from test_neox_baseline import prepared, small_cpu
    from test_distributed import invoke
    data = tmp_path / 'tokens'; prepared(data)
    runs = {}
    for workers in (2, 1):
        root = tmp_path / f'workers{workers}'
        cfg = small_cpu(root, model='grt')
        cfg.model.router.write_source = source
        cfg.model.router.write_detach_source = source == 'candidate'
        cfg.data.options['path'] = str(data)
        cfg.training.batch_size = 1
        cfg.training.grad_accum_steps = 4 // workers
        cfg.training.global_batch_size = 4
        cfg.model.register.write_gate_bias_init = 6
        cfg.model.register.write_logit_threshold = .1
        cfg.model.register.write_logit_coef = .01
        path = tmp_path / f'workers{workers}.yaml'
        path.write_text(yaml.safe_dump(asdict(cfg)))
        invoke(path, root, workers=workers)
        runs[workers] = torch.load(root / 'last.pt', weights_only=False)
        rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
        assert all(row['train/gate_regularization_loss'] > 0 for row in rows)
        if workers == 2:
            assert runs[workers]['target_tokens_seen'] == 64
            distributed_rows = rows
        else:
            for a, b in zip(distributed_rows, rows):
                for key in a:
                    if key.startswith('train/gate') or key in ('train/loss', 'train/total_loss'):
                        assert a[key] == pytest.approx(b[key], rel=2e-5, abs=2e-6)
    for key, value in runs[2]['model'].items():
        torch.testing.assert_close(value, runs[1]['model'][key], rtol=1e-4, atol=2e-5)
