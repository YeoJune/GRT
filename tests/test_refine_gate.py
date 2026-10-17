"""Refined writes: neutral pairing, recurrent credit and observable gate semantics."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from grt.config import RouterConfig
from grt.models.factory import create_model
from grt.models.router import GlobalRouterUnit
from grt.memory_analysis import analyze_stream, probe_metrics
from grt.register_analysis import RegisterObserver
from test_memory_analysis import config


@pytest.mark.parametrize('bias', [-2., 0., 2.])
def test_neutral_refinement_preserves_parameters_rng_and_initial_forward(tmp_path, bias):
    cfg = config(tmp_path)
    cfg.model.register.write_gate_bias_init = bias
    cfg.model.register.dropout_prob = .2
    cfg.model.alu.dropout = .1
    torch.manual_seed(54)
    baseline = create_model(deepcopy(cfg)).train()
    expected_rng = torch.get_rng_state().clone()
    cfg.model.register.refine_gate = True
    torch.manual_seed(54)
    refined = create_model(cfg).train()
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, refined.state_dict()[name])
    assert not any('refine_head' in name for name in baseline.state_dict())
    assert refined.router.refine_head.weight.count_nonzero() == 0
    assert refined.router.refine_head.bias.count_nonzero() == 0
    ids = torch.arange(24).reshape(2, 12) % cfg.model.vocab_size
    rng = torch.get_rng_state().clone()
    expected = baseline(ids, target_ids=ids)
    final_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual = refined(ids, target_ids=ids)
    assert torch.equal(final_rng, torch.get_rng_state())
    for attr in ('logits', 'memory_state', 'loss'):
        torch.testing.assert_close(getattr(actual, attr), getattr(expected, attr), rtol=0, atol=0)
    actual.loss.backward()
    assert refined.router.refine_head.weight.grad.abs().sum() > 0
    assert refined.router.mlp_r[-1].weight.grad is None
    optimizer = torch.optim.Adam(refined.parameters(), lr=.001)
    optimizer.step()
    assert refined.router.refine_head.weight.count_nonzero() > 0


def test_refined_write_bounds_and_dropout_mask_the_effective_gate():
    router = GlobalRouterUnit(RouterConfig(pool_nhead=2, mlp_hidden=8), 3, 8,
                              refine_gate=True)
    with torch.no_grad():
        router.mlp_w[-1].weight.zero_()
        router.mlp_w[-1].bias.copy_(torch.tensor([-2., 0., 2.]))
        router.refine_head.bias.copy_(torch.tensor([-2., 0., 2.]))
    x, state = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
    router.eval()
    _, write, _ = router(x, state)
    base = router.mlp_w[-1].bias.sigmoid()
    assert (write.squeeze(-1) >= base.square()).all()
    assert (write.squeeze(-1) <= 1-(1-base).square()).all()
    assert write[0, 0] < base[0] and write[0, 2] > base[2]
    router.train()
    router.dropout_prob = 1.
    _, write, _ = router(x, state)
    assert write.count_nonzero() == 0
    from grt.models.neox import write_back
    assert torch.equal(write_back(state, state+1, write), state)


def test_analysis_distinguishes_base_refine_and_effective_writes(tmp_path):
    cfg = config(tmp_path)
    cfg.model.register.refine_gate = True
    model = create_model(cfg)
    with torch.no_grad():
        model.router.refine_head.bias.fill_(1.5)
    observer = RegisterObserver()
    summary, arrays = analyze_stream(model, cfg, torch.arange(48) % 13,
        horizon=2, trace_examples=True, observer=observer)
    diagnostics = observer.diagnostics(model)
    internal = diagnostics['refine_gate']
    base, refine = internal['base_write_gates'], internal['raw_gates']
    effective = diagnostics['write_gates']
    np.testing.assert_allclose(effective, base+(2*refine-1)*base*(1-base), atol=1e-7)
    assert not np.allclose(effective, base)
    np.testing.assert_array_equal(effective, arrays['trace_w_gates_per_example'])
    np.testing.assert_allclose(diagnostics['gate_logits']['mean_sigmoid_slope_per_slot'],
                               (base*(1-base)).mean((0, 1)))
    # Local sensitivity reports must differentiate the two learned logits.
    logits = torch.tensor([-.7, 1.2], requires_grad=True)
    p, q = logits.sigmoid()
    result = p+(2*q-1)*p*(1-p)
    da, db = torch.autograd.grad(result, logits)[0]
    torch.testing.assert_close(da, p*(1-p)*(1+(2*q-1)*(1-2*p)))
    torch.testing.assert_close(db, 2*p*(1-p)*q*(1-q))
    metrics = probe_metrics(summary, arrays)
    assert metrics['grt/write_mean'] == pytest.approx(float(effective.mean()))
    assert metrics['grt/refine/base_write_mean'] == pytest.approx(float(base.mean()))
    assert metrics['grt/refine/refine_mean'] == pytest.approx(float(refine.mean()))
    assert metrics['grt/refine/refine_logit_derivative_mean'] > 0
    assert not model.router.refine_head._forward_hooks
    assert not model.router.mlp_w[0]._forward_pre_hooks


def test_refine_flag_is_typed_and_changes_notebook_run_identity(tmp_path):
    from grt.utils.kaggle import WikiTextRun
    root = Path(__file__).resolve().parents[1]
    jobs = [WikiTextRun(repo_dir=root, work_dir=tmp_path, models=('grt',),
        budget='20k', seed=54, num_registers=20, router_hidden=512,
        overrides={'model': {'segment_len': 75, 'register': {
            'conditional_read': False, 'write_gate_bias_init': 0.,
            'dropout_prob': 0., 'refine_gate': enabled}}}) for enabled in (False, True)]
    assert jobs[0].run_dir != jobs[1].run_dir
    for enabled, job in zip((False, True), jobs):
        cfg = job.config('grt', calibrated=False)
        assert cfg.model.register.refine_gate is enabled
        assert cfg.training.model_seed == 54 and cfg.training.max_steps == 20000
        assert cfg.training.batch_size*cfg.training.grad_accum_steps*job.world_size == 256
    with pytest.raises(ValueError, match='refine_gate'):
        WikiTextRun(repo_dir=root, work_dir=tmp_path, models=('grt',),
                    overrides={'model': {'register': {'refine_gate': 'true'}}})
