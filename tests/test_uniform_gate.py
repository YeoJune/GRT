"""Uniform bias initialization must isolate the intended experimental change."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from grt.checkpoint import restore_model
from grt.config import validate_model
from grt.models.factory import create_model
from grt.utils.kaggle import WikiTextRun
from test_memory_analysis import config


@pytest.mark.parametrize('refine', [False, True])
def test_uniform_changes_only_write_bias_and_preserves_rng(tmp_path, refine):
    cfg = config(tmp_path)
    cfg.model.num_registers = 20
    cfg.model.register.refine_gate = refine
    cfg.model.register.write_gate_bias_init = 0.
    torch.manual_seed(54)
    baseline = create_model(deepcopy(cfg))
    expected_rng = torch.get_rng_state().clone()
    cfg.model.register.uniform_gate_init = True
    torch.manual_seed(54)
    uniform = create_model(cfg)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in baseline.state_dict().items():
        if name != 'router.mlp_w.2.bias':
            assert torch.equal(value, uniform.state_dict()[name]), name
    bias = uniform.router.mlp_w[-1].bias
    probability = bias.sigmoid()
    assert ((probability >= .05) & (probability <= .95)).all()
    assert probability.std() > .1 and probability.min() < .5 < probability.max()
    # Match the inverse-CDF rule with an isolated copy of the common RNG.
    generator = torch.Generator().set_state(expected_rng)
    expected = torch.empty(20).uniform_(.05, .95, generator=generator).logit()
    torch.testing.assert_close(bias, expected, rtol=0, atol=0)
    torch.manual_seed(54)
    repeated = create_model(deepcopy(cfg))
    torch.testing.assert_close(bias, repeated.router.mlp_w[-1].bias, rtol=0, atol=0)
    torch.manual_seed(55)
    different = create_model(deepcopy(cfg))
    assert not torch.equal(bias, different.router.mlp_w[-1].bias)
    ids = torch.arange(24).reshape(2, 12) % cfg.model.vocab_size
    loss = uniform(ids, target_ids=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert bias.grad.abs().sum() > 0
    assert uniform.router.mlp_w[-1].weight.grad.abs().sum() > 0
    # A trained bias is loaded verbatim, not replaced by initialization on restore.
    torch.optim.Adam(uniform.parameters(), lr=.001).step()
    restored, restored_cfg = restore_model(dict(config=asdict(cfg), model=uniform.state_dict()))
    assert restored_cfg.model.register.uniform_gate_init is True
    for name, value in uniform.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    assert sum(p.numel() for p in uniform.parameters()) == sum(p.numel() for p in baseline.parameters())


@pytest.mark.parametrize('memory', [1, 2])
def test_uniform_rejects_degenerate_truncation(tmp_path, memory):
    cfg = config(tmp_path)
    cfg.model.num_registers = memory
    cfg.model.register.uniform_gate_init = True
    with pytest.raises(ValueError, match='uniform_gate_init'):
        validate_model(cfg.model)


def test_uniform_is_typed_and_has_distinct_run_identity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    jobs = [WikiTextRun(repo_dir=root, work_dir=tmp_path, models=('grt',),
        budget='20k', seed=54, num_registers=20, router_hidden=512,
        overrides={'model': {'segment_len': 75, 'register': {
            'conditional_read': False, 'write_gate_bias_init': 0.,
            'dropout_prob': 0., 'refine_gate': False,
            'uniform_gate_init': enabled}}}) for enabled in (False, True)]
    assert jobs[0].run_dir != jobs[1].run_dir
    for enabled, job in zip((False, True), jobs):
        cfg = job.config('grt', calibrated=False)
        assert cfg.model.register.uniform_gate_init is enabled
        assert cfg.model.register.refine_gate is False
    with pytest.raises(ValueError, match='uniform_gate_init'):
        WikiTextRun(repo_dir=root, work_dir=tmp_path, models=('grt',),
            overrides={'model': {'register': {'uniform_gate_init': 'true'}}})
