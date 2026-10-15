"""Independent controls for register grouping, geometry and joint interventions."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grt.memory_analysis import analyze_stream, token_nll
from grt.register_analysis import (RegisterObserver, analyze_groups, classify_gates,
    group_stability, intervention_forward_calls, intervention_plan, slot_cka, spectrum)
from grt.models.factory import create_model
from test_memory_analysis import config


def test_group_definition_and_threshold_sensitivity():
    gates = np.array([[[0., 1., .5, .005]], [[.001, .999, 1., .02]]])
    labels = classify_gates(gates)
    assert labels.tolist() == [0, 1, 2, 2]
    evaluation = gates.copy()
    evaluation[..., 0], evaluation[..., 1] = 1., 0.
    stats = group_stability(evaluation, labels)
    np.testing.assert_array_equal(stats['transition_from_calibration'], [[0, 1, 0], [1, 0, 0], [0, 0, 2]])
    assert stats['fraction_same_type'] == .5
    assert stats['threshold_sensitivity']['0.05']['counts']['closed'] == 2


def test_spectrum_and_cka_distinguish_constant_and_redundant_slots():
    values = torch.stack([torch.eye(2), torch.zeros(2, 2)])
    stats = spectrum(values)
    assert stats['entropy_effective_rank']['mean'] == pytest.approx(1.)
    assert stats['stable_rank']['mean'] == pytest.approx(1.)
    assert stats['zero_energy_fraction'] == .5
    x = torch.arange(12).reshape(6, 2).float()
    states = torch.stack([x, x*3+7, torch.ones_like(x)], dim=1)
    cka = slot_cka(states)
    assert cka[0, 1] == pytest.approx(1.)
    assert np.isnan(cka[2]).all()


def test_masks_are_reproducible_size_matched_and_nested():
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    groups, plans = intervention_plan(labels, fractions=(.25, .5, 1.), repeats=2)
    assert (groups, plans) == intervention_plan(labels, fractions=(.25, .5, 1.), repeats=2)
    assert plans != intervention_plan(labels, fractions=(.25, .5, 1.), repeats=2, seed=55)[1]
    for group in [p for p in plans if p['kind'] == 'gate_group' and 0 < p['size'] < 8]:
        controls = [p for p in plans if p['kind'] == 'size_matched_random' and p['size'] == group['size']]
        assert len(controls) == 2
    for repeat in range(2):
        masks = [p['slots'] for p in plans if p['kind'] == 'random_fraction' and p['repeat'] == repeat]
        assert set(masks[0]) <= set(masks[1]) <= set(masks[2])
    mask_ids = {tuple(p['slots']): p['mask'] for p in plans}
    assert len(set(mask_ids.values())) == len(mask_ids)


class RedundantMemory(torch.nn.Module):
    """Either of two slots suffices; joint reset fails, then the state recovers."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def initial_state(self, batch):
        return torch.zeros(batch, 2, 1)

    def forward(self, ids, memory_state, _trace=None):
        logits = torch.zeros(*ids.shape, 3)
        index = memory_state.amax(1).long()
        logits.scatter_(-1, index[:, None].expand(-1, ids.shape[1], -1), 6.)
        torch.rand(1)
        return SimpleNamespace(logits=logits, memory_state=ids[:, -1:, None].expand(-1, 2, -1).float())


def test_group_damage_exposes_redundancy_recovers_and_weights_tail():
    model = RedundantMemory().train()
    state = torch.tensor([[[1.], [1.]], [[2.], [2.]]])
    records = []
    for n in (4, 2, 1):
        ids = torch.tensor([[1]*n, [2]*n])
        baseline, _ = token_nll(model, ids, ids, state)
        records.append((ids, ids, state, baseline))
    _, plans = intervention_plan(np.array([0, 1]), fractions=(.5,), repeats=1)
    calls = []
    rng = torch.get_rng_state().clone()
    result = analyze_groups(model, SimpleNamespace(records=records), 2, plans, 4,
                            progress=lambda done, total: calls.append((done, total)))
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert calls[-1] == (len(calls), len(calls))
    assert len(calls) == intervention_forward_calls(plans, 2, 3, 4)
    by_name = {p['name']: result['effects'][p['mask']] for p in plans}
    assert by_name['all']['reset']['delta_nll_by_lag'][0] == pytest.approx(6.)
    assert by_name['closed']['reset']['delta_nll_by_lag'][0] == pytest.approx(0.)
    assert by_name['open']['reset']['delta_nll_by_lag'][0] == pytest.approx(0.)
    np.testing.assert_allclose(by_name['all']['swap']['delta_nll_by_lag'][:3], [6., 0., 0.], atol=1e-6)
    assert np.isnan(by_name['all']['swap']['delta_nll_by_lag'][3])
    assert by_name['all']['swap']['target_tokens_by_lag'].tolist() == [12, 6, 2, 0]
    np.testing.assert_allclose(by_name['other']['reset']['delta_nll_by_lag'][:3], 0.)


def test_observer_preserves_forward_and_captures_true_saturated_logits(tmp_path):
    cfg = config(tmp_path)
    model = create_model(cfg).train()
    with torch.no_grad():
        model.router.mlp_w[-1].weight.zero_()
        model.router.mlp_w[-1].bias.copy_(torch.tensor([-100., 100.]))
    tokens = torch.arange(48)%13
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    summary, arrays = analyze_stream(model, cfg, tokens, horizon=2, trace_examples=True)
    observer = RegisterObserver()
    observed, observed_arrays = analyze_stream(model, cfg, tokens, horizon=2,
        trace_examples=True, observer=observer)
    assert summary == observed
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], observed_arrays[name])
    diagnostics = observer.diagnostics(model)
    np.testing.assert_array_equal(diagnostics['gate_logits']['per_slot']['mean'], [-100., 100.])
    assert diagnostics['gate_logits']['fraction_exact_one'] == .5
    assert diagnostics['gate_logits']['fraction_exact_zero'] == .5
    np.testing.assert_array_equal(diagnostics['gate_logits']['mean_sigmoid_slope_per_slot'], [0., 0.])
    assert torch.equal(rng, torch.get_rng_state()) and model.training
    assert not model.router._forward_hooks and not model.router.mlp_w._forward_pre_hooks
    assert not model.router.mlp_w[-1]._forward_hooks
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
