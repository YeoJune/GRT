"""Checkpoint-only register diagnostics; no changes to the model or training path.

Groups are defined on a calibration window, then tested on separate windows.
Only compact statistics and intervention effects leave this module, not states.
"""
from contextlib import contextmanager
from itertools import combinations
import math

import numpy as np
import torch

from grt.checkpoint import evaluation_context
from grt.memory_analysis import token_nll


GROUP_NAMES = ('closed', 'open', 'other')


def classify_gates(gates, threshold=.01):
    gates = np.asarray(gates)
    closed = gates.max((0, 1)) <= threshold
    opened = gates.min((0, 1)) >= 1-threshold
    labels = np.full(gates.shape[-1], 2, dtype=int)
    labels[closed], labels[opened] = 0, 1
    return labels


def distribution(values, axis=None):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=values.mean(axis), std=values.std(axis),
                min=values.min(axis), max=values.max(axis))


def spectrum(values):
    """SVD of each [slots, features] state, with explicitly zero-rank handling."""
    values = values.double()
    energy = torch.linalg.svdvals(values).square()
    total = energy.sum(-1)
    p = energy / total.clamp_min(1e-30)[:, None]
    entropy_rank = torch.exp(-(p * p.clamp_min(1e-30).log()).sum(-1))
    entropy_rank = torch.where(total > 1e-30, entropy_rank, 0.)
    stable_rank = total / energy[:, 0].clamp_min(1e-30)
    rank90 = (p.cumsum(-1) < .9).sum(-1)+1
    rank90 = torch.where(total > 1e-30, rank90, 0)
    return dict(entropy_effective_rank=distribution(entropy_rank.numpy()),
                stable_rank=distribution(stable_rank.numpy()),
                dimensions_for_90pct_energy=distribution(rank90.numpy()),
                mean_normalized_squared_singular_values=p.mean(0).numpy(),
                zero_energy_fraction=float((total <= 1e-30).double().mean()))


def slot_cka(states):
    """Linear CKA across slots; observations are the same segment/example pairs.

    This is a descriptive, biased finite-sample estimator, not a significance test.
    Undefined similarities (constant slots) are NaN, exported as JSON null.
    """
    x = states.double()-states.double().mean(0, keepdim=True)
    x = x.permute(1, 0, 2)
    gram = x @ x.transpose(-1, -2)
    flat = gram.flatten(1)
    norm = flat.norm(dim=1)
    denominator = norm[:, None]*norm[None, :]
    result = (flat @ flat.T)/denominator.clamp_min(1e-30)
    result[denominator <= 1e-30] = float('nan')
    return result.numpy()


class RegisterObserver:
    """Read-only hooks, installed only during measured normal forwards."""
    def __init__(self):
        self.records = []
        self.samples = []
        self._captured = {}

    @contextmanager
    def capture(self, model):
        self._captured = {}

        def context_hook(module, args):
            self._captured['context'] = args[0].detach().cpu().clone()

        def logits_hook(module, args, output):
            self._captured['logits'] = output.detach().cpu().clone()
            self._captured['hidden_norm'] = args[0].detach().float().norm(dim=-1).cpu()

        def router_hook(module, args, output):
            self._captured['gates'] = output[1].detach().squeeze(-1).cpu().clone()

        handles = [model.router.mlp_w[0].register_forward_pre_hook(context_hook),
                   model.router.mlp_w[-1].register_forward_hook(logits_hook),
                   model.router.register_forward_hook(router_hook)]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def record(self, ids, targets, state, baseline, next_state, *, measured):
        self.records.append((ids.detach().cpu(), targets.detach().cpu(), state, baseline))
        if measured:
            self.samples.append(dict(**self._captured, state=state,
                update=next_state.detach().cpu()-state))

    def diagnostics(self, model):
        states = torch.stack([r['state'] for r in self.samples]).flatten(0, 1)
        updates = torch.stack([r['update'] for r in self.samples]).flatten(0, 1)
        initial = model.initial_state(1).detach().cpu()
        gates = torch.stack([r['gates'] for r in self.samples]).numpy()
        logits = torch.stack([r['logits'] for r in self.samples]).numpy().astype(np.float64)
        context = torch.stack([r['context'] for r in self.samples]).double()
        head = model.router.mlp_w[-1]
        bias = head.bias.detach().cpu().double().numpy()
        norms = head.weight.detach().float().norm(dim=-1).cpu().numpy()
        centered = states-states.mean(0, keepdim=True)
        result = dict(
            gate_logits=dict(raw=logits, per_slot=distribution(logits, (0, 1)),
                output_bias=bias, output_weight_row_norm=norms,
                context_contribution=distribution(logits-bias, (0, 1)),
                fraction_abs_over_10=float((np.abs(logits) > 10).mean()),
                fraction_abs_over_20=float((np.abs(logits) > 20).mean()),
                fraction_exact_zero=float((gates == 0).mean()),
                fraction_exact_one=float((gates == 1).mean()),
                mean_sigmoid_slope_per_slot=(gates*(1-gates)).mean((0, 1))),
            router_input=dict(
                context_norm=distribution(context.norm(dim=-1).numpy()),
                token_context_norm=distribution(context[..., :context.shape[-1]//2].norm(dim=-1).numpy()),
                memory_context_norm=distribution(context[..., context.shape[-1]//2:].norm(dim=-1).numpy()),
                mlp_hidden_norm=distribution(torch.stack([r['hidden_norm'] for r in self.samples]).numpy())),
            representations=dict(
                observations=states.shape[0], slots=states.shape[1], features=states.shape[2],
                scope='per-state slot x feature SVD; squared singular values define energy ranks; '
                      'actual_update=S_next-S, context centering is per slot over observations',
                raw=spectrum(states), minus_initial=spectrum(states-initial),
                context_centered=spectrum(centered), actual_update=spectrum(updates),
                context_variation_norm_per_slot=centered.double().square().mean(0).sum(-1).sqrt().numpy(),
                actual_update_norm_per_slot=updates.double().norm(dim=-1).mean(0).numpy(),
                slot_linear_cka=slot_cka(states),
                cka_scope='same segment/example observations, centered per slot; constant=null; '
                          'finite-sample bias, especially observations<features; no independent-sample claim'),
            write_gates=gates)
        return result


def group_stability(gates, calibration_labels, thresholds=(.001, .01, .05)):
    results = {}
    for threshold in thresholds:
        labels = classify_gates(gates, threshold)
        results[str(threshold)] = dict(labels=labels,
            counts={name: int((labels == i).sum()) for i, name in enumerate(GROUP_NAMES)})
    labels = classify_gates(gates)
    transition = np.zeros((3, 3), dtype=np.int64)
    np.add.at(transition, (calibration_labels, labels), 1)
    return dict(threshold_sensitivity=results, transition_from_calibration=transition,
                fraction_same_type=float((labels == calibration_labels).mean()),
                category_order=list(GROUP_NAMES))


def intervention_plan(labels, fractions=(.1, .25, .5, .75), repeats=3, seed=54):
    """Use one fixed mask set on all windows, independent of evaluation outcomes."""
    m = len(labels)
    groups = {name: np.flatnonzero(labels == i).tolist() for i, name in enumerate(GROUP_NAMES)}
    plans = []
    masks = {}

    def add(name, slots, kind, **extra):
        key = tuple(sorted(slots))
        # Empty and full group unions stay explicit, without redundant forwards.
        masks.setdefault(key, f'mask_{len(masks)}')
        plans.append(dict(name=name, mask=masks[key], slots=list(key), size=len(key), kind=kind, **extra))

    add('all', range(m), 'reference')
    add('none', [], 'reference')
    for count in range(1, 4):
        for names in combinations(GROUP_NAMES, count):
            slots = sorted(slot for name in names for slot in groups[name])
            add('+'.join(names), slots, 'gate_group', groups=list(names))
    rng = np.random.default_rng(seed)
    sizes = sorted({p['size'] for p in plans if p['kind'] == 'gate_group' and 0 < p['size'] < m})
    for size in sizes:
        for repeat in range(repeats):
            add(f'matched_{size}_{repeat}', rng.choice(m, size, replace=False).tolist(),
                'size_matched_random', repeat=repeat)
    for repeat in range(repeats):
        order = rng.permutation(m)
        for fraction in fractions:
            size = min(m, max(1, math.ceil(fraction*m)))
            add(f'fraction_{fraction:g}_{repeat}', order[:size].tolist(), 'random_fraction',
                requested_fraction=fraction, actual_fraction=size/m, repeat=repeat)
    return groups, plans


def intervention_forward_calls(plans, measured, records, horizon):
    masks = {tuple(p['slots']) for p in plans if p['size']}
    return 2*len(masks)*sum(min(max(1, horizon), records-i) for i in range(measured))


def analyze_groups(model, observer, measured, plans, horizon, progress=None):
    """Paired effects per intervention start, lag and example, token weighted.

    Each branch is independently reset/swapped once at the normal start state.
    Subsequent segments use the altered branch; unrelated starts never share it.
    """
    horizon = max(1, horizon)
    records = observer.records
    batch = records[0][0].shape[0]
    device = next(model.parameters()).device
    initial = model.initial_state(batch).detach()
    masks = {p['mask']: p['slots'] for p in plans}
    total = intervention_forward_calls(plans, measured, len(records), horizon)
    done = 0
    effects = {}
    with evaluation_context(model):
        for key, slots in masks.items():
            effects[key] = {}
            for mode in ('reset', 'swap'):
                delta = np.full((measured, horizon, batch), np.nan)
                counts = np.zeros((measured, horizon), dtype=np.int64)
                for start in range(measured):
                    normal = records[start][2].to(device)
                    branch = normal.clone()
                    replacement = initial if mode == 'reset' else normal.roll(1, 0)
                    branch[:, slots] = replacement[:, slots]
                    for lag in range(min(horizon, len(records)-start)):
                        ids, targets, _, baseline = records[start+lag]
                        if slots:
                            nll, branch = token_nll(model, ids.to(device), targets.to(device), branch)
                            done += 1
                            if progress is not None:
                                progress(done, total)
                        else:
                            nll = baseline
                        delta[start, lag] = (nll-baseline).mean(1).numpy()
                        counts[start, lag] = targets.shape[1]
                weight = counts.sum(0)*batch
                summed = np.nansum(delta*counts[..., None], axis=(0, 2))
                mean = np.divide(summed, weight, out=np.full(horizon, np.nan), where=weight > 0)
                effects[key][mode] = dict(delta_nll_by_lag=mean, target_tokens_by_lag=weight,
                    delta_nll_by_start_lag_example=delta, target_tokens_per_example=counts)
    comparisons = []
    by_name = {p['name']: p for p in plans}
    for plan in plans:
        if plan['kind'] != 'gate_group':
            continue
        controls = [p for p in plans if p['kind'] == 'size_matched_random' and p['size'] == plan['size']]
        comparison = dict(name=plan['name'], size=plan['size'])
        for mode in ('reset', 'swap'):
            group_delta = effects[plan['mask']][mode]['delta_nll_by_lag']
            all_delta = effects[by_name['all']['mask']][mode]['delta_nll_by_lag']
            complement = set(range(initial.shape[1]))-set(plan['slots'])
            complement_key = next(p['mask'] for p in plans if set(p['slots']) == complement)
            restored_delta = effects[complement_key][mode]['delta_nll_by_lag']
            individual_sum = sum((effects[by_name[name]['mask']][mode]['delta_nll_by_lag']
                                  for name in plan['groups']), np.zeros(horizon))
            comparison[mode] = dict(delta_nll_by_lag=group_delta,
                recovery_from_all_altered=all_delta-restored_delta,
                union_minus_sum_of_group_deltas=group_delta-individual_sum,
                random_control_count=len(controls))
            if controls:
                values = np.array([effects[p['mask']][mode]['delta_nll_by_lag'] for p in controls])
                # Some lags can be absent; preserve null rather than inventing zero.
                comparison[mode]['size_matched_random'] = dict(
                    mean=values.mean(0), min=values.min(0), max=values.max(0))
                comparison[mode]['group_minus_random_mean'] = group_delta-values.mean(0)
        comparisons.append(comparison)
    return dict(effects=effects, group_comparisons=comparisons, forward_calls=done,
        scope='positive=NLL intervention minus normal; reset to learned S0 or cyclic donor stream; '
              'fixed masks across windows; individual/group effects are not additive; '
              'recovery=all altered NLL minus only complement altered NLL; '
              'random range is a descriptive range, not a confidence interval; '
              'starts/lags overlap and are not independent samples')
