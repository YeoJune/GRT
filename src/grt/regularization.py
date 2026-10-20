"""Write-logit tail penalty, separate from task likelihood and evaluation."""
import torch


def supervised_positions(labels, labels_mask, target_ids, objective):
    if target_ids is not None:
        return target_ids.ne(-100)
    if objective == 'parallel':
        return labels_mask & labels.ne(-100)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    valid[:, :-1] = labels[:, 1:].ne(-100)
    if labels_mask is not None:
        valid[:, :-1] &= labels_mask[:, :-1]
    return valid


def write_logit_regularization(task_loss, observations, valid, segment_len, threshold, coef):
    """Token-weighted slot means; masked BPTT history contributes zero weight.

    Observations contain raw logits BEFORE write dropout.
    No per-run scale normalization or M/L factor is applied.
    """
    logits = torch.stack(observations, dim=1).float()
    counts = torch.stack([part.sum(-1) for part in valid.split(segment_len, 1)], dim=1)
    weights = counts.to(logits.dtype) / counts.sum()
    absolute = logits.abs()
    penalty = (weights * (absolute - threshold).clamp_min(0).square().mean(-1)).sum()
    weighted = coef * penalty
    total = task_loss + weighted

    def average(values):
        return (weights * values.mean(-1)).sum().detach()

    gates = logits.detach().sigmoid()
    metrics = {
        'train/loss': task_loss.detach(),
        'train/total_loss': total.detach(),
        'train/gate_logit_penalty': penalty.detach(),
        'train/gate_regularization_loss': weighted.detach(),
        'train/gate_fraction_abs_over20': average((absolute.detach() > 20).float()),
        'train/gate_raw_write_mean': average(gates),
    }
    return total, metrics
