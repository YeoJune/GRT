"""The author's predictor-mask convention, shared by local model implementations."""

import torch.nn.functional as F


def causal_loss(logits, labels, labels_mask=None):
    predictions = logits[:, :-1].reshape(-1, logits.shape[-1])
    targets = labels[:, 1:].reshape(-1)
    if labels_mask is not None:
        mask = labels_mask[:, :-1].reshape(-1)
        predictions = predictions[mask]
        targets = targets[mask]
    if not targets.numel():
        raise ValueError("No supervised next-token targets")
    return F.cross_entropy(predictions, targets)
