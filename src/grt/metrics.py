from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

def masked_ce(logits, labels, reduction="mean"):
    if labels.shape != logits.shape[:-1] or labels.dtype != torch.int64:
        raise ValueError("labels must be int64 and match logit positions")
    mask = labels != -100
    if not mask.any():
        raise ValueError("A batch must have supervised labels")
    return F.cross_entropy(logits[mask].float(), labels[mask], reduction=reduction)

@dataclass
class MetricAccumulator:
    nll: float = 0.0
    tokens: int = 0
    correct: int = 0
    samples: int = 0
    exact: int = 0

    def update(self, logits, labels):
        with torch.no_grad():
            mask = labels != -100
            if not mask.any(dim=1).all():
                raise ValueError("Every sample needs at least one supervised label")
            self.nll += masked_ce(logits, labels, "sum").item()
            match = logits.argmax(-1) == labels
            self.tokens += mask.sum().item()
            self.correct += (match & mask).sum().item()
            self.samples += labels.shape[0]
            self.exact += (match | ~mask).all(dim=1).sum().item()

    def compute(self):
        if not self.tokens or not self.samples:
            raise ValueError("Cannot evaluate an empty dataset")
        ce = self.nll / self.tokens
        return {"loss": ce, "token_accuracy": self.correct / self.tokens,
                "exact_match": self.exact / self.samples,
                "perplexity": math.exp(ce) if ce < 709 else None,
                "samples": self.samples, "supervised_tokens": self.tokens}
