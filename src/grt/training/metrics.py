from __future__ import annotations

import math

from torch import Tensor


def compute_ppl(loss: float | Tensor) -> float:
    value = float(loss.item() if isinstance(loss, Tensor) else loss)
    return float(math.exp(min(value, 20.0)))


def compute_throughput(num_tokens: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return float(num_tokens / duration_sec)
