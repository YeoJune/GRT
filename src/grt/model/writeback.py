from __future__ import annotations

import torch
from torch import Tensor, nn


class RegisterWriteback(nn.Module):
    def forward(self, s: Tensor, delta_s: Tensor, w_gate: Tensor) -> tuple[Tensor, Tensor]:
        if w_gate.shape != (s.shape[0], s.shape[1], 1):
            raise ValueError(f"Writeback: w_gate shape mismatch, got {w_gate.shape}")
        if (w_gate < 0).any() or (w_gate > 1).any():
            raise ValueError("Writeback: w_gate must be in [0, 1]")

        s_next = (1.0 - w_gate) * s + w_gate * delta_s
        pred_error = torch.norm(delta_s - s, dim=-1)
        return s_next, pred_error
