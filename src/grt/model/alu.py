from __future__ import annotations

import torch
from torch import Tensor, nn

from grt.config import ALUConfig


class ALU(nn.Module):
    def __init__(self, cfg: ALUConfig, segment_len: int, num_registers: int, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.segment_len = segment_len
        self.num_registers = num_registers
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

    def forward(self, x: Tensor, s_masked: Tensor) -> tuple[Tensor, Tensor]:
        if x.dim() != 3 or s_masked.dim() != 3:
            raise ValueError("ALU: expected [B, N, D] and [B, M, D] inputs")

        seq = torch.cat([x, s_masked], dim=1)
        expected_len = self.segment_len + self.num_registers
        if seq.shape[1] != expected_len:
            raise ValueError(f"ALU: seq length must be N+M={expected_len}, got {seq.shape[1]}")

        out = self.encoder(seq)
        y_tokens = out[:, : self.segment_len, :]
        delta_s = out[:, self.segment_len :, :]
        return y_tokens, delta_s
