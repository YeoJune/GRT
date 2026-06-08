from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from grt.config import ModelConfig
from grt.model.alu import ALU
from grt.model.router import GlobalRouterUnit
from grt.model.writeback import RegisterWriteback


@dataclass
class TraceBuffer:
    r_gates: list[Tensor] = field(default_factory=list)
    w_gates: list[Tensor] = field(default_factory=list)
    s_norms: list[Tensor] = field(default_factory=list)
    pred_errors: list[Tensor] = field(default_factory=list)
    attn_weights: list[Tensor] = field(default_factory=list)

    def record(self, r_gate: Tensor, w_gate: Tensor, s: Tensor, pred_error: Tensor, attn_w: Tensor) -> None:
        self.r_gates.append(r_gate.detach().cpu())
        self.w_gates.append(w_gate.detach().cpu())
        self.s_norms.append(s.detach().norm(dim=-1).cpu())
        self.pred_errors.append(pred_error.detach().cpu())
        self.attn_weights.append(attn_w.detach().cpu())


@dataclass
class GRTOutput:
    logits: Tensor
    loss: Tensor | None = None
    trace: TraceBuffer | None = None


class GRTModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.router = GlobalRouterUnit(cfg.router, cfg.num_registers, cfg.d_model)
        self.alu = ALU(cfg.alu, cfg.segment_len, cfg.num_registers, cfg.d_model)
        self.writeback = RegisterWriteback()
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.s0 = nn.Parameter(torch.zeros(1, cfg.num_registers, cfg.d_model))

    def init_registers(self, batch_size: int) -> Tensor:
        if self.cfg.register.s0_learnable:
            return self.s0.expand(batch_size, -1, -1).clone()
        return torch.zeros(batch_size, self.cfg.num_registers, self.cfg.d_model, device=self.s0.device)

    def forward(self, input_ids: Tensor, return_trace: bool = False) -> GRTOutput:
        if input_ids.dim() != 2:
            raise ValueError("GRTModel: input_ids must be [B, L]")
        if input_ids.shape[1] % self.cfg.segment_len != 0:
            raise ValueError(
                f"GRTModel: sequence length must be divisible by segment_len={self.cfg.segment_len}"
            )

        token_emb = self.embedding(input_ids)
        segments = token_emb.split(self.cfg.segment_len, dim=1)
        batch_size = input_ids.shape[0]
        s = self.init_registers(batch_size)

        all_logits: list[Tensor] = []
        trace = TraceBuffer() if return_trace else None

        for x_t in segments:
            r_gate, w_gate, attn_w = self.router(x_t, s)
            s_masked = r_gate * s
            y_t, delta_s = self.alu(x_t, s_masked)
            s, pred_error = self.writeback(s, delta_s, w_gate)
            logits_t = self.lm_head(y_t)
            all_logits.append(logits_t)

            if trace is not None:
                trace.record(r_gate, w_gate, s, pred_error, attn_w)

        logits = torch.cat(all_logits, dim=1)
        return GRTOutput(logits=logits, trace=trace)
