from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np
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

    def _stack(self, values: list[Tensor], batch_idx: int) -> np.ndarray:
        arrays = [value[batch_idx].detach().cpu().numpy() for value in values]
        return np.stack(arrays, axis=0)

    def to_numpy(self, batch_idx: int = 0) -> dict[str, np.ndarray]:
        return {
            "r_gates": self._stack(self.r_gates, batch_idx).squeeze(-1),
            "w_gates": self._stack(self.w_gates, batch_idx).squeeze(-1),
            "s_norms": self._stack(self.s_norms, batch_idx),
            "pred_errors": self._stack(self.pred_errors, batch_idx),
            "attn_weights": self._stack(self.attn_weights, batch_idx),
        }

    def save(self, path: str, batch_idx: int = 0) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path_obj, **self.to_numpy(batch_idx=batch_idx))

    def gate_stats(self, batch_idx: int = 0) -> dict[str, np.ndarray]:
        arrays = self.to_numpy(batch_idx=batch_idx)
        mean_w = arrays["w_gates"].mean(axis=0)
        mean_r = arrays["r_gates"].mean(axis=0)
        mean_norm = arrays["s_norms"].mean(axis=0)
        mean_err = arrays["pred_errors"].mean(axis=0)
        tier = np.where(mean_w < 0.2, "long-term", np.where(mean_w < 0.7, "working", "scratch"))
        return {
            "slot_id": np.arange(len(mean_w)),
            "mean_w": mean_w,
            "mean_r": mean_r,
            "mean_norm": mean_norm,
            "mean_pred_err": mean_err,
            "tier": tier,
        }


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

        self.router = GlobalRouterUnit(
            cfg.router,
            cfg.num_registers,
            cfg.d_model,
            write_gate_bias_init=cfg.register.write_gate_bias_init,
            dropout_prob=cfg.register.dropout_prob,
        )
        self.alu = ALU(cfg.alu, cfg.segment_len, cfg.num_registers, cfg.d_model)
        self.writeback = RegisterWriteback()
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.s0 = nn.Parameter(torch.zeros(1, cfg.num_registers, cfg.d_model))

        # 세그먼트 내 위치 인코딩 (0..segment_len-1). 트랜스포머가 토큰 순서를
        # 인식하려면 필수. 세그먼트 간(전역) 위치는 레지스터 메모리가 담당.
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.segment_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # 아키텍처 개선 토글
        self.register_id = cfg.register_id
        self.segment_pos = cfg.segment_pos
        self.conditional_read = cfg.conditional_read
        if cfg.register_id:
            self.reg_emb = nn.Parameter(torch.zeros(1, cfg.num_registers, cfg.d_model))
            nn.init.normal_(self.reg_emb, std=0.02)
        if cfg.segment_pos:
            self._max_seg = 64
            self.seg_emb = nn.Parameter(torch.zeros(self._max_seg, cfg.d_model))
            nn.init.normal_(self.seg_emb, std=0.02)

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

        for seg_idx, x_t in enumerate(segments):
            x_t = x_t + self.pos_emb
            if self.segment_pos:
                x_t = x_t + self.seg_emb[min(seg_idx, self._max_seg - 1)]
            r_gate, w_gate, attn_w = self.router(x_t, s)
            s_read = r_gate * s if self.conditional_read else s
            if self.register_id:
                s_read = s_read + self.reg_emb
            y_t, delta_s = self.alu(x_t, s_read)
            s, pred_error = self.writeback(s, delta_s, w_gate)
            logits_t = self.lm_head(y_t)
            all_logits.append(logits_t)

            if trace is not None:
                trace.record(r_gate, w_gate, s, pred_error, attn_w)

        logits = torch.cat(all_logits, dim=1)
        return GRTOutput(logits=logits, trace=trace)
