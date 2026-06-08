from __future__ import annotations

import torch
from torch import Tensor, nn

from grt.config import RouterConfig


class GateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GlobalRouterUnit(nn.Module):
    def __init__(self, cfg: RouterConfig, num_registers: int, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_registers = num_registers
        self.d_model = d_model

        self.q_pool = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.xavier_uniform_(self.q_pool)

        self.pool_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=cfg.pool_nhead, batch_first=True)
        self.reg_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=cfg.pool_nhead, batch_first=True)
        self.mlp_r = GateMLP(2 * d_model, cfg.mlp_hidden, num_registers)
        self.mlp_w = GateMLP(2 * d_model, cfg.mlp_hidden, num_registers)
        nn.init.zeros_(self.mlp_r.net[-1].bias)
        nn.init.constant_(self.mlp_w.net[-1].bias, -2.0)

    def forward(self, x: Tensor, s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.dim() != 3 or s.dim() != 3:
            raise ValueError("GlobalRouter: expected [B, N, D] and [B, M, D] inputs")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"GlobalRouter: x shape mismatch, got {x.shape}")
        if s.shape != (x.shape[0], self.num_registers, self.d_model):
            raise ValueError(f"GlobalRouter: s shape mismatch, got {s.shape}")

        batch_size, seq_len, _ = x.shape
        if self.cfg.mode == "mean_pool":
            x_rich = x.mean(dim=1, keepdim=True)
            attn_w = torch.full((batch_size, seq_len), 1.0 / seq_len, device=x.device, dtype=x.dtype)
        else:
            query = self.q_pool.expand(batch_size, -1, -1)
            x_rich, attn_w = self.pool_attn(query, x, x, need_weights=True, average_attn_weights=True)
            attn_w = attn_w.squeeze(1)

        context, _ = self.reg_attn(x_rich, s, s, need_weights=False)
        router_context = torch.cat([x_rich, context], dim=-1).squeeze(1)
        read_logit = self.mlp_r(router_context)
        write_logit = self.mlp_w(router_context)

        r_gate = torch.sigmoid(read_logit).unsqueeze(-1)
        if self.training and getattr(self.cfg, "dropout_prob", 0.0) > 0:
            dropout_mask = torch.bernoulli(torch.full_like(write_logit, self.cfg.dropout_prob)).bool()
            write_logit = write_logit.masked_fill(dropout_mask, float("-inf"))
        w_gate = torch.sigmoid(write_logit).unsqueeze(-1)
        return r_gate, w_gate, attn_w
