import torch
from torch import nn


class GlobalRouterUnit(nn.Module):
    def __init__(
        self, cfg, num_registers, d_model, write_gate_bias_init=-2.0, dropout_prob=0.0
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.q_pool = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.xavier_uniform_(self.q_pool)
        self.pool_attn = nn.MultiheadAttention(
            d_model, cfg.pool_nhead, dropout=0.0, batch_first=True
        )
        self.reg_attn = nn.MultiheadAttention(
            d_model, cfg.pool_nhead, dropout=0.0, batch_first=True
        )
        self.mlp_r = nn.Sequential(
            nn.Linear(2 * d_model, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, num_registers),
        )
        self.mlp_w = nn.Sequential(
            nn.Linear(2 * d_model, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, num_registers),
        )
        nn.init.zeros_(self.mlp_r[-1].bias)
        nn.init.constant_(self.mlp_w[-1].bias, write_gate_bias_init)

    def forward(self, x, s):
        rich, alpha = self.pool_attn(
            self.q_pool.expand(x.shape[0], -1, -1), x, x, average_attn_weights=True
        )
        context, _ = self.reg_attn(rich, s, s, need_weights=False)
        g = torch.cat([rich, context], dim=-1).squeeze(1)
        read = self.mlp_r(g).sigmoid().unsqueeze(-1)
        write_logits = self.mlp_w(g)
        if self.training and self.dropout_prob:
            write_logits = write_logits.masked_fill(
                torch.rand_like(write_logits) < self.dropout_prob, -torch.inf
            )
        return read, write_logits.sigmoid().unsqueeze(-1), alpha.squeeze(1)
