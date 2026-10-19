import torch
from torch import nn


class SlotWriteLinear(nn.Linear):
    """Keep the existing slot-specific output rows for global or slot inputs."""
    def forward(self, hidden):
        if hidden.ndim == 2:
            return super().forward(hidden)
        return torch.einsum('bmh,mh->bm', hidden, self.weight) + self.bias


class GlobalRouterUnit(nn.Module):
    def __init__(
        self, cfg, num_registers, d_model, write_gate_bias_init=-2.0, dropout_prob=0.0,
        conditional_read=False,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.write_source = cfg.write_source
        self.write_detach_source = cfg.write_detach_source
        self.conditional_read = conditional_read
        self.num_registers = num_registers
        # The independent route retains its original modules and initialization.
        # Token pooling is independent-only; state attention is common to all routes.
        if self.write_source == 'independent' or conditional_read:
            self.q_pool = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.xavier_uniform_(self.q_pool)
            self.pool_attn = nn.MultiheadAttention(
                d_model, cfg.pool_nhead, dropout=0.0, batch_first=True
            )
        self.reg_attn = nn.MultiheadAttention(
            d_model, cfg.pool_nhead, dropout=0.0, batch_first=True
        )
        if self.write_source == 'independent' or conditional_read:
            self.mlp_r = nn.Sequential(
                nn.Linear(2 * d_model, cfg.mlp_hidden),
                nn.GELU(),
                nn.Linear(cfg.mlp_hidden, num_registers),
            )
            nn.init.zeros_(self.mlp_r[-1].bias)
        source_name = {'independent': 'token_context', 'separate_head': 'gate_head',
                       'candidate': 'candidate'}[self.write_source]
        self.input_blocks = [(source_name, d_model), ('memory_context', d_model)]
        self.mlp_w = nn.Sequential(
            nn.Linear(2 * d_model, cfg.mlp_hidden),
            nn.GELU(),
            SlotWriteLinear(cfg.mlp_hidden, num_registers),
        )
        nn.init.constant_(self.mlp_w[-1].bias, write_gate_bias_init)
        if self.write_source == 'separate_head':
            self.gate_head = nn.Linear(d_model, d_model)
            nn.init.normal_(self.gate_head.weight, std=0.02)
            nn.init.zeros_(self.gate_head.bias)

    def context(self, x, s):
        rich, alpha = self.pool_attn(
            self.q_pool.expand(x.shape[0], -1, -1), x, x, average_attn_weights=True
        )
        return self.with_state(rich, s).squeeze(1), alpha.squeeze(1)

    def with_state(self, features, s):
        context, _ = self.reg_attn(features, s, s, need_weights=False)
        return torch.cat([features, context], dim=-1)

    def read(self, s):
        if self.write_source != 'independent' and not self.conditional_read:
            return s.new_zeros(s.shape[0], self.num_registers, 1)
        read_context, _ = self.context(s, s)
        return self.mlp_r(read_context).sigmoid().unsqueeze(-1)

    def forward(self, x, s, *, features=None, read=None, return_gate_inputs=False):
        read = self.read(s) if read is None else read
        if self.write_source == 'independent':
            write_context, alpha = self.context(x, s)
        else:
            if features is None or features.shape != s.shape:
                raise ValueError("Shared write source must have shape [B,M,D]")
            # Stop the backbone edge BEFORE the optional trainable gate head.
            # The added head and state attention must still learn when detached.
            if self.write_detach_source:
                features = features.detach()
            if self.write_source == 'separate_head':
                features = self.gate_head(features)
            write_context, alpha = self.with_state(features, s), None
        if return_gate_inputs:
            hidden = self.mlp_w[1](self.mlp_w[0](write_context))
            raw_logits = self.mlp_w[-1](hidden)
        else:
            raw_logits = self.mlp_w(write_context)
        write_logits = raw_logits
        if self.training and self.dropout_prob:
            write_logits = write_logits.masked_fill(
                torch.rand_like(write_logits) < self.dropout_prob, -torch.inf
            )
        write = write_logits.sigmoid()
        if return_gate_inputs:
            hidden_norm = hidden.float().norm(dim=-1)
            if hidden_norm.ndim == 2:
                hidden_norm = hidden_norm.mean(-1)
            return read, write.unsqueeze(-1), alpha, raw_logits, hidden_norm
        return read, write.unsqueeze(-1), alpha
