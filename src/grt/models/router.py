import torch
from torch import nn


class ZeroInitLinear(nn.Linear):
    """A neutral output head that does not consume initialization RNG."""
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class GlobalRouterUnit(nn.Module):
    def __init__(
        self, cfg, num_registers, d_model, write_gate_bias_init=-2.0, dropout_prob=0.0,
        refine_gate=False, uniform_gate_init=False,
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
        self.refine_head = ZeroInitLinear(cfg.mlp_hidden, num_registers) if refine_gate else None
        if uniform_gate_init:
            if num_registers < 3:
                raise ValueError("uniform_gate_init requires at least 3 registers (epsilon=1/M)")
            # Gu et al. (ICML 2020), Listing 2: uniform activations, not logits.
            # Only biases change; preserve common weights and subsequent training RNG.
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                epsilon = 1.0 / num_registers
                probabilities = torch.empty(num_registers, device='cpu').uniform_(
                    epsilon, 1 - epsilon)
                self.mlp_w[-1].bias.copy_(probabilities.logit())

    def context(self, x, s):
        rich, alpha = self.pool_attn(
            self.q_pool.expand(x.shape[0], -1, -1), x, x, average_attn_weights=True
        )
        context, _ = self.reg_attn(rich, s, s, need_weights=False)
        g = torch.cat([rich, context], dim=-1).squeeze(1)
        return g, alpha.squeeze(1)

    def forward(self, x, s, trace=None):
        read_context, _ = self.context(s, s)
        write_context, alpha = self.context(x, s)
        read = self.mlp_r(read_context).sigmoid().unsqueeze(-1)
        if self.refine_head is None:
            write_logits = self.mlp_w(write_context)
            if self.training and self.dropout_prob:
                write_logits = write_logits.masked_fill(
                    torch.rand_like(write_logits) < self.dropout_prob, -torch.inf
                )
            write = write_logits.sigmoid()
        else:
            hidden = self.mlp_w[1](self.mlp_w[0](write_context))
            write_logits = self.mlp_w[-1](hidden)
            refine_logits = self.refine_head(hidden)
            base, refine = write_logits.sigmoid(), refine_logits.sigmoid()
            # Gu et al. (ICML 2020), Eq. 13, expressed as a write gate.
            write = base + (2 * refine - 1) * base * (1 - base)
            if trace is not None:
                trace.record_refine(base, refine, write_logits, refine_logits)
            if self.training and self.dropout_prob:
                write = write.masked_fill(torch.rand_like(write) < self.dropout_prob, 0.)
        return read, write.unsqueeze(-1), alpha
