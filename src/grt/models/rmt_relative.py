"""Short-Copy RMT adapted from booydar/LM-RMT (Apache-2.0).

Changed: no Transformer-XL cache/adaptive softmax; batch-first project interface,
external loss, and explicit segment/generation APIs. Relative shift, memory mask,
Post-LN and attention projection widths retain the original computations.
Source: LM-RMT e2895080868afc390386220f2b68a56c4a218126,
pytorch/mem_transformer.py. See third_party/LM-RMT-LICENSE and NOTICE.md.
"""
import math
import torch
from torch import nn
from grt.config import validate_model
from .interface import ModelOutput, validate_inputs


class RelativeLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, k = cfg.d_model, cfg.alu.nhead, cfg.head_dim
        self.nhead, self.head_dim = h, k
        self.qkv_net = nn.Linear(d, 3*h*k, bias=False)
        self.r_net = nn.Linear(d, h*k, bias=False)
        self.o_net = nn.Linear(h*k, d, bias=False)
        self.attn_norm = nn.LayerNorm(d)
        self.ff_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.alu.dropout)
        self.ff = nn.Sequential(nn.Linear(d, cfg.alu.d_ff), nn.ReLU(),
                                nn.Dropout(cfg.alu.dropout), nn.Linear(cfg.alu.d_ff, d),
                                nn.Dropout(cfg.alu.dropout))

    def forward(self, x, position, content_bias, position_bias, mask):
        # Time-major internally, matching the original relative-shift indexing.
        x = x.transpose(0, 1)
        length, batch, _ = x.shape
        q, k, v = [t.reshape(length, batch, self.nhead, self.head_dim)
                   for t in self.qkv_net(x).chunk(3, dim=-1)]
        r = self.r_net(position).reshape(length, self.nhead, self.head_dim)
        content = torch.einsum("ibhd,jbhd->ijbh", q + content_bias, k)
        relative = torch.einsum("ibhd,jhd->ijbh", q + position_bias, r)
        zero = relative.new_zeros(length, 1, batch, self.nhead)
        shifted = torch.cat([zero, relative], dim=1).reshape(length+1, length, batch, self.nhead)
        relative = shifted[1:].reshape_as(relative)
        scores = (content + relative) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask[:, :, None, None], float("-inf"))
        probability = scores.softmax(dim=1)  # Original dropatt=0.
        attended = torch.einsum("ijbh,jbhd->ibhd", probability, v)
        attended = attended.reshape(length, batch, -1)
        x = self.attn_norm(x + self.drop(self.o_net(attended)))
        x = self.ff_norm(x + self.ff(x))
        return x.transpose(0, 1)


class RelativeRMTModel(nn.Module):
    """Causal relative-position, Post-LN RMT without a Transformer-XL cache."""
    def __init__(self, cfg):
        super().__init__()
        validate_model(cfg)
        self.cfg = cfg
        self.internal_segment_length = cfg.segment_len + 2*cfg.num_registers
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.mem0 = nn.Parameter(torch.randn(1, 1, cfg.d_model).repeat(1, cfg.num_registers, 1))
        self.r_w_bias = nn.Parameter(torch.empty(cfg.alu.nhead, cfg.head_dim))
        self.r_r_bias = nn.Parameter(torch.empty(cfg.alu.nhead, cfg.head_dim))
        self.layers = nn.ModuleList([RelativeLayer(cfg) for _ in range(cfg.alu.num_layers)])
        self.drop = nn.Dropout(cfg.alu.dropout)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.r_w_bias, std=0.02)
        nn.init.normal_(self.r_r_bias, std=0.02)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, cfg.d_model, 2.0) / cfg.d_model))
        self.register_buffer("inv_freq", inv_freq)
        m, n = cfg.num_registers, cfg.segment_len
        mask = torch.ones(self.internal_segment_length, self.internal_segment_length, dtype=torch.bool).triu(1)
        mask[:m, :m] = False
        mask[m+n:, m+n:] = False
        self.register_buffer("causal_mask", mask, persistent=False)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.zeros_(module.bias)

    def initial_memory(self, batch_size):
        return self.mem0.expand(batch_size, -1, -1)

    def forward_segment(self, ids, memory):
        m, n = self.cfg.num_registers, self.cfg.segment_len
        tokens = self.embedding(ids) * math.sqrt(self.cfg.d_model)
        sequence = self.drop(torch.cat([memory, tokens, memory], dim=1))
        distance = torch.arange(self.internal_segment_length-1, -1, -1.0, device=ids.device)
        sinusoid = distance[:, None] * self.inv_freq[None, :]
        position = self.drop(torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1))
        for layer in self.layers:
            sequence = layer(sequence, position, self.r_w_bias, self.r_r_bias, self.causal_mask)
        sequence = self.drop(sequence)
        return self.lm_head(sequence[:, m:m+n]), sequence[:, m+n:]

    def forward(self, input_ids, attention_mask=None):
        validate_inputs(input_ids, attention_mask, self.cfg.segment_len, self.cfg.vocab_size)
        memory = self.initial_memory(input_ids.shape[0])
        logits = []
        for ids in input_ids.split(self.cfg.segment_len, dim=1):
            output, memory = self.forward_segment(ids, memory)
            logits.append(output)
        return ModelOutput(torch.cat(logits, dim=1))

    def generate_copy(self, source):
        """Greedy two-copy rollout; only source and start token are supplied."""
        n = self.cfg.segment_len
        if source.ndim != 2 or source.shape[1] != n:
            raise ValueError("generate_copy requires one complete source segment")
        _, memory = self.forward_segment(source, self.initial_memory(source.shape[0]))
        previous = torch.ones(source.shape[0], dtype=torch.long, device=source.device)
        answers = []
        for _ in range(2):
            ids = torch.zeros_like(source)
            ids[:, 0] = previous
            for position in range(n):
                logits, next_memory = self.forward_segment(ids, memory)
                previous = logits[:, position].argmax(dim=-1)
                answers.append(previous)
                if position+1 < n:
                    ids[:, position+1] = previous
            # Commit only the fully generated segment, never an intermediate state.
            memory = next_memory
        return torch.stack(answers, dim=1)
