"""Shared NeoX blocks with explicit memory masks."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPTNeoXConfig
from transformers.utils import ModelOutput
from grt.vendor.armt.modeling_gpt_neox import (
    GPTNeoXAttention, GPTNeoXLayer, apply_rotary_pos_emb,
)
from .router import GlobalRouterUnit
from grt.metrics import causal_loss
from grt.regularization import supervised_positions, write_logit_regularization


def write_back(state, candidate, write):
    return (1 - write) * state + write * candidate


@dataclass
class RecurrentOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    memory_state: torch.Tensor | None = None
    training_metrics: dict[str, torch.Tensor] | None = None


class MemoryAttention(GPTNeoXAttention):
    def attend(self, query, key, value, mask):
        scores = query @ key.transpose(-1, -2) / self.norm_factor
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.attention_dropout(scores.float().softmax(-1).to(value.dtype))
        return weights @ value

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                head_mask=None, layer_past=None, use_cache=False,
                output_attentions=False, read_state=None, token_length=0, rotary_length=None):
        if layer_past is not None or use_cache:
            raise ValueError("MemoryAttention uses explicit segment state, not KV cache")
        b, length, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states).view(
            b, length, self.num_attention_heads, 3 * self.head_size)
        q, k, v = [x.permute(0, 2, 1, 3) for x in qkv.split(self.head_size, -1)]
        cos, sin = self.rotary_emb(v, seq_len=rotary_length or length)
        qr, kr = apply_rotary_pos_emb(q[..., :self.rotary_ndims],
            k[..., :self.rotary_ndims], cos, sin, position_ids)
        q = torch.cat([qr, q[..., self.rotary_ndims:]], -1)
        k = torch.cat([kr, k[..., self.rotary_ndims:]], -1)
        if read_state is None:  # Plain decoder or RMT read/write blocks.
            attended = self.attend(q, k, v, attention_mask)
        else:
            fixed = self.query_key_value(read_state).view(
                b, read_state.shape[1], self.num_attention_heads, 3 * self.head_size)
            _, fk, fv = [x.permute(0, 2, 1, 3) for x in fixed.split(self.head_size, -1)]
            # Apply the same positional rotation to fixed-source register keys.
            dummy = torch.zeros_like(fk[..., :self.rotary_ndims])
            _, fkr = apply_rotary_pos_emb(dummy, fk[..., :self.rotary_ndims],
                cos, sin, position_ids[:, token_length:])
            fk = torch.cat([fkr, fk[..., self.rotary_ndims:]], -1)
            fixed_keys = torch.cat([k[:, :, :token_length], fk], 2)
            fixed_values = torch.cat([v[:, :, :token_length], fv], 2)
            # Token K/V always read the segment-start state. Candidate hidden
            # states can collect the whole segment but are never read by tokens.
            tokens = self.attend(q[:, :, :token_length], fixed_keys, fixed_values,
                                 attention_mask[:, :, :token_length])
            candidates = self.attend(q[:, :, token_length:], k, v,
                                     attention_mask[:, :, token_length:])
            attended = torch.cat([tokens, candidates], 2)
        out = attended.transpose(1, 2).reshape(b, length, self.hidden_size)
        return (self.dense(out), None)


class NeoXBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        config = GPTNeoXConfig(hidden_size=m.d_model, num_hidden_layers=m.alu.num_layers,
            num_attention_heads=m.alu.nhead, intermediate_size=m.alu.d_ff,
            hidden_act=m.alu.activation, hidden_dropout=m.alu.dropout,
            attention_dropout=m.attention_dropout, rotary_pct=m.rotary_pct,
            rotary_emb_base=m.rotary_base, max_position_embeddings=2048,
            layer_norm_eps=1e-5, use_parallel_residual=True)
        self.embedding = nn.Embedding(m.vocab_size, m.d_model)
        self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(m.alu.num_layers)])
        for layer in self.layers:
            layer.attention = MemoryAttention(config)
        self.final_norm = nn.LayerNorm(m.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(m.alu.dropout)
        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask, read_state=None, token_length=0,
                position_ids=None, rotary_length=None, return_register_hidden=False):
        h = self.dropout(x)
        positions = torch.arange(h.shape[1], device=h.device)[None] if position_ids is None else position_ids
        for layer in self.layers:
            attn = layer.attention(layer.input_layernorm(h), attention_mask=mask,
                position_ids=positions,
                read_state=layer.input_layernorm(read_state) if read_state is not None else None,
                token_length=token_length, rotary_length=rotary_length)[0]
            # NeoX parallel residual: both branches see the same layer input.
            h = h + layer.post_attention_dropout(attn) + layer.post_mlp_dropout(
                layer.mlp(layer.post_attention_layernorm(h)))
        output = self.final_norm(h)
        # A separate gate head branches from the SAME input to the existing
        # final output path, not from the already-produced candidate.
        return (output, h[:, token_length:]) if return_register_hidden else output


class RecurrentNeoX(nn.Module):
    trace_scope = "all segment write-backs, including masked query; once after the stack"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.objective = cfg.data.options.get("objective", "next_token")
        self.backbone = NeoXBackbone(self.cfg)
        m = self.cfg
        if m.name != 'neox':
            self.initial_memory = nn.Parameter(torch.zeros(1, m.num_registers, m.d_model))
        if m.name == "rmt":
            nn.init.normal_(self.initial_memory, std=0.02)
        elif m.name == 'grt':
            self.router = GlobalRouterUnit(m.router, m.num_registers, m.d_model,
                m.register.write_gate_bias_init, m.register.dropout_prob,
                m.register.conditional_read)

    def initial_state(self, batch_size):
        if self.cfg.name == 'neox':
            return self.backbone.embedding.weight.new_empty(batch_size, 0, self.cfg.d_model)
        return self.initial_memory.expand(batch_size, -1, -1)

    def attention_mask(self, n, device):
        m = self.cfg.num_registers
        rmt = self.cfg.name == "rmt"
        length = n + (2 * m if rmt else m)
        mask = torch.ones(length, length, dtype=torch.bool, device=device)
        if self.cfg.attention_mode == "decoder":
            start = m if rmt else 0
            mask[start:start+n, start:start+n] = torch.ones(n, n,
                dtype=torch.bool, device=device).tril()
            if rmt:
                # Read block cannot see this segment; tokens cannot see write block.
                mask[:m, m:] = False
                mask[m:m+n, m+n:] = False
        return mask[None, None]

    def cell(self, ids, state, trace=None, gate_observations=None):
        if not 0 < ids.shape[1] <= self.cfg.segment_len:
            raise ValueError("Cell length must be within segment_len")
        x = self.backbone.embedding(ids)
        n, m = ids.shape[1], self.cfg.num_registers
        if self.cfg.name == 'neox':
            y = self.backbone(x, self.attention_mask(n, ids.device))
            next_state = state
        elif self.cfg.name == "rmt":
            h = self.backbone(torch.cat([state, x, state], 1),
                self.attention_mask(n, ids.device))
            y, candidate = h[:, m:m+n], h[:, -m:]
            next_state = candidate
        else:
            shared = self.cfg.router.write_source != 'independent'
            separate_head = self.cfg.router.write_source == 'separate_head'
            if shared:
                read = self.router.read(state)
            else:
                routed = self.router(x, state, return_gate_inputs=gate_observations is not None)
                read, write, alpha = routed[:3]
            source = read * state if self.cfg.register.conditional_read else state
            # Reserve memory positions after a full segment even for a shorter prefix.
            # Otherwise future sequence length would change token-to-memory RoPE.
            positions = torch.cat([torch.arange(n, device=ids.device),
                torch.arange(self.cfg.segment_len, self.cfg.segment_len+m, device=ids.device)])[None]
            h = self.backbone(torch.cat([x, source], 1),
                self.attention_mask(n, ids.device), source, n, positions, self.cfg.segment_len+m,
                return_register_hidden=separate_head)
            if separate_head:
                h, register_hidden = h
            y, candidate = h[:, :n], h[:, n:]
            if shared:
                features = register_hidden if separate_head else candidate
                routed = self.router(x, state, features=features, read=read,
                    return_gate_inputs=gate_observations is not None)
                read, write, alpha = routed[:3]
            if gate_observations is not None:
                gate_observations.append(routed[3:])
            next_state = write_back(state, candidate, write)
            if trace is not None:
                trace.record(read, write, state, candidate, alpha)
        return y, next_state

    def forward(self, input_ids, attention_mask=None, labels=None, labels_mask=None,
                memory_state=None, target_ids=None, _trace=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("Expected int64 [B,L] IDs")
        if not input_ids.numel():
            raise ValueError("Expected a nonempty token batch")
        if attention_mask is not None and (attention_mask.shape != input_ids.shape
                or attention_mask.dtype != torch.bool or not attention_mask.all()):
            raise ValueError("Explicit stream batches are unpadded")
        if labels is not None and target_ids is not None:
            raise ValueError("Choose raw labels or adapter-aligned target_ids")
        state = self.initial_state(input_ids.shape[0]) if memory_state is None else memory_state
        if state.shape != (input_ids.shape[0], self.cfg.num_registers, self.cfg.d_model):
            raise ValueError("Invalid segment memory state shape")
        regularizing = (self.cfg.name == 'grt' and self.training and torch.is_grad_enabled()
            and self.cfg.register.write_logit_coef > 0 and (labels is not None or target_ids is not None))
        observations = [] if regularizing else None
        hidden = []
        for segment in input_ids.split(self.cfg.segment_len, 1):
            y, state = self.cell(segment, state, _trace, observations)
            hidden.append(y)
        combined = torch.cat(hidden, 1)
        if target_ids is not None:
            if target_ids.shape != input_ids.shape or target_ids.dtype != torch.long:
                raise ValueError("Aligned targets must be int64 [B,L]")
            valid = target_ids != -100
            if not valid.any():
                raise ValueError("At least one aligned target is required")
            logits = self.backbone.lm_head(combined[valid])
            loss = F.cross_entropy(logits, target_ids[valid])
            return self.training_output(loss, state, observations, labels, labels_mask, target_ids)
        logits = self.backbone.lm_head(combined)
        loss = None
        if labels is not None:
            if self.objective == "parallel":
                if labels_mask is None or not labels_mask.any():
                    raise ValueError("Parallel targets require aligned slot mask")
                loss = F.cross_entropy(logits[labels_mask], labels[labels_mask])
            else:
                loss = causal_loss(logits, labels, labels_mask)
        return self.training_output(loss, state, observations, labels, labels_mask, target_ids, logits)

    def training_output(self, loss, state, observations, labels, labels_mask, target_ids, logits=None):
        metrics = None
        if observations is not None:
            valid = supervised_positions(labels, labels_mask, target_ids, self.objective)
            r = self.cfg.register
            loss, metrics = write_logit_regularization(loss, observations, valid,
                self.cfg.segment_len, r.write_logit_threshold, r.write_logit_coef)
        return RecurrentOutput(loss=loss, logits=logits, memory_state=state, training_metrics=metrics)

    def forward_with_trace(self, **kwargs):
        if self.cfg.name != "grt":
            raise ValueError("RMT has no gate trace")
        from grt.rtla import TraceBuffer
        trace = TraceBuffer(read_active=self.cfg.register.conditional_read)
        return self(**kwargs, _trace=trace), trace

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=2,
                 pad_token_id=0, eos_token_id=None):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.objective == "parallel":
            return self(input_ids, attention_mask).logits[:, -max_new_tokens:].argmax(-1)
        # Slow reference generation without a KV cache; no per-position training reruns.
        if self.cfg.attention_mode != "decoder":
            raise ValueError("Encoder generation requires a parallel-slot objective")
        prompt = input_ids
        result = []
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            token = self(prompt).logits[:, -1].argmax(-1)
            token = torch.where(finished, pad_token_id, token)
            result.append(token)
            if eos_token_id is not None:
                finished |= token == eos_token_id
            prompt = torch.cat([prompt, token[:, None]], 1)
        return torch.stack(result, 1)
