"""GRT router/ALU/write-back with the same forward/generate contract as RMT."""

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from grt.config import validate_model
from grt.metrics import causal_loss
from .alu import ALU
from .router import GlobalRouterUnit


def write_back(state, candidate, write):
    return (1 - write) * state + write * candidate


class GRTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        validate_model(cfg)
        if cfg.name != "grt":
            raise ValueError("GRTModel requires name=grt")
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.pos_emb = nn.Parameter(
            torch.empty(1, cfg.position_capacity or cfg.segment_len, cfg.d_model)
        )
        nn.init.normal_(self.pos_emb, std=0.02)
        self.s0 = nn.Parameter(torch.zeros(1, cfg.num_registers, cfg.d_model))
        self.router = GlobalRouterUnit(
            cfg.router,
            cfg.num_registers,
            cfg.d_model,
            cfg.register.write_gate_bias_init,
            cfg.register.dropout_prob,
        )
        self.alu = ALU(cfg.alu, cfg.segment_len, cfg.num_registers, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def _validate(self, ids, mask):
        if (
            ids.ndim != 2
            or min(ids.shape) <= 0
            or ids.dtype != torch.long
            or (ids < 0).any()
            or (ids >= self.cfg.vocab_size).any()
        ):
            raise ValueError("Expected nonempty int64 [B,L] token IDs")
        if mask is not None and (
            mask.shape != ids.shape
            or mask.dtype != torch.bool
            or mask.device != ids.device
            or not mask.all()
        ):
            raise ValueError(
                "This experiment supports unpadded bool attention masks only"
            )

    def _cell(self, ids, state):
        if ids.shape[1] > self.cfg.segment_len:
            raise ValueError("Query prefix exceeds configured segment_len")
        x = self.embedding(ids) + self.pos_emb[:, : ids.shape[1]]
        read, write, alpha = self.router(x, state)
        y, candidate = self.alu(
            x, read * state if self.cfg.register.conditional_read else state
        )
        return self.lm_head(y), candidate, read, write, alpha

    def _forward(self, input_ids, attention_mask, labels, labels_mask, trace=None):
        self._validate(input_ids, attention_mask)
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels shape mismatch")
        if labels_mask is not None and (
            labels_mask.shape != input_ids.shape or labels_mask.dtype != torch.bool
        ):
            raise ValueError("labels_mask must be bool [B,L]")
        state = self.s0.expand(input_ids.shape[0], -1, -1)
        all_logits = []
        segments = list(input_ids.split(self.cfg.segment_len, 1))
        offset = 0
        supervised_positions = (
            labels_mask.any(0).tolist()
            if labels is not None and labels_mask is not None
            else None
        )
        for index, ids in enumerate(segments):
            before = state
            # Only a completed segment whose successor exists commits a memory update.
            if index < len(segments) - 1:
                logits, candidate, read, write, alpha = self._cell(ids, before)
                if trace is not None:
                    trace.record(read, write, before, candidate, alpha)
                state = write_back(before, candidate, write)
            else:
                logits = self.embedding.weight.new_zeros(
                    (ids.shape[0], ids.shape[1], self.cfg.vocab_size)
                )
            if supervised_positions is None:
                needed = [True] * ids.shape[1]
            else:
                needed = supervised_positions[offset : offset + ids.shape[1]]
                if index == len(segments) - 1:
                    needed[-1] = False
            # Pooling and ALU both receive only the available prefix. No target can leak
            # through a global router or bidirectional attention into an earlier prediction.
            pieces = []
            for position in range(ids.shape[1]):
                if needed[position]:
                    prediction = self._cell(ids[:, : position + 1], before)[0][:, -1:]
                else:
                    prediction = logits[:, position : position + 1]
                pieces.append(prediction)
            all_logits.append(torch.cat(pieces, 1))
            offset += ids.shape[1]
        logits = torch.cat(all_logits, 1)
        loss = causal_loss(logits, labels, labels_mask) if labels is not None else None
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)

    def forward(self, input_ids, attention_mask=None, labels=None, labels_mask=None):
        return self._forward(input_ids, attention_mask, labels, labels_mask)

    def forward_with_trace(
        self, input_ids, attention_mask=None, labels=None, labels_mask=None
    ):
        from grt.rtla import TraceBuffer

        trace = TraceBuffer(read_active=self.cfg.register.conditional_read)
        return self._forward(
            input_ids, attention_mask, labels, labels_mask, trace
        ), trace

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=2,
        pad_token_id=0,
        eos_token_id=None,
    ):
        self._validate(input_ids, attention_mask)
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        boundary = input_ids.shape[1] // self.cfg.segment_len * self.cfg.segment_len
        if boundary == input_ids.shape[1]:
            raise ValueError("Generation requires an incomplete final segment")
        state = self.s0.expand(input_ids.shape[0], -1, -1)
        if boundary:
            for ids in input_ids[:, :boundary].split(self.cfg.segment_len, 1):
                _, candidate, _, write, _ = self._cell(ids, state)
                state = write_back(state, candidate, write)
        prefix = input_ids[:, boundary:]
        tokens = []
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        for _ in range(max_new_tokens):
            token = self._cell(prefix, state)[0][:, -1].argmax(-1)
            token = token.masked_fill(finished, pad_token_id)
            tokens.append(token)
            if eos_token_id is not None:
                finished |= token.eq(eos_token_id)
            if finished.all():
                break
            prefix = torch.cat([prefix, token[:, None]], 1)
        return torch.stack(tokens, 1)
