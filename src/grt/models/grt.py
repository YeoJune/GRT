import torch
from torch import nn
from grt.config import validate_model
from .interface import ModelOutput, validate_inputs
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
        self.internal_segment_length = cfg.segment_len + cfg.num_registers
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.pos_emb = nn.Parameter(torch.empty(1, cfg.segment_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.s0 = nn.Parameter(torch.zeros(1, cfg.num_registers, cfg.d_model))
        self.router = GlobalRouterUnit(cfg.router, cfg.num_registers, cfg.d_model,
                                       cfg.register.write_gate_bias_init, cfg.register.dropout_prob)
        self.alu = ALU(cfg.alu, cfg.segment_len, cfg.num_registers, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def _forward(self, input_ids, attention_mask, trace):
        validate_inputs(input_ids, attention_mask, self.cfg.segment_len, self.cfg.vocab_size)
        state = self.s0.expand(input_ids.shape[0], -1, -1)
        logits = []
        for ids in input_ids.split(self.cfg.segment_len, dim=1):
            x = self.embedding(ids) + self.pos_emb
            read, write, alpha = self.router(x, state)
            y, candidate = self.alu(x, read * state if self.cfg.register.conditional_read else state)
            if trace is not None:
                trace.record(read, write, state, candidate, alpha)
            state = write_back(state, candidate, write)
            logits.append(self.lm_head(y))
        return ModelOutput(torch.cat(logits, dim=1))

    def forward(self, input_ids, attention_mask=None):
        return self._forward(input_ids, attention_mask, None)

    def forward_with_trace(self, input_ids, attention_mask=None):
        from grt.rtla import TraceBuffer
        trace = TraceBuffer(read_active=self.cfg.register.conditional_read)
        return self._forward(input_ids, attention_mask, trace), trace
