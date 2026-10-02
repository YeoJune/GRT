import torch
from torch import nn
from grt.config import validate_model
from .interface import ModelOutput, validate_inputs

class RMTModel(nn.Module):
    """Independent bidirectional RMT for synthetic recovery."""
    def __init__(self, cfg):
        super().__init__()
        validate_model(cfg)
        if cfg.name != "rmt":
            raise ValueError("RMTModel requires name=rmt")
        self.cfg = cfg
        self.internal_segment_length = cfg.segment_len + 2 * cfg.num_registers
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.empty(1, cfg.segment_len, cfg.d_model))
        self.mem0 = nn.Parameter(torch.empty(1, cfg.num_registers, cfg.d_model))
        self.read_marker = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.write_marker = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        for p in (self.embedding.weight, self.pos_emb, self.mem0, self.read_marker, self.write_marker):
            nn.init.normal_(p, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(cfg.d_model, cfg.alu.nhead, cfg.alu.d_ff, cfg.alu.dropout,
                                       activation=cfg.alu.activation, batch_first=True, norm_first=True)
            for _ in range(cfg.alu.num_layers)
        ])
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, attention_mask=None):
        validate_inputs(input_ids, attention_mask, self.cfg.segment_len, self.cfg.vocab_size)
        m, n = self.cfg.num_registers, self.cfg.segment_len
        mem = self.mem0.expand(input_ids.shape[0], -1, -1)
        logits = []
        for ids in input_ids.split(n, dim=1):
            x = self.embedding(ids) + self.pos_emb
            seq = torch.cat([mem + self.read_marker, x, mem + self.write_marker], dim=1)
            for layer in self.layers:
                seq = layer(seq)
            mem = seq[:, m+n:m+n+m]
            logits.append(self.lm_head(seq[:, m:m+n]))
        return ModelOutput(torch.cat(logits, dim=1))
