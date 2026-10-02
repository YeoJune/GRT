import torch
from torch import nn

class ALU(nn.Module):
    def __init__(self, cfg, segment_len, num_registers, d_model):
        super().__init__()
        self.segment_len = segment_len
        self.num_registers = num_registers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, cfg.nhead, cfg.d_ff, cfg.dropout,
                                       activation=cfg.activation, batch_first=True, norm_first=True)
            for _ in range(cfg.num_layers)
        ])

    def forward(self, x, s_read):
        seq = torch.cat([x, s_read], dim=1)
        if seq.shape[1] != self.segment_len + self.num_registers:
            raise ValueError("ALU input must have N+M positions")
        for layer in self.layers:
            seq = layer(seq)
        return seq[:, :self.segment_len], seq[:, self.segment_len:]
