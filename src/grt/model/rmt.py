from __future__ import annotations

import torch
from torch import Tensor, nn

from grt.config import ModelConfig
from grt.model.grt import GRTOutput


class RMTModel(nn.Module):
    """Recurrent Memory Transformer (Bulatov et al., 2022) — GRT 비교용 베이스라인.

    세그먼트 단위로 입력을 끊고, 각 세그먼트 앞뒤에 고정 개수의 메모리 토큰을 붙인다.
    [mem_read | tokens | mem_write] 를 트랜스포머 인코더에 통과시키고, 출력의
    뒤쪽 mem_write 블록을 다음 세그먼트의 메모리로 넘긴다(재귀). KV 캐시 없이
    고정 크기 메모리로 장문맥을 처리한다는 점은 GRT와 같지만, GRT의 read/write
    게이트·라우터·writeback 이 전혀 없는 '평범한' 재귀 메모리라는 점이 다르다.

    공정 비교를 위해 백본(층수/헤드/d_ff/dropout/activation)은 GRT의 ALU와 동일하게
    구성하고, 메모리 토큰 수는 GRT의 레지스터 수(num_registers)와 맞춘다.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.segment_len = cfg.segment_len
        # 메모리 토큰 수: rmt.num_mem_tokens 가 0이면 GRT 레지스터 수를 그대로 사용.
        self.num_mem = cfg.rmt.num_mem_tokens or cfg.num_registers

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying (GRT와 동일)

        # 세그먼트 내 위치 인코딩 (GRT와 동일). 세그먼트 간 위치는 재귀 메모리가 담당.
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.segment_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # 초기 메모리 상태(학습 가능) — GRT 의 s0 에 대응.
        self.mem0 = nn.Parameter(torch.zeros(1, self.num_mem, cfg.d_model))
        nn.init.normal_(self.mem0, std=0.02)

        # 앞(read)·뒤(write) 메모리 블록을 구분하는 마커.
        self.read_marker = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.write_marker = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.read_marker, std=0.02)
        nn.init.normal_(self.write_marker, std=0.02)

        # 백본: GRT 의 ALU 와 동일 스펙의 트랜스포머 인코더(세그먼트 내 양방향 어텐션).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.alu.nhead,
            dim_feedforward=cfg.alu.d_ff,
            dropout=cfg.alu.dropout,
            activation=cfg.alu.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.alu.num_layers)

    def init_memory(self, batch_size: int) -> Tensor:
        return self.mem0.expand(batch_size, -1, -1).clone()

    def forward(self, input_ids: Tensor, return_trace: bool = False) -> GRTOutput:
        if input_ids.dim() != 2:
            raise ValueError("RMTModel: input_ids must be [B, L]")
        if input_ids.shape[1] % self.segment_len != 0:
            raise ValueError(
                f"RMTModel: sequence length must be divisible by segment_len={self.segment_len}"
            )

        token_emb = self.embedding(input_ids)
        segments = token_emb.split(self.segment_len, dim=1)
        batch_size = input_ids.shape[0]
        mem = self.init_memory(batch_size)

        all_logits: list[Tensor] = []
        for x_t in segments:
            x_t = x_t + self.pos_emb
            seq = torch.cat(
                [mem + self.read_marker, x_t, mem + self.write_marker], dim=1
            )
            out = self.encoder(seq)
            token_out = out[:, self.num_mem : self.num_mem + self.segment_len, :]
            mem = out[:, self.num_mem + self.segment_len :, :]  # 다음 세그먼트로 재귀 전달
            all_logits.append(self.lm_head(token_out))

        logits = torch.cat(all_logits, dim=1)
        # RMT 에는 GRT 의 게이트/레지스터 트레이스가 없으므로 trace 는 항상 None.
        return GRTOutput(logits=logits, trace=None)
