from __future__ import annotations

import torch

from grt.config import GRTConfig


class SyntheticBatchStream:
    def __init__(
        self,
        cfg: GRTConfig,
        num_batches: int,
        batch_size: int | None = None,
        seq_len: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.num_batches = num_batches
        self.batch_size = batch_size or cfg.training.batch_size
        self.seq_len = seq_len or cfg.data.seq_len
        self.device = device or "cpu"

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self._make_batch()

    def _make_batch(self) -> dict[str, torch.Tensor]:
        input_ids = torch.randint(
            0,
            self.cfg.model.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
        )
        if self.cfg.data.task == "synthetic_copy":
            labels = input_ids.clone()
        else:
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            labels[:, -1] = -100
        return {"input_ids": input_ids, "labels": labels}


def build_synthetic_batches(
    cfg: GRTConfig,
    num_batches: int,
    batch_size: int | None = None,
    seq_len: int | None = None,
    device: torch.device | str | None = None,
) -> SyntheticBatchStream:
    return SyntheticBatchStream(cfg, num_batches, batch_size=batch_size, seq_len=seq_len, device=device)


def build_dataloaders(cfg: GRTConfig, device: torch.device | str | None = None):
    train_batches = build_synthetic_batches(cfg, num_batches=max(cfg.training.max_steps, 1), device=device)
    eval_batches = list(build_synthetic_batches(cfg, num_batches=8, device=device))
    return train_batches, eval_batches
