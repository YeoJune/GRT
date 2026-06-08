from __future__ import annotations

import torch

from grt.config import GRTConfig


def _build_passkey_retrieval_batch(cfg: GRTConfig) -> dict[str, torch.Tensor]:
    batch_size = cfg.training.batch_size
    seq_len = cfg.data.seq_len
    vocab_size = cfg.model.vocab_size

    if vocab_size < 8:
        raise ValueError("passkey_retrieval requires a vocabulary large enough for special tokens")

    query_token_id = vocab_size - 1
    sep_token_id = vocab_size - 2
    content_max_id = vocab_size - 3

    facts_tokens = cfg.data.num_facts * 3
    query_tokens = cfg.data.num_queries * 3
    if seq_len < facts_tokens + query_tokens:
        raise ValueError(
            f"passkey_retrieval requires seq_len >= {facts_tokens + query_tokens}, got {seq_len}"
        )

    input_ids = torch.randint(0, content_max_id, (batch_size, seq_len), dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for batch_idx in range(batch_size):
        keys: list[int] = []
        values: list[int] = []
        cursor = 0

        # Store multiple key-value facts near the front.
        for _ in range(cfg.data.num_facts):
            key = int(torch.randint(0, content_max_id, (1,)).item())
            value = int(torch.randint(0, content_max_id, (1,)).item())
            while value == key:
                value = int(torch.randint(0, content_max_id, (1,)).item())

            keys.append(key)
            values.append(value)
            input_ids[batch_idx, cursor] = key
            input_ids[batch_idx, cursor + 1] = value
            input_ids[batch_idx, cursor + 2] = sep_token_id
            cursor += 3

        # Fill the middle with noise so the model must retain the mapping.
        remaining = seq_len - facts_tokens - query_tokens
        if remaining > 0:
            noise_len = int(max(0, round(remaining * cfg.data.noise_token_fraction)))
            noise_len = min(noise_len, remaining)
            if noise_len > 0:
                noise = torch.randint(0, content_max_id, (noise_len,), dtype=torch.long)
                input_ids[batch_idx, cursor : cursor + noise_len] = noise
                cursor += noise_len
            # Leave the rest as random content; it still adds interference.
            if cursor < seq_len - query_tokens:
                filler = torch.randint(0, content_max_id, (seq_len - query_tokens - cursor,), dtype=torch.long)
                input_ids[batch_idx, cursor : seq_len - query_tokens] = filler

        # Ask several retrieval queries at the end. The model must output the value after the query token.
        query_indices = torch.randperm(len(keys))[: cfg.data.num_queries].tolist()
        for idx in query_indices:
            query_key = keys[idx]
            query_value = values[idx]
            input_ids[batch_idx, cursor] = query_key
            input_ids[batch_idx, cursor + 1] = query_token_id
            input_ids[batch_idx, cursor + 2] = query_value
            labels[batch_idx, cursor + 1] = query_value
            cursor += 3

    return {"input_ids": input_ids, "labels": labels}


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
        if self.cfg.data.task == "passkey_retrieval":
            return _build_passkey_retrieval_batch(self.cfg)

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
