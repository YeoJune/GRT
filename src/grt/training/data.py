from __future__ import annotations

import torch

from grt.config import GRTConfig


def _build_passkey_retrieval_batch(cfg: GRTConfig, device: torch.device | str) -> dict[str, torch.Tensor]:
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

    input_ids = torch.randint(0, content_max_id, (batch_size, seq_len), dtype=torch.long, device=device)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for batch_idx in range(batch_size):
        keys: list[int] = []
        values: list[int] = []
        cursor = 0

        # Store multiple key-value facts near the front.
        for _ in range(cfg.data.num_facts):
            key = int(torch.randint(0, content_max_id, (1,), device=device).item())
            value = int(torch.randint(0, content_max_id, (1,), device=device).item())
            while value == key:
                value = int(torch.randint(0, content_max_id, (1,), device=device).item())

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
                noise = torch.randint(0, content_max_id, (noise_len,), dtype=torch.long, device=device)
                input_ids[batch_idx, cursor : cursor + noise_len] = noise
                cursor += noise_len
            # Leave the rest as random content; it still adds interference.
            if cursor < seq_len - query_tokens:
                filler = torch.randint(
                    0,
                    content_max_id,
                    (seq_len - query_tokens - cursor,),
                    dtype=torch.long,
                    device=device,
                )
                input_ids[batch_idx, cursor : seq_len - query_tokens] = filler

        # Ask several retrieval queries at the end. The model must output the value after the query token.
        query_indices = torch.randperm(len(keys))[: cfg.data.num_queries].tolist()
        for idx in query_indices:
            query_key = keys[idx]
            query_value = values[idx]
            input_ids[batch_idx, cursor] = query_key
            input_ids[batch_idx, cursor + 1] = query_token_id
            # Do NOT place the true value into the next input slot — fill with a non-answer filler
            filler = int(torch.randint(0, content_max_id, (1,), device=device).item())
            # ensure filler is not equal to the true value
            if filler == query_value:
                filler = (filler + 1) % content_max_id
            input_ids[batch_idx, cursor + 2] = filler
            # The model should predict the value when it sees the query token (causal LM alignment)
            labels[batch_idx, cursor + 1] = query_value
            cursor += 3

    return {"input_ids": input_ids, "labels": labels}


def _build_entity_tracking_batch(cfg: GRTConfig, device: torch.device | str) -> dict[str, torch.Tensor]:
    batch_size = cfg.training.batch_size
    seq_len = cfg.data.seq_len
    vocab_size = cfg.model.vocab_size
    content_max_id = vocab_size - 4
    query_token_id = vocab_size - 1
    upd_token_id = vocab_size - 2
    sep_token_id = vocab_size - 3

    update_tokens = cfg.data.num_updates * 3
    query_tokens = cfg.data.num_queries * 3
    if seq_len < update_tokens + query_tokens:
        raise ValueError(
            f"entity_tracking requires seq_len >= {update_tokens + query_tokens}, got {seq_len}"
        )

    input_ids = torch.randint(0, content_max_id, (batch_size, seq_len), dtype=torch.long, device=device)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        cursor = 0
        latest_value_by_entity: dict[int, int] = {}
        entities = torch.randint(0, content_max_id, (cfg.data.num_entities,), device=device).tolist()

        for _ in range(cfg.data.num_updates):
            ent = int(entities[int(torch.randint(0, len(entities), (1,), device=device).item())])
            val = int(torch.randint(0, content_max_id, (1,), device=device).item())
            latest_value_by_entity[ent] = val
            input_ids[b, cursor] = ent
            input_ids[b, cursor + 1] = upd_token_id
            input_ids[b, cursor + 2] = val
            cursor += 3

        gap = seq_len - update_tokens - query_tokens
        if gap > 0:
            input_ids[b, cursor : cursor + gap] = torch.randint(0, content_max_id, (gap,), device=device)
            cursor += gap

        query_entities = list(latest_value_by_entity.keys())
        if len(query_entities) == 0:
            query_entities = entities[:1]

        for _ in range(cfg.data.num_queries):
            ent = int(query_entities[int(torch.randint(0, len(query_entities), (1,), device=device).item())])
            answer = int(latest_value_by_entity.get(ent, 0))
            input_ids[b, cursor] = ent
            input_ids[b, cursor + 1] = query_token_id
            # do not leak the answer into the next input slot; use random filler instead
            filler = int(torch.randint(0, content_max_id, (1,), device=device).item())
            if filler == answer:
                filler = (filler + 1) % content_max_id
            input_ids[b, cursor + 2] = filler
            labels[b, cursor + 1] = answer
            cursor += 3

    return {"input_ids": input_ids, "labels": labels}


def _build_in_context_arithmetic_batch(cfg: GRTConfig, device: torch.device | str) -> dict[str, torch.Tensor]:
    batch_size = cfg.training.batch_size
    seq_len = cfg.data.seq_len
    vocab_size = cfg.model.vocab_size
    content_max_id = vocab_size - 4
    plus_token_id = vocab_size - 1
    eq_token_id = vocab_size - 2
    sep_token_id = vocab_size - 3

    fact_tokens = cfg.data.num_equations * 4
    query_tokens = cfg.data.num_queries * 4
    if seq_len < fact_tokens + query_tokens:
        raise ValueError(
            f"in_context_arithmetic requires seq_len >= {fact_tokens + query_tokens}, got {seq_len}"
        )

    input_ids = torch.randint(0, content_max_id, (batch_size, seq_len), dtype=torch.long, device=device)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    modulus = min(512, content_max_id)

    for b in range(batch_size):
        cursor = 0
        pairs: list[tuple[int, int, int]] = []
        for _ in range(cfg.data.num_equations):
            a = int(torch.randint(0, modulus, (1,), device=device).item())
            c = int(torch.randint(0, modulus, (1,), device=device).item())
            y = (a + c) % modulus
            pairs.append((a, c, y))
            input_ids[b, cursor] = a
            input_ids[b, cursor + 1] = plus_token_id
            input_ids[b, cursor + 2] = c
            input_ids[b, cursor + 3] = y
            cursor += 4

        gap = seq_len - fact_tokens - query_tokens
        if gap > 0:
            input_ids[b, cursor : cursor + gap] = torch.randint(0, content_max_id, (gap,), device=device)
            cursor += gap

        for _ in range(cfg.data.num_queries):
            a, c, y = pairs[int(torch.randint(0, len(pairs), (1,), device=device).item())]
            input_ids[b, cursor] = a
            input_ids[b, cursor + 1] = plus_token_id
            input_ids[b, cursor + 2] = c
            input_ids[b, cursor + 3] = eq_token_id
            labels[b, cursor + 3] = y
            cursor += 4

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
            return _build_passkey_retrieval_batch(self.cfg, self.device)
        if self.cfg.data.task == "entity_tracking":
            return _build_entity_tracking_batch(self.cfg, self.device)
        if self.cfg.data.task == "in_context_arithmetic":
            return _build_in_context_arithmetic_batch(self.cfg, self.device)

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
