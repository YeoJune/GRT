"""Integer-token benchmarks; data RNG never touches the model RNG."""
import hashlib
from typing import TypedDict
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

class Batch(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor

def generator(seed, task, split, sample_id, component):
    encoded = f"{seed}/{task}/{split}/{sample_id}/{component}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % 2**63
    return torch.Generator().manual_seed(value)

def generate_sample(task, split, sample_id, segments, data_seed=20260916) -> Batch:
    if task not in ("copy", "reverse", "passkey") or split not in ("train", "validation", "test"):
        raise ValueError("Unknown task/split")
    if type(sample_id) is not int or sample_id < 0 or type(segments) is not int or segments < 2:
        raise ValueError("sample_id must be nonnegative; segments must be >=2")
    length = segments * 128
    payload_rng = generator(data_seed, task, split, sample_id, "payload")
    noise_rng = generator(data_seed, task, split, sample_id, f"noise:{segments}")
    ids = torch.randint(4, 1024, (length,), generator=noise_rng)
    labels = torch.full((length,), -100, dtype=torch.int64)
    if task in ("copy", "reverse"):
        target = torch.randint(4, 1024, (20,), generator=payload_rng)
        ids[:20] = target
        ids[-20:] = 3
        labels[-20:] = target if task == "copy" else target.flip(0)
    else:
        keys = torch.randperm(1020, generator=payload_rng)[:4] + 4
        values = torch.randint(4, 1024, (4,), generator=payload_rng)
        query = torch.randint(0, 4, (), generator=payload_rng).item()
        ids[:12:3], ids[1:12:3], ids[2:12:3] = keys, 1, values
        ids[-2], ids[-1], labels[-1] = keys[query], 2, values[query]
    return {"input_ids": ids, "attention_mask": torch.ones(length, dtype=torch.bool), "labels": labels}

class SyntheticDataset(Dataset):
    def __init__(self, cfg, split, segments, samples, start_id=0, batch_size=1):
        self.cfg, self.split, self.segments = cfg, split, segments
        self.samples, self.start_id = samples, start_id
        self.batch_size = batch_size

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        if index < 0 or index >= self.samples:
            raise IndexError(index)
        sample_id = self.start_id + index
        if self.cfg.protocol == "paper_ar":
            pairs = self.segments - 1
            if self.split == "train" and self.cfg.vary_n_pairs:
                # One length per microbatch; validation always uses the stated length.
                length_rng = generator(self.cfg.data_seed, "remember", self.split,
                                       sample_id // self.batch_size, f"length:{pairs}")
                pairs = torch.randint(1, pairs + 1, (), generator=length_rng).item()
            if self.split == "train":
                sample_id %= self.cfg.train_samples
            return generate_remember(self.split, sample_id, pairs, self.cfg.key_size,
                                     self.cfg.value_size, self.cfg.data_seed)
        if self.cfg.protocol == "paper_copy":
            if self.segments != 3:
                raise ValueError("paper_copy requires 3 segments")
            if self.split == "train":
                sample_id %= self.cfg.train_samples
            return generate_paper_copy(self.split, sample_id, self.cfg.data_seed)
        return generate_sample(self.cfg.task, self.split, sample_id, self.segments, self.cfg.data_seed)

def generate_paper_copy(split, sample_id, data_seed=20260916) -> Batch:
    """Published short Copy: X + [start] + X + X, shifted next-token labels."""
    source = torch.randint(2, 12, (24,), generator=generator(data_seed, "paper_copy", split, sample_id, "source"))
    sequence = torch.cat([source, torch.tensor([1]), source, source])
    ids, labels = sequence[:-1].clone(), sequence[1:].clone()
    labels[:24] = -100
    return {"input_ids": ids, "attention_mask": torch.ones(72, dtype=torch.bool), "labels": labels}


def generate_remember(split, sample_id, pairs, key_size=1, value_size=1, data_seed=20260916):
    """ARMT Appendix E/I: unique facts followed by a separate query segment."""
    if pairs <= 0 or key_size <= 0 or value_size <= 0 or pairs > 16 ** key_size:
        raise ValueError("Remember requires positive sizes and enough unique keys")
    rng = generator(data_seed, "remember", split, sample_id, f"pairs:{pairs}:key:{key_size}:value:{value_size}")
    # Rejection sampling avoids allocating all 16**key_size possible keys.
    selected = []
    seen = set()
    while len(selected) < pairs:
        key = torch.randint(0, 16, (key_size,), generator=rng)
        identity = tuple(key.tolist())
        if identity not in seen:
            seen.add(identity)
            selected.append(key)
    keys = torch.stack(selected)
    values = torch.randint(0, 16, (pairs, value_size), generator=rng)
    target = torch.randint(pairs, (), generator=rng).item()
    sep, gen, eos = torch.tensor([100]), torch.tensor([101]), torch.tensor([102])
    facts = [torch.cat([key, sep, value, eos]) for key, value in zip(keys, values)]
    query = torch.cat([keys[target], gen, values[target], eos])
    ids = torch.cat([*facts, query])
    labels = torch.full_like(ids, -100)
    # Labels are already next-token aligned; GEN predicts the first value.
    answer_start = ids.numel() - value_size - 1
    labels[answer_start-1:-1] = ids[answer_start:]
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool), "labels": labels}

def collate(samples) -> Batch:
    if not samples or len({s["input_ids"].shape for s in samples}) != 1:
        raise ValueError("A batch must contain nonempty samples of one length")
    return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}

def make_loader(cfg, split, segments, samples, batch_size, start_id=0):
    return DataLoader(SyntheticDataset(cfg, split, segments, samples, start_id, batch_size),
                      batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers,
                      collate_fn=collate, generator=torch.Generator().manual_seed(cfg.data_seed))

def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
