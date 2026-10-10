"""Dataset adapter boundary: models and trainer never interpret task tokens."""

from typing import Protocol
from types import SimpleNamespace
import importlib
import torch
from torch.utils.data import DataLoader
from grt.vendor.armt.data import ARDataset, make_collator


class DatasetAdapter(Protocol):
    def validate(self, cfg): ...
    def configure_stage(self, cfg, stage): ...
    def dataset(self, cfg, split, pairs): ...
    def collator(self, cfg, vary): ...
    def generation_kwargs(self, cfg): ...
    def generation_target(self, cfg, batch): ...


class RememberAdapter:
    def validate(self, cfg):
        d = cfg.data
        if (
            d.key_size <= 0
            or d.value_size <= 0
            or d.train_segments < 2
            or d.data_seed < 0
        ):
            raise ValueError("Invalid Remember dimensions")
        if (
            cfg.model.vocab_size != 128
            or cfg.model.segment_len != d.key_size + d.value_size + 2
        ):
            raise ValueError(
                "Remember requires V128 and segment_len=key_size+value_size+2"
            )
        if min(d.train_samples, d.validation_samples, d.test_samples) <= 0:
            raise ValueError("Dataset sizes must be positive")
        if d.train_segments - 1 > 16**d.key_size:
            raise ValueError("Remember requires unique keys")
        for s in cfg.training.curriculum:
            if (
                s.num_pairs <= 0
                or s.key_size <= 0
                or s.max_steps <= 0
                or s.num_pairs > 16**s.key_size
            ):
                raise ValueError("Invalid curriculum stage")

    def configure_stage(self, cfg, stage):
        cfg.data.key_size = stage.key_size
        cfg.data.train_segments = stage.num_pairs + 1
        cfg.model.segment_len = stage.key_size + cfg.data.value_size + 2

    def dataset(self, cfg, split, pairs):
        size = getattr(
            cfg.data,
            {
                "train": "train_samples",
                "validation": "validation_samples",
                "test": "test_samples",
            }[split],
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                cfg.data.data_seed + {"train": 0, "validation": 1, "test": 2}[split]
            )
            return ARDataset(cfg.data.key_size, cfg.data.value_size, pairs, size)

    def collator(self, cfg, vary):
        return make_collator(
            SimpleNamespace(vary_n_segments=vary, value_size=cfg.data.value_size)
        )

    def generation_kwargs(self, cfg):
        return dict(
            max_new_tokens=cfg.data.value_size + 1, pad_token_id=0, eos_token_id=102
        )

    def generation_target(self, cfg, batch):
        return batch["labels"][:, -cfg.data.value_size - 1 :]


_ADAPTERS = {"remember": RememberAdapter()}


def register_adapter(name, adapter):
    if name in _ADAPTERS:
        raise ValueError(f"Dataset adapter already registered: {name}")
    _ADAPTERS[name] = adapter


def get_adapter(name):
    if name in _ADAPTERS:
        return _ADAPTERS[name]
    if ":" in name:
        module, attr = name.split(":", 1)
        adapter = getattr(importlib.import_module(module), attr)
        return adapter() if isinstance(adapter, type) else adapter
    raise ValueError(f"Unknown dataset adapter: {name}")


def make_dataset(cfg, split, pairs):
    return get_adapter(cfg.data.name).dataset(cfg, split, pairs)


def make_loader(cfg, dataset, training=False, vary=None):
    if vary is None:
        vary = cfg.data.vary_n_pairs
    batch = (
        cfg.training.batch_size * cfg.training.grad_accum_steps
        if training
        else cfg.evaluation.batch_size
    )
    return DataLoader(
        dataset,
        batch_size=batch,
        collate_fn=get_adapter(cfg.data.name).collator(cfg, vary),
        num_workers=0,
        generator=torch.Generator().manual_seed(cfg.training.model_seed),
    )


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def forward_batch(model, batch, device):
    return model(
        **{
            k: batch[k].to(device)
            for k in ("input_ids", "attention_mask", "labels", "labels_mask")
        }
    )
