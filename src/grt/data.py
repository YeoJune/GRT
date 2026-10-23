"""Dataset adapter boundary: models and trainer never interpret task tokens."""

from types import SimpleNamespace
import importlib
import torch
from torch.utils.data import DataLoader
from grt.vendor.armt.data import ARDataset, make_collator


class RememberAdapter:
    def validate(self, cfg):
        if cfg.model.name == 'neox':
            raise ValueError('The memory-free NeoX baseline is supported for WikiText only')
        d = cfg.data
        if d.options.get('objective', 'next_token') not in ('next_token', 'parallel'):
            raise ValueError('Unknown Remember objective')
        if d.options.get("objective") == "parallel" and cfg.model.attention_mode != "encoder":
            raise ValueError("Parallel Remember requires the new encoder")
        if cfg.model.attention_mode == "encoder" and d.options.get("objective") != "parallel":
            raise ValueError("Remember encoder requires masked parallel answer slots")
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
        author_collate = make_collator(
            SimpleNamespace(vary_n_segments=vary, value_size=cfg.data.value_size)
        )
        if cfg.data.options.get("objective") != "parallel":
            return author_collate
        # Independent RNG prevents model initialization from changing selected queries.
        generator = torch.Generator().manual_seed(cfg.data.data_seed + int(vary))
        def collate(batch):
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(generator.get_state())
                result = author_collate(batch)
                generator.set_state(torch.get_rng_state())
            n = cfg.data.value_size + 1
            mask = torch.zeros_like(result["input_ids"], dtype=torch.bool)
            mask[:, -n:] = True
            ids = result["input_ids"].clone()
            ids[:, -n:] = 103  # Reserved V128 symbol; one shared masked answer token.
            result.update(input_ids=ids, input_ids_generate=ids.clone(), labels_mask=mask,
                attention_mask_generate=torch.ones_like(ids, dtype=torch.bool))
            return result
        collate.generator = generator
        return collate

    def generation_kwargs(self, cfg):
        return dict(
            max_new_tokens=cfg.data.value_size + 1, pad_token_id=0, eos_token_id=102
        )

    def generation_target(self, cfg, batch):
        return batch["labels"][:, -cfg.data.value_size - 1 :]


_ADAPTERS = {"remember": RememberAdapter()}


def register_builtin_adapters():
    from grt.wikitext import WikiTextAdapter
    _ADAPTERS.setdefault("wikitext103", WikiTextAdapter())


def register_adapter(name, adapter):
    if name in _ADAPTERS:
        raise ValueError(f"Dataset adapter already registered: {name}")
    _ADAPTERS[name] = adapter


def get_adapter(name):
    if name == "wikitext103":
        register_builtin_adapters()
    if name in _ADAPTERS:
        return _ADAPTERS[name]
    if ":" in name:
        module, attr = name.split(":", 1)
        adapter = getattr(importlib.import_module(module), attr)
        return adapter() if isinstance(adapter, type) else adapter
    raise ValueError(f"Unknown dataset adapter: {name}")


def make_dataset(cfg, split, pairs):
    return get_adapter(cfg.data.name).dataset(cfg, split, pairs)


def make_loader(cfg, dataset, training=False, vary=None, world_size=1):
    if vary is None:
        vary = cfg.data.vary_n_pairs
    batch = (
        cfg.training.batch_size * cfg.training.grad_accum_steps
        if training
        else cfg.evaluation.batch_size
    )
    return DataLoader(
        dataset,
        batch_size=batch * world_size,
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
