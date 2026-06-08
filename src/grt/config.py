from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ALUConfig:
    num_layers: int = 2
    nhead: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class RouterConfig:
    mode: str = "cross_attn"
    pool_nhead: int = 4
    mlp_hidden: int = 512


@dataclass
class RegisterConfig:
    write_gate_bias_init: float = -2.0
    dropout_prob: float = 0.10
    s0_learnable: bool = True


@dataclass
class ModelConfig:
    segment_len: int = 128
    num_registers: int = 32
    d_model: int = 256
    vocab_size: int = 32000
    alu: ALUConfig = field(default_factory=ALUConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    register: RegisterConfig = field(default_factory=RegisterConfig)


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "grt-research"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    group: str | None = None
    log_freq: int = 50
    watch_model: bool = False
    save_code: bool = True
    checkpoint_artifact: bool = True


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    split_train: str = "train"
    split_eval: str = "validation"
    tokenizer: str = "gpt2"
    num_workers: int = 0
    task: str = "synthetic_copy"
    seq_len: int = 512


@dataclass
class TrainingConfig:
    batch_size: int = 16
    max_segments: int = 64
    optimizer: str = "adamw"
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"
    seed: int = 1234


@dataclass
class RTLAConfig:
    enabled: bool = False
    trace_every_n_steps: int = 500
    output_dir: str = "traces/"
    float_dtype: str = "float32"
    upload_panels: bool = True
    upload_table: bool = True


@dataclass
class LoggingConfig:
    save_dir: str = "checkpoints/"
    save_every_n_steps: int = 2000
    eval_every_n_steps: int = 1000


@dataclass
class GRTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rtla: RTLAConfig = field(default_factory=RTLAConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_config(path: str) -> GRTConfig:
    raw = _load_yaml(Path(path).resolve())
    return _build_dataclass(GRTConfig, raw)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw

    base_path = (path.parent / base_ref).resolve()
    base_raw = _load_yaml(base_path)
    return _merge_dicts(base_raw, raw)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_value(field_obj) -> Any:
    if field_obj.default is not MISSING:
        return field_obj.default
    if field_obj.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return field_obj.default_factory()
    return None


def _build_dataclass(cls, data: dict[str, Any] | None):
    if data is None or not is_dataclass(cls):
        return data

    values: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for field_obj in fields(cls):
        if field_obj.name not in data:
            values[field_obj.name] = _default_value(field_obj)
            continue

        value = data[field_obj.name]
        field_type = hints.get(field_obj.name, field_obj.type)
        if isinstance(field_type, type) and is_dataclass(field_type):
            values[field_obj.name] = _build_dataclass(field_type, value)
        else:
            values[field_obj.name] = value

    return cls(**values)
