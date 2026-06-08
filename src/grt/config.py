from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any

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
    log_freq: int = 50
    watch_model: bool = False
    save_code: bool = True
    checkpoint_artifact: bool = True


@dataclass
class GRTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_config(path: str) -> GRTConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _build_dataclass(GRTConfig, raw)


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
    for field_obj in fields(cls):
        if field_obj.name not in data:
            values[field_obj.name] = _default_value(field_obj)
            continue

        value = data[field_obj.name]
        field_type = field_obj.type
        if isinstance(field_type, type) and is_dataclass(field_type):
            values[field_obj.name] = _build_dataclass(field_type, value)
        else:
            values[field_obj.name] = value

    return cls(**values)
