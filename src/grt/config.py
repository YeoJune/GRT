"""Typed experiment configuration; dataset adapters own task-specific validation."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints, Union
import math
import types
import yaml

EXPERIMENT_VERSION = "remember/2"


@dataclass
class ALUConfig:
    num_layers: int = 4
    nhead: int = 4
    d_ff: int = 128
    dropout: float = 0.0
    activation: str = "gelu"


@dataclass
class RouterConfig:
    pool_nhead: int = 4
    mlp_hidden: int = 512


@dataclass
class RegisterConfig:
    conditional_read: bool = False
    dropout_prob: float = 0.0
    write_gate_bias_init: float = -2.0
    s0_learnable: bool = True


@dataclass
class ModelConfig:
    name: str = "rmt"
    vocab_size: int = 128
    segment_len: int = 4
    num_registers: int = 32
    d_model: int = 128
    tie_word_embeddings: bool = False
    position_capacity: int | None = None
    alu: ALUConfig = field(default_factory=ALUConfig)
    router: RouterConfig | None = None
    register: RegisterConfig | None = None


@dataclass
class DataConfig:
    name: str = "remember"
    data_seed: int = 54
    train_segments: int = 2
    key_size: int = 1
    value_size: int = 1
    vary_n_pairs: bool = True
    train_samples: int = 1000000
    validation_samples: int = 1000
    test_samples: int = 10000
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumStage:
    num_pairs: int = 1
    key_size: int = 1
    max_steps: int = 2000
    batch_size: int | None = None
    grad_accum_steps: int | None = None


@dataclass
class TrainingConfig:
    model_seed: int = 54
    batch_size: int = 512
    grad_accum_steps: int = 1
    global_batch_size: int | None = None
    max_steps: int = 2000
    lr: float = 0.0003
    weight_decay: float = 0.001
    adam_beta2: float = 0.999
    grad_clip: float = 1.0
    mixed_precision: str = "fp32"
    stop_on_convergence: bool = False
    convergence_exact_match: float = 0.99
    max_seconds: float | None = None
    curriculum: list[CurriculumStage] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    batch_size: int = 512
    every_steps: int = 250
    pair_counts: list[int] = field(default_factory=list)
    generalization_samples: int = 1000


@dataclass
class RTLAConfig:
    enabled: bool = False
    batch_size: int = 8


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "grt-research"
    run_name: str | None = None
    mode: str = "online"
    every_steps: int = 50


@dataclass
class RunConfig:
    output_dir: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    rtla: RTLAConfig = field(default_factory=RTLAConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _check_type(value, hint):
    origin, args = get_origin(hint), get_args(hint)
    if hint is Any:
        return True
    if origin in (Union, types.UnionType):
        return any(_check_type(value, a) for a in args)
    if origin is list:
        return isinstance(value, list) and all(_check_type(x, args[0]) for x in value)
    if origin is dict:
        return isinstance(value, dict) and all(
            _check_type(k, args[0]) and _check_type(v, args[1])
            for k, v in value.items()
        )
    if hint is float:
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is hint


def build_config(cls, raw, path="config"):
    if not isinstance(raw, dict) or any(not isinstance(k, str) for k in raw):
        raise ValueError(f"{path} must be a mapping with string keys")
    names = {f.name for f in fields(cls)}
    if raw.keys() - names:
        raise ValueError(f"Unknown keys in {path}: {sorted(raw.keys() - names)}")
    hints = get_type_hints(cls)
    values = {}
    for key, value in raw.items():
        hint = hints[key]
        dc = (
            hint
            if is_dataclass(hint)
            else next((a for a in get_args(hint) if is_dataclass(a)), None)
        )
        if get_origin(hint) is list and is_dataclass(get_args(hint)[0]):
            if not isinstance(value, list):
                raise ValueError(f"{path}.{key} must be a list")
            values[key] = [
                build_config(get_args(hint)[0], item, f"{path}.{key}") for item in value
            ]
        elif dc and value is not None:
            values[key] = build_config(dc, value, f"{path}.{key}")
        elif _check_type(value, hint):
            values[key] = value
        else:
            raise ValueError(f"Invalid type/value for {path}.{key}: {value!r}")
    return cls(**values)


def merge_dicts(base, override):
    result = dict(base)
    for k, v in override.items():
        result[k] = (
            merge_dicts(result[k], v)
            if isinstance(v, dict) and isinstance(result.get(k), dict)
            else v
        )
    return result


def validate_model(m):
    build_config(ModelConfig, asdict(m))
    if m.name not in ("rmt", "grt"):
        raise ValueError("model.name must be rmt or grt")
    dims = (
        m.vocab_size,
        m.segment_len,
        m.num_registers,
        m.d_model,
        m.alu.num_layers,
        m.alu.nhead,
        m.alu.d_ff,
    )
    if any(type(v) is not int or v <= 0 for v in dims) or m.d_model % m.alu.nhead:
        raise ValueError("Invalid model dimensions")
    if not 0 <= m.alu.dropout <= 1 or m.alu.activation != "gelu":
        raise ValueError("Invalid ALU settings")
    if m.name == "rmt":
        if (
            m.tie_word_embeddings
            or m.position_capacity is not None
            or m.router
            or m.register
            or (m.d_model // m.alu.nhead) % 8
        ):
            raise ValueError(
                "Author RMT requires untied embeddings, rotary head dimension divisible by 8 and no router"
            )
    else:
        if m.position_capacity is not None and m.position_capacity < m.segment_len:
            raise ValueError("position_capacity must cover segment_len")
        m.router = m.router or RouterConfig()
        m.register = m.register or RegisterConfig()
        if (
            m.router.pool_nhead <= 0
            or m.d_model % m.router.pool_nhead
            or m.router.mlp_hidden <= 0
        ):
            raise ValueError("Invalid router dimensions")
        if (
            not m.tie_word_embeddings
            or not m.register.s0_learnable
            or not 0 <= m.register.dropout_prob <= 1
            or not math.isfinite(m.register.write_gate_bias_init)
        ):
            raise ValueError("GRT retains tied embeddings and learnable registers")
    return m


def validate_config(cfg, require_output=True):
    build_config(Config, asdict(cfg))
    validate_model(cfg.model)
    t = cfg.training
    counts = (
        t.batch_size,
        t.grad_accum_steps,
        t.max_steps,
        cfg.evaluation.batch_size,
        cfg.evaluation.every_steps,
        cfg.rtla.batch_size,
        cfg.wandb.every_steps,
    )
    if any(type(v) is not int or v <= 0 for v in counts):
        raise ValueError("Counts must be positive integers")
    if t.global_batch_size is not None and t.global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if (
        t.lr <= 0
        or t.weight_decay < 0
        or t.grad_clip <= 0
        or not 0 < t.adam_beta2 < 1
        or not 0 < t.convergence_exact_match <= 1
    ):
        raise ValueError("Invalid training settings")
    if t.model_seed < 0 or t.mixed_precision != "fp32":
        raise ValueError(
            "This reference experiment requires nonnegative seeds and fp32"
        )
    if t.max_seconds is not None and t.max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    for stage in t.curriculum:
        if any(
            v is not None and v <= 0 for v in (stage.batch_size, stage.grad_accum_steps)
        ):
            raise ValueError("Stage batch settings must be positive")
    if cfg.evaluation.generalization_samples <= 0 or any(
        p <= 0 for p in cfg.evaluation.pair_counts
    ):
        raise ValueError("Evaluation lengths/sample count must be positive")
    if cfg.wandb.mode not in ("online", "offline"):
        raise ValueError("Invalid wandb.mode")
    if require_output and not cfg.run.output_dir:
        raise ValueError("run.output_dir is required")
    from grt.data import get_adapter

    get_adapter(cfg.data.name).validate(cfg)
    return cfg


def load_config(paths, output_dir=None):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    raw = {}
    for path in paths:
        part = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(part, dict):
            raise ValueError(f"{path} must contain a mapping")
        raw = merge_dicts(raw, part)
    cfg = build_config(Config, raw)
    if output_dir is not None:
        cfg.run.output_dir = str(output_dir)
    return cfg


def save_config(cfg, path):
    Path(path).write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8"
    )


def legacy_config(raw):
    """Explicit read-only import of author_rmt/1 configuration."""
    if raw["model"].get("rmt_backbone") != "author_neox":
        raise ValueError("Only native author RMT legacy checkpoints are supported")
    selected = {}
    for key, cls in [
        ("model", ModelConfig),
        ("data", DataConfig),
        ("training", TrainingConfig),
        ("evaluation", EvaluationConfig),
        ("rtla", RTLAConfig),
        ("wandb", WandbConfig),
        ("run", RunConfig),
    ]:
        names = {f.name for f in fields(cls)}
        selected[key] = {k: v for k, v in raw.get(key, {}).items() if k in names}
    selected["data"]["name"] = "remember"
    return build_config(Config, selected)
