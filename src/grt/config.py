from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints, get_origin, get_args, Union
import types
import math
import yaml

SPEC_VERSION = "20260916/1.0"

@dataclass
class ALUConfig:
    num_layers: int = 4
    nhead: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
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
    name: str = ""
    size: str = "small"
    vocab_size: int = 1024
    segment_len: int = 128
    num_registers: int = 32
    d_model: int = 256
    tie_word_embeddings: bool = True
    alu: ALUConfig = field(default_factory=ALUConfig)
    router: RouterConfig | None = None
    register: RegisterConfig | None = None

@dataclass
class DataConfig:
    task: str = ""
    data_seed: int = 20260916
    train_segments: int = 4
    eval_segments: list[int] = field(default_factory=lambda: [4, 10, 20])
    target_len: int | None = None
    num_facts: int | None = None
    validation_samples: int = 1024
    test_samples: int = 4096
    num_workers: int = 0

@dataclass
class TrainingConfig:
    model_seed: int = 1234
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 50000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    mixed_precision: str = "fp32"

@dataclass
class EvaluationConfig:
    batch_size: int = 8
    every_steps: int = 500
    warmup: int = 10
    iterations: int = 50

@dataclass
class CheckpointConfig:
    every_steps: int = 2000

@dataclass
class RTLAConfig:
    enabled: bool = True
    every_steps: int = 500
    batch_size: int = 8

@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "grt-research"
    run_name: str | None = None

@dataclass
class RunConfig:
    output_dir: str | None = None

@dataclass
class GRTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    rtla: RTLAConfig = field(default_factory=RTLAConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    run: RunConfig = field(default_factory=RunConfig)

def _check_type(value, hint):
    origin, args = get_origin(hint), get_args(hint)
    if origin in (Union, types.UnionType):
        return any(_check_type(value, a) for a in args)
    if origin is list:
        return isinstance(value, list) and all(_check_type(x, args[0]) for x in value)
    if hint is float:
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is hint

def build_config(cls, raw, path="config"):
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{path} keys must be strings")
    names = {f.name for f in fields(cls)}
    if raw.keys() - names:
        raise ValueError(f"Unknown keys in {path}: {sorted(raw.keys() - names)}")
    hints, values = get_type_hints(cls), {}
    for key, value in raw.items():
        hint = hints[key]
        dc = hint if is_dataclass(hint) else next((a for a in get_args(hint) if is_dataclass(a)), None)
        if dc and value is not None:
            values[key] = build_config(dc, value, f"{path}.{key}")
        elif _check_type(value, hint):
            values[key] = value
        else:
            raise ValueError(f"Invalid type/value for {path}.{key}: {value!r}")
    return cls(**values)

def merge_dicts(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = merge_dicts(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result

def validate_model(m):
    build_config(ModelConfig, asdict(m))
    if m.name not in ("grt", "rmt"):
        raise ValueError("model.name must be grt or rmt")
    if m.size not in ("small", "large") or not m.tie_word_embeddings:
        raise ValueError("Use small/large and tied word embeddings")
    for value in (m.vocab_size, m.segment_len, m.num_registers, m.d_model, m.alu.nhead, m.alu.num_layers, m.alu.d_ff):
        if type(value) is not int or value <= 0:
            raise ValueError("Model dimensions must be positive integers")
    if m.d_model % m.alu.nhead or not 0 <= m.alu.dropout <= 1 or m.alu.activation != "gelu":
        raise ValueError("Invalid ALU heads, dropout or activation")
    if m.name == "rmt" and (m.router is not None or m.register is not None):
        raise ValueError("RMT does not accept GRT router/register settings")
    if m.name == "grt":
        m.router = m.router or RouterConfig()
        m.register = m.register or RegisterConfig()
        if m.router.pool_nhead <= 0 or m.d_model % m.router.pool_nhead or m.router.mlp_hidden <= 0:
            raise ValueError("Invalid router dimensions")
        if not 0 <= m.register.dropout_prob <= 1 or not m.register.s0_learnable or not math.isfinite(m.register.write_gate_bias_init):
            raise ValueError("Invalid register configuration")

def validate_config(cfg, *, benchmark=True, require_output=True):
    build_config(GRTConfig, asdict(cfg))
    validate_model(cfg.model)
    d, t = cfg.data, cfg.training
    if d.task not in ("copy", "reverse", "passkey"):
        raise ValueError("data.task must be copy, reverse or passkey")
    if d.task == "passkey":
        if d.target_len is not None or d.num_facts != 4:
            raise ValueError("passkey requires num_facts=4 and no target_len")
        expected = (5, [5, 15, 30])
    else:
        if d.num_facts is not None or d.target_len != 20:
            raise ValueError("copy/reverse requires target_len=20 and no num_facts")
        expected = (4, [4, 10, 20])
    if benchmark:
        m = cfg.model
        dims = (256, 4, 4, 1024) if m.size == "small" else (512, 8, 8, 2048)
        if (m.vocab_size, m.segment_len, m.num_registers) != (1024, 128, 32) or (m.d_model, m.alu.num_layers, m.alu.nhead, m.alu.d_ff) != dims:
            raise ValueError("Model dimensions differ from the standard benchmark")
        if (d.train_segments, d.eval_segments) != expected:
            raise ValueError("Task lengths differ from the standard benchmark")
    positives = [d.train_segments, *d.eval_segments, d.validation_samples, d.test_samples, t.batch_size, t.grad_accum_steps, t.max_steps, cfg.evaluation.batch_size, cfg.evaluation.every_steps, cfg.evaluation.iterations, cfg.checkpoint.every_steps, cfg.rtla.every_steps, cfg.rtla.batch_size]
    if any(type(x) is not int or x <= 0 for x in positives) or not d.eval_segments:
        raise ValueError("Counts/intervals must be positive integers")
    if d.train_segments < 2 or any(x < 2 for x in d.eval_segments) or min(t.warmup_steps, d.num_workers, cfg.evaluation.warmup, d.data_seed, t.model_seed) < 0 or t.lr <= 0 or t.grad_clip <= 0 or t.weight_decay < 0:
        raise ValueError("Invalid training/data controls")
    if t.mixed_precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("mixed_precision must be fp32/bf16/fp16")
    if require_output and not cfg.run.output_dir:
        raise ValueError("run.output_dir or --output-dir is required")
    return cfg

def load_config(paths, *, output_dir=None):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    raw = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            part = yaml.safe_load(handle) or {}
        if not isinstance(part, dict):
            raise ValueError(f"{path} must contain a mapping")
        raw = merge_dicts(raw, part)
    cfg = build_config(GRTConfig, raw)
    if output_dir is not None:
        cfg.run.output_dir = str(output_dir)
    return cfg

def save_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
