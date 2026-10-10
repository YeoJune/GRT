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
    rmt_backbone: str = "bidirectional"
    head_dim: int | None = None

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
    protocol: str = "recovery"
    train_samples: int | None = None
    key_size: int = 1
    value_size: int = 1
    vary_n_pairs: bool = False
    remember_sampling: str = "random"

@dataclass
class CurriculumStage:
    num_pairs: int = 1
    key_size: int = 1
    max_steps: int = 1000

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
    optimizer: str = "adamw"
    adam_beta2: float = 0.95
    scheduler: str = "cosine"
    scheduler_steps: int | None = None
    plateau_patience: int = 8
    plateau_every_steps: int | None = None
    min_lr: float = 1e-6
    stop_on_convergence: bool = False
    curriculum: list[CurriculumStage] = field(default_factory=list)
    max_seconds: float | None = None
    convergence_exact_match: float = 0.99
    grad_clip_type: str = "norm"

@dataclass
class EvaluationConfig:
    batch_size: int = 8
    every_steps: int = 500
    warmup: int = 10
    iterations: int = 50
    autoregressive_samples: int = 0

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
        if get_origin(hint) is list and is_dataclass(get_args(hint)[0]):
            if not isinstance(value, list):
                raise ValueError(f"{path}.{key} must be a list")
            values[key] = [build_config(get_args(hint)[0], item, f"{path}.{key}") for item in value]
        elif dc and value is not None:
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
    if m.size not in ("small", "large") or (not m.tie_word_embeddings and m.rmt_backbone not in ("gpt_neox", "author_neox")):
        raise ValueError("Use small/large and tied word embeddings")
    for value in (m.vocab_size, m.segment_len, m.num_registers, m.d_model, m.alu.nhead, m.alu.num_layers, m.alu.d_ff):
        if type(value) is not int or value <= 0:
            raise ValueError("Model dimensions must be positive integers")
    if m.d_model % m.alu.nhead or not 0 <= m.alu.dropout <= 1 or m.alu.activation not in ("gelu", "relu"):
        raise ValueError("Invalid ALU heads, dropout or activation")
    if m.rmt_backbone not in ("bidirectional", "relative_postln", "gpt_neox", "author_neox"):
        raise ValueError("Invalid rmt_backbone")
    if m.rmt_backbone == "relative_postln":
        if m.name != "rmt" or m.head_dim is None or m.head_dim <= 0 or m.d_model % 2 or m.alu.activation != "relu":
            raise ValueError("relative_postln requires RMT, positive head_dim, even d_model and relu")
    elif m.head_dim is not None or m.alu.activation != "gelu":
        raise ValueError("The recovery backbone requires gelu and no head_dim override")
    if m.rmt_backbone in ("gpt_neox", "author_neox") and (m.name != "rmt" or (m.d_model // m.alu.nhead) % 8):
        raise ValueError("GPT-NeoX RMT requires a head dimension divisible by 8 for rotary_pct=0.25")
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
    if d.task not in ("copy", "reverse", "passkey", "remember"):
        raise ValueError("data.task must be copy, reverse or passkey")
    if d.protocol not in ("recovery", "paper_copy", "paper_ar"):
        raise ValueError("Unknown data.protocol")
    if d.protocol == "paper_copy":
        if d.task != "copy" or d.target_len != 24 or d.num_facts is not None:
            raise ValueError("paper_copy requires copy, target_len=24 and no num_facts")
        if cfg.model.name != "rmt" or cfg.model.rmt_backbone != "relative_postln":
            raise ValueError("paper_copy stage 1 requires the relative_postln RMT; GRT is not implemented yet")
        m = cfg.model
        if (m.vocab_size, m.segment_len, m.num_registers, m.d_model, m.alu.num_layers, m.alu.nhead, m.head_dim, m.alu.d_ff) != (12, 24, 24, 128, 4, 4, 64, 256):
            raise ValueError("paper_copy requires the published short Copy model dimensions")
        if d.train_segments != 3 or d.eval_segments != [3]:
            raise ValueError("paper_copy currently evaluates only the published 3-segment length")
        if d.train_samples is None or d.train_samples <= 0 or cfg.evaluation.autoregressive_samples <= 0:
            raise ValueError("paper_copy requires a finite train set and autoregressive evaluation")
    elif d.protocol == "paper_ar":
        m = cfg.model
        if d.task != "remember" or m.name != "rmt" or m.rmt_backbone not in ("gpt_neox", "author_neox"):
            raise ValueError("paper_ar stage 1 requires Remember and GPT-NeoX RMT")
        if d.key_size <= 0 or d.value_size <= 0 or m.segment_len != d.key_size + d.value_size + 2:
            raise ValueError("paper_ar segment length must be key_size + value_size + 2")
        if m.vocab_size != 128 or m.tie_word_embeddings or d.target_len is not None or d.num_facts is not None:
            raise ValueError("paper_ar uses V128, untied embeddings and key/value lengths")
        if d.train_samples is None or d.train_samples <= 0 or cfg.evaluation.autoregressive_samples <= 0:
            raise ValueError("paper_ar requires finite data and generation evaluation")
        if max(d.train_segments, *d.eval_segments) - 1 > 16 ** d.key_size:
            raise ValueError("Remember requires unique keys; increase key_size")
        if any(s.num_pairs <= 0 or s.key_size <= 0 or s.num_pairs > 16 ** s.key_size or s.max_steps <= 0 for s in t.curriculum):
            raise ValueError("Invalid Remember curriculum stage")
        if d.remember_sampling not in ("random", "balanced_contexts"):
            raise ValueError("Unknown Remember sampling")
        if m.rmt_backbone == "author_neox" and d.remember_sampling != "random":
            raise ValueError("Author reference uses the original random Remember data")
        if d.remember_sampling == "balanced_contexts" and (d.key_size != 1 or d.value_size != 1 or d.vary_n_pairs or
                any(s.key_size != 1 or s.num_pairs > 16 for s in t.curriculum)):
            raise ValueError("Balanced Remember uses single-token distinct keys/values and fixed pair counts")
    elif d.task == "passkey":
        if d.target_len is not None or d.num_facts != 4:
            raise ValueError("passkey requires num_facts=4 and no target_len")
        expected = (5, [5, 15, 30])
    else:
        if d.num_facts is not None or d.target_len != 20:
            raise ValueError("copy/reverse requires target_len=20 and no num_facts")
        expected = (4, [4, 10, 20])
    if benchmark and d.protocol == "recovery":
        m = cfg.model
        if m.rmt_backbone != "bidirectional":
            raise ValueError("The recovery benchmark requires its bidirectional backbone")
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
    if t.optimizer not in ("adam", "adamw") or not 0 < t.adam_beta2 < 1 or t.scheduler not in ("cosine", "plateau", "linear"):
        raise ValueError("Invalid optimizer/scheduler controls")
    if t.plateau_patience < 0 or t.min_lr <= 0 or (t.scheduler == "plateau" and t.min_lr > t.lr) or cfg.evaluation.autoregressive_samples < 0:
        raise ValueError("Invalid plateau/evaluation controls")
    if d.protocol == "recovery" and (d.train_samples is not None or cfg.evaluation.autoregressive_samples or t.stop_on_convergence):
        raise ValueError("Finite training, generation evaluation and early stopping require a paper protocol")
    if d.protocol != "paper_ar" and (t.curriculum or d.vary_n_pairs or d.task == "remember"):
        raise ValueError("Remember curriculum belongs to paper_ar")
    if d.protocol != "paper_ar" and d.remember_sampling != "random":
        raise ValueError("Remember sampling belongs to paper_ar")
    if t.grad_clip_type not in ("norm", "value") or not 0 < t.convergence_exact_match <= 1:
        raise ValueError("Invalid gradient clipping or convergence threshold")
    if t.max_seconds is not None and t.max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if t.scheduler_steps is not None and t.scheduler_steps <= 0:
        raise ValueError("scheduler_steps must be positive")
    if t.scheduler == "plateau" and t.warmup_steps:
        raise ValueError("plateau currently requires warmup_steps=0")
    if t.plateau_every_steps is not None and (t.plateau_every_steps <= 0 or t.plateau_every_steps % cfg.evaluation.every_steps):
        raise ValueError("plateau_every_steps must be a positive multiple of evaluation.every_steps")
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
