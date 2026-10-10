"""Local result files and optional W&B; logging must not consume model RNG."""

from dataclasses import asdict
from pathlib import Path
import json
import platform
import subprocess
import warnings
from uuid import uuid4
import torch
import transformers
from grt.checkpoint import preserve_rng
from grt.config import EXPERIMENT_VERSION


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_result_files(root, name, result):
    for suffix in ("json", "txt"):
        write_json(Path(root) / f"{name}.{suffix}", result)


def append_metrics(stage_dir, row):
    with (stage_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    v = row["validation"]
    fixed = v["fixed_length"]
    text = (
        f"Step {row['step']}\n  Train: loss={row['train/loss']:.6f}, lr={row['train/lr']:.8g}\n"
        f"  Random-length validation: loss={v['loss']:.6f}, exact_match={v['exact_match']:.2%}\n"
        f"  Fixed-length validation:  loss={fixed['loss']:.6f}, exact_match={fixed['exact_match']:.2%}\n"
        f"  Training time={row['train/seconds']:.2f}s, peak VRAM={row['train/peak_gpu_memory_bytes'] / 2**30:.2f} GiB\n\n"
    )
    with (stage_dir / "metrics.txt").open("a", encoding="utf-8") as f:
        f.write(text)


def metadata(model, cfg):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    device = next(model.parameters()).device
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "dtype": cfg.training.mixed_precision,
        "model_seed": cfg.training.model_seed,
        "data_seed": cfg.data.data_seed,
        "sampling_rng": "author collator uses model-seeded global torch RNG; identical distribution, not guaranteed identical batch sequence across architectures",
    }


class Monitor:
    def __init__(self, root, cfg):
        self.root = Path(root)
        self.run = None
        self.enabled = cfg.wandb.enabled
        if not self.enabled:
            return
        try:
            with preserve_rng():
                import wandb

                path = self.root / "wandb_run.json"
                identity = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"id": uuid4().hex[:8]}
                )
                write_json(path, identity)
                self.run = wandb.init(
                    project=cfg.wandb.project,
                    name=cfg.wandb.run_name,
                    id=identity["id"],
                    resume="allow",
                    mode=cfg.wandb.mode,
                    config=asdict(cfg),
                    dir=str(self.root),
                )
        except Exception as error:
            self.warn(
                f"W&B initialization failed; local results remain enabled: {error}"
            )

    def warn(self, message):
        warnings.warn(message, RuntimeWarning)
        with (self.root / "warnings.txt").open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log(self, step, payload):
        if self.run is None:
            return
        try:
            with preserve_rng():
                self.run.log({"global_step": step, **payload}, step=step)
        except Exception as error:
            self.warn(f"W&B upload failed: {error}")

    def result(self, payload):
        if self.run is None:
            return
        try:
            with preserve_rng():
                self.run.summary.update(payload)
        except Exception as error:
            self.warn(f"W&B summary failed: {error}")

    def finish(self):
        if self.run is None:
            return
        try:
            with preserve_rng():
                self.run.finish()
        except Exception as error:
            self.warn(f"W&B finalization failed: {error}")
