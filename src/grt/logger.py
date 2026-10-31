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
from grt.monitoring import configure_wandb, history_metrics, summary_metrics


class TrainingMetrics:
    """Sample-weighted means across microbatches and ranks."""
    def __init__(self):
        self.values = {}

    def update(self, output, weight):
        metrics = getattr(output, 'training_metrics', None) or {'train/loss': output.loss.detach()}
        keys = list(metrics)
        values = torch.stack([metrics[k].detach().float() * weight for k in keys]).cpu().tolist()
        for key, value in zip(keys, values):
            self.values[key] = self.values.get(key, 0) + value

    def reduce(self, ctx):
        return dict(zip(self.values, [v / ctx.world_size for v in ctx.sum(list(self.values.values()))]))


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


def metadata(model, cfg, *, repo_dir=None):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=repo_dir, stderr=subprocess.DEVNULL
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
        "parameters_router": (sum(p.numel() for p in model.router.parameters())
                              if cfg.model.name == 'grt' else 0),
        "parameters_write_head": (sum(p.numel() for p in model.router.mlp_w.parameters())
                                  if cfg.model.name == 'grt' else 0),
        "parameters_gate_head": (sum(p.numel() for p in model.router.gate_head.parameters())
                                 if cfg.model.name == 'grt' and hasattr(model.router, 'gate_head') else 0),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "dtype": cfg.training.mixed_precision,
        "model_seed": cfg.training.model_seed,
        "data_seed": cfg.data.data_seed,
        "sampling_rng": ("ordered token streams; no shuffle" if cfg.data.name == "wikitext103" else
            "data-seeded private collator RNG" if cfg.data.options.get("objective") == "parallel" else
            "author collator uses model-seeded global torch RNG; identical distribution, not guaranteed identical batch sequence across architectures"),
        "implementation": cfg.model.implementation,
        "attention_mode": cfg.model.attention_mode,
        "memory_read": {"neox": "none", "grt": "segment_start_state", "rmt": "read_block"}[cfg.model.name],
        "objective": cfg.data.options.get("objective", "next_token"),
    }


class Monitor:
    def __init__(self, root, cfg):
        self.root = Path(root)
        self.run = None
        self.enabled = cfg.wandb.enabled
        self.stream = cfg.data.name == 'wikitext103'
        if not self.enabled:
            return
        self.status = dict(enabled=True, project=cfg.wandb.project, mode=cfg.wandb.mode, failures=0)
        try:
            with preserve_rng():
                import wandb

                previous = self.root / 'wandb_status.json'
                if previous.exists():
                    self.status['last_logged_step'] = json.loads(previous.read_text()).get('last_logged_step', 0)

                path = self.root / "wandb_run.json"
                identity = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"id": uuid4().hex[:8]}
                )
                write_json(path, identity)
                self.run = wandb.init(
                    project=cfg.wandb.project,
                    name=cfg.wandb.run_name or f'{cfg.data.name}-{cfg.model.name}-{self.root.name}',
                    id=identity["id"],
                    resume="allow",
                    mode=cfg.wandb.mode,
                    config=asdict(cfg),
                    dir=str(self.root),
                )
                configure_wandb(self.run, self.stream)
                mode = getattr(getattr(self.run, 'settings', None), 'mode', cfg.wandb.mode)
                url = getattr(self.run, 'url', None) if mode == 'online' else None
                self._status('connected', id=identity['id'], mode=mode, url=url)
                print(f"W&B {mode}: {url or identity['id']}", flush=True)
        except Exception as error:
            self._status('initialization_failed', error=str(error))
            self.warn(
                f"W&B initialization failed; local results remain enabled: {error}"
            )

    def _status(self, state, **details):
        if state.endswith('_failed'):
            self.status['failures'] += 1
            self.status['last_error'] = details.get('error')
        self.status.update(state=state, **details)
        write_json(self.root / 'wandb_status.json', self.status)

    def warn(self, message):
        warnings.warn(message, RuntimeWarning)
        with (self.root / "warnings.txt").open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log(self, step, payload):
        if self.run is None:
            return
        try:
            with preserve_rng():
                self.run.log({"global_step": step, **history_metrics(payload, self.stream)}, step=step)
            self._status('active', last_logged_step=step, error=None)
        except Exception as error:
            self._status('upload_failed', error=str(error))
            self.warn(f"W&B upload failed: {error}")

    def result(self, payload):
        if self.run is None:
            return
        try:
            with preserve_rng():
                self.run.summary.update(summary_metrics(payload))
            self._status('active', summary_updated=True)
        except Exception as error:
            self._status('summary_failed', error=str(error))
            self.warn(f"W&B summary failed: {error}")

    def finish(self, *, exit_code=0):
        if self.run is None:
            return
        try:
            with preserve_rng():
                self.run.finish(exit_code=exit_code)
            self._status('finished' if exit_code == 0 else 'failed', exit_code=exit_code)
        except Exception as error:
            self._status('finish_failed', error=str(error))
            self.warn(f"W&B finalization failed: {error}")
