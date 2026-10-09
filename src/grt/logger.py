import json
import platform
import subprocess
import warnings
from dataclasses import asdict
from pathlib import Path
import torch
from grt.config import SPEC_VERSION

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def metadata(model, cfg):
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit, git_error = result.stdout.strip(), None
    except (OSError, subprocess.CalledProcessError) as error:
        commit, git_error = None, str(error)
    device = next(model.parameters()).device
    return {"spec_version": SPEC_VERSION, "git_commit": commit, "git_error": git_error,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "dtype": cfg.training.mixed_precision, "model_seed": cfg.training.model_seed,
            "data_seed": cfg.data.data_seed, "rtla_supported": cfg.model.name == "grt",
            "cuda_determinism": "Backend determinism is not guaranteed" if device.type == "cuda" else None}

class Logger:
    def __init__(self, output_dir, cfg):
        self.path = Path(output_dir) / "metrics.jsonl"
        self.wandb = None
        self.run = None
        self.pending_media = {}
        self.warning_path = Path(output_dir) / "warnings.jsonl"
        if cfg.wandb.enabled:
            try:
                import wandb
                self.wandb = wandb
                self.run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config=asdict(cfg))
            except Exception as error:
                self.warn(f"W&B initialization failed: {error}")

    def warn(self, message):
        warnings.warn(message, RuntimeWarning)
        with self.warning_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"warning": message}, ensure_ascii=False) + "\n")

    def log(self, step, payload):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"global_step": step, **payload}, allow_nan=False) + "\n")
        self.upload(step, {**payload, **self.pending_media.pop(step, {})})
        if "val/loss" in payload:
            print(f"step={step} val_loss={payload['val/loss']:.5f} "
                  f"token_accuracy={payload['val/token_accuracy']:.3%} "
                  f"exact_match={payload['val/exact_match']:.3%}", flush=True)
            if "val/autoregressive/token_accuracy" in payload:
                print(f"  autoregressive_accuracy={payload['val/autoregressive/token_accuracy']:.3%} "
                      f"autoregressive_exact_match={payload['val/autoregressive/exact_match']:.3%}", flush=True)
        elif "train/loss" in payload and (step == 1 or step % 100 == 0):
            print(f"step={step} train_loss={payload['train/loss']:.5f} "
                  f"lr={payload['train/lr']:.3g} grad_norm={payload['train/grad_norm']:.3g}", flush=True)

    def queue_trace(self, step, arrays, panel):
        self.pending_media.setdefault(step, {}).update(self.trace_payload(arrays, panel))

    def upload(self, step, payload):
        if self.run is not None:
            try:
                self.run.log(payload, step=step)
            except Exception as error:
                self.warn(f"W&B upload failed: {error}")

    def trace_payload(self, arrays, panel):
        if self.run is None:
            return {}
        try:
            table = [[t, m, float(arrays['r_gates'][t,m]), float(arrays['w_gates'][t,m]),
                      float(arrays['s_norms'][t,m]), float(arrays['update_distances'][t,m])]
                     for t in range(arrays['w_gates'].shape[0]) for m in range(arrays['w_gates'].shape[1])]
            attention = [[t, n, float(v)] for t, row in enumerate(arrays['attn_weights']) for n, v in enumerate(row)]
            return {"rtla/panels": self.wandb.Image(str(panel)),
                    "rtla/registers": self.wandb.Table(columns=["timestep", "register", "read", "write", "state_norm", "update_distance"], data=table),
                    "rtla/attention": self.wandb.Table(columns=["timestep", "token_position", "weight"], data=attention)}
        except Exception as error:
            self.warn(f"W&B trace conversion failed: {error}")
            return {}

    def finish(self):
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as error:
                self.warn(f"W&B finalization failed: {error}")
