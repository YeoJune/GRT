from __future__ import annotations

import dataclasses
import math

import wandb

from grt.config import GRTConfig, WandbConfig


class WandbLogger:
    def __init__(self, cfg: WandbConfig, full_cfg: GRTConfig) -> None:
        self.enabled = cfg.enabled
        self.run = None
        if not self.enabled:
            return

        self.run = wandb.init(
            project=cfg.project,
            entity=cfg.entity or None,
            name=cfg.run_name or None,
            tags=cfg.tags or [],
            notes=cfg.notes or "",
            group=cfg.group or None,
            config=dataclasses.asdict(full_cfg),
            save_code=cfg.save_code,
        )

    def log_train_step(self, step: int, loss: float, lr: float, grad_norm: float) -> None:
        if not self.enabled:
            return
        wandb.log(
            {
                "train/loss": loss,
                "train/ppl": math.exp(min(loss, 20.0)),
                "train/lr": lr,
                "train/grad_norm": grad_norm,
            },
            step=step,
        )

    def log_eval(self, step: int, loss: float, ppl: float, tokens_per_sec: float, peak_mem_gb: float) -> None:
        if not self.enabled:
            return
        wandb.log(
            {
                "eval/loss": loss,
                "eval/ppl": ppl,
                "eval/tokens_per_sec": tokens_per_sec,
                "eval/peak_mem_gb": peak_mem_gb,
            },
            step=step,
        )

    def log_checkpoint(self, ckpt_path: str, step: int) -> None:
        if not self.enabled:
            return
        artifact = wandb.Artifact(name=f"checkpoint-step-{step}", type="model", metadata={"step": step})
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)

    def alert(self, title: str, text: str, level: str = "warn") -> None:
        if self.enabled:
            wandb.alert(title=title, text=text, level=level)

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()
