from __future__ import annotations

import dataclasses
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grt.config import GRTConfig
from grt.model.grt import GRTModel
from grt.training.metrics import compute_ppl, compute_throughput


class Trainer:
    def __init__(
        self,
        model: GRTModel,
        cfg: GRTConfig,
        dataloader: Iterable[dict[str, Tensor]] | None = None,
        eval_dataloader: Iterable[dict[str, Tensor]] | None = None,
        wandb_logger=None,
        rtla_uploader=None,
        analyzer=None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device or next(model.parameters()).device
        self.wandb_logger = wandb_logger
        self.rtla_uploader = rtla_uploader
        self.analyzer = analyzer

        self.train_loader = dataloader
        self.eval_loader = eval_dataloader
        self._train_iter = None

        no_decay = {"bias", "norm", "ln"}
        decay_params = []
        nodecay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name.lower() for token in no_decay):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": cfg.training.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=cfg.training.lr,
            betas=(0.9, 0.95),
        )

        def lr_lambda(step: int) -> float:
            if step < cfg.training.warmup_steps:
                return float(step + 1) / float(max(cfg.training.warmup_steps, 1))
            progress = (step - cfg.training.warmup_steps) / float(
                max(cfg.training.max_steps - cfg.training.warmup_steps, 1)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _ensure_iter(self, loader):
        if loader is None:
            raise RuntimeError("Trainer: dataloader is required")
        return iter(loader)

    def _next_batch(self) -> dict[str, Tensor]:
        if self._train_iter is None:
            self._train_iter = self._ensure_iter(self.train_loader)
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = self._ensure_iter(self.train_loader)
            return next(self._train_iter)

    def _mixed_precision_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.cfg.training.mixed_precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.cfg.training.mixed_precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def train(self, max_steps: int | None = None) -> list[dict[str, float]]:
        max_steps = max_steps or self.cfg.training.max_steps
        self.model.to(self.device)
        self.model.train()

        metrics_history: list[dict[str, float]] = []
        for step in range(max_steps):
            batch = self._next_batch()
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            do_trace = self.cfg.rtla.enabled and (step % max(self.cfg.rtla.trace_every_n_steps, 1) == 0)

            start_time = time.perf_counter()
            with self._mixed_precision_context():
                out = self.model(input_ids, return_trace=do_trace)
                loss = F.cross_entropy(
                    out.logits.view(-1, self.cfg.model.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            duration = time.perf_counter() - start_time
            train_metrics = {
                "loss": float(loss.item()),
                "ppl": compute_ppl(loss),
                "lr": float(self.scheduler.get_last_lr()[0]),
                "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                "throughput": compute_throughput(int(input_ids.numel()), duration),
            }
            metrics_history.append(train_metrics)

            if self.wandb_logger is not None:
                self.wandb_logger.log_train_step(
                    step=step,
                    loss=train_metrics["loss"],
                    lr=train_metrics["lr"],
                    grad_norm=train_metrics["grad_norm"],
                )

            if do_trace and out.trace is not None:
                trace_path = Path(self.cfg.rtla.output_dir) / f"step_{step:06d}.npz"
                out.trace.save(str(trace_path))
                if self.rtla_uploader is not None:
                    self.rtla_uploader.upload(out.trace, step=step, trace_path=str(trace_path))

            if step % max(self.cfg.logging.eval_every_n_steps, 1) == 0 and self.eval_loader is not None:
                eval_metrics = self.evaluate(max_batches=4)
                if self.wandb_logger is not None:
                    self.wandb_logger.log_eval(step=step, **eval_metrics)

            if step % max(self.cfg.logging.save_every_n_steps, 1) == 0:
                ckpt_path = Path(self.cfg.logging.save_dir) / f"step_{step:06d}.pt"
                save_checkpoint(self.model, self.optimizer, self.scheduler, step, str(ckpt_path))
                if self.wandb_logger is not None and self.cfg.wandb.checkpoint_artifact:
                    self.wandb_logger.log_checkpoint(str(ckpt_path), step=step)

        return metrics_history

    def evaluate(self, max_batches: int | None = None) -> dict[str, float]:
        if self.eval_loader is None:
            raise RuntimeError("Trainer: eval loader is required")

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        total_tokens = 0
        start_time = time.perf_counter()

        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                out = self.model(input_ids, return_trace=False)
                loss = F.cross_entropy(
                    out.logits.view(-1, self.cfg.model.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )
                total_loss += float(loss.item())
                total_batches += 1
                total_tokens += int(input_ids.numel())
                if max_batches is not None and total_batches >= max_batches:
                    break

        duration = time.perf_counter() - start_time
        avg_loss = total_loss / max(total_batches, 1)
        peak_mem_gb = 0.0
        if self.device.type == "cuda":
            peak_mem_gb = float(torch.cuda.max_memory_allocated(self.device) / 1e9)

        self.model.train()
        return {
            "loss": avg_loss,
            "ppl": compute_ppl(avg_loss),
            "tokens_per_sec": compute_throughput(total_tokens, duration),
            "peak_mem_gb": peak_mem_gb,
        }


def save_checkpoint(
    model: GRTModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    ckpt_path: str,
) -> None:
    path_obj = Path(ckpt_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path_obj,
    )


def load_checkpoint(path: str, model: GRTModel, optimizer=None, scheduler=None) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
