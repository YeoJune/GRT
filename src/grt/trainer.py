import math
import time
from pathlib import Path
import torch
from torch import nn
from grt.checkpoint import save_checkpoint, restore_rng, evaluation_context
from grt.data import make_loader, move_batch
from grt.evaluator import evaluate, evaluate_copy_generation, evaluate_remember_generation, autocast_context, check_precision
from grt.metrics import masked_ce
from grt.logger import write_json

def optimizer_groups(model, weight_decay):
    exempt = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if isinstance(module, nn.LayerNorm) or name == "bias" or name.endswith("_bias"):
                exempt.add(id(parameter))
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if id(p) in exempt else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

class Trainer:
    def __init__(self, model, cfg, logger=None, on_evaluate=None):
        self.model, self.cfg = model, cfg
        self.device = next(model.parameters()).device
        check_precision(self.device, cfg.training.mixed_precision)
        self.logger, self.on_evaluate = logger, on_evaluate
        t = cfg.training
        optimizer_cls = torch.optim.Adam if t.optimizer == "adam" else torch.optim.AdamW
        self.optimizer = optimizer_cls(optimizer_groups(model, t.weight_decay), lr=t.lr, betas=(0.9, t.adam_beta2))
        horizon = t.scheduler_steps or t.max_steps
        def schedule(step):
            if step >= horizon:
                return 0.0
            if step < t.warmup_steps:
                return (step + 1) / max(1, t.warmup_steps)
            progress = min(1.0, max(0.0, (step - t.warmup_steps) / max(1, horizon - t.warmup_steps)))
            return 1 - progress if t.scheduler == "linear" else 0.5 * (1 + math.cos(math.pi * progress))
        self.scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=t.plateau_patience, min_lr=t.min_lr)
            if t.scheduler == "plateau" else torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule))
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=t.mixed_precision == "fp16")
        else:  # PyTorch 2.2 compatibility; newer versions use the current API.
            self.scaler = torch.cuda.amp.GradScaler(enabled=t.mixed_precision == "fp16")
        self.global_step = 0
        self.next_train_sample_id = 0
        self.best_validation_loss = math.inf
        self.training_seconds = 0.0
        self.first_target_step = None
        self.first_target_seconds = None
        self.converged = False
        self.stop_reason = "update_budget"
        self.peak_training_gpu_memory_bytes = None
        self.output_dir = Path(cfg.run.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def progress(self):
        return {key: getattr(self, key) for key in ("global_step", "next_train_sample_id", "best_validation_loss",
                "training_seconds", "first_target_step", "first_target_seconds", "converged", "stop_reason",
                "peak_training_gpu_memory_bytes")}

    def resume(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        for key in self.progress():
            setattr(self, key, state.get(key, getattr(self, key)))
        restore_rng(state["rng"])

    def save(self, name):
        save_checkpoint(self.output_dir / name, self.model, self.cfg, self.optimizer,
                        self.scheduler, self.scaler, self.progress())

    def validate(self):
        d = self.cfg.data
        loader = make_loader(d, "validation", d.train_segments, d.validation_samples, self.cfg.evaluation.batch_size)
        metrics = evaluate(self.model, loader, self.cfg.training.mixed_precision)
        if not math.isfinite(metrics["loss"]):
            self.fail("nonfinite validation loss")
        payload = {f"val/{k}": v for k, v in metrics.items()}
        if d.protocol == "paper_copy":
            generation_loader = make_loader(d, "validation", d.train_segments,
                self.cfg.evaluation.autoregressive_samples, self.cfg.evaluation.batch_size)
            generation = evaluate_copy_generation(self.model, generation_loader, self.cfg.training.mixed_precision)
            payload.update({f"val/autoregressive/{k}": v for k, v in generation.items()})
            self.converged = (metrics["loss"] <= 0.05 and metrics["token_accuracy"] >= 0.99
                              and generation["token_accuracy"] >= 0.99 and generation["exact_match"] >= 0.95)
        elif d.protocol == "paper_ar":
            generation_loader = make_loader(d, "validation", d.train_segments,
                self.cfg.evaluation.autoregressive_samples, self.cfg.evaluation.batch_size)
            generation = evaluate_remember_generation(self.model, generation_loader, d.value_size,
                                                      self.cfg.training.mixed_precision)
            payload.update({f"val/autoregressive/{k}": v for k, v in generation.items()})
            self.converged = generation["exact_match"] >= self.cfg.training.convergence_exact_match
        interval = self.cfg.training.plateau_every_steps or self.cfg.evaluation.every_steps
        if self.cfg.training.scheduler == "plateau" and self.global_step % interval == 0:
            self.scheduler.step(metrics["loss"])
        if metrics["loss"] <= 0.05 and self.first_target_step is None:
            self.first_target_step, self.first_target_seconds = self.global_step, self.training_seconds
        if metrics["loss"] < self.best_validation_loss:
            self.best_validation_loss = metrics["loss"]
            self.save("best.pt")
        if self.converged:
            self.save("converged.pt")
        return payload

    def fail(self, reason, **details):
        write_json(self.output_dir / "failure.json", {"reason": reason, "global_step": self.global_step,
                                                       "next_train_sample_id": self.next_train_sample_id, **details})
        raise FloatingPointError(reason)

    def train(self, stop_after=None, time_limit_seconds=None):
        # stop_after is a testable interruption boundary; it does not alter the LR budget.
        stop = self.cfg.training.max_steps if stop_after is None else min(stop_after, self.cfg.training.max_steps)
        t, d = self.cfg.training, self.cfg.data
        self.model.train()
        count = t.batch_size * t.grad_accum_steps
        train_batches = iter(make_loader(d, "train", d.train_segments,
                             max(0, stop-self.global_step)*count, t.batch_size, self.next_train_sample_id))
        session_started = time.perf_counter()
        remaining = (None if t.max_seconds is None else max(0.0, t.max_seconds-self.training_seconds))
        if time_limit_seconds is not None:
            remaining = time_limit_seconds if remaining is None else min(remaining, time_limit_seconds)
        while self.global_step < stop and not (t.stop_on_convergence and self.converged):
            if remaining is not None and time.perf_counter() - session_started >= remaining:
                self.stop_reason = "time_budget"
                break
            started = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            # Materialize only one effective batch so the denominator is exact.
            batches = [move_batch(next(train_batches), self.device) for _ in range(t.grad_accum_steps)]
            self.next_train_sample_id += count
            denominator = sum((b["labels"] != -100).sum().item() for b in batches)
            if not denominator:
                raise ValueError("Effective batch has no labels")
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for b in batches:
                with autocast_context(self.device, t.mixed_precision):
                    out = self.model(b["input_ids"], b["attention_mask"])
                    loss = masked_ce(out.logits, b["labels"], "sum") / denominator
                if not torch.isfinite(loss):
                    self.fail("nonfinite training loss")
                loss_sum += loss.item()
                self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if t.grad_clip_type == "value":
                grad = torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm()
                    for p in self.model.parameters() if p.grad is not None]))
                nn.utils.clip_grad_value_(self.model.parameters(), t.grad_clip)
            else:
                grad = nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
            if not torch.isfinite(grad) and not self.scaler.is_enabled():
                self.fail("nonfinite gradient")
            lr = self.optimizer.param_groups[0]["lr"]
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            skipped = self.scaler.is_enabled() and self.scaler.get_scale() < previous_scale
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                self.peak_training_gpu_memory_bytes = max(self.peak_training_gpu_memory_bytes or 0, peak)
            self.training_seconds += time.perf_counter() - started
            if skipped:
                if self.logger:
                    self.logger.log(self.global_step, {"train/overflow_skip": True, "train/scaler": self.scaler.get_scale()})
                # An overflow is recorded and training stops rather than silently retrying.
                self.fail("fp16 overflow skipped optimizer update", scaler=self.scaler.get_scale())
            if t.scheduler in ("cosine", "linear"):
                self.scheduler.step()
            self.global_step += 1
            payload = {"train/loss": loss_sum, "train/lr": lr, "train/grad_norm": grad.item(),
                       "train/next_sample_id": self.next_train_sample_id, "train/seconds": self.training_seconds}
            if self.peak_training_gpu_memory_bytes is not None:
                payload["train/peak_gpu_memory_bytes"] = self.peak_training_gpu_memory_bytes
            if self.global_step % self.cfg.evaluation.every_steps == 0 or self.global_step == stop:
                payload.update(self.validate())
            if self.on_evaluate and self.cfg.rtla.enabled and self.global_step % self.cfg.rtla.every_steps == 0:
                # The callback handles eval mode; preserve RNG even for custom callbacks.
                with evaluation_context(self.model):
                    callback_metrics = self.on_evaluate(self.model, self.global_step)
                payload.update(callback_metrics or {})
            if self.logger:
                self.logger.log(self.global_step, payload)
            if self.global_step % self.cfg.checkpoint.every_steps == 0:
                self.save(f"step_{self.global_step:06d}.pt")
                self.save("last.pt")
        if self.converged:
            self.stop_reason = "converged"
        if not (self.output_dir / "best.pt").exists() or (self.global_step % self.cfg.evaluation.every_steps and not self.converged):
            payload = self.validate()
            if self.logger:
                self.logger.log(self.global_step, payload)
        self.save("last.pt")
        return self.progress()
