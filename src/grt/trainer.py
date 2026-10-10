"""Common curriculum trainer, preserving the native author RMT update order."""

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import time
import torch
from transformers import get_linear_schedule_with_warmup
from grt.checkpoint import (
    seed_all,
    capture_rng,
    restore_rng,
    load_checkpoint,
    atomic_save,
    CHECKPOINT_FORMAT,
)
from grt.config import save_config, validate_config
from grt.logger import metadata, write_json, write_result_files, append_metrics, Monitor
from grt.models.factory import create_model
from grt.models.rmt import REFERENCE_COMMIT as COMMIT
from grt.data import get_adapter, make_dataset, make_loader, forward_batch
from grt.evaluator import evaluate, evaluate_lengths
from grt.distributed import DistributedContext


def run(cfg, device, distributed=None):
    ctx = distributed or DistributedContext(device)
    device = ctx.device
    validate_config(cfg)
    root = Path(cfg.run.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Check configuration before initializing optional external monitoring.
    stored = root / "experiment_config.json"
    if ctx.main and stored.exists() and json.loads(stored.read_text()) != asdict(cfg):
        raise ValueError("Configuration changed; use a new output directory")
    if ctx.main and not stored.exists() and any(root.iterdir()):
        raise ValueError("Nonempty output directory without experiment configuration")
    if ctx.main:
        stored.write_text(json.dumps(asdict(cfg), indent=2))
    ctx.barrier()
    logging_cfg = copy.deepcopy(cfg)
    if not ctx.main:
        logging_cfg.wandb.enabled = False
    monitor = Monitor(root, logging_cfg)
    try:
        return _run(cfg, device, monitor, ctx)
    finally:
        monitor.finish()


def _run(cfg, device, monitor, ctx):
    if cfg.training.mixed_precision != "fp32":
        raise ValueError("The author reference path currently requires fp32")
    root = Path(cfg.run.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if ctx.main:
        save_config(cfg, root / "resolved_config.yaml")
    stages = cfg.training.curriculum or [
        SimpleNamespace(
            num_pairs=cfg.data.train_segments - 1,
            key_size=cfg.data.key_size,
            max_steps=cfg.training.max_steps,
        )
    ]
    results = []
    previous = None
    started = time.perf_counter()
    global_offset = 0
    adapter = get_adapter(cfg.data.name)
    old_summary = root / "summary.json"
    prior_wall = (
        json.loads(old_summary.read_text()).get("wall_seconds", 0)
        if old_summary.exists()
        else 0
    )
    for index, stage in enumerate(stages):
        current = copy.deepcopy(cfg)
        adapter.configure_stage(current, stage)
        current.training.max_steps = stage.max_steps
        current.training.batch_size = (
            getattr(stage, "batch_size", None) or current.training.batch_size
        )
        current.training.grad_accum_steps = (
            getattr(stage, "grad_accum_steps", None)
            or current.training.grad_accum_steps
        )
        actual_batch = (
            current.training.batch_size
            * current.training.grad_accum_steps
            * ctx.world_size
        )
        if (
            current.training.global_batch_size is not None
            and actual_batch != current.training.global_batch_size
        ):
            raise ValueError(
                f"Expected global batch {current.training.global_batch_size}, got {actual_batch}; check torchrun world_size and stage batch settings"
            )
        if current.data.train_samples % ctx.world_size:
            raise ValueError(
                "Distributed training requires train_samples divisible by world_size"
            )
        current.training.curriculum = []
        stage_dir = (
            root / f"stage_{index + 1:02d}_pairs{stage.num_pairs}_key{stage.key_size}"
        )
        stage_dir.mkdir(exist_ok=True)
        if ctx.main:
            save_config(current, stage_dir / "resolved_config.yaml")
        seed_all(current.training.model_seed)
        model = create_model(current).to(device)
        provenance = metadata(model, current)
        provenance["world_size"] = ctx.world_size
        provenance["global_batch_size"] = (
            current.training.batch_size
            * current.training.grad_accum_steps
            * ctx.world_size
        )
        provenance["reference_source"] = json.loads(
            (Path(__file__).parent / "vendor/armt/SOURCE.json").read_text()
        )
        if ctx.main and not (root / "metadata.json").exists():
            write_json(root / "metadata.json", provenance)
        provenance["loss_convention"] = "raw labels and shifted predictor mask"
        if ctx.main and not (stage_dir / "metadata.json").exists():
            write_json(stage_dir / "metadata.json", provenance)
        monitor.result({"metadata": provenance})
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=current.training.lr,
            weight_decay=current.training.weight_decay,
            betas=(0.9, current.training.adam_beta2),
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, stage.max_steps // 10, stage.max_steps * 2
        )
        train = make_dataset(current, "train", stage.num_pairs)
        valid = make_dataset(current, "validation", stage.num_pairs)
        loader = make_loader(current, train, training=True, world_size=ctx.world_size)
        step = epoch_batch = samples_seen = 0
        best = -1.0
        last_validation = None
        done = False
        training_seconds = 0.0
        peak_memory = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        path = stage_dir / "last.pt"
        if path.exists():
            state = load_checkpoint(path)
            if state["format"] != CHECKPOINT_FORMAT:
                raise ValueError(
                    "Legacy checkpoint resume requires its original checkout"
                )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            saved_world = state.get("world_size", 1)
            if saved_world != ctx.world_size:
                raise ValueError("Resume requires unchanged world_size")
            restore_rng(state.get("rng_by_rank", [state["rng"]])[ctx.rank])
            step = state["step"]
            epoch_batch = state["epoch_batch"]
            best = state["best_exact_match"]
            done = state["done"]
            last_validation = state["validation"]
            training_seconds = state.get("training_seconds", 0.0)
            peak_memory = state.get("peak_training_gpu_memory_bytes", 0)
            samples_seen = state.get("train_samples_seen", 0)
        elif previous is not None:
            model.load_state_dict(previous)
        training_model = ctx.wrap(model)
        iterator = iter(loader)
        # Rebuild the loader cursor without changing the checkpoint RNG used by the collator.
        if epoch_batch:
            rng = capture_rng()
            for _ in range(epoch_batch):
                next(iterator)
            restore_rng(rng)

        def save(target, done_flag):
            states = ctx.rng_states(capture_rng())
            if ctx.main:
                atomic_save(
                    {
                        "format": CHECKPOINT_FORMAT,
                        "reference_commit": COMMIT,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "rng": states[0],
                        "rng_by_rank": states,
                        "world_size": ctx.world_size,
                        "step": step,
                        "epoch_batch": epoch_batch,
                        "best_exact_match": best,
                        "validation": last_validation,
                        "done": done_flag,
                        "training_seconds": training_seconds,
                        "peak_training_gpu_memory_bytes": peak_memory,
                        "train_samples_seen": samples_seen,
                        "config": asdict(current),
                    },
                    target,
                )
            ctx.barrier()

        model.train()
        while step < stage.max_steps and not done:
            if ctx.decision(
                cfg.training.max_seconds is not None
                and prior_wall + time.perf_counter() - started
                >= cfg.training.max_seconds
            ):
                break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            try:
                effective_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                epoch_batch = 0
                effective_batch = next(iterator)
            epoch_batch += 1
            samples_seen += len(effective_batch["input_ids"])
            effective_batch = {
                k: v[ctx.rank :: ctx.world_size] for k, v in effective_batch.items()
            }
            local_samples = len(effective_batch["input_ids"])
            for start in range(
                0, len(effective_batch["input_ids"]), current.training.batch_size
            ):
                batch = {
                    k: v[start : start + current.training.batch_size]
                    for k, v in effective_batch.items()
                }
                last_microbatch = start + current.training.batch_size >= local_samples
                with ctx.synchronize(training_model, last_microbatch):
                    output = forward_batch(training_model, batch, device)
                    loss = output.loss * (len(batch["input_ids"]) / local_samples)
                    loss.backward()
                loss_total += loss.item()
            loss_total = ctx.sum([loss_total])[0] / ctx.world_size
            torch.nn.utils.clip_grad_value_(
                model.parameters(), current.training.grad_clip
            )
            optimizer.step()
            scheduler.step()
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
            training_seconds += ctx.max(time.perf_counter() - update_started)
            peak_memory = int(ctx.max(peak_memory))
            if step % current.evaluation.every_steps == 0 or step == stage.max_steps:
                # Original random-length validation plus an explicit maximum-length functional check.
                last_validation = evaluate(
                    model,
                    current,
                    valid,
                    device,
                    vary=current.data.vary_n_pairs,
                    distributed=ctx if ctx.world_size > 1 else None,
                )
                fixed = evaluate(
                    model,
                    current,
                    valid,
                    device,
                    vary=False,
                    distributed=ctx if ctx.world_size > 1 else None,
                )
                last_validation["fixed_length"] = fixed
                score = (
                    fixed["exact_match"]
                    if current.training.stop_on_convergence
                    else last_validation["exact_match"]
                )
                if score > best:
                    best = score
                    save(stage_dir / "best.pt", False)
                done = (
                    current.training.stop_on_convergence
                    and fixed["exact_match"] >= current.training.convergence_exact_match
                )
                row = {
                    "step": step,
                    "train/loss": loss_total,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/seconds": training_seconds,
                    "train/peak_gpu_memory_bytes": peak_memory,
                    "validation": last_validation,
                }
                row["train/samples"] = samples_seen
                if ctx.main:
                    append_metrics(stage_dir, row)
                payload = {
                    "stage_step": step,
                    "num_pairs": stage.num_pairs,
                    "key_size": stage.key_size,
                    **{k: v for k, v in row.items() if k.startswith("train/")},
                    "val/random/loss": last_validation["loss"],
                    "val/random/exact_match": last_validation["exact_match"],
                    "val/fixed/loss": fixed["loss"],
                    "val/fixed/exact_match": fixed["exact_match"],
                }
                if (
                    ctx.main
                    and cfg.rtla.enabled
                    and callable(getattr(model, "forward_with_trace", None))
                ):
                    from grt.rtla import log_trace

                    payload.update(
                        log_trace(model, current, valid, device, stage_dir, step)
                    )
                monitor.log(global_offset + step, payload)
                if ctx.main:
                    print(
                        f"{cfg.model.name.upper()} {stage.num_pairs} pairs step {step}: loss={loss_total:.4f}, fixed EM={fixed['exact_match']:.3f}",
                        flush=True,
                    )
                save(path, done)
            elif step % cfg.wandb.every_steps == 0:
                monitor.log(
                    global_offset + step,
                    {
                        "stage_step": step,
                        "num_pairs": stage.num_pairs,
                        "train/loss": loss_total,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/samples": samples_seen,
                        "train/seconds": training_seconds,
                        "train/peak_gpu_memory_bytes": peak_memory,
                    },
                )
        if last_validation is None:
            last_validation = evaluate(
                model,
                current,
                valid,
                device,
                vary=False,
                distributed=ctx if ctx.world_size > 1 else None,
            )
            last_validation["fixed_length"] = dict(last_validation)
            best = last_validation["exact_match"]
            save(stage_dir / "best.pt", done)
        done = done or step >= stage.max_steps
        save(path, done)
        selected = torch.load(
            stage_dir / "best.pt", map_location="cpu", weights_only=False
        )
        previous = selected["model"]
        converged = (
            last_validation["fixed_length"]["exact_match"]
            >= current.training.convergence_exact_match
        )
        results.append(
            {
                "num_pairs": stage.num_pairs,
                "key_size": stage.key_size,
                "step": step,
                "completed": done,
                "converged": converged,
                "train_samples_seen": samples_seen,
                "checkpoint": str(stage_dir / "best.pt"),
                "training_seconds": training_seconds,
                "peak_training_gpu_memory_bytes": peak_memory,
                "stop_reason": "converged"
                if current.training.stop_on_convergence and converged
                else ("update_budget" if done else "time_budget"),
                "validation": last_validation,
            }
        )
        status = "completed" if done and index == len(stages) - 1 else "interrupted"
        if current.training.stop_on_convergence and done and not converged:
            status = "stage_not_converged"
        if ctx.main:
            write_result_files(
                root,
                "summary",
                {
                    "reference_commit": COMMIT,
                    "status": status,
                    "stages": results,
                    "wall_seconds": prior_wall + time.perf_counter() - started,
                },
            )
        monitor.result({"status": status, "stages": results})
        global_offset += step
        if not done or (current.training.stop_on_convergence and not converged):
            break
        del training_model, train, valid, loader, iterator
        optimizer = scheduler = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model.load_state_dict(previous)
    test = make_dataset(current, "test", stage.num_pairs)
    evaluation = evaluate(
        model,
        current,
        test,
        device,
        vary=False,
        distributed=ctx if ctx.world_size > 1 else None,
    )
    if ctx.main:
        write_result_files(
            root,
            "evaluation",
            {
                "checkpoint": str(stage_dir / "best.pt"),
                "checkpoint_step": selected["step"],
                "num_pairs": stage.num_pairs,
                **evaluation,
            },
        )
    if current.evaluation.pair_counts:
        lengths = evaluate_lengths(
            model, current, device, ctx if ctx.world_size > 1 else None
        )
        if ctx.main:
            write_result_files(
                root,
                "length_evaluation",
                {
                    "checkpoint": str(stage_dir / "best.pt"),
                    "checkpoint_step": selected["step"],
                    "trained_num_pairs": stage.num_pairs,
                    "key_size": current.data.key_size,
                    "results": lengths,
                },
            )
    monitor.result(
        {
            "test/loss": evaluation["loss"],
            "test/exact_match": evaluation["exact_match"],
            "test/samples": evaluation["samples"],
        }
    )
    return results
