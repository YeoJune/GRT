"""Preflight actual forward/backward/AdamW through scripts/train.py."""

import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import time
import torch
from grt.checkpoint import seed_all, preserve_rng, evaluation_context
from grt.config import save_config
from grt.data import get_adapter, make_dataset, make_loader, forward_batch
from grt.models.factory import create_model
from grt.logger import write_result_files


def plan_curriculum(
    reports, available_seconds, total_seconds, max_training_seconds=23400
):
    """Select one common paper-budget prefix that fits both measured models."""
    models = list(reports)
    if not models or not reports[models[0]]["stages"]:
        raise ValueError("No measured curriculum")
    stages = reports[models[0]]["stages"]
    signature = lambda rows: [
        (r["num_pairs"], r["key_size"], r["max_steps"]) for r in rows
    ]
    if any(
        signature(report["stages"]) != signature(stages) for report in reports.values()
    ):
        raise ValueError("Models must measure the same curriculum and update budgets")
    costs = {model: 0.0 for model in models}
    count = 0
    overhead, data_reserve, final_reserve = 1.25, 600, 1200
    for index in range(len(stages)):
        for model in models:
            row = reports[model]["stages"][index]
            costs[model] += (
                row["estimated_training_seconds"] + row["estimated_validation_seconds"]
            )
        fits_models = all(
            costs[model] * overhead + data_reserve
            <= min(max_training_seconds, available_seconds[model] - final_reserve)
            for model in models
        )
        fits_total = (
            sum(costs.values()) * overhead
            + len(models) * (data_reserve + final_reserve)
            <= total_seconds
        )
        if not fits_models or not fits_total:
            break
        count = index + 1
    return {
        "models": models,
        "stage_count": count,
        "pairs": [r["num_pairs"] for r in stages[:count]],
        "estimated_seconds": {
            model: sum(
                r["estimated_training_seconds"] + r["estimated_validation_seconds"]
                for r in reports[model]["stages"][:count]
            )
            for model in models
        },
        "overhead_factor": overhead,
        "data_reserve_seconds": data_reserve,
        "final_evaluation_reserve_seconds": final_reserve,
    }


def _optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, cfg.training.adam_beta2),
    )


def _update(model, batch, optimizer, cfg, ctx):
    optimizer.zero_grad(set_to_none=True)
    samples = len(batch["input_ids"])
    micro = cfg.training.batch_size
    for offset in range(0, samples, micro):
        local = {k: v[offset : offset + micro] for k, v in batch.items()}
        with ctx.synchronize(model, offset + micro >= samples):
            loss = forward_batch(model, local, ctx.device).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite preflight loss")
            (loss * (len(local["input_ids"]) / samples)).backward()
    underlying = model.module if hasattr(model, "module") else model
    torch.nn.utils.clip_grad_value_(underlying.parameters(), cfg.training.grad_clip)
    optimizer.step()


def _time_update(model, batch, optimizer, cfg, ctx, steps):
    _update(model, batch, optimizer, cfg, ctx)
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)
    start = time.perf_counter()
    for _ in range(steps):
        _update(model, batch, optimizer, cfg, ctx)
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)
    return ctx.max((time.perf_counter() - start) / steps)


def _eval_time(model, batch, cfg, ctx, steps):
    adapter = get_adapter(cfg.data.name)

    def execute():
        forward_batch(model, batch, ctx.device)
        model.generate(
            batch["input_ids_generate"].to(ctx.device),
            attention_mask=batch["attention_mask_generate"].to(ctx.device),
            **adapter.generation_kwargs(cfg),
        )

    with evaluation_context(model):
        execute()
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
        start = time.perf_counter()
        for _ in range(steps):
            execute()
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
    return ctx.max((time.perf_counter() - start) / steps)


def benchmark(cfg, ctx, steps=3, memory_fraction=0.8):
    from grt.config import validate_config

    validate_config(cfg)
    root = Path(cfg.run.output_dir)
    if ctx.main:
        root.mkdir(parents=True, exist_ok=True)
        save_config(cfg, root / "resolved_config.yaml")
    ctx.barrier()
    rows = []
    calibrated = []
    stages = cfg.training.curriculum or [
        SimpleNamespace(
            num_pairs=cfg.data.train_segments - 1,
            key_size=cfg.data.key_size,
            max_steps=cfg.training.max_steps,
        )
    ]
    with preserve_rng():
        for stage in stages:
            current = copy.deepcopy(cfg)
            get_adapter(cfg.data.name).configure_stage(current, stage)
            current.training.curriculum = []
            current.training.batch_size = (
                getattr(stage, "batch_size", None) or cfg.training.batch_size
            )
            current.training.grad_accum_steps = (
                getattr(stage, "grad_accum_steps", None)
                or cfg.training.grad_accum_steps
            )
            local_batch = (
                current.training.batch_size * current.training.grad_accum_steps
            )
            global_batch = local_batch * ctx.world_size
            if (
                current.training.global_batch_size is not None
                and global_batch != current.training.global_batch_size
            ):
                raise ValueError(
                    f"Expected global batch {current.training.global_batch_size}, got {global_batch}"
                )
            current.data.train_samples = global_batch
            # Both maximum length and one-pair inputs use the native dataset/collator.
            batches = []
            eval_batches = []
            collate_times = []
            for pairs in (1, stage.num_pairs):
                dataset = make_dataset(current, "train", pairs)
                collate_started = time.perf_counter()
                batch = next(
                    iter(
                        make_loader(
                            current,
                            dataset,
                            training=True,
                            vary=False,
                            world_size=ctx.world_size,
                        )
                    )
                )
                batches.append(
                    {k: v[ctx.rank :: ctx.world_size] for k, v in batch.items()}
                )
                collate_times.append(ctx.max(time.perf_counter() - collate_started))
                ev = copy.deepcopy(current)
                ev.data.test_samples = current.evaluation.batch_size * ctx.world_size
                data = make_dataset(ev, "test", pairs)
                batch = next(
                    iter(make_loader(ev, data, vary=False, world_size=ctx.world_size))
                )
                eval_batches.append(
                    {k: v[ctx.rank :: ctx.world_size] for k, v in batch.items()}
                )
            while True:
                seed_all(current.training.model_seed)
                model = create_model(current).to(ctx.device)
                optimizer = _optimizer(model, current)
                if ctx.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(ctx.device)
                error = None
                try:
                    _update(
                        model,
                        batches[-1],
                        optimizer,
                        current,
                        SimpleNamespace(
                            device=ctx.device, synchronize=lambda *args: nullcontext()
                        ),
                    )
                except torch.cuda.OutOfMemoryError:
                    error = "out_of_memory"
                except RuntimeError as e:
                    if ctx.device.type == "cuda" and "out of memory" in str(e).lower():
                        error = "out_of_memory"
                    else:
                        raise
                peak = (
                    torch.cuda.max_memory_allocated(ctx.device)
                    if ctx.device.type == "cuda"
                    else 0
                )
                fraction = (
                    peak / torch.cuda.get_device_properties(ctx.device).total_memory
                    if ctx.device.type == "cuda"
                    else 0
                )
                retry = ctx.max(error is not None or fraction > memory_fraction) > 0
                if not retry:
                    break
                del model, optimizer
                gc.collect()
                if ctx.device.type == "cuda":
                    torch.cuda.empty_cache()
                if current.training.batch_size == 1:
                    raise RuntimeError(
                        "Preflight cannot fit the effective batch at microbatch1"
                    )
                current.training.batch_size //= 2
                while local_batch % current.training.batch_size:
                    current.training.batch_size -= 1
                current.training.grad_accum_steps = (
                    local_batch // current.training.batch_size
                )
                if ctx.main:
                    print(
                        f"Preflight {cfg.model.name} pairs{stage.num_pairs}: reduce microbatch to{current.training.batch_size}",
                        flush=True,
                    )
            trained = ctx.wrap(model)
            short_time = _time_update(
                trained, batches[0], optimizer, current, ctx, steps
            )
            full_time = _time_update(
                trained, batches[1], optimizer, current, ctx, steps
            )
            eval_short = _eval_time(model, eval_batches[0], current, ctx, steps)
            eval_full = _eval_time(model, eval_batches[1], current, ctx, steps)
            peak = (
                int(ctx.max(torch.cuda.max_memory_allocated(ctx.device)))
                if ctx.device.type == "cuda"
                else 0
            )
            expected = (short_time + full_time + sum(collate_times)) / 2
            validations = math.ceil(stage.max_steps / current.evaluation.every_steps)
            evaluation_cost = (
                validations
                * math.ceil(
                    current.data.validation_samples
                    / (current.evaluation.batch_size * ctx.world_size)
                )
                * ((eval_short + eval_full) / 2 + eval_full)
            )
            rows.append(
                {
                    "num_pairs": stage.num_pairs,
                    "key_size": stage.key_size,
                    "max_steps": stage.max_steps,
                    "microbatch_per_gpu": current.training.batch_size,
                    "accumulation": current.training.grad_accum_steps,
                    "world_size": ctx.world_size,
                    "global_batch_size": global_batch,
                    "short_step_seconds": short_time,
                    "maximum_step_seconds": full_time,
                    "estimated_random_step_seconds": expected,
                    "short_collation_seconds": collate_times[0],
                    "maximum_collation_seconds": collate_times[1],
                    "estimated_training_seconds": expected * stage.max_steps,
                    "estimated_validation_seconds": evaluation_cost,
                    "evaluation_batch_seconds": eval_full,
                    "peak_bytes_per_gpu": peak,
                }
            )
            calibrated.append(
                {
                    "num_pairs": stage.num_pairs,
                    "key_size": stage.key_size,
                    "max_steps": stage.max_steps,
                    "batch_size": current.training.batch_size,
                    "grad_accum_steps": current.training.grad_accum_steps,
                }
            )
            if ctx.main:
                print(
                    f"Preflight {cfg.model.name} pairs{stage.num_pairs}: max {full_time:.3f}s/update, peak {peak / 2**30:.2f}GiB",
                    flush=True,
                )
            del trained, model, optimizer, dataset, data, batches, eval_batches
            gc.collect()
            if ctx.device.type == "cuda":
                torch.cuda.empty_cache()
    result = {
        "model": cfg.model.name,
        "stages": rows,
        "estimated_training_and_validation_seconds": sum(
            r["estimated_training_seconds"] + r["estimated_validation_seconds"]
            for r in rows
        ),
        "estimate_note": "Linear interpolation between one and maximum pairs; data preparation, checkpoint I/O and final tests excluded. Wall-time caps remain authoritative.",
        "calibrated_override": {"training": {"curriculum": calibrated}},
    }
    if ctx.main:
        write_result_files(root, "benchmark", result)
    ctx.barrier()
    return result
