import time
from contextlib import nullcontext
import torch
from grt.checkpoint import evaluation_context
from grt.data import make_loader, move_batch
from grt.metrics import MetricAccumulator

def check_precision(device, precision):
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("Unsupported precision")
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed precision currently requires CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")

def autocast_context(device, precision):
    check_precision(device, precision)
    return nullcontext() if precision == "fp32" else torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)

def evaluate(model, loader, precision="fp32"):
    device = next(model.parameters()).device
    totals = MetricAccumulator()
    with evaluation_context(model):
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast_context(device, precision):
                output = model(batch["input_ids"], batch["attention_mask"])
            totals.update(output.logits, batch["labels"])
    return totals.compute()

def evaluate_copy_generation(model, loader, precision="fp32"):
    """Free-running Copy accuracy, with no answer-prefix teacher forcing."""
    device = next(model.parameters()).device
    correct = exact = samples = tokens = first_correct = second_correct = 0
    with evaluation_context(model):
        for batch in loader:
            batch = move_batch(batch, device)
            n = model.cfg.segment_len
            with autocast_context(device, precision):
                prediction = model.generate_copy(batch["input_ids"][:, :n])
            target = batch["labels"][:, n:]
            matches = prediction == target
            correct += matches.sum().item()
            first_correct += matches[:, :n].sum().item()
            second_correct += matches[:, n:].sum().item()
            exact += matches.all(dim=1).sum().item()
            samples += matches.shape[0]
            tokens += matches.numel()
    return {"token_accuracy": correct/tokens, "exact_match": exact/samples,
            "first_copy_accuracy": first_correct/(tokens/2),
            "second_copy_accuracy": second_correct/(tokens/2), "samples": samples}


def evaluate_remember_generation(model, loader, value_size, precision="fp32"):
    device = next(model.parameters()).device
    tokens = samples = correct = exact = value_exact = 0
    with evaluation_context(model):
        for batch in loader:
            batch = move_batch(batch, device)
            target = batch["input_ids"][:, -value_size-1:]
            prompt = batch["input_ids"][:, :-value_size-1]
            with autocast_context(device, precision):
                prediction = model.generate_answer(prompt, value_size + 1)
            matches = prediction == target
            correct += matches.sum().item()
            exact += matches.all(dim=1).sum().item()
            value_exact += matches[:, :value_size].all(dim=1).sum().item()
            samples += matches.shape[0]
            tokens += matches.numel()
    return {"token_accuracy": correct/tokens, "exact_match": exact/samples,
            "value_exact_match": value_exact/samples, "samples": samples}

def measure_performance(model, batch, precision="fp32", warmup=10, iterations=50):
    if warmup < 0 or iterations <= 0:
        raise ValueError("Invalid measurement counts")
    device = next(model.parameters()).device
    batch = move_batch(batch, device)  # Transfer outside timing.
    with evaluation_context(model), torch.inference_mode(), autocast_context(device, precision):
        for _ in range(warmup):
            model(batch["input_ids"], batch["attention_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(batch["input_ids"], batch["attention_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    b, l = batch["input_ids"].shape
    return {"input_tokens_per_sec": b*l*iterations/elapsed, "samples_per_sec": b*iterations/elapsed,
            "peak_allocated_gpu_memory_bytes": peak, "batch_size": b, "precision": precision,
            "warmup": warmup, "iterations": iterations, "measurement_seconds": elapsed}

def evaluate_lengths(model, cfg, progress=None):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = []
    for segments in cfg.data.eval_segments:
        loader = make_loader(cfg.data, "test", segments, cfg.data.test_samples, cfg.evaluation.batch_size)
        metrics = evaluate(model, loader, cfg.training.mixed_precision)
        performance = measure_performance(model, next(iter(loader)), cfg.training.mixed_precision,
                                           cfg.evaluation.warmup, cfg.evaluation.iterations)
        row = {"task": cfg.data.task, "model": cfg.model.name, "size": cfg.model.size,
               "protocol": cfg.data.protocol,
               "seed": cfg.training.model_seed, "T": segments, "L": segments*cfg.model.segment_len,
               "total_parameters": total, "trainable_parameters": trainable,
               "internal_segment_length": model.internal_segment_length,
               **metrics, **performance}
        if cfg.data.protocol == "paper_copy":
            generation_loader = make_loader(cfg.data, "test", segments, cfg.evaluation.autoregressive_samples,
                                            cfg.evaluation.batch_size)
            row.update({f"autoregressive/{k}": v for k, v in
                        evaluate_copy_generation(model, generation_loader, cfg.training.mixed_precision).items()})
        elif cfg.data.protocol == "paper_ar":
            generation_loader = make_loader(cfg.data, "test", segments, cfg.evaluation.autoregressive_samples,
                                            cfg.evaluation.batch_size)
            row.update({f"autoregressive/{k}": v for k, v in evaluate_remember_generation(
                model, generation_loader, cfg.data.value_size, cfg.training.mixed_precision).items()})
            row["num_pairs"] = segments - 1
        if progress:
            row.update({k: progress.get(k) for k in ("first_target_step", "first_target_seconds", "global_step", "best_validation_loss", "training_seconds")})
        rows.append(row)
    return rows
