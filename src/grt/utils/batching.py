"""CUDA microbatch selection with a fixed effective batch."""
import copy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import torch
from grt.checkpoint import seed_all
from grt.config import Config, build_config, load_config, merge_dicts, validate_config
from grt.models.factory import create_model


def peak_fraction(ctx):
    local = (torch.cuda.max_memory_reserved(ctx.device) /
             torch.cuda.get_device_properties(ctx.device).total_memory) if ctx.device.type == 'cuda' else 0
    return ctx.max(local)


def select_batch(training, device, probe):
    """Try divisors largest-first; probe returns an all-rank fit decision."""
    if not training.auto_batch or device.type != 'cuda':
        return
    local = training.batch_size * training.grad_accum_steps
    for micro in range(local, 0, -1):
        if local % micro:
            continue
        training.batch_size, training.grad_accum_steps = micro, local // micro
        if probe():
            return
    raise RuntimeError('No microbatch fits the VRAM target; effective/evaluation batches were kept fixed')


def calibrate_batch(cfg, ctx, optimizer_factory, update, evaluate):
    t, device = cfg.training, ctx.device
    policy = dict(enabled=t.auto_batch, applied=t.auto_batch and device.type == 'cuda',
                  memory_fraction=t.memory_fraction,
                  configured_microbatch=t.batch_size, configured_accumulation=t.grad_accum_steps)

    def probe():
        model = optimizer = wrapped = None
        seed_all(t.model_seed)
        torch.cuda.reset_peak_memory_stats(device)
        capacity = torch.cuda.get_device_properties(device).total_memory
        def fits(error=False):
            return not ctx.max(error or torch.cuda.max_memory_reserved(device) > capacity * t.memory_fraction)
        try:
            error = False
            try:
                model = create_model(cfg).to(device)
                optimizer = optimizer_factory(model, cfg)
                update(model, model, optimizer, False)
                evaluate(model)
            except torch.cuda.OutOfMemoryError:
                error = True
            # Agree before entering DDP collectives. Leave space for DDP buckets.
            free = (torch.cuda.mem_get_info(device)[0] + torch.cuda.memory_reserved(device)
                    - torch.cuda.memory_allocated(device))
            bucket_bytes = 2 * sum(p.numel() * p.element_size() for p in model.parameters()) if model is not None else 0
            if not fits(error or (ctx.world_size > 1 and free < bucket_bytes)):
                return False
            wrapped = ctx.wrap(model)
            error = False
            try:
                # Local gradients during probing keep rank-local OOM off all-reduce.
                with ctx.synchronize(wrapped, False):
                    for _ in range(2):
                        update(model, wrapped, optimizer, True)
                evaluate(model)
            except torch.cuda.OutOfMemoryError:
                error = True
            return fits(error)
        finally:
            del wrapped, optimizer, model
            gc.collect()
            torch.cuda.empty_cache()

    select_batch(t, device, probe)
    return policy


def apply_batch_override(cfg, override, world_size):
    """Reject a calibration that changes the effective batch or task budget."""
    if set(override) - {'training'} or set(override.get('training', {})) - {'batch_size', 'grad_accum_steps', 'curriculum'}:
        raise ValueError('Calibration may only change microbatch and accumulation')
    result = build_config(Config, merge_dicts(asdict(cfg), override))
    def batches(config):
        t = config.training
        return [(s.num_pairs, s.key_size, s.max_steps,
                 (s.batch_size or t.batch_size) * (s.grad_accum_steps or t.grad_accum_steps))
                for s in t.curriculum] or [(config.data.train_segments-1, config.data.key_size,
                                           t.max_steps, t.batch_size*t.grad_accum_steps)]
    before, after = batches(cfg), batches(result)
    if before != after or any(result.training.global_batch_size is not None
                              and row[-1]*world_size != result.training.global_batch_size for row in after):
        raise ValueError('Calibration must preserve effective batches and task budgets')
    return validate_config(result)


def prepare_batch(cfg, ctx):
    """Reuse current-run calibration or measure it once before CUDA training."""
    if not cfg.training.auto_batch or ctx.device.type != 'cuda':
        return cfg
    root = Path(cfg.run.output_dir)
    stored = root / 'experiment_config.json'
    if stored.exists():
        saved = build_config(Config, json.loads(stored.read_text()))
        override = {'training': {'batch_size': saved.training.batch_size,
                                'grad_accum_steps': saved.training.grad_accum_steps}}
        if saved.training.curriculum:
            override['training']['curriculum'] = asdict(saved)['training']['curriculum']
        result = apply_batch_override(cfg, override, ctx.world_size)
        if asdict(result) != asdict(saved):
            raise ValueError('Existing training conditions changed; use a new run')
        return result
    probe = root.with_name(root.name + '_preflight')
    report_path = probe / 'benchmark.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        requested = load_config(probe / 'resolved_config.yaml', output_dir=root)
        result = apply_batch_override(cfg, report['calibrated_override'], ctx.world_size)
        previous = apply_batch_override(requested, report['calibrated_override'], ctx.world_size)
        if (not report['batch_tuning']['applied']
            or report['world_size'] != ctx.world_size or asdict(result) != asdict(previous)):
            raise ValueError('Preflight conditions changed; use a new run')
        return result
    from grt.benchmark import benchmark
    requested = copy.deepcopy(cfg)
    requested.run.output_dir = str(probe)
    report = benchmark(requested, ctx)
    return apply_batch_override(cfg, report['calibrated_override'], ctx.world_size)
