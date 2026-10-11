"""Stateful token-stream training selected through a dataset capability."""
from collections import deque
from contextlib import nullcontext
import copy
from dataclasses import asdict
import gc
import math
from pathlib import Path
import json
import time
import torch
from transformers import get_cosine_schedule_with_warmup
from grt.checkpoint import (CHECKPOINT_FORMAT, seed_all, capture_rng, restore_rng,
                            atomic_save, load_checkpoint, preserve_rng, evaluation_context)
from grt.config import save_config
from grt.logger import metadata, write_json, write_result_files
from grt.models.factory import create_model
from grt.utils.batching import calibrate_batch, peak_fraction


def train_stream(adapter, cfg, device, monitor, ctx):
    root = Path(cfg.run.output_dir)
    t = cfg.training
    global_batch = t.batch_size * t.grad_accum_steps * ctx.world_size
    if t.global_batch_size is not None and global_batch != t.global_batch_size:
        raise ValueError("microbatch * accumulation * world_size must equal global batch")
    seed_all(t.model_seed)
    model = create_model(cfg).to(device)
    wrapped = ctx.wrap(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=t.lr,
        betas=(0.9, t.adam_beta2), weight_decay=t.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, t.max_steps)
    train_tokens = adapter.dataset(cfg, 'train')
    validation = adapter.dataset(cfg, 'validation')
    train = adapter.streams(train_tokens, global_batch)[ctx.rank::ctx.world_size]
    manifest = json.loads((Path(cfg.data.options['path'])/'manifest.json').read_text())
    if ctx.main:
        save_config(cfg, root/'resolved_config.yaml')
        info = metadata(model, cfg)
        info.update(tokenizer=manifest, global_batch_size=global_batch,
            bptt_previous_segments=t.bptt_segments, loss_convention='adapter-aligned next token targets',
            score_kind=adapter.score_kind(cfg), stream_reset='split and epoch; continuous across document EOS',
            parameters_backbone=sum(p.numel() for p in model.backbone.parameters()),
            discarded_train_tail=train_tokens.numel()-train.numel()*ctx.world_size)
        write_json(root/'metadata.json', info)
    step = offset = tokens_seen = 0
    best = float('inf')
    boundaries = deque([None], maxlen=t.bptt_segments+1)
    history = deque(maxlen=t.bptt_segments)
    path = root/'last.pt'
    elapsed = 0.0
    peak = 0
    if path.exists():
        saved = load_checkpoint(path)
        if saved['world_size'] != ctx.world_size:
            raise ValueError('Resume requires the same world_size')
        if saved['dataset_manifest'] != manifest:
            raise ValueError('Prepared dataset changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        local = saved['stream_by_rank'][ctx.rank]
        boundaries = deque([s.to(device) if s is not None else None for s in local['boundaries']],
            maxlen=t.bptt_segments+1)
        history = deque(local['history'], maxlen=t.bptt_segments)
        restore_rng(saved['rng_by_rank'][ctx.rank])
        step, offset, tokens_seen, best, elapsed, peak = (
            saved[k] for k in ['step','offset','target_tokens_seen','best_loss','elapsed_seconds','peak_gpu_bytes'])
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def save(destination):
        rngs = ctx.rng_states(capture_rng())
        local = dict(boundaries=[s.detach().cpu() if s is not None else None for s in boundaries],
            history=list(history))
        streams = ctx.rng_states(local)
        if ctx.main:
            atomic_save(dict(format=CHECKPOINT_FORMAT, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), config=asdict(cfg),
                step=step, offset=offset, target_tokens_seen=tokens_seen, best_loss=best,
                elapsed_seconds=elapsed+time.perf_counter()-started, peak_gpu_bytes=peak,
                world_size=ctx.world_size, rng_by_rank=rngs, stream_by_rank=streams,
                dataset_manifest=manifest), destination)
        ctx.barrier()

    last_validation = None
    validation_step = -1
    model.train()
    while step < t.max_steps:
        if ctx.decision(t.max_seconds is not None and elapsed+time.perf_counter()-started >= t.max_seconds):
            break
        if offset >= train.shape[1]-1:
            offset = 0
            history.clear()
            boundaries.clear()
            boundaries.append(None)
        n = min(cfg.model.segment_len, train.shape[1]-1-offset)
        ids = train[:, offset:offset+n]
        targets = train[:, offset+1:offset+n+1]
        window = torch.cat([*history, ids], 1)
        aligned = torch.full_like(window, -100)
        aligned[:, -n:] = targets
        initial = boundaries[0]
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        final_states = []
        local_batch = train.shape[0]
        for start in range(0, local_batch, t.batch_size):
            end = min(start+t.batch_size, local_batch)
            with ctx.synchronize(wrapped, end == local_batch):
                out = wrapped(input_ids=window[start:end].to(device),
                    target_ids=aligned[start:end].to(device),
                    memory_state=initial[start:end] if initial is not None else None)
                if not torch.isfinite(out.loss):
                    raise RuntimeError('Non-finite stream loss')
                loss = out.loss * ((end-start)/local_batch)
                loss.backward()
            batch_loss += loss.item()
            final_states.append(out.memory_state.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        history.append(ids.clone())
        boundaries.append(torch.cat(final_states))
        offset += n
        step += 1
        tokens_seen += global_batch*n
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            peak = int(ctx.max(torch.cuda.max_memory_allocated(device)))
        loss_value = ctx.sum([batch_loss])[0]/ctx.world_size
        validation_due = step % cfg.evaluation.every_steps == 0 or step == t.max_steps
        payload = {'train/loss':loss_value, 'train/lr':optimizer.param_groups[0]['lr'],
            'train/target_tokens':tokens_seen,
            'train/elapsed_seconds':elapsed+time.perf_counter()-started,
            'train/peak_gpu_memory_bytes':peak}
        if validation_due:
            last_validation = adapter.evaluate(model, cfg, validation, device)
            validation_step = step
            if last_validation['loss'] < best:
                best = last_validation['loss']
                save(root/'best.pt')
            payload.update({'val/loss':last_validation['loss'], 'val/perplexity':last_validation['perplexity']})
        if validation_due or step == 1 or step % cfg.wandb.every_steps == 0:
            if ctx.main:
                row = dict(step=step, **payload)
                if validation_due:
                    row['validation'] = last_validation
                with (root/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                with (root/'metrics.txt').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                suffix = f", val PPL={last_validation['perplexity']:.3f}" if validation_due else ''
                print(f"Step {step}: loss={loss_value:.5f}{suffix}", flush=True)
            monitor.log(step, payload)
        if validation_due:
            save(path)
    if validation_step != step:
        last_validation = adapter.evaluate(model, cfg, validation, device)
        if last_validation['loss'] < best or not (root/'best.pt').exists():
            best = last_validation['loss']
            save(root/'best.pt')
    save(path)
    model.load_state_dict(load_checkpoint(root/'best.pt')['model'])
    result = adapter.evaluate(model, cfg, adapter.dataset(cfg, 'test'), device)
    summary = dict(status='completed' if step == t.max_steps else 'interrupted',
        step=step, target_tokens_seen=tokens_seen, validation=last_validation, validation_step=step,
        wall_seconds=elapsed+time.perf_counter()-started, peak_gpu_memory_bytes=peak,
        updates_remaining=t.max_steps-step, **result)
    if ctx.main:
        write_result_files(root, 'summary', summary)
        write_result_files(root, 'evaluation', result)
    monitor.result(summary)
    return summary


def benchmark_stream(adapter, cfg, ctx, steps):
    """Measure the same replay, full-softmax, Adam and clipping as stream training."""
    current = copy.deepcopy(cfg)
    t, device = current.training, ctx.device
    local_batch = t.batch_size * t.grad_accum_steps
    global_batch = local_batch * ctx.world_size
    if t.global_batch_size is not None and t.global_batch_size != global_batch:
        raise ValueError('microbatch * accumulation * world_size must equal global batch')
    root = Path(cfg.run.output_dir)
    if ctx.main:
        root.mkdir(parents=True, exist_ok=True)
        save_config(cfg, root/'resolved_config.yaml')
    ctx.barrier()
    train = adapter.streams(adapter.dataset(cfg, 'train'), global_batch)[ctx.rank::ctx.world_size]
    length = cfg.model.segment_len * (t.bptt_segments+1)
    if train.shape[1] <= length:
        raise ValueError('Preflight requires at least a full BPTT window per stream')
    ids = train[:, :length]
    targets = torch.full_like(ids, -100)
    targets[:, -cfg.model.segment_len:] = train[:, length-cfg.model.segment_len+1:length+1]
    validation = adapter.streams(adapter.dataset(cfg, 'validation'), cfg.evaluation.batch_size)
    eval_n = min(cfg.model.segment_len, validation.shape[1]-1)

    def update(model, wrapped, optimizer, distributed):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, local_batch, t.batch_size):
            end = start+t.batch_size
            sync = ctx.synchronize(wrapped, end == local_batch) if distributed else nullcontext()
            with sync:
                out = wrapped(input_ids=ids[start:end].to(device), target_ids=targets[start:end].to(device),
                    memory_state=model.initial_state(t.batch_size).detach())
                if not torch.isfinite(out.loss):
                    raise RuntimeError('Non-finite preflight loss')
                (out.loss/t.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()

    def eval_segment(model):
        with evaluation_context(model):
            model(validation[:, :eval_n].to(device),
                target_ids=validation[:, 1:eval_n+1].to(device))

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    with preserve_rng():
        def optimizer_factory(model, config):
            return torch.optim.Adam(model.parameters(), lr=t.lr,
                betas=(0.9, t.adam_beta2), weight_decay=t.weight_decay)
        policy = calibrate_batch(current, ctx, optimizer_factory, update, eval_segment)
        seed_all(t.model_seed)
        model = create_model(current).to(device)
        optimizer = optimizer_factory(model, current)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        wrapped = ctx.wrap(model)
        update(model, wrapped, optimizer, True)
        eval_segment(model)
        synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            update(model, wrapped, optimizer, True)
        synchronize()
        update_seconds = ctx.max((time.perf_counter()-started)/steps)
        started = time.perf_counter()
        for _ in range(steps):
            eval_segment(model)
        synchronize()
        eval_seconds = ctx.max((time.perf_counter()-started)/steps)
        peak = int(ctx.max(torch.cuda.max_memory_allocated(device))) if device.type == 'cuda' else 0
        reserved = int(ctx.max(torch.cuda.max_memory_reserved(device))) if device.type == 'cuda' else 0
        fraction = peak_fraction(ctx)
        if policy['applied'] and fraction > t.memory_fraction:
            raise RuntimeError('Measured DDP peak exceeds the VRAM target; calibration cannot be used')
        eval_batches = math.ceil((validation.shape[1]-1)/cfg.model.segment_len)
        validations = math.ceil(t.max_steps/cfg.evaluation.every_steps)
        result = dict(model=cfg.model.name, objective='next_token', score_kind=adapter.score_kind(cfg),
            batch_tuning=policy, microbatch_per_gpu=t.batch_size, accumulation=t.grad_accum_steps, world_size=ctx.world_size,
            global_batch_size=global_batch, previous_segments=t.bptt_segments,
            replay_tokens_per_stream=length, target_tokens_per_update=global_batch*cfg.model.segment_len,
            update_seconds=update_seconds, evaluation_segment_seconds=eval_seconds,
            peak_bytes_per_gpu=peak, reserved_bytes_per_gpu=reserved, peak_reserved_fraction=fraction,
            estimated_training_seconds=update_seconds*t.max_steps,
            estimated_validation_seconds=eval_seconds*eval_batches*validations,
            estimate_note='Steady-state replay; excludes preparation, checkpoint I/O and final test. Evaluation repeats the full stream on each rank.',
            calibrated_override=({'training':{'batch_size':t.batch_size,'grad_accum_steps':t.grad_accum_steps}}
                                 if policy['applied'] else {}))
        if ctx.main:
            write_result_files(root, 'benchmark', result)
            print(f'Preflight: {update_seconds:.3f}s/update, microbatch={t.batch_size}, '
                  f'accumulation={t.grad_accum_steps}, global={global_batch}, '
                  f'peak {peak/2**30:.2f} GiB, reserved {fraction:.1%}', flush=True)
        del wrapped, model, optimizer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    ctx.barrier()
    return result
