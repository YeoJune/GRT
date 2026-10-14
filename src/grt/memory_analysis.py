"""Paired GRT state interventions on fixed WikiText streams, without training."""
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
from grt.checkpoint import evaluation_context
from grt.config import EXPERIMENT_VERSION
from grt.logger import write_result_files
from grt.rtla import TraceBuffer, trace_metrics
from grt.wikitext import WikiTextAdapter


def token_nll(model, ids, targets, state, trace=None):
    out = model(ids, memory_state=state, _trace=trace)
    loss = F.cross_entropy(out.logits.flatten(0, 1).float(), targets.flatten(),
                           reduction='none').view_as(targets)
    return loss.detach().cpu().double(), out.memory_state


def analyze_stream(model, cfg, tokens, *, segments=None, warmup_segments=None,
                   horizon=0, slots=False, trace_examples=False, progress=None):
    if cfg.model.name != 'grt' or cfg.model.attention_mode != 'decoder':
        raise ValueError('Memory interventions require the GRT decoder')
    segments = cfg.rtla.segments if segments is None else segments
    warmup = cfg.rtla.warmup_segments if warmup_segments is None else warmup_segments
    if segments <= 0 or warmup <= 0 or horizon < 0 or cfg.rtla.batch_size < 2:
        raise ValueError('Require segments/warmup > 0, horizon >= 0 and batch >= 2')
    stream = WikiTextAdapter.streams(tokens, cfg.rtla.batch_size)
    length, memory_slots = cfg.model.segment_len, cfg.model.num_registers
    available = math.ceil((stream.shape[1]-1) / length)
    measured = min(segments, available-warmup)
    if measured <= 0:
        raise ValueError('Probe needs at least one segment after warmup')
    device = next(model.parameters()).device
    trace = TraceBuffer(read_active=cfg.model.register.conditional_read, per_example=trace_examples)
    records = []
    normal = np.full((measured, length), np.nan)
    reset, swap = normal.copy(), normal.copy()
    counts = np.zeros((measured, length), dtype=np.int64)
    slot_delta = np.full((measured, memory_slots if slots else 0), np.nan)
    retention = np.full((measured, horizon), np.nan)
    retention_counts = np.zeros((measured, horizon), dtype=np.int64)
    stop = min(available, warmup+measured+max(0, horizon-1))
    retention_calls = sum(max(0, min(horizon, stop-warmup-index)-1)
                          for index in range(measured))
    total_calls = stop + 2*measured + (measured*memory_slots if slots else 0) + retention_calls
    completed_calls = 0

    def forward(ids, targets, state, trace=None):
        nonlocal completed_calls
        result = token_nll(model, ids, targets, state, trace)
        completed_calls += 1
        if progress is not None:
            progress(completed_calls, total_calls)
        return result

    with evaluation_context(model):
        state = model.initial_state(cfg.rtla.batch_size)
        initial = state.clone()
        for index in range(stop):
            offset = index*length
            n = min(length, stream.shape[1]-1-offset)
            ids = stream[:, offset:offset+n].to(device)
            targets = stream[:, offset+1:offset+n+1].to(device)
            start_state = state.detach().cpu().clone()
            nll, state = forward(ids, targets, state,
                trace if warmup <= index < warmup+measured else None)
            if index >= warmup:
                records.append((offset, n, start_state, nll))
        for index in range(measured):
            offset, n, state, baseline = records[index]
            state = state.to(device)
            ids = stream[:, offset:offset+n].to(device)
            targets = stream[:, offset+1:offset+n+1].to(device)
            normal[index, :n] = baseline.mean(0).numpy()
            counts[index, :n] = cfg.rtla.batch_size
            reset_nll, _ = forward(ids, targets, initial)
            swap_nll, branch = forward(ids, targets, state.roll(1, 0))
            reset[index, :n] = (reset_nll-baseline).mean(0).numpy()
            swap[index, :n] = (swap_nll-baseline).mean(0).numpy()
            if slots:
                for slot in range(memory_slots):
                    altered = state.clone()
                    altered[:, slot] = initial[:, slot]
                    nll, _ = forward(ids, targets, altered)
                    slot_delta[index, slot] = (nll-baseline).mean().item()
            if horizon:
                retention[index, 0] = (swap_nll-baseline).mean().item()
                retention_counts[index, 0] = baseline.numel()
                for lag in range(1, min(horizon, len(records)-index)):
                    offset, n, _, baseline = records[index+lag]
                    nll, branch = forward(
                        stream[:, offset:offset+n].to(device),
                        stream[:, offset+1:offset+n+1].to(device), branch)
                    retention[index, lag] = (nll-baseline).mean().item()
                    retention_counts[index, lag] = baseline.numel()
    total = int(counts.sum())
    normal_mean = float(np.nansum(normal*counts)/total)
    conditions = {'normal': dict(loss=normal_mean, perplexity=math.exp(normal_mean), delta_nll=0.)}
    for name, delta in [('reset', reset), ('swap', swap)]:
        mean = float(np.nansum(delta*counts)/total)
        conditions[name] = dict(loss=normal_mean+mean, perplexity=math.exp(normal_mean+mean), delta_nll=mean)
    weights = counts.sum(1)
    summary = dict(experiment_version=EXPERIMENT_VERSION,
        conditions=conditions, target_tokens=total, batch_size=cfg.rtla.batch_size,
        segment_len=length, num_registers=memory_slots, requested_segments=segments,
        measured_segments=measured, warmup_segments=warmup, horizon=horizon,
        slot_delta_nll=(np.nansum(slot_delta*weights[:, None], axis=0)/total).tolist(),
        retention_delta_nll=[float(np.nansum(retention[:, lag]*retention_counts[:, lag]) /
                                  retention_counts[:, lag].sum())
                             if retention_counts[:, lag].sum() else None for lag in range(horizon)],
        retention_target_tokens=retention_counts.sum(0).tolist(),
        read_active=trace.read_active, score_kind='autoregressive',
        tokenization=cfg.data.options.get('tokenization', 'word'),
        scope='fixed stream probe; interventions start from the normal trajectory',
        discarded_tail_tokens=tokens.numel()-stream.numel())
    arrays = dict(normal_nll=normal, reset_delta_nll=reset, swap_delta_nll=swap,
        target_counts=counts, slot_delta_nll=slot_delta,
        retention_delta_nll=retention, retention_target_counts=retention_counts,
        **{'trace_'+key: value for key, value in trace.to_numpy().items()})
    return summary, arrays


def probe_metrics(summary, arrays):
    result = trace_metrics({key[6:]: value for key, value in arrays.items() if key.startswith('trace_')},
                           summary['read_active'])
    for name, condition in summary['conditions'].items():
        for key, value in condition.items():
            result[f'grt/rtla/{name}_{key}'] = value
    middle = (summary['segment_len']+1)//2
    for name in ('reset', 'swap'):
        for part, region in [('early', slice(None, middle)), ('late', slice(middle, None))]:
            counts = arrays['target_counts'][:, region]
            if counts.sum():
                result[f'grt/rtla/{name}_{part}_delta_nll'] = float(
                    np.nansum(arrays[name+'_delta_nll'][:, region]*counts)/counts.sum())
    result['grt/rtla/target_tokens'] = summary['target_tokens']
    return result


def save_analysis(root, summary, arrays, metadata, *, plots=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_result_files(root, 'summary', {**summary, **metadata})
    np.savez_compressed(root/'arrays.npz', **arrays)
    if plots:
        from grt.plots import plot_memory_analysis, plot_trace
        plot_memory_analysis(summary, arrays, root/'memory.png')
        plot_trace({key[6:]: value for key, value in arrays.items() if key.startswith('trace_')},
                   root/'trace.png', summary['read_active'])


def log_stream_analysis(model, cfg, tokens, root, step):
    if cfg.model.name != 'grt':
        return {}
    summary, arrays = analyze_stream(model, cfg, tokens)
    save_analysis(Path(root)/'rtla'/f'step_{step:06d}', summary, arrays,
                  dict(step=step, split='validation', data_path=cfg.data.options['path']))
    return probe_metrics(summary, arrays)
