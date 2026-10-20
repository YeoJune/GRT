"""Core monitoring contract shared by W&B and notebook result summaries.

No model/SDK imports: training diagnostics and full posthoc arrays stay in local
reports; the live dashboard contains only the fields listed here.
"""

COMMON_FIELDS = {
    'train/loss': ('01_quality/train_nll', 1),
    'train/lr': ('02_optimization/learning_rate', 1),
    'train/grad_norm': ('02_optimization/pre_clip_grad_norm', 1),
    'train/gate_regularization_loss': ('02_optimization/gate_penalty', 1),
    'train/gate_raw_write_mean': ('03_memory/train_raw_write_mean', 1),
    'train/gate_fraction_abs_over20': ('03_memory/train_abs_logit_over20', 1),
    'grt/write_mean': ('03_memory/probe_write_mean', 1),
    'grt/relative_update': ('03_memory/probe_relative_update', 1),
    'grt/rtla/reset_delta_nll': ('03_memory/reset_delta_nll', 1),
    'grt/rtla/swap_delta_nll': ('03_memory/swap_delta_nll', 1),
    'train/peak_gpu_memory_bytes': ('04_runtime/peak_gpu_gib', 1 / 2**30),
}
STREAM_FIELDS = {
    'val/loss': ('01_quality/validation_nll', 1),
    'val/perplexity': ('01_quality/validation_ppl', 1),
    'train/elapsed_seconds': ('04_runtime/elapsed_hours', 1 / 3600),
}
REMEMBER_FIELDS = {
    'val/random/loss': ('01_quality/random_validation_nll', 1),
    'val/random/exact_match': ('01_quality/random_validation_exact_match', 1),
    'val/fixed/loss': ('01_quality/fixed_validation_nll', 1),
    'val/fixed/exact_match': ('01_quality/fixed_validation_exact_match', 1),
    'train/seconds': ('04_runtime/training_hours', 1 / 3600),
}
COUNTERS = {'train/target_tokens': 'progress/target_tokens',
            'train/samples': 'progress/samples', 'stage_step': 'progress/stage_step',
            'num_pairs': 'progress/num_pairs', 'key_size': 'progress/key_size'}


def history_fields(stream):
    return {**COMMON_FIELDS, **(STREAM_FIELDS if stream else REMEMBER_FIELDS)}


def configure_wandb(run, stream):
    """Four ordered prefix sections, explicit axes, no automatic diagnostic plots."""
    run.define_metric('global_step', hidden=True)
    for key in COUNTERS.values():
        run.define_metric(key, hidden=True)
    names = [name for name, _ in history_fields(stream).values()]
    if stream:
        names.append('04_runtime/target_tokens_per_second')
    for name in names:
        axis = 'progress/target_tokens' if stream and not name.startswith('04_runtime/') else 'global_step'
        run.define_metric(name, step_metric=axis, step_sync=True, hidden=False)


def history_metrics(payload, stream):
    values = {name: payload[source] * scale for source, (name, scale) in history_fields(stream).items()
              if payload.get(source) is not None}
    values.update({name: payload[source] for source, name in COUNTERS.items() if source in payload})
    # Cumulative stream throughput includes preceding evaluation/checkpoint time.
    if stream and payload.get('train/elapsed_seconds', 0) > 0 and 'train/target_tokens' in payload:
        values['04_runtime/target_tokens_per_second'] = payload['train/target_tokens'] / payload['train/elapsed_seconds']
    return values


def training_overview(summary):
    """Keep best-validation selection, final-budget quality and costs separate."""
    if not summary:
        return {}
    overview = {key: summary[source] for key, source in (
        ('status', 'status'), ('completed_step', 'step'), ('selected_step', 'selected_step'),
        ('target_tokens', 'target_tokens_seen'), ('evaluation_split', 'evaluation_split')) if source in summary}
    for prefix, source in (('best_val', 'selected_validation'), ('final_val', 'validation')):
        metrics = summary.get(source) or {}
        for key, field in (('nll', 'loss'), ('ppl', 'perplexity')):
            if field in metrics:
                overview[prefix + '_' + key] = metrics[field]
    for source, key, scale in (('wall_seconds', 'wall_hours', 1 / 3600),
                                ('peak_gpu_memory_bytes', 'peak_gpu_gib', 1 / 2**30)):
        if source in summary:
            overview[key] = summary[source] * scale
    probe = summary.get('selected_rtla') or {}
    for source, key in (('grt/write_mean', 'probe_write_mean'),
                        ('grt/rtla/reset_delta_nll', 'reset_delta_nll'),
                        ('grt/rtla/swap_delta_nll', 'swap_delta_nll')):
        if source in probe:
            overview[key] = probe[source]
    if summary.get('evaluation_split') == 'test':
        for field, key in (('loss', 'test_nll'), ('perplexity', 'test_ppl')):
            if field in summary:
                overview[key] = summary[field]
    return overview


def summary_metrics(payload):
    values = {'result/' + key: value for key, value in training_overview(payload).items()}
    for source, key in (('test/loss', 'test_nll'), ('test/exact_match', 'test_exact_match'),
                        ('test/samples', 'test_samples')):
        if source in payload:
            values['result/' + key] = payload[source]
    metadata = payload.get('metadata') or {}
    for key in ('git_commit', 'python', 'torch', 'transformers', 'gpu', 'dtype'):
        if key in metadata:
            values['provenance/' + key] = metadata[key]
    return values


def format_number(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def print_training_overview(experiments):
    print('\nQuality — full validation; best checkpoint / final budget')
    print('| model | status | best step | best NLL / PPL | final step | final NLL / PPL |')
    print('|---|---|---:|---|---:|---|')
    for e in experiments:
        v = e['overview']
        print(f"| {e['model']} | {e['status']} | {v.get('selected_step', '—')} | "
              f"{format_number(v.get('best_val_nll'))} / {format_number(v.get('best_val_ppl'))} | "
              f"{v.get('completed_step', '—')} | {format_number(v.get('final_val_nll'))} / {format_number(v.get('final_val_ppl'))} |")
    print('\nMemory — selected-checkpoint validation probe; cost — entire training run')
    print('| model | L / M | probe write | reset / swap ΔNLL | targets (M) | hours / peak GiB |')
    print('|---|---|---:|---|---:|---|')
    for e in experiments:
        v, m = e['overview'], e['config']['model']
        tokens = v.get('target_tokens')
        print(f"| {e['model']} | {m['segment_len']} / {m['num_registers']} | {format_number(v.get('probe_write_mean'))} | "
              f"{format_number(v.get('reset_delta_nll'))} / {format_number(v.get('swap_delta_nll'))} | "
              f"{format_number(tokens / 1e6 if tokens is not None else None, 3)} | "
              f"{format_number(v.get('wall_hours'), 3)} / {format_number(v.get('peak_gpu_gib'), 3)} |")


def posthoc_overview(report):
    """Scalar reading guide; detailed groups/arrays remain in the posthoc JSON."""
    analysis = report.get('register_analysis')
    if not analysis:
        return {'checkpoint_step': report['checkpoint']['step'], 'evaluation_windows': []}
    windows = [w for w in analysis['windows'] if w['role'] == 'evaluation']
    if not windows:
        return {'checkpoint_step': report['checkpoint']['step'], 'evaluation_windows': []}
    def mean(values):
        return sum(values) / len(values)
    return dict(checkpoint_step=report['checkpoint']['step'],
        evaluation_windows=[w['start_segment'] for w in windows],
        write_mean=mean([w['gate_statistics']['mean'] for w in windows]),
        within_slot_gate_std=mean([mean(w['gate_statistics']['per_slot']['std']) for w in windows]),
        state_effective_rank=mean([w['diagnostics']['representations']['context_centered']['entropy_effective_rank']['mean'] for w in windows]),
        update_effective_rank=mean([w['diagnostics']['representations']['actual_update']['entropy_effective_rank']['mean'] for w in windows]))


def print_posthoc_overview(reports):
    print('\nPosthoc — held-out evaluation windows; scalar means across windows')
    print('| model | checkpoint step | windows | write / within-slot std | state / update rank |')
    print('|---|---:|---|---|---|')
    for model, report in reports.items():
        v = posthoc_overview(report)
        print(f"| {model} | {v['checkpoint_step']} | {v['evaluation_windows']} | "
              f"{format_number(v.get('write_mean'))} / {format_number(v.get('within_slot_gate_std'))} | "
              f"{format_number(v.get('state_effective_rank'), 3)} / {format_number(v.get('update_effective_rank'), 3)} |")
