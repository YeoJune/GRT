"""Regenerate gate report tables/figures from received GPU JSON, without model imports.

Add a received training report to RUNS and the relevant GROUPS, then run this script.
Only marked Markdown blocks, current figures and the derived metrics JSON are written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

# Reporting metadata only: retired methods are read as data, never constructed.
RUNS = {
    'm8_off': ('ccf041592f3e4ebe8f287e69195205a5', 'Off'),
    'm8_dropout': ('746ab4ab41e84ec0bdd0f3f0ed6bffdb', 'Dropout'),
    'm8_a': ('1006dcd4d956442097dba6780d07ab69', 'A'),
    'm8_a_dropout': ('d3b3358f21f949bd84e58b7b1ed0aee5', 'A+Dropout'),
    'm20_off': ('86fb2481ac1e45f29da0164863b28d66', 'Off'),
    'm20_dropout': ('51bafa3f611d48c2808a7a55d732c91d', 'Dropout'),
    'm20_a': ('b0420e8b63f64d0094ed84a8d273f14b', 'A'),
    'm20_a_dropout': ('8c2eec49e4cc43989a9847ccd7b942c9', 'A+Dropout'),
    'm20_b': ('31a7d55d4d64400e81578e055d88b2e1', 'B'),
    'm20_c': ('93d019b472434814b3e057ea5732f616', 'C'),
    'm20_bias_minus': ('3001e7328e87456395275745ada7b946', 'Bias-2'),
    'm20_bias_plus': ('0bb9e51192d54a398d771bab611e9fb1', 'Bias+2'),
    'm20_refine': ('89571cd181a4465b8e818a2407be8924', 'Refine'),
    'm20_uniform': ('51a6bf7d37db4e5b82ce96709c15885f', 'Uniform'),
    'm75_off': ('b4b4414578524de0a6e9de50851b91b2', 'Off'),
    'm75_dropout': ('06fe396a3e6b4e258b779481409de1ba', 'Dropout'),
    'm75_a': ('70bea6e32c584668ad3078de871cd66c', 'A'),
    'm75_a_dropout': ('ba51ccddc849423a9e7aa5bc26729b9c', 'A+Dropout'),
    'l150_m10': ('22ab31a56cb546f993197e5ce937e5de', 'L150'),
    'l150_m40': ('22cfd16b44964befbe42912c4aba28c5', 'L150'),
    'l150_m160': ('0e0ead4744594b59a7d9a771049c79a1', 'L150'),
    'gate_m20_independent': ('c4b4660b71164924a0bef45863d42a92', 'Independent'),
    'gate_m20_separate_head': ('d598c992d4734bd9aa327cbff78b96c5', 'Separate head'),
    'gate_m20_candidate': ('73881fe675194b53b499eab9e754a6d2', 'Candidate'),
    'gate_m8_independent': ('b847ae2264864871b0f132a1ee0ed61e', 'Independent'),
    'gate_m8_separate_head': ('e1aad6f646664a52b986d0cf59eaff24', 'Separate head'),
    'gate_m8_candidate': ('8908b33d329c4264b8c8f96872cada64', 'Candidate'),
    'gate_m75_independent': ('bbd0eca2b36a43be94684b7226118291', 'Independent'),
    'gate_m75_candidate': ('1e0143312a004ef9b28e6c104f1dbe89', 'Candidate'),
}
GROUPS = {
    'penalty': ('Penalty settings: L75/M20', ['m20_off', 'm20_a', 'm20_b', 'm20_c']),
    'transfer': ('Penalty A transfer: L75/M8, M20 and M75', ['m8_off', 'm8_a', 'm20_off', 'm20_a', 'm75_off', 'm75_a']),
    'dropout': ('Write dropout: L75, penalty off', ['m8_off', 'm8_dropout', 'm20_off', 'm20_dropout', 'm75_off', 'm75_dropout']),
    'combined': ('Write dropout with penalty A: L75', ['m8_a', 'm8_a_dropout', 'm20_a', 'm20_a_dropout', 'm75_a', 'm75_a_dropout']),
    'bias': ('Bias initialization: L75/M20', ['m20_off', 'm20_bias_minus', 'm20_bias_plus']),
    'initialization': ('Retired gate variants: L75/M20', ['m20_off', 'm20_refine', 'm20_uniform']),
    'reference': ('Reference: L150 vs L75, bias -2', ['l150_m10', 'l150_m40', 'l150_m160', 'm20_bias_minus']),
    'gate_paths': ('Gate paths: L75/M20, 40k updates',
                   ['gate_m20_independent', 'gate_m20_separate_head', 'gate_m20_candidate']),
    'gate_transfer_m8': ('Gate paths: L75/M8 transfer, 40k updates',
                         ['gate_m8_independent', 'gate_m8_separate_head', 'gate_m8_candidate']),
    'gate_transfer_m75': ('Gate paths: L75/M75, 40k updates (separate head pending)',
                          ['gate_m75_independent', 'gate_m75_candidate']),
}
GATE_PATH_GROUPS = ('gate_paths', 'gate_transfer_m8', 'gate_transfer_m75')
COLORS = {'Off': '#6b7280', 'Dropout': '#D55E00', 'A+Dropout': '#D55E00', 'A': '#0072B2', 'B': '#009E73',
          'C': '#8b5cf6', 'Bias-2': '#009E73', 'Bias+2': '#CC79A7',
          'Refine': '#E69F00', 'Uniform': '#CC79A7', 'L150': '#56B4E9',
          'Independent': '#0072B2', 'Separate head': '#D55E00', 'Candidate': '#009E73'}
METRICS = [
    ('write_mean', 'Write 평균', '.6f'), ('slot_std', '슬롯 내 gate std', '.6f'),
    ('abs_median', '절대 logit 중앙값', '.3f'), ('over20', '절대 logit >20 (%)', '.2f'),
    ('hidden_norm', 'MLP hidden L2 norm', '.3f'), ('state_rank', '문맥 중심화 state rank', '.3f'),
    ('update_rank', '실제 update rank', '.3f'), ('cka', '평균 slot CKA', '.3f'),
    ('reset', '전체 reset ΔNLL, lag0', '+.6f'), ('swap', '전체 swap ΔNLL, lag0', '+.6f'),
    ('swap_lag3', '전체 swap ΔNLL, lag3', '+.6f'),
]


def summarize_window(window, M):
    d = window['diagnostics']; z = np.asarray(d['gate_logits']['raw'], dtype=float)
    gates = np.asarray(d['write_gates'], dtype=float)
    if z.shape != gates.shape or gates.shape[-1] != M or not np.isfinite(z).all():
        raise ValueError('Invalid saved gate/logit arrays')
    np.testing.assert_allclose(gates.mean(), window['gate_statistics']['mean'], atol=1e-8)
    rep = d['representations']; cka = np.asarray(rep['slot_linear_cka'], dtype=float)
    all_group = next(g for g in window['interventions']['group_comparisons'] if g['name'] == 'closed+open+other')
    out = dict(write_mean=float(gates.mean()), slot_std=float(gates.std((0, 1)).mean()),
        abs_median=float(np.median(abs(z))), over20=float((abs(z) > 20).mean()) * 100,
        hidden_norm=d['router_input']['mlp_hidden_norm']['mean'],
        state_rank=rep['context_centered']['entropy_effective_rank']['mean'],
        update_rank=rep['actual_update']['entropy_effective_rank']['mean'],
        cka=float(np.nanmean(cka[~np.eye(M, dtype=bool)])),
        reset=window['summary']['conditions']['reset']['delta_nll'],
        swap=window['summary']['conditions']['swap']['delta_nll'],
        swap_lag3=all_group['swap']['delta_nll_by_lag'][3])
    groups = {}
    for g in window['interventions']['group_comparisons']:
        groups[g['name']] = dict(size=g['size'], **{mode: g[mode]['delta_nll_by_lag'] for mode in ('reset', 'swap')},
            **{mode + '_random': g[mode].get('size_matched_random', {}).get('mean') for mode in ('reset', 'swap')})
    return out, abs(z).ravel(), gates.std((0, 1)), groups


def comparable_config(cfg):
    # A resumed best.pt may retain the previous attempt's output/display names.
    # Keep every other field exact, including data path and physical batch size.
    return {**cfg, 'run': {**cfg['run'], 'output_dir': None},
            'wandb': {**cfg['wandb'], 'run_name': None}}


def load_runs(root):
    posts = [json.loads(p.read_text()) for p in (root / 'results/posthoc').glob('*.json')]
    runs = {}
    for key, (report_id, method) in RUNS.items():
        path = root / 'results' / (report_id + '.json'); raw = path.read_bytes(); t = json.loads(raw)
        e, = t['experiments']; cfg = e['config']; m = cfg['model']; reg = m['register']; s = e['summary']
        matches = [p for p in posts if comparable_config(p['training_config']) == comparable_config(cfg)]
        if len(matches) != 1:
            raise ValueError(f'{key}: expected one matching posthoc report')
        p, = matches
        if (p['dataset_manifest'] != t['dataset_manifest'] or p['checkpoint']['step'] != s['selected_step']
            or any(p['metrics'][k] != v for k, v in s['selected_rtla'].items())):
            raise ValueError(f'{key}: mismatched training/posthoc source')
        if not e['metadata']['device'].startswith('cuda') or p['environment']['precision'] != 'fp32':
            raise ValueError(f'{key}: expected supplied CUDA/FP32 measurements')
        windows = p['register_analysis']['windows']
        if [w['start_segment'] for w in windows] != [0, 16, 32]:
            raise ValueError(f'{key}: unsupported evaluation windows')
        summaries = [summarize_window(w, m['num_registers']) for w in windows[1:]]
        stats = {k: float(np.mean([w[0][k] for w in summaries])) for k, _, _ in METRICS}
        counts = {k: len(v) for k, v in p['register_analysis']['calibration_groups'].items()}
        groups = {}
        for name, g in summaries[0][3].items():
            groups[name] = {'size': g['size']}
            for kind in ('reset', 'swap', 'reset_random', 'swap_random'):
                values = [w[3][name][kind] for w in summaries]
                groups[name][kind] = None if values[0] is None else np.mean(values, axis=0).tolist()
        head = windows[0]['diagnostics']['gate_logits']
        provenance = dict(train_id=report_id, post_id=p['report_id'],
            training_sha256=hashlib.sha256(raw).hexdigest(), checkpoint_sha256=p['checkpoint']['sha256'],
            train_commit=e['metadata']['git_commit'], post_commit=p['environment']['git_commit'],
            train_python=e['metadata']['python'], train_torch=e['metadata']['torch'],
            post_python=p['environment']['python'], post_torch=p['environment']['torch'],
            gpu=e['metadata']['gpu'], precision=e['metadata']['dtype'], world_size=t['environment']['world_size'],
            microbatch=cfg['training']['batch_size'], accumulation=cfg['training']['grad_accum_steps'],
            target_tokens=s['target_tokens_seen'])
        label = f"M{m['num_registers']} {method}"
        runs[key] = dict(key=key, method=method, label=label, L=m['segment_len'], M=m['num_registers'],
            bias=reg['write_gate_bias_init'], dropout=reg['dropout_prob'],
            threshold=reg.get('write_logit_threshold'), coef=reg.get('write_logit_coef', 0),
            best_step=s['selected_step'], best_target_tokens=p['checkpoint'].get('target_tokens_seen'),
            best_nll=s['selected_validation']['loss'],
            best_ppl=s['selected_validation']['perplexity'], final_nll=s['validation']['loss'],
            final_ppl=s['validation']['perplexity'], wall_h=s['wall_seconds'] / 3600,
            peak_gib=s['peak_gpu_memory_bytes'] / 2**30, counts=counts, stats=stats, groups=groups,
            output_bias_range=[min(head['output_bias']), max(head['output_bias'])],
            output_weight_norm=float(np.mean(head['output_weight_row_norm'])), source=provenance,
            history=sorted({v['step']: v for v in e['metrics']}.values(), key=lambda v: v['step']),
            abs_logits=np.concatenate([w[1] for w in summaries]),
            slot_stds=np.mean([w[2] for w in summaries], axis=0))
    return runs


def table_block(group, runs):
    selected = [runs[k] for k in GROUPS[group][1]]
    lines = ['| 조건 | best step / PPL | 최종 PPL / NLL | wall h / peak GiB | 학습 / 사후 원본 |',
             '|---|---|---|---|---|']
    for r in selected:
        src = r['source']
        lines.append(f"| {r['label']} | {r['best_step']:,} / {r['best_ppl']:.4f} | {r['final_ppl']:.4f} / {r['final_nll']:.6f} | {r['wall_h']:.3f} / {r['peak_gib']:.3f} | [학습](../../results/{src['train_id']}.json) / [사후](../../results/posthoc/{src['post_id']}.json) |")
    lines += ['', '| 공통 지표 | ' + ' | '.join(r['label'] for r in selected) + ' |',
              '|---|' + '---:|' * len(selected)]
    lines.append('| C/O/T (window0) | ' + ' | '.join('/'.join(str(r['counts'][k]) for k in ('closed', 'open', 'other')) for r in selected) + ' |')
    for key, label, fmt in METRICS:
        lines.append('| ' + label + ' | ' + ' | '.join(format(r['stats'][key], fmt) for r in selected) + ' |')
    lines += ['', f'![{GROUPS[group][0]} — 공통 6패널](../figures/gate_stabilization/{group}.png)']
    return '\n'.join(lines)


def source_block(runs):
    lines = ['| 조건 (L/M) | best step | train / post commit | train / post PyTorch | micro×accum×GPU | checkpoint SHA256 |', '|---|---:|---|---|---|---|']
    for r in runs.values():
        s = r['source']
        lines.append(f"| {r['method']} ({r['L']}/{r['M']}) | {r['best_step']:,} | `{s['train_commit'][:7]}` / `{s['post_commit'][:7]}` | {s['train_torch']} / {s['post_torch']} | {s['microbatch']}×{s['accumulation']}×{s['world_size']} | `{s['checkpoint_sha256']}` |")
    return '\n'.join(lines)


def dropout_effect_table(runs):
    lines = ['| M | Off: dropout0 → .1 최종 PPL | A: dropout0 → .1 최종 PPL | dropout ΔNLL (Off) | dropout ΔNLL (A) | 차이의 차이 |',
             '|---|---|---|---:|---:|---:|']
    for memory in (8, 20, 75):
        off, drop, a, combined = [runs[f'm{memory}_{suffix}'] for suffix in ('off', 'dropout', 'a', 'a_dropout')]
        off_delta = drop['final_nll'] - off['final_nll']
        a_delta = combined['final_nll'] - a['final_nll']
        lines.append(f"| {memory} | {off['final_ppl']:.4f} → {drop['final_ppl']:.4f} | {a['final_ppl']:.4f} → {combined['final_ppl']:.4f} | {off_delta:+.6f} | {a_delta:+.6f} | {a_delta - off_delta:+.6f} |")
    return '\n'.join(lines)


def style(run):
    return dict(color=COLORS[run['method']], linestyle={8: '--', 20: '-', 75: ':', 10: '--', 40: '-', 160: ':'}[run['M']])


def common_figure(group, runs, destination):
    import matplotlib.pyplot as plt
    selected = [runs[k] for k in GROUPS[group][1]]
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.2))
    fig.subplots_adjust(left=.06, right=.985, bottom=.115, top=.90, wspace=.32, hspace=.35)
    for r in selected:
        hist = [v for v in r['history'] if 'val/perplexity' in v]
        xx = [v['train/target_tokens'] / 1e6 for v in hist]
        axes[0, 0].plot(xx, [v['val/perplexity'] for v in hist], label=r['label'], **style(r))
        if r['best_step'] not in {v['step'] for v in hist} and r['best_target_tokens'] is not None:
            axes[0, 0].scatter([r['best_target_tokens'] / 1e6], [r['best_ppl']],
                               marker='*', s=70, color=COLORS[r['method']], zorder=4)
        axes[0, 1].plot(xx, [v['grt/write_mean'] for v in hist], **style(r))
        zz = np.sort(r['abs_logits'])
        axes[0, 2].plot(zz, np.arange(1, len(zz) + 1) / len(zz), **style(r))
    labels = [r['label'].replace(' ', '\n', 1) for r in selected]; xx = np.arange(len(selected))
    boxes = axes[1, 0].boxplot([r['slot_stds'] for r in selected], positions=xx,
        widths=.6, showfliers=False, patch_artist=True, medianprops={'color': 'black'})
    for i, (box, r) in enumerate(zip(boxes['boxes'], selected)):
        box.set_facecolor(COLORS[r['method']]); box.set_alpha(.4)
        axes[1, 0].plot(i, r['stats']['slot_std'], 'D', color=COLORS[r['method']], ms=4)
    for ax, pairs in [(axes[1, 1], [('state_rank', 'State centered'), ('update_rank', 'Actual update')]),
                      (axes[1, 2], [('reset', 'Reset'), ('swap', 'Swap')])]:
        for j, (key, label) in enumerate(pairs):
            ax.bar(xx + (j - .5) * .34, [r['stats'][key] for r in selected], width=.32,
                color=[COLORS[r['method']] for r in selected], alpha=.55 if j == 0 else 1,
                hatch='' if j == 0 else '//', label=label)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8)
    axes[0, 0].set(title='A. Full validation PPL', yscale='log', xlabel='Trained target tokens (M)', ylabel='PPL')
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 1].set(title='B. Validation probe mean write', xlabel='Trained target tokens (M)', ylabel='Mean write', ylim=(-.02, 1.02))
    axes[0, 2].set(title='C. Absolute base-logit ECDF', xlabel='Absolute raw logit (symlog)', ylabel='Cumulative fraction', ylim=(0, 1.02))
    axes[0, 2].set_xscale('symlog', linthresh=1)
    axes[0, 2].set_xlim(left=0)
    axes[1, 0].set(title='D. Within-slot gate variation', ylabel='Std over contexts (diamond: mean)', ylim=(-.01, .51))
    axes[1, 1].set(title='E. Representation effective rank', ylabel='Effective rank (M sets its ceiling)')
    axes[1, 2].set(title='F. Functional memory intervention', ylabel='Delta NLL, lag0')
    for ax in axes.flat:
        ax.title.set_fontsize(11)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.grid(alpha=.2); ax.set_axisbelow(True)
    fig.suptitle(GROUPS[group][0], fontsize=16)
    footer = ('GPU measurements, seed54. Posthoc: windows16/32; same 40k schedule. Repeated steps keep the last logged row.'
              if group in GATE_PATH_GROUPS else
              'GPU measurements, seed54. Posthoc: windows16/32; Refine logits are base p, gates are effective w. Conditions and versions differ.')
    if any(r['best_step'] not in {v['step'] for v in r['history'] if 'val/perplexity' in v} for r in selected):
        footer += ' Star: selected best outside periodic logs.'
    fig.supxlabel(footer, fontsize=9, y=.025)
    fig.savefig(destination / (group + '.png'), dpi=150); plt.close(fig)


def penalty_figure(runs, destination):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.7), constrained_layout=True)
    fields = [('train/gate_raw_write_mean', 'Training mean write: early plateau and escape', 'Mean raw write'),
              ('train/gate_abs_logit_mean', 'Training absolute raw logit', 'Mean |z|'),
              ('train/gate_regularization_loss', 'Weighted penalty', 'coef * R'),
              ('train/gate_hidden_norm', 'Write MLP hidden norm', 'L2 norm')]
    for key in ('m8_a', 'm20_a', 'm75_a', 'm20_b', 'm20_c'):
        r = runs[key]
        for i, (field, title, ylabel) in enumerate(fields):
            hist = [v for v in r['history'] if i != 0 or v['step'] <= 2500]
            axes.flat[i].plot([v['step'] / 1000 for v in hist], [v[field] for v in hist], label=r['label'], lw=1, **style(r))
    for ax, (_, title, ylabel) in zip(axes.flat, fields):
        ax.set(title=title, xlabel='Optimizer steps (k)', ylabel=ylabel); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8); axes[0, 0].set_ylim(-.02, 1.02)
    axes[1, 0].set_yscale('symlog', linthresh=1e-8)
    fig.suptitle('Penalty-specific training diagnostics (not validation probes)', fontsize=15)
    fig.savefig(destination / 'penalty_training.png', dpi=150); plt.close(fig)


def update_block(document, name, body):
    start, end = f'<!-- gate-report:{name}:start -->', f'<!-- gate-report:{name}:end -->'
    if (document.count(start) != 1 or document.count(end) != 1
        or document.index(start) > document.index(end)):
        raise ValueError(f'Expected exactly one marked block: {name}')
    return re.sub(re.escape(start) + r'.*?' + re.escape(end), lambda _: start + '\n' + body + '\n' + end,
                  document, flags=re.S)


def generate(root):
    import matplotlib
    matplotlib.use('Agg')
    runs = load_runs(root)
    destination = root / 'docs/figures/gate_stabilization'; destination.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        common_figure(group, runs, destination)
    penalty_figure(runs, destination)
    report = root / 'docs/experiments/20261018_l75_gate_stabilization.md'
    text = report.read_text()
    for group in GROUPS:
        if group not in GATE_PATH_GROUPS:
            text = update_block(text, group, table_block(group, runs))
    text = update_block(text, 'effects', dropout_effect_table(runs))
    path_runs = {key: runs[key] for group in GATE_PATH_GROUPS for key in GROUPS[group][1]}
    text = update_block(text, 'sources', source_block({key: r for key, r in runs.items() if key not in path_runs}))
    report.write_text(text)
    path_report = root / 'docs/experiments/20261019_l75_gate_paths.md'
    text = path_report.read_text()
    for group in GATE_PATH_GROUPS:
        text = update_block(text, group, table_block(group, runs))
    text = update_block(text, 'gate_path_sources', source_block(path_runs))
    path_report.write_text(text)
    summary = {key: {k: v for k, v in r.items() if k not in ('history', 'abs_logits', 'slot_stds')} for key, r in runs.items()}
    path = root / 'docs/tables/gate_stabilization_metrics.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(format='grt.gate_summary/1', source='User GPU JSON only; no model execution',
        aggregation='Continuous posthoc values: mean of windows16/32. Types: window0. ECDF: pooled equal-sized evaluation windows.', runs=summary), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return runs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    data = generate(args.repo_dir)
    print(f'Updated {len(data)} source pairs, {len(GROUPS) + 1} figures and marked report tables.')
