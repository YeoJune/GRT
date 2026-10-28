"""Saved GPU measurements keep the same definitions across every report section."""
import importlib.util
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gate_report', REPO / 'scripts/report_gate_stabilization.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture(scope='module')
def runs():
    return report.load_runs(REPO)


def test_received_sources_match_training_checkpoint_and_common_tables(runs):
    # load_runs rejects mismatched manifests, selected steps, metrics and non-GPU provenance.
    assert len(runs) == 30
    for group, (_, keys) in report.GROUPS.items():
        table = report.table_block(group, runs)
        for _, label, _ in report.METRICS:
            assert table.count('| ' + label + ' |') == 1
        for key in keys:
            row = runs[key]
            assert row['source']['precision'] == 'fp32'
            assert len(row['source']['checkpoint_sha256']) == 64
            assert sum(row['counts'].values()) == row['M']
            assert row['source']['train_id'] in table
            assert row['source']['post_id'] in table


def test_gate_variation_is_within_slot_and_evaluation_only(runs):
    # M8 Off has both context-invariant gates and very large logits. Pooling slots
    # or including calibration would give a different answer in other conditions.
    assert runs['m8_off']['stats']['slot_std'] == 0
    assert runs['m20_a']['stats']['slot_std'] == pytest.approx(.18339697, abs=1e-8)
    assert runs['m20_a']['stats']['update_rank'] == pytest.approx(2.812683, abs=1e-6)
    assert runs['m8_a']['stats']['slot_std'] == pytest.approx(.16998296, abs=1e-8)
    assert runs['m8_a']['counts'] == {'closed': 0, 'open': 0, 'other': 8}
    assert runs['m75_a']['stats']['slot_std'] == pytest.approx(.23412104, abs=1e-8)
    assert runs['m75_a']['stats']['update_rank'] == pytest.approx(8.700139, abs=1e-6)
    assert runs['m75_a']['counts'] == {'closed': 0, 'open': 3, 'other': 72}
    post = json.loads((REPO / 'results/posthoc' / (runs['m20_a']['source']['post_id'] + '.json')).read_text())
    gates = [np.asarray(w['diagnostics']['write_gates']) for w in post['register_analysis']['windows'][1:]]
    pooled_std = np.mean([g.std() for g in gates])
    assert abs(pooled_std - runs['m20_a']['stats']['slot_std']) > .01


def test_empty_random_control_is_missing_and_refine_uses_effective_gate(runs):
    assert runs['m20_dropout']['groups']['other']['size'] == 0
    assert runs['m20_dropout']['groups']['other']['swap_random'] is None
    assert runs['m75_dropout']['groups']['other']['swap_random'][0] == pytest.approx(.002645149, abs=1e-8)
    row = runs['m20_refine']
    post = json.loads((REPO / 'results/posthoc' / (row['source']['post_id'] + '.json')).read_text())
    windows = post['register_analysis']['windows'][1:]
    effective_mean = np.mean([np.asarray(w['diagnostics']['write_gates']).mean() for w in windows])
    assert row['stats']['write_mean'] == effective_mean
    # Refine's raw logits describe base p, not its corrected effective w.
    assert any(np.max(abs(np.asarray(w['diagnostics']['write_gates']) -
        1 / (1 + np.exp(-np.asarray(w['diagnostics']['gate_logits']['raw']))))) > 1e-5 for w in windows)


def test_received_40k_gate_paths_preserve_best_final_and_resume_history(runs):
    for key, best_step, best_ppl, final_ppl, raw_rows in [
        ('gate_m20_independent', 38000, 28.25498946758573, 28.28760396048914, 4016),
        ('gate_m20_separate_head', 40000, 28.41196133215214, 28.41196133215214, 4087),
        ('gate_m20_candidate', 39000, 28.178114591242053, 28.203008750372558, 4027),
        ('gate_m8_independent', 40000, 29.006642171298886, 29.006642171298886, 4001),
        ('gate_m8_candidate', 38000, 28.92683921352185, 29.027446118291582, 4001),
        ('gate_m75_independent', 39768, 28.326013746946423, 28.475445144947905, 4001),
        ('gate_m8_separate_head', 37755, 29.21633061766568, 29.466509892782312, 4001),
        ('gate_m75_candidate', 39000, 27.176870468047007, 27.177628482547842, 4001),
        ('gate_m75_separate_head', 38000, 29.748144450871784, 29.895041094460645, 4001),
    ]:
        row = runs[key]
        raw = json.loads((REPO/'results'/f"{row['source']['train_id']}.json").read_text())
        experiment, = raw['experiments']
        cfg = experiment['config']
        assert experiment['status'] == 'completed' and experiment['summary']['step'] == 40000
        assert cfg['training']['warmup_ratio'] == .05 and cfg['training']['min_lr_ratio'] == .01
        assert row['source']['target_tokens'] == 767950848
        assert row['best_step'] == best_step and row['best_ppl'] == best_ppl and row['final_ppl'] == final_ppl
        assert len(experiment['metrics']) == raw_rows and len(row['history']) == 4001
        latest = {v['step']: v for v in experiment['metrics']}
        assert row['history'] == [latest[step] for step in sorted(latest)]
        if key == 'gate_m75_candidate':
            expected_counts = {'closed': 0, 'open': 3, 'other': 72}
        elif row['M'] in (8, 75):
            expected_counts = {'closed': 0, 'open': 0, 'other': row['M']}
        else:
            expected_counts = {'closed': 0, 'open': 0, 'other': 20} if key.endswith('separate_head') else {'closed': 0, 'open': 1, 'other': 19}
        assert row['counts'] == expected_counts
        assert row['stats']['over20'] == pytest.approx(0.010416666666666668 if key in ('gate_m75_independent', 'gate_m75_separate_head') else 0)

    # M8 reverses the ranking between best-checkpoint and final-budget PPL.
    independent, candidate = [runs['gate_m8_' + source] for source in ('independent', 'candidate')]
    assert candidate['best_ppl'] < independent['best_ppl']
    assert candidate['final_ppl'] > independent['final_ppl']


@pytest.mark.parametrize('key,target_tokens,off_cadence', [
    ('gate_m75_independent', 763496448, True), ('gate_m8_separate_head', 724846848, True),
    ('gate_m75_separate_head', 729550848, False),
])
def test_resumed_sources_match_only_attempt_labels_and_preserve_best(runs, key, target_tokens, off_cadence):
    row = runs[key]
    train = json.loads((REPO/'results'/f"{row['source']['train_id']}.json").read_text())['experiments'][0]
    post = json.loads((REPO/'results/posthoc'/f"{row['source']['post_id']}.json").read_text())
    assert train['config'] != post['training_config']
    assert report.comparable_config(train['config']) == report.comparable_config(post['training_config'])
    for field in ('batch_size', 'grad_accum_steps', 'lr'):
        changed = deepcopy(post['training_config'])
        changed['training'][field] *= 2
        assert report.comparable_config(train['config']) != report.comparable_config(changed)
    changed = deepcopy(post['training_config'])
    changed['data']['options']['path'] += '-other'
    assert report.comparable_config(train['config']) != report.comparable_config(changed)
    assert (row['best_step'] not in {v['step'] for v in row['history']}) == off_cadence
    assert row['best_target_tokens'] == post['checkpoint']['target_tokens_seen'] == target_tokens
    periodic_best = min(v['val/perplexity'] for v in row['history'] if 'val/perplexity' in v)
    assert row['best_ppl'] < periodic_best if off_cadence else row['best_ppl'] == periodic_best


def test_regeneration_preserves_narrative_and_rejects_ambiguous_markers():
    text = 'Manual intro\n<!-- gate-report:x:start -->\nold table\n<!-- gate-report:x:end -->\nManual interpretation\n'
    updated = report.update_block(text, 'x', 'new table')
    assert updated == text.replace('old table', 'new table')
    assert report.update_block(updated, 'x', 'new table') == updated
    for invalid in (text + text, text.replace('x:end', 'missing:end'),
                    '<!-- gate-report:x:end --><!-- gate-report:x:start -->'):
        with pytest.raises(ValueError, match='marked block'):
            report.update_block(invalid, 'x', 'new table')


@pytest.mark.parametrize('memory,delta,interaction', [
    (8, -.017353918402487345, -.02245326202835729),
    (20, .02513037121590589, .012153103744894533),
    (75, .013854796198694341, .039796086605711434),
])
def test_completed_dropout_comparison_uses_final_budget(runs, memory, delta, interaction):
    combined, a = [runs[f'm{memory}_{suffix}'] for suffix in ('a_dropout', 'a')]
    assert combined['best_step'] == 19000
    assert combined['history'][-1]['step'] == a['history'][-1]['step'] == 20000
    assert combined['source']['target_tokens'] == a['source']['target_tokens'] == 383975424
    assert combined['final_nll'] - a['final_nll'] == pytest.approx(delta)
    line, = [v for v in report.dropout_effect_table(runs).splitlines() if v.startswith(f'| {memory} |')]
    assert line.endswith(f'| {delta:+.6f} | {interaction:+.6f} |')
