"""Saved GPU measurements keep the same definitions across every report section."""
import importlib.util
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
    assert len(runs) == 18
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


def test_regeneration_preserves_narrative_and_rejects_ambiguous_markers():
    text = 'Manual intro\n<!-- gate-report:x:start -->\nold table\n<!-- gate-report:x:end -->\nManual interpretation\n'
    updated = report.update_block(text, 'x', 'new table')
    assert updated == text.replace('old table', 'new table')
    assert report.update_block(updated, 'x', 'new table') == updated
    for invalid in (text + text, text.replace('x:end', 'missing:end'),
                    '<!-- gate-report:x:end --><!-- gate-report:x:start -->'):
        with pytest.raises(ValueError, match='marked block'):
            report.update_block(invalid, 'x', 'new table')
