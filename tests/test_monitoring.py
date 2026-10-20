"""Core dashboard filtering, units, explicit scopes and non-duplicated results."""
import json
from pathlib import Path

import pytest

from grt.monitoring import configure_wandb, history_metrics, training_overview, summary_metrics, posthoc_overview


def test_dashboard_axes_and_counter_visibility_are_explicit():
    from types import SimpleNamespace
    definitions = {}
    run = SimpleNamespace(define_metric=lambda name, **settings: definitions.update({name: settings}))
    configure_wandb(run, stream=True)
    assert definitions['progress/target_tokens']['hidden']
    assert definitions['global_step']['hidden']
    for name in ('01_quality/validation_ppl', '02_optimization/learning_rate', '03_memory/probe_write_mean'):
        assert definitions[name]['step_metric'] == 'progress/target_tokens'
        assert definitions[name]['hidden'] is False
    assert definitions['04_runtime/elapsed_hours']['step_metric'] == 'global_step'


def test_stream_dashboard_keeps_core_fields_with_readable_units():
    values = history_metrics({
        'train/loss': 4., 'train/total_loss': 4.01, 'train/gate_regularization_loss': .01,
        'train/grad_norm': 3., 'train/gate_raw_write_mean': .4,
        'train/gate_hidden_norm': 200., 'train/gate_abs_logit_max': 90.,
        'grt/write_mean': .3, 'grt/relative_update': .2, 'grt/slot_cosine': .8,
        'grt/rtla/normal_perplexity': 80., 'grt/rtla/swap_delta_nll': .5,
        'train/target_tokens': 2000, 'train/elapsed_seconds': 4,
        'train/peak_gpu_memory_bytes': 3 * 2**30, 'val/perplexity': 40.,
    }, stream=True)
    assert values['01_quality/train_nll'] == 4
    assert values['01_quality/validation_ppl'] == 40
    assert values['02_optimization/gate_penalty'] == .01
    assert values['02_optimization/pre_clip_grad_norm'] == 3
    assert values['03_memory/probe_write_mean'] == .3
    assert values['03_memory/train_raw_write_mean'] == .4
    assert values['04_runtime/peak_gpu_gib'] == 3
    assert values['04_runtime/target_tokens_per_second'] == 500
    assert values['progress/target_tokens'] == 2000
    assert not any(word in k for k in values for word in ('hidden_norm', 'logit_max', 'slot_cosine', 'normal_perplexity', 'total_loss'))
    assert '03_memory/reset_delta_nll' not in values  # Missing is not zero.


def test_remember_exact_match_stays_separate_from_stream_perplexity():
    values = history_metrics({'val/fixed/exact_match': .9, 'val/random/loss': .3,
        'train/seconds': 7200, 'num_pairs': 5, 'train/samples': 100}, stream=False)
    assert values['01_quality/fixed_validation_exact_match'] == .9
    assert values['04_runtime/training_hours'] == 2
    assert values['progress/num_pairs'] == 5
    assert not any('ppl' in k or 'tokens_per_second' in k for k in values)


def test_result_summary_distinguishes_best_final_and_test():
    summary = dict(status='completed', step=20, selected_step=19,
        selected_validation=dict(loss=3., perplexity=20.),
        validation=dict(loss=3.1, perplexity=22.), evaluation_split='test',
        loss=3.2, perplexity=24., target_tokens_seen=10000,
        wall_seconds=3600, peak_gpu_memory_bytes=2**30,
        selected_rtla={'grt/rtla/swap_delta_nll': .4})
    overview = training_overview(summary)
    assert overview['best_val_ppl'] == 20 and overview['final_val_ppl'] == 22
    assert overview['test_ppl'] == 24
    assert overview['selected_step'] == 19 and overview['completed_step'] == 20
    assert overview['wall_hours'] == overview['peak_gpu_gib'] == 1
    assert 'probe_write_mean' not in overview
    values = summary_metrics(summary)
    assert values['result/final_val_ppl'] == 22
    assert 'selected_rtla' not in values and 'validation' not in values
    assert training_overview(None) == {}


def test_posthoc_overview_uses_held_out_windows_and_within_slot_variation():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/'results/posthoc/e9d2ee22c7344315be1a64b908873405.json').read_text())
    values = posthoc_overview(report)
    assert values['checkpoint_step'] == 20000
    assert values['evaluation_windows'] == [16, 32]
    assert values['within_slot_gate_std'] == pytest.approx(.2341210434306239)
    assert values['update_effective_rank'] == pytest.approx(8.700138950905536)
    assert posthoc_overview({'checkpoint': {'step': 3}, 'register_analysis': None}) == {
        'checkpoint_step': 3, 'evaluation_windows': []}
