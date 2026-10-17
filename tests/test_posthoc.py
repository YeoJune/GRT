"""Portable posthoc results from a two-rank checkpoint, without training data."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from grt.checkpoint import CHECKPOINT_FORMAT
from grt.experiments import load_experiment
from grt.memory_analysis import analyze_stream
from grt.models.factory import create_model
from grt.rtla import TraceBuffer
from grt.utils.kaggle import file_hash
from grt.utils.posthoc import WikiTextPosthoc, gate_statistics


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path/'prepared'
    (data/'tokenizer').mkdir(parents=True)
    tokenizer = data/'tokenizer/tokenizer.json'
    tokenizer.write_text('{}')
    tokens = torch.arange(23, dtype=torch.long)
    torch.save(tokens, data/'validation.pt')
    manifest = dict(vocab_size=128, tokenization='word',
        sha256={'validation': file_hash(data/'validation.pt')},
        prepared_artifacts_sha256={'tokenizer/tokenizer.json': file_hash(tokenizer)})
    (data/'manifest.json').write_text(json.dumps(manifest))
    cfg = load_experiment(task='wikitext103', model='grt', size='tiny',
        tokenization='word', runtime='cpu', steps=10, output_dir=tmp_path/'training', overrides={
            'model': {'vocab_size': 128, 'd_model': 32, 'segment_len': 4, 'num_registers': 3,
                      'alu': {'num_layers': 1, 'nhead': 4, 'd_ff': 64},
                      'router': {'mlp_hidden': 32}},
            'data': {'options': {'path': '/original/kaggle/prepared'}},
            'rtla': {'batch_size': 2, 'segments': 2, 'warmup_segments': 1}})
    model = create_model(cfg)
    checkpoint = tmp_path/'best.pt'
    torch.save(dict(format=CHECKPOINT_FORMAT, model=model.state_dict(), config=asdict(cfg),
                    dataset_manifest=manifest, step=7, world_size=2, target_tokens_seen=1234), checkpoint)
    return checkpoint, data, cfg, model, tokens


def job_for(inputs, tmp_path, **options):
    checkpoint, data, *_ = inputs
    options.setdefault('registers', False)
    return WikiTextPosthoc(checkpoint, data, tmp_path/'outputs', device='cpu',
                          batch_size=2, segments=2, warmup_segments=1, **options)


def test_roundtrip_preserves_checkpoint_and_exports_raw_gates(inputs, tmp_path, monkeypatch):
    checkpoint, data, cfg, model, tokens = inputs
    before = file_hash(checkpoint)
    original_config = asdict(cfg)
    job = job_for(inputs, tmp_path, horizon=4, slots=True)
    result = job.run()
    expected, _ = analyze_stream(model, cfg, tokens, horizon=4, slots=True)
    assert result['summary']['conditions'] == expected['conditions']
    assert result['summary']['slot_delta_nll'] == expected['slot_delta_nll']
    assert result['summary']['retention_target_tokens'] == [12, 4, 0, 0]
    assert result['summary']['retention_delta_nll'][2:] == [None, None]
    assert result['arrays']['normal_nll'][1][2:] == [None, None]
    assert np.array(result['arrays']['trace_w_gates_per_example']).shape == (2, 2, 3)
    assert result['checkpoint']['training_world_size'] == 2
    assert result['environment']['analysis_world_size'] == 1
    assert result['training_config'] == original_config
    assert file_hash(checkpoint) == before
    assert not (data/'train.pt').exists() and not (data/'test.pt').exists()
    text = job.report().read_text()
    assert 'NaN' not in text and 'Infinity' not in text
    assert json.loads(text) == result
    for filename in ['memory.png', 'trace.png', 'arrays.npz', 'summary.json']:
        assert (job.analysis_dir/filename).stat().st_size > 0
    google, colab = ModuleType('google'), ModuleType('google.colab')
    downloads = []
    colab.files = SimpleNamespace(download=downloads.append)
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    job.report(download=True)
    assert downloads == [str(job.report_path)]
    rerun = job_for(inputs, tmp_path, horizon=4, slots=True)
    assert rerun.run()['report_id'] == result['report_id']
    other = job_for(inputs, tmp_path, horizon=4, slots=False)
    other.inspect()
    assert other.analysis_dir != job.analysis_dir


@pytest.mark.parametrize('changed', ['manifest', 'validation', 'tokenizer'])
def test_input_mismatch_rejected_before_analysis(inputs, tmp_path, changed):
    _, data, *_ = inputs
    if changed == 'manifest':
        manifest = json.loads((data/'manifest.json').read_text())
        manifest['vocab_size'] += 1
        (data/'manifest.json').write_text(json.dumps(manifest))
        message = 'manifest differs'
    else:
        filename = data/('validation.pt' if changed == 'validation' else 'tokenizer/tokenizer.json')
        with filename.open('ab') as stream:
            stream.write(b'changed')
        message = 'hash mismatch'
    with pytest.raises(ValueError, match=message):
        job_for(inputs, tmp_path).inspect()
    assert not (tmp_path/'outputs').exists()


def test_raw_gates_expose_saturation_hidden_by_batch_mean():
    state = torch.ones(2, 2, 3)
    write = torch.tensor([[[0.], [1.]], [[1.], [0.]]])
    trace = TraceBuffer(per_example=True)
    plain = TraceBuffer()
    for buffer in (trace, plain):
        buffer.record(torch.ones_like(write), write, state, state+1, torch.ones(2, 4)/4)
    arrays = trace.to_numpy()
    np.testing.assert_allclose(arrays['w_gates'], [[.5, .5]])
    stats = gate_statistics(arrays['w_gates_per_example'])
    assert stats['fraction_near_closed'] == .5 and stats['fraction_near_open'] == .5
    assert stats['per_slot']['mean_within_segment_example_std'] == [.5, .5]
    assert 'w_gates_per_example' not in plain.to_numpy()
    np.testing.assert_allclose(arrays['update_distances'], plain.to_numpy()['update_distances'])


@pytest.mark.parametrize('refine', [False, True])
def test_extended_report_covers_held_out_windows_and_cache(inputs, tmp_path, refine):
    checkpoint, data, cfg, model, _ = inputs
    # Three disjoint four-segment windows in each of the two fixed streams.
    torch.save(torch.arange(160, dtype=torch.long)%128, data/'validation.pt')
    manifest = json.loads((data/'manifest.json').read_text())
    manifest['sha256']['validation'] = file_hash(data/'validation.pt')
    (data/'manifest.json').write_text(json.dumps(manifest))
    state = torch.load(checkpoint, weights_only=False)
    if refine:
        cfg.model.register.refine_gate = True
        state['config'] = asdict(cfg)
        state['model'] = create_model(cfg).state_dict()
        state['model']['router.refine_head.bias'].fill_(1.)
    state['dataset_manifest'] = manifest
    # Exercise all three groups, including the empty complement of their union.
    state['model']['router.mlp_w.2.weight'].zero_()
    state['model']['router.mlp_w.2.bias'].copy_(torch.tensor([-100., 100., 0.]))
    torch.save(state, checkpoint)
    before = file_hash(checkpoint)
    options = dict(registers=True, window_starts=(0, 6, 12), group_repeats=2,
                   lesion_fractions=(.5,), horizon=2)
    job = job_for(inputs, tmp_path, **options)
    info = job.inspect()
    original = job.training_config
    report = job.run()
    assert report['training_config'] == original == asdict(cfg)
    assert file_hash(checkpoint) == before
    result = report['register_analysis']
    assert result['calibration_groups'] == dict(closed=[0], open=[1], other=[2])
    assert [w['role'] for w in result['windows']] == ['calibration', 'evaluation', 'evaluation']
    assert [w['start_segment'] for w in result['windows']] == [0, 6, 12]
    assert report['summary'] == result['windows'][0]['summary']
    for window in result['windows']:
        assert window['summary']['target_tokens'] == 16
        assert np.array(window['diagnostics']['gate_logits']['raw']).shape == (2, 2, 3)
        if refine:
            assert np.array(window['diagnostics']['refine_gate']['raw_logits']).shape == (2, 2, 3)
            assert report['metrics']['grt/refine/refine_mean'] > .5
        assert sum(window['stability']['threshold_sensitivity']['0.01']['counts'].values()) == 3
        all_key = result['intervention_plan'][0]['mask']
        for mode in ('reset', 'swap'):
            effect = window['interventions']['effects'][all_key][mode]
            assert effect['delta_nll_by_lag'][0] == pytest.approx(window['summary']['conditions'][mode]['delta_nll'])
            assert effect['target_tokens_by_lag'] == [16, 16]
            assert np.array(effect['delta_nll_by_start_lag_example']).shape == (2, 2, 2)
    assert (job.analysis_dir/'registers.png').stat().st_size > 0
    assert report['analysis_source_sha256']['register_analysis.py']
    assert info['forward_count_kind'] == 'upper_bound_until_calibration'
    assert 0 < report['environment']['analysis_forward_calls'] <= info['forward_calls']
    assert 'NaN' not in job.report().read_text()
    assert job_for(inputs, tmp_path, **options).run()['report_id'] == report['report_id']
    changed = job_for(inputs, tmp_path, **{**options, 'analysis_seed': 55})
    changed.inspect()
    assert changed.analysis_dir != job.analysis_dir


@pytest.mark.parametrize('options', [dict(window_starts=(0, 2)), dict(window_starts=(0,)),
    dict(window_starts=(1, 16)), dict(group_repeats=0), dict(lesion_fractions=(0.,)),
    dict(analysis_seed=-1)])
def test_invalid_extended_settings(inputs, tmp_path, options):
    with pytest.raises(ValueError):
        job_for(inputs, tmp_path, registers=True, **options)


def test_short_validation_rejects_requested_windows(inputs, tmp_path):
    with pytest.raises(ValueError, match='too short'):
        job_for(inputs, tmp_path, registers=True).inspect()
