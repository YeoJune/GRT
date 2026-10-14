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
