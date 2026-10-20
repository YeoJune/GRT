"""Known memory effects, true updates, and monitoring without changing training."""
import hashlib
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from grt import trainer
from grt.experiments import load_experiment
from grt.memory_analysis import analyze_stream, probe_metrics, save_analysis
from grt.models.factory import create_model
from grt.rtla import TraceBuffer


def config(root, data=None, enabled=False):
    return load_experiment(task='wikitext103', model='grt', size='tiny',
        tokenization='word', runtime='cpu', steps=2, output_dir=root, overrides={
            'model': {'vocab_size': 128, 'd_model': 32, 'segment_len': 4, 'num_registers': 2,
                      'alu': {'num_layers': 1, 'nhead': 4, 'd_ff': 64, 'dropout': .1},
                      'router': {'mlp_hidden': 32}},
            'data': {'options': {'path': str(data or root)}},
            'training': {'batch_size': 2, 'grad_accum_steps': 1, 'global_batch_size': 2, 'bptt_segments': 1},
            'evaluation': {'batch_size': 2, 'every_steps': 1},
            'rtla': {'enabled': enabled, 'batch_size': 2, 'segments': 2, 'warmup_segments': 1}})


class KnownMemory(torch.nn.Module):
    def __init__(self, ignore=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.ignore = ignore

    def initial_state(self, batch_size):
        return torch.zeros(batch_size, 1, 1)

    def forward(self, ids, memory_state, _trace=None):
        # History remembers which constant stream this is. After one step it recovers.
        index = ids[:, :1] if self.ignore else memory_state[:, 0].long()
        logits = torch.zeros(*ids.shape, 3)
        logits.scatter_(-1, index[:, None].expand(-1, ids.shape[1], -1), 6.)
        next_state = ids[:, -1:].float().unsqueeze(-1)
        if _trace is not None:
            _trace.record(torch.ones_like(memory_state), torch.ones_like(memory_state),
                          memory_state, next_state, torch.ones_like(ids).float()/ids.shape[1])
        torch.rand(1)  # Even a diagnostic model that consumes RNG must be isolated.
        return SimpleNamespace(logits=logits, memory_state=next_state)


@pytest.mark.parametrize('ignore', [False, True])
def test_known_memory_effect_and_recovery(tmp_path, ignore):
    cfg = config(tmp_path)
    cfg.model.num_registers = 1
    model = KnownMemory(ignore).train()
    rng = torch.get_rng_state().clone()
    tokens = torch.cat([torch.ones(11), torch.full((11,), 2)]).long()
    calls = []
    summary, arrays = analyze_stream(model, cfg, tokens, segments=99, horizon=3, slots=True,
        trace_examples=True, progress=lambda done, total: calls.append((done, total)))
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert summary['target_tokens'] == 12 and summary['measured_segments'] == 2
    assert summary['retention_target_tokens'] == [12, 4, 0]
    assert summary['retention_delta_nll'][2] is None
    assert np.isnan(arrays['normal_nll'][1, 2:]).all()
    for name in ['reset', 'swap']:
        assert summary['conditions'][name]['delta_nll'] == pytest.approx(0. if ignore else 6.)
    assert summary['slot_delta_nll'][0] == pytest.approx(0. if ignore else 6.)
    assert summary['retention_delta_nll'][1] == pytest.approx(0.)
    assert 'grt/read_mean' not in probe_metrics(summary, arrays)
    assert calls[-1] == (len(calls), len(calls))
    assert arrays['trace_w_gates_per_example'].shape == (2, 2, 1)
    save_analysis(tmp_path/'analysis', summary, arrays, {}, plots=True)
    assert (tmp_path/'analysis/memory.png').stat().st_size > 0
    assert (tmp_path/'analysis/trace.png').stat().st_size > 0


def test_trace_measures_actual_gated_update():
    trace = TraceBuffer()
    state = torch.ones(1, 2, 2)
    candidate = state+1
    trace.record(torch.ones_like(state[:, :, :1]), torch.tensor([[[0.], [.5]]]),
                 state, candidate, torch.ones(1, 4)/4)
    arrays = trace.to_numpy()
    np.testing.assert_allclose(arrays['candidate_distances'], [[2**.5, 2**.5]])
    np.testing.assert_allclose(arrays['update_distances'], [[0., .5*2**.5]])
    np.testing.assert_allclose(arrays['relative_updates'], [[0., 1/6]])


def test_stream_monitor_logs_and_preserves_training(tmp_path, monkeypatch):
    data = tmp_path/'data'
    data.mkdir()
    hashes = {}
    for split, size in [('train', 100), ('validation', 31), ('test', 31)]:
        path = data/f'{split}.pt'
        torch.save(torch.arange(size)%13, path)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    (data/'manifest.json').write_text(json.dumps(dict(vocab_size=128, tokenization='word', sha256=hashes)))
    runs = []
    class Run:
        settings = SimpleNamespace(mode='online')
        url = 'https://example.invalid/mock-wandb'
        def __init__(self):
            self.summary, self.logs = {}, []
        def define_metric(self, name, **settings):
            torch.rand(3)
        def log(self, payload, step):
            self.logs.append((step, payload))
            torch.rand(3)
        def finish(self):
            pass
    def init(**kwargs):
        run = Run()
        runs.append(run)
        return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init))
    for enabled in [False, True]:
        cfg = config(tmp_path/str(enabled), data, enabled)
        cfg.wandb.enabled = enabled
        trainer.run(cfg, torch.device('cpu'))
    plain = torch.load(tmp_path/'False/last.pt', weights_only=False)
    logged = torch.load(tmp_path/'True/last.pt', weights_only=False)
    assert all(torch.equal(value, logged['model'][key]) for key, value in plain['model'].items())
    assert torch.equal(plain['rng_by_rank'][0]['torch'], logged['rng_by_rank'][0]['torch'])
    assert [step for step, _ in runs[0].logs] == [1, 2]
    key = '03_memory/reset_delta_nll'
    assert all(key in payload for _, payload in runs[0].logs)
    rows = [json.loads(row) for row in (tmp_path/'True/metrics.jsonl').read_text().splitlines()]
    assert rows[-1]['grt/rtla/reset_delta_nll'] == runs[0].logs[-1][1][key]
    assert (tmp_path/'True/rtla/step_000002/arrays.npz').exists()
    # Same checkpoint trajectory and short tail as production WikiText evaluation.
    model = create_model(cfg)
    model.load_state_dict(logged['model'])
    summary, _ = analyze_stream(model, cfg, torch.arange(31)%13, slots=True, horizon=2)
    assert summary['conditions']['normal']['loss'] > 0
    status = json.loads((tmp_path/'True/wandb_status.json').read_text())
    assert status['state'] == 'finished' and status['failures'] == 0
