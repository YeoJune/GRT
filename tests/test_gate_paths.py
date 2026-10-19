"""CPU contracts for write-source experiments, not convergence measurements."""
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grt.config import validate_config
from grt.models.factory import create_model
from grt.register_analysis import RegisterObserver
from grt.memory_analysis import analyze_stream
from grt.utils.kaggle import WikiTextRun
from conftest import tiny_config


SOURCES = ('independent', 'separate_head', 'candidate')


def configuration(tmp_path, source, detach=False):
    cfg = tiny_config(tmp_path, 'grt')
    cfg.model.router.write_source = source
    cfg.model.router.write_detach_source = detach
    return validate_config(cfg)


@pytest.mark.parametrize('source', SOURCES)
def test_selected_source_and_fixed_state_attention_use_one_stack(tmp_path, source):
    cfg = configuration(tmp_path, source)
    model = create_model(cfg).eval()
    state = torch.randn(2, cfg.model.num_registers, cfg.model.d_model)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    captured = {}
    handles = [
        model.backbone.final_norm.register_forward_pre_hook(
            lambda module, args: captured.update(register_hidden=args[0][:, ids.shape[1]:])),
        model.backbone.final_norm.register_forward_hook(
            lambda module, args, out: captured.update(candidate=out[:, ids.shape[1]:])),
        model.router.mlp_w[0].register_forward_pre_hook(
            lambda module, args: captured.update(gate_input=args[0])),
    ]
    calls = []
    handles.append(model.backbone.register_forward_hook(lambda *args: calls.append(1)))
    try:
        out = model(ids, memory_state=state)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 1
    assert torch.isfinite(out.logits).all()
    assert out.memory_state.shape == state.shape
    from grt.logger import metadata
    info = metadata(model, cfg)
    assert info['parameters_router'] == sum(p.numel() for p in model.router.parameters())
    assert info['parameters_write_head'] == sum(p.numel() for p in model.router.mlp_w.parameters())
    assert info['parameters_gate_head'] == (32 * 32 + 32 if source == 'separate_head' else 0)
    gate_input = captured['gate_input']
    if source == 'independent':
        expected, _ = model.router.context(model.backbone.embedding(ids), state)
        assert hasattr(model.router, 'pool_attn')
    else:
        features = captured['register_hidden' if source == 'separate_head' else 'candidate']
        if source == 'separate_head':
            features = model.router.gate_head(features)
        expected = model.router.with_state(features, state)
        assert not hasattr(model.router, 'pool_attn')
    assert hasattr(model.router, 'gate_head') == (source == 'separate_head')
    assert gate_input.shape[-1] == 2 * cfg.model.d_model
    torch.testing.assert_close(gate_input, expected, rtol=0, atol=0)
    hidden = model.router.mlp_w[1](model.router.mlp_w[0](gate_input))
    head = model.router.mlp_w[-1]
    if hidden.ndim == 3:
        expected_logits = torch.stack([
            torch.nn.functional.linear(hidden[:, i], head.weight[i:i+1], head.bias[i:i+1])
            for i in range(state.shape[1])], 1).squeeze(-1)
    else:
        expected_logits = torch.nn.functional.linear(hidden, head.weight, head.bias)
    torch.testing.assert_close(model.router.mlp_w(gate_input), expected_logits)


def test_only_second_route_adds_head_and_gate_head_cannot_change_candidate(tmp_path):
    cfg = configuration(tmp_path, 'separate_head')
    model = create_model(cfg).eval()
    shared = create_model(configuration(tmp_path, 'candidate')).eval()
    assert set(model.state_dict()) - set(shared.state_dict()) == {
        'router.gate_head.weight', 'router.gate_head.bias'}
    assert set(shared.state_dict()) - set(model.state_dict()) == set()
    assert model.router.gate_head.weight.shape == (32, 32)
    ids = torch.tensor([[1, 2, 3]])
    state = torch.randn(1, 4, 32)
    candidates = []
    handle = model.backbone.final_norm.register_forward_hook(
        lambda module, args, out: candidates.append(out[:, ids.shape[1]:].clone()))
    try:
        before = model(ids, memory_state=state)
        with torch.no_grad():
            model.router.gate_head.weight.add_(1.)
        after = model(ids, memory_state=state)
    finally:
        handle.remove()
    torch.testing.assert_close(candidates[0], candidates[1], rtol=0, atol=0)
    torch.testing.assert_close(before.logits, after.logits, rtol=0, atol=0)
    assert not torch.allclose(before.memory_state, after.memory_state)


@pytest.mark.parametrize('source', SOURCES[1:])
@pytest.mark.parametrize('detach', [False, True])
def test_detach_blocks_only_source_edge_and_gate_head_still_learns(
        tmp_path, source, detach):
    cfg = configuration(tmp_path, source, detach)
    model = create_model(cfg)
    state = torch.randn(2, 4, 32, requires_grad=True)
    features = torch.randn_like(state, requires_grad=True)
    result = model.router(torch.randn(2, 3, 32), state,
                          features=features, return_gate_inputs=True)
    result[3].square().mean().backward()
    assert (features.grad is None) == detach
    if not detach:
        assert features.grad.abs().sum() > 0
    assert state.grad.abs().sum() > 0
    assert model.router.reg_attn.in_proj_weight.grad.abs().sum() > 0
    if source == 'separate_head':
        assert model.router.gate_head.weight.grad.abs().sum() > 0
    assert model.router.mlp_w[0].weight.grad.abs().sum() > 0
    assert model.router.mlp_w[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('read', [False, True])
def test_shared_sources_preserve_causality_prefix_and_cross_segment_gradient(tmp_path, source, read):
    cfg = configuration(tmp_path, source)
    cfg.model.register.conditional_read = read
    model = create_model(cfg).eval()
    ids = torch.arange(1, 11)[None]
    state = torch.randn(1, 4, 32)
    altered = ids.clone(); altered[:, 2:4] = torch.tensor([[21, 22]])
    out = model(ids, memory_state=state)
    torch.testing.assert_close(out.logits[:, :2],
        model(altered, memory_state=state).logits[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(out.logits[:, :2],
        model(ids[:, :2], memory_state=state).logits, rtol=1e-5, atol=1e-7)
    current, logits = state, []
    for segment in ids.split(cfg.model.segment_len, 1):
        step = model(segment, memory_state=current)
        current = step.memory_state
        logits.append(step.logits)
    torch.testing.assert_close(out.logits, torch.cat(logits, 1))
    torch.testing.assert_close(out.memory_state, current)
    embeddings = []
    def capture(module, args, output):
        output.retain_grad()
        embeddings.append(output)
    handle = model.backbone.embedding.register_forward_hook(capture)
    targets = ids.clone(); targets[:, :8] = -100
    try:
        model(ids, memory_state=state, target_ids=targets).loss.backward()
    finally:
        handle.remove()
    assert embeddings[0].grad.abs().sum() > 0


@pytest.mark.parametrize('source', SOURCES)
def test_all_sources_keep_raw_tail_penalty_before_write_dropout(tmp_path, source):
    cfg = configuration(tmp_path, source)
    cfg.model.register.dropout_prob = 1.
    cfg.model.register.write_logit_threshold = 5.
    cfg.model.register.write_logit_coef = .1
    model = create_model(cfg).train()
    with torch.no_grad():
        model.router.mlp_w[-1].weight.zero_()
        model.router.mlp_w[-1].bias.fill_(6.)
    ids = torch.arange(8)[None]
    out = model(ids, target_ids=ids)
    torch.testing.assert_close(out.memory_state, model.initial_state(1))
    assert out.training_metrics['train/gate_logit_penalty'] == 1.
    assert out.training_metrics['train/gate_hidden_norm'].ndim == 0
    out.loss.backward()
    torch.testing.assert_close(model.router.mlp_w[-1].bias.grad,
                               torch.full((4,), .2 / 4))


@pytest.mark.parametrize('source', SOURCES)
def test_observer_and_trace_report_real_feature_blocks(tmp_path, source):
    cfg = configuration(tmp_path, source)
    cfg.data.name = 'wikitext103'
    cfg.data.options = {'tokenization': 'word'}
    cfg.rtla.batch_size = 2
    observer = RegisterObserver()
    model = create_model(cfg).eval()
    summary, arrays = analyze_stream(model, cfg, torch.arange(80) % 128,
        segments=2, warmup_segments=1, observer=observer, trace_examples=True)
    diagnostics = observer.diagnostics(model)
    assert np.asarray(diagnostics['gate_logits']['raw']).shape == (2, 2, 4)
    assert np.asarray(diagnostics['write_gates']).shape == (2, 2, 4)
    info = diagnostics['router_input']
    assert info['source'] == source
    assert info['state_reference'] == 'attention(source, segment_start_state, segment_start_state)'
    name = {'independent': 'token_context', 'separate_head': 'gate_head', 'candidate': 'candidate'}[source]
    expected = {name, 'memory_context'}
    assert set(info['feature_blocks']) == expected
    assert ('gate_head' in info) == (source == 'separate_head')
    assert ('trace_write_pool_attention' in arrays) == (source == 'independent')
    assert summary['conditions']['normal']['loss'] > 0


def test_invalid_independent_detach_and_unknown_source(tmp_path):
    with pytest.raises(ValueError, match='shared candidate source'):
        configuration(tmp_path, 'independent', detach=True)
    with pytest.raises(ValueError, match='write_source'):
        configuration(tmp_path, 'unknown')
    with pytest.raises(ValueError, match='write_source'):
        configuration(tmp_path, 'candidate_attention')
    from grt.experiments import load_experiment
    with pytest.raises(ValueError, match='Unknown keys'):
        load_experiment(task='remember', model='grt',
            overrides={'model': {'router': {'write_include_state': True}}})


@pytest.mark.parametrize('source', SOURCES[1:])
def test_shared_gate_training_resumes_exactly_with_dropout_and_penalty(tmp_path, monkeypatch, source):
    from test_neox import stream_config
    from grt import stream_training
    from grt.trainer import run

    cfg = stream_config(tmp_path)
    cfg.model.router.write_source = source
    cfg.model.router.write_detach_source = source == 'candidate'
    cfg.model.alu.dropout = .1
    cfg.model.register.dropout_prob = .2
    cfg.model.register.write_gate_bias_init = 6.
    cfg.model.register.write_logit_threshold = .1
    cfg.model.register.write_logit_coef = .01
    save = stream_training.atomic_save
    early = tmp_path/'early.pt'

    def snapshot(state, path):
        save(state, path)
        if state['step'] == 2 and Path(path).name == 'last.pt':
            save(state, early)

    monkeypatch.setattr(stream_training, 'atomic_save', snapshot)
    run(cfg, torch.device('cpu'))
    checkpoint = Path(cfg.run.output_dir)/'last.pt'
    full = torch.load(checkpoint, weights_only=False)
    torch.save(torch.load(early, weights_only=False), checkpoint)
    run(cfg, torch.device('cpu'))
    resumed = torch.load(checkpoint, weights_only=False)
    for name, expected in full['model'].items():
        torch.testing.assert_close(resumed['model'][name], expected, rtol=0, atol=0)
    assert full['target_tokens_seen'] == resumed['target_tokens_seen']


@pytest.mark.parametrize('runtime', ['colab', 'kaggle'])
def test_notebook_gate_conditions_have_distinct_configs_and_run_paths(tmp_path, monkeypatch, runtime):
    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo/'notebooks'/f'{runtime}.ipynb').read_text())
    code = ast.parse(''.join(notebook['cells'][3]['source']))
    monkeypatch.setattr(WikiTextRun, 'setup', lambda self: None)
    paths = set()
    for source in SOURCES:
        for detach in (False, True) if source != 'independent' else (False,):
            selection = copy.deepcopy(code)
            values = dict(write_source=source, write_detach_source=detach)
            for node in selection.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in values:
                        node.value = ast.Constant(values[node.targets[0].id])
            namespace = dict(Path=Path, WikiTextRun=WikiTextRun, repo=repo)
            exec(compile(ast.fix_missing_locations(selection), '<selection>', 'exec'), namespace)
            job = namespace['job']; cfg = job.config('grt', calibrated=False)
            assert cfg.model.router.write_source == source
            assert cfg.model.router.write_detach_source == detach
            assert not hasattr(cfg.model.router, 'write_include_state')
            assert cfg.model.segment_len == 75 and cfg.model.num_registers == 20
            assert cfg.model.register.write_logit_coef == 1e-4
            paths.add(job.run_dir)
    assert len(paths) == 5
