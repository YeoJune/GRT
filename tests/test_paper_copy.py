import copy
from pathlib import Path
import pytest
import torch
from grt.config import load_config, validate_config
from grt.data import generate_paper_copy, make_loader
from grt.checkpoint import seed_all, load_checkpoint, restore_model
from grt.models.factory import create_model
from grt.metrics import masked_ce
from grt.evaluator import evaluate_copy_generation
from grt.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def paper_config(path):
    return validate_config(load_config(ROOT / "configs/rmt_paper_copy.yaml", output_dir=path))


def tiny_relative(cfg):
    m = copy.deepcopy(cfg.model)
    m.d_model, m.head_dim, m.num_registers = 16, 8, 4
    m.alu.num_layers, m.alu.d_ff = 2, 32
    return m


def test_published_dimensions_and_parameter_count(tmp_path):
    cfg = paper_config(tmp_path)
    model = create_model(cfg.model)
    assert sum(p.numel() for p in model.parameters()) == 926220
    torch.testing.assert_close(model.mem0, model.mem0[:, :1].expand_as(model.mem0))
    assert cfg.training.batch_size * cfg.training.grad_accum_steps == 32
    assert cfg.training.lr == 1e-4 and cfg.training.scheduler == "plateau"


def test_shifted_copy_mask_and_cyclic_train_set(tmp_path):
    cfg = paper_config(tmp_path)
    rng = torch.get_rng_state().clone()
    sample = generate_paper_copy("train", 7)
    assert torch.equal(rng, torch.get_rng_state())
    x, y = sample["input_ids"], sample["labels"]
    source = x[:24]
    assert x.shape == y.shape == (72,)
    assert x[24] == 1 and (source >= 2).all() and (source < 12).all()
    assert torch.equal(x[25:49], source)
    assert torch.equal(x[49:], source[:-1])
    assert (y[:24] == -100).all()
    assert torch.equal(y[24:], source.repeat(2))
    a = next(iter(make_loader(cfg.data, "train", 3, 1, 1, start_id=7)))
    b = next(iter(make_loader(cfg.data, "train", 3, 1, 1, start_id=100007)))
    assert torch.equal(a["input_ids"], b["input_ids"])
    val = generate_paper_copy("validation", 7)
    assert not torch.equal(x, val["input_ids"])


@pytest.mark.parametrize("changed_position", [30, 48, 60, 71])
def test_future_answers_cannot_leak_into_earlier_predictions(tmp_path, changed_position):
    cfg = paper_config(tmp_path)
    model = create_model(tiny_relative(cfg)).eval()
    ids = generate_paper_copy("validation", 0)["input_ids"][None]
    changed = ids.clone()
    changed[:, changed_position] = (changed[:, changed_position] + 3) % 12
    with torch.no_grad():
        before, after = model(ids).logits, model(changed).logits
    torch.testing.assert_close(before[:, :changed_position], after[:, :changed_position], rtol=0, atol=0)


def test_memory_carries_gradient_and_has_slot_identity(tmp_path):
    cfg = paper_config(tmp_path)
    seed_all(123)
    model = create_model(tiny_relative(cfg)).double().eval()
    sample = generate_paper_copy("validation", 0)
    memories = []
    original = model.forward_segment
    def capture(ids, memory):
        logits, state = original(ids, memory)
        state.retain_grad()
        memories.append(state)
        return logits, state
    model.forward_segment = capture
    loss = masked_ce(model(sample["input_ids"][None]).logits, sample["labels"][None])
    loss.backward()
    assert memories[0].grad.abs().sum() > 0
    assert model.mem0.grad.abs().sum() > 0
    # Despite equal initial memory content, relative attention distinguishes slots.
    assert (memories[0][:, 0] - memories[0][:, -1]).abs().max() > 1e-10
    # Removing relative terms restores equal memory rows: identity is positional.
    with torch.no_grad():
        for layer in model.layers:
            layer.r_net.weight.zero_()
        _, no_position = original(sample["input_ids"][None, :24], model.initial_memory(1))
    torch.testing.assert_close(no_position[:, :1].expand_as(no_position), no_position, rtol=0, atol=1e-12)


def test_generation_feeds_predictions_and_commits_only_complete_segments(tmp_path):
    cfg = paper_config(tmp_path)
    model = create_model(tiny_relative(cfg)).eval()
    calls = []
    def fake_segment(ids, memory):
        calls.append((ids.clone(), memory.clone()))
        logits = torch.zeros(ids.shape[0], 24, 12)
        logits[..., 7] = 1
        return logits, memory+1
    model.forward_segment = fake_segment
    output = model.generate_copy(torch.full((2, 24), 5, dtype=torch.long))
    assert output.shape == (2, 48) and (output == 7).all()
    assert len(calls) == 49
    initial = model.initial_memory(2)
    for i in range(24):
        ids, state = calls[1+i]
        assert (ids[:, 0] == 1).all()
        assert (ids[:, 1:i+1] == 7).all()
        assert (ids[:, i+1:] == 0).all()
        torch.testing.assert_close(state, initial+1)
    for i in range(24):
        ids, state = calls[25+i]
        assert (ids[:, :i+1] == 7).all()
        assert (ids[:, i+1:] == 0).all()
        torch.testing.assert_close(state, initial+2)


def test_generation_metric_does_not_accept_teacher_forced_outputs(tmp_path):
    cfg = paper_config(tmp_path)
    model = create_model(tiny_relative(cfg))
    # An oracle uses source only; deliberately fail the second copy.
    model.generate_copy = lambda source: torch.cat([source, torch.zeros_like(source)], dim=1)
    loader = make_loader(cfg.data, "validation", 3, 3, 2)
    rng = torch.get_rng_state().clone()
    metrics = evaluate_copy_generation(model, loader)
    assert metrics["token_accuracy"] == 0.5 and metrics["exact_match"] == 0
    assert metrics["first_copy_accuracy"] == 1 and metrics["second_copy_accuracy"] == 0
    assert model.training and torch.equal(rng, torch.get_rng_state())


def test_paper_optimizer_plateau_and_resume_next_update(tmp_path):
    cfg = paper_config(tmp_path / "full")
    cfg.model = tiny_relative(cfg)
    cfg.training.max_steps, cfg.training.batch_size = 3, 2
    cfg.training.stop_on_convergence = False
    cfg.data.validation_samples = cfg.evaluation.autoregressive_samples = 2
    cfg.evaluation.every_steps = 1
    cfg.training.plateau_patience = 0
    cfg.training.plateau_every_steps = 1
    seed_all(123)
    full = Trainer(create_model(cfg.model), cfg)
    full.train()
    split_cfg = copy.deepcopy(cfg)
    split_cfg.run.output_dir = str(tmp_path / "split")
    seed_all(123)
    split = Trainer(create_model(split_cfg.model), split_cfg)
    assert isinstance(split.optimizer, torch.optim.Adam)
    assert split.optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    split.train(stop_after=1)
    state = load_checkpoint(tmp_path / "split/last.pt")
    model, restored_cfg = restore_model(state)
    resumed = Trainer(model, restored_cfg)
    resumed.resume(state)
    resumed.train()
    assert full.scheduler.state_dict() == resumed.scheduler.state_dict()
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("change", ["grt", "dimensions", "protocol", "length"])
def test_stage_one_rejects_mixed_protocols(tmp_path, change):
    cfg = paper_config(tmp_path)
    if change == "grt": cfg.model.name = "grt"
    elif change == "dimensions": cfg.model.segment_len = 128
    elif change == "protocol": cfg.data.protocol = "recovery"
    else: cfg.data.eval_segments = [3, 6]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_early_stop_requires_free_running_success(tmp_path, monkeypatch):
    cfg = paper_config(tmp_path)
    cfg.model = tiny_relative(cfg)
    trainer = Trainer(create_model(cfg.model), cfg)
    monkeypatch.setattr("grt.trainer.evaluate", lambda *args: {"loss": 0.01, "token_accuracy": 1.0, "exact_match": 1.0})
    monkeypatch.setattr("grt.trainer.evaluate_copy_generation", lambda *args: {"token_accuracy": 0.1, "exact_match": 0.0})
    trainer.validate()
    assert not trainer.converged and not (tmp_path / "converged.pt").exists()
    monkeypatch.setattr("grt.trainer.evaluate_copy_generation", lambda *args: {"token_accuracy": 1.0, "exact_match": 1.0})
    trainer.validate()
    assert trainer.converged and (tmp_path / "converged.pt").exists()
    trainer.train()
    assert trainer.global_step == 0  # No further update after a saved convergence decision.
