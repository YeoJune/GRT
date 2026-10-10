import pytest
import torch
from grt.models.factory import create_model
from grt.models.grt import write_back
from grt.data import forward_batch, make_dataset, make_loader
from grt.metrics import causal_loss
from grt import trainer
from conftest import tiny_config


@pytest.mark.parametrize("name", ["rmt", "grt"])
def test_common_forward_loss_and_teacher_forced_generation_prefix(tmp_path, name):
    cfg = tiny_config(tmp_path, name)
    cfg.data.train_segments = 3
    model = create_model(cfg).eval()
    batch = next(iter(make_loader(cfg, make_dataset(cfg, "validation", 2), vary=False)))
    output = forward_batch(model, batch, "cpu")
    assert output.logits.shape == (*batch["input_ids"].shape, cfg.model.vocab_size)
    torch.testing.assert_close(
        output.loss, causal_loss(output.logits, batch["labels"], batch["labels_mask"])
    )
    prompt = batch["input_ids_generate"]
    next_logits = model(
        prompt, attention_mask=torch.ones_like(prompt, dtype=torch.bool)
    ).logits[:, -1]
    torch.testing.assert_close(output.logits[:, prompt.shape[1] - 1], next_logits)
    generated = model.generate(
        prompt,
        attention_mask=torch.ones_like(prompt, dtype=torch.bool),
        max_new_tokens=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    assert torch.equal(generated[:, -1], next_logits.argmax(-1))
    model(
        torch.ones((2, 8), dtype=torch.long),
        attention_mask=torch.ones((2, 8), dtype=torch.bool),
    )
    torch.testing.assert_close(output.logits, forward_batch(model, batch, "cpu").logits)


@pytest.mark.parametrize("name", ["rmt", "grt"])
def test_future_targets_do_not_change_value_prediction(tmp_path, name):
    cfg = tiny_config(tmp_path, name)
    cfg.data.train_segments = 3
    model = create_model(cfg).eval()
    batch = next(iter(make_loader(cfg, make_dataset(cfg, "validation", 2), vary=False)))
    index = batch["input_ids_generate"].shape[1] - 1
    router_inputs = []
    if name == "grt":
        hook = model.router.register_forward_pre_hook(
            lambda m, args: router_inputs.append(args[0].detach().clone())
        )
    first = forward_batch(model, batch, "cpu")
    before = [x.clone() for x in router_inputs]
    router_inputs.clear()
    changed = {k: v.clone() for k, v in batch.items()}
    changed["input_ids"][:, -2] = (changed["input_ids"][:, -2] + 1) % 16
    changed["labels"] = changed["input_ids"].clone()
    second = forward_batch(model, changed, "cpu")
    torch.testing.assert_close(
        first.logits[:, index], second.logits[:, index], rtol=0, atol=0
    )
    if name == "grt":
        # Final value/EOS prefixes are the last two calls; value prefix must exclude the answer.
        torch.testing.assert_close(before[-2], router_inputs[-2], rtol=0, atol=0)
        hook.remove()


@pytest.mark.parametrize("active", [False, True])
def test_grt_write_read_and_fact_gradient(tmp_path, active):
    cfg = tiny_config(tmp_path, "grt")
    cfg.data.train_segments = 3
    cfg.model.register.conditional_read = active
    model = create_model(cfg).eval()
    states = []
    hook = model.alu.register_forward_hook(
        lambda m, args, out: (out[1].retain_grad(), states.append(out[1])) and None
    )
    batch = next(iter(make_loader(cfg, make_dataset(cfg, "validation", 2), vary=False)))
    out = forward_batch(model, batch, "cpu")
    out.loss.backward()
    hook.remove()
    assert states[0].grad is not None and states[0].grad.abs().sum() > 0
    assert model.s0.grad.abs().sum() > 0
    grad = model.router.mlp_r[-1].weight.grad
    assert grad is not None and grad.abs().sum() > 0 if active else grad is None
    s, c = torch.randn(2, 4, 32), torch.randn(2, 4, 32)
    torch.testing.assert_close(
        write_back(s, c, torch.zeros(2, 4, 1)), s, rtol=0, atol=0
    )
    torch.testing.assert_close(write_back(s, c, torch.ones(2, 4, 1)), c, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["rmt", "grt"])
def test_common_training_and_exact_resume(tmp_path, name):
    cfg = tiny_config(tmp_path, name)
    cfg.training.curriculum = cfg.training.curriculum[:1]
    cfg.training.curriculum[0].max_steps = 4
    trainer.run(cfg, torch.device("cpu"))
    directory = tmp_path / "stage_01_pairs1_key1"
    uninterrupted = torch.load(directory / "last.pt", weights_only=False)
    earlier = torch.load(directory / "best.pt", weights_only=False)
    assert earlier["step"] < 4
    torch.save(earlier, directory / "last.pt")
    trainer.run(cfg, torch.device("cpu"))
    resumed = torch.load(directory / "last.pt", weights_only=False)
    assert all(
        torch.equal(v, resumed["model"][k]) for k, v in uninterrupted["model"].items()
    )


def test_grt_trace_only_counts_committed_facts(tmp_path):
    cfg = tiny_config(tmp_path, "grt")
    cfg.data.train_segments = 3
    model = create_model(cfg)
    batch = next(iter(make_loader(cfg, make_dataset(cfg, "validation", 2), vary=False)))
    _, trace = model.forward_with_trace(
        **{
            k: batch[k]
            for k in ["input_ids", "attention_mask", "labels", "labels_mask"]
        }
    )
    assert trace.to_numpy()["w_gates"].shape == (2, 4)
    assert trace.to_numpy()["attn_weights"].shape == (2, 4)
