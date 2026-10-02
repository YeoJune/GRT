import pytest
import torch
from grt.models.factory import create_model
from grt.models.interface import ModelOutput
from conftest import tiny_model_config

@pytest.mark.parametrize("name", ["grt","rmt"])
def test_common_contract_positions_and_reset(name):
    cfg = tiny_model_config(name)
    model = create_model(cfg).eval()
    x = torch.randint(0,32,(2,24))
    out = model(x, torch.ones_like(x,dtype=torch.bool))
    assert isinstance(out, ModelOutput) and out.logits.shape == (2,24,32) and out.logits.isfinite().all()
    assert model.lm_head.weight is model.embedding.weight
    model(torch.randint(0,32,(2,16)))
    torch.testing.assert_close(out.logits, model(x).logits, rtol=0, atol=0)
    with torch.no_grad():
        model.pos_emb.zero_()
    assert not torch.equal(out.logits, model(x).logits)
    for bad in (torch.zeros(8,dtype=torch.int64), torch.zeros(2,0,dtype=torch.int64),
                torch.zeros(2,9,dtype=torch.int64), torch.full((2,8),32,dtype=torch.int64),
                torch.zeros(2,8)):
        with pytest.raises(ValueError):
            model(bad)
    with pytest.raises(ValueError, match="Padding"):
        model(x, torch.zeros_like(x,dtype=torch.bool))
    with pytest.raises(ValueError):
        model(x, torch.ones_like(x))

@pytest.mark.parametrize("name", ["grt","rmt"])
def test_full_bptt_and_independent_layers(name):
    cfg = tiny_model_config(name)
    cfg.alu.num_layers = 2
    model = create_model(cfg)
    memories = []
    if name == "grt":
        def capture(module, inputs, outputs):
            outputs[1].retain_grad()
            memories.append(outputs[1])
        handle = model.alu.register_forward_hook(capture)
        layers = model.alu.layers
    else:
        def capture(module, inputs, outputs):
            outputs.retain_grad()
            memories.append(outputs)
        handle = model.layers[-1].register_forward_hook(capture)
        layers = model.layers
    assert layers[0].linear1.weight is not layers[1].linear1.weight
    assert not torch.equal(layers[0].linear1.weight,layers[1].linear1.weight)
    x = torch.randint(0,32,(2,24))
    model(x).logits[:,-1].square().sum().backward()
    handle.remove()
    assert memories[0].grad is not None and memories[0].grad.abs().sum() > 0
    initial = model.s0 if name == "grt" else model.mem0
    assert initial.grad is not None and initial.grad.abs().sum() > 0

def test_unknown_factory():
    with pytest.raises(ValueError):
        create_model(tiny_model_config("other"))
