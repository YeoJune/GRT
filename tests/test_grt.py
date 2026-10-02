import pytest
import torch
from grt.models.factory import create_model
from grt.models.grt import write_back
from conftest import tiny_model_config

def test_write_boundaries():
    s,c = torch.randn(2,3,8),torch.randn(2,3,8)
    torch.testing.assert_close(write_back(s,c,torch.zeros(2,3,1)),s,rtol=0,atol=0)
    torch.testing.assert_close(write_back(s,c,torch.ones(2,3,1)),c,rtol=0,atol=0)

@pytest.mark.parametrize("active", [False,True])
def test_read_connectivity_and_alu_length(active):
    cfg = tiny_model_config()
    model = create_model(cfg).eval()
    cfg.register.conditional_read = active
    assert model.s0.requires_grad and not model.s0.detach().any()
    lengths = []
    hook = model.alu.layers[0].register_forward_pre_hook(lambda module,args: lengths.append(args[0].shape[1]))
    x = torch.randint(0,32,(2,24))
    before = model(x).logits
    before[:,-1].sum().backward()
    grad = model.router.mlp_r[-1].weight.grad
    assert (grad is not None and grad.abs().sum() > 0) if active else grad is None
    with torch.no_grad():
        model.router.mlp_r[-1].bias.add_(10)
    after = model(x).logits
    if active:
        assert not torch.equal(before,after)
    else:
        torch.testing.assert_close(before,after,rtol=0,atol=0)
    hook.remove()
    assert set(lengths) == {cfg.segment_len+cfg.num_registers}

def test_write_dropout_preserves_state():
    model = create_model(tiny_model_config()).train()
    s,x = torch.randn(2,2,8),torch.randn(2,8,8)
    model.router.dropout_prob = 1.0
    _,write,_ = model.router(x,s)
    assert not write.any()
    torch.testing.assert_close(write_back(s,torch.randn_like(s),write),s,rtol=0,atol=0)
    model.eval()
    assert model.router(x,s)[1].gt(0).all()
    model.train()
    model.router.dropout_prob = 0.0
    assert model.router(x,s)[1].gt(0).all()
