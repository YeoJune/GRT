import json
import numpy as np
import pytest
import torch
from grt.models.factory import create_model
from grt.rtla import TraceBuffer, analyze
from grt.data import collate, generate_sample
from grt.checkpoint import evaluation_context
from conftest import tiny_model_config

def test_mean_of_fp32_norms_before_update():
    trace = TraceBuffer()
    state = torch.tensor([[[3.,4.]], [[-3.,-4.]]],dtype=torch.bfloat16)
    candidate = torch.zeros_like(state)
    r,w = torch.tensor([[[0.2]],[[0.8]]]),torch.tensor([[[0.4]],[[0.6]]])
    alpha = torch.tensor([[0.2,0.8],[0.4,0.6]])
    trace.record(r,w,state,candidate,alpha)
    arrays = trace.to_numpy()
    assert all(a.dtype == np.float32 for a in arrays.values())
    assert arrays["s_norms"].item() == 5.0  # norm(mean state) would be zero.
    assert arrays["update_distances"].item() == 5.0
    assert arrays["r_gates"].item() == pytest.approx(0.5)
    np.testing.assert_allclose(arrays["attn_weights"],[[0.3,0.7]])

def test_forward_trace_equality_shapes_and_initial_state(tmp_path):
    model = create_model(tiny_model_config()).eval()
    x = torch.randint(0,32,(3,24))
    with torch.no_grad():
        ordinary = model(x)
        traced,trace = model.forward_with_trace(x)
    torch.testing.assert_close(ordinary.logits,traced.logits,rtol=0,atol=0)
    arrays = trace.to_numpy()
    for key in ("r_gates","w_gates","s_norms","update_distances"):
        assert arrays[key].shape == (3,2)
    assert arrays["attn_weights"].shape == (3,8)
    assert not arrays["s_norms"][0].any()
    path = tmp_path / "trace.npz"
    trace.save(path,{"sample_ids": [0,1,2]})
    with np.load(path,allow_pickle=False) as saved:
        assert saved["s_norms"].dtype == np.float32
        assert json.loads(saved["metadata"].item())["read_active"] is False
        assert "update_distances" in saved and "pred_errors" not in saved
    with pytest.raises(FileExistsError):
        trace.save(path,{})

def test_analysis_outputs_and_mode_rng_restore(tmp_path):
    cfg = tiny_model_config(segment_len=128)
    cfg.vocab_size = 1024
    model = create_model(cfg).train()
    batch = collate([generate_sample("copy","validation",i,4) for i in range(2)])
    rng = torch.get_rng_state().clone()
    trace,panel = analyze(model,batch,tmp_path / "step_1_copy_T4.npz",{"sample_ids": [0,1]})
    assert panel.exists() and panel.stat().st_size > 1000
    assert model.training and torch.equal(rng,torch.get_rng_state())
    from grt.plots import plot_trace
    plot_trace(trace.to_numpy(),tmp_path / "replot.png",trace.read_active)
    assert (tmp_path / "replot.png").exists()

def test_rmt_analysis_rejected(tmp_path):
    model = create_model(tiny_model_config("rmt"))
    with pytest.raises(ValueError,match="GRT"):
        analyze(model,{},tmp_path / "no.npz",{})

def test_mode_restore_on_callback_error():
    model = create_model(tiny_model_config()).train()
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with evaluation_context(model):
            assert not model.training
            torch.rand(3)
            raise RuntimeError("fixture")
    assert model.training and torch.equal(rng,torch.get_rng_state())
