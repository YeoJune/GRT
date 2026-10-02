from pathlib import Path
import ast
import torch
from grt.models.factory import create_model
from conftest import tiny_model_config

def test_tail_memory_is_reused_at_both_ends():
    cfg = tiny_model_config("rmt")
    model = create_model(cfg).eval()
    inputs,outputs = [],[]
    pre = model.layers[0].register_forward_pre_hook(lambda module,args: inputs.append(args[0].detach().clone()))
    post = model.layers[0].register_forward_hook(lambda module,args,out: outputs.append(out.detach().clone()))
    model(torch.randint(0,32,(2,24)))
    pre.remove(); post.remove()
    m,n = cfg.num_registers,cfg.segment_len
    assert inputs[0].shape[1] == n+2*m
    for t in range(1,len(inputs)):
        previous = outputs[t-1][:,m+n:]
        torch.testing.assert_close(inputs[t][:,:m], previous+model.read_marker)
        torch.testing.assert_close(inputs[t][:,m+n:], previous+model.write_marker)

def test_independent_imports():
    root = Path(__file__).resolve().parents[1] / "src/grt/models"
    for name,forbidden in [("rmt", {"grt","router","alu"}), ("grt", {"rmt"})]:
        tree = ast.parse((root/f"{name}.py").read_text())
        imports = [node.module.split(".")[-1] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module]
        assert not forbidden.intersection(imports)
