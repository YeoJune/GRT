import pytest
import torch
from grt.config import DataConfig
from grt.data import generate_sample, collate, make_loader

@pytest.mark.parametrize("task,lengths", [("copy", [4,10,20]), ("reverse", [4,10,20]), ("passkey", [5,15,30])])
def test_oracle_and_exact_positions(task, lengths):
    payloads = []
    for t in lengths:
        for sample_id in range(5):
            s = generate_sample(task, "test", sample_id, t)
            x, y = s["input_ids"], s["labels"]
            assert x.shape == (t*128,) and s["attention_mask"].all()
            assert x.dtype == y.dtype == torch.int64 and s["attention_mask"].dtype == torch.bool
            assert ((x >= 0) & (x < 1024)).all()
            noise = torch.ones_like(x, dtype=torch.bool)
            if task == "passkey":
                noise[:12], noise[-2:] = False, False
                assert (x[1:12:3] == 1).all()
                assert x[:12:3].unique().numel() == 4
                mapping = dict(zip(x[:12:3].tolist(), x[2:12:3].tolist()))
                assert x[-1] == 2 and y[-1].item() == mapping[x[-2].item()]
                assert (y != -100).nonzero().flatten().tolist() == [t*128-1]
                payload = x[:12].clone(), x[-2:].clone(), y[-1:].clone()
            else:
                noise[:20], noise[-20:] = False, False
                assert (x[-20:] == 3).all()
                assert torch.equal(y[-20:], x[:20] if task == "copy" else x[:20].flip(0))
                assert (y != -100).nonzero().flatten().tolist() == list(range(t*128-20,t*128))
                payload = x[:20].clone(), y[-20:].clone()
            assert ((x[noise] >= 4) & (x[noise] < 1024)).all()
            if sample_id == 0:
                payloads.append(payload)
    for payload in payloads[1:]:
        assert all(torch.equal(a,b) for a,b in zip(payloads[0], payload))

def test_rng_and_worker_independence():
    cfg = DataConfig(task="reverse", target_len=20)
    torch.manual_seed(19)
    state = torch.get_rng_state().clone()
    sample = generate_sample("reverse", "test", 3, 4)
    assert torch.equal(state, torch.get_rng_state())
    torch.manual_seed(99)
    repeat = generate_sample("reverse", "test", 3, 4)
    assert all(torch.equal(sample[k],repeat[k]) for k in sample)
    one = torch.cat([b["input_ids"] for b in make_loader(cfg,"test",4,5,1)])
    cfg.num_workers = 2
    two = torch.cat([b["input_ids"] for b in make_loader(cfg,"test",4,5,3)])
    assert torch.equal(one,two)
    assert not torch.equal(sample["input_ids"], generate_sample("reverse","train",3,4)["input_ids"])

@pytest.mark.parametrize("args", [("bad","test",0,4),("copy","bad",0,4),("copy","test",-1,4),("copy","test",0,1)])
def test_invalid_sample(args):
    with pytest.raises(ValueError):
        generate_sample(*args)

def test_mixed_length_collate():
    with pytest.raises(ValueError):
        collate([generate_sample("copy","test",0,t) for t in (4,10)])
