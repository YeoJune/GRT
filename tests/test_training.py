import copy
import math
import pytest
import torch
from grt.checkpoint import seed_all, load_checkpoint, restore_model, capture_rng
from grt.models.factory import create_model
from grt.trainer import Trainer, optimizer_groups
from grt.metrics import masked_ce, MetricAccumulator
from grt.evaluator import evaluate, measure_performance, check_precision
from grt.data import make_loader
from grt.logger import Logger
from conftest import tiny_model_config, tiny_run_config

def test_manual_masked_metrics():
    logits = torch.zeros(2,3,3)
    labels = torch.tensor([[0,-100,1],[-100,0,-100]])
    totals = MetricAccumulator()
    totals.update(logits,labels)
    result = totals.compute()
    assert result["loss"] == pytest.approx(math.log(3))
    assert result["token_accuracy"] == pytest.approx(2/3)
    assert result["exact_match"] == 0.5
    changed = logits.clone()
    changed[labels == -100] = 100
    torch.testing.assert_close(masked_ce(changed,labels),masked_ce(logits,labels))
    with pytest.raises(ValueError):
        masked_ce(logits,torch.full_like(labels,-100))

@pytest.mark.parametrize("name", ["grt","rmt"])
def test_training_resume_next_update_with_dropout(tmp_path,name):
    cfg = tiny_run_config(tmp_path / "full", name, steps=3)
    cfg.evaluation.every_steps = 1
    seed_all(123)
    full = Trainer(create_model(cfg.model),cfg)
    full.train()
    full_state = {k:v.clone() for k,v in full.model.state_dict().items()}
    full_rng = capture_rng()
    split_cfg = copy.deepcopy(cfg)
    split_cfg.run.output_dir = str(tmp_path / "split")
    seed_all(123)
    split = Trainer(create_model(split_cfg.model),split_cfg)
    split.train(stop_after=1)
    state = load_checkpoint(tmp_path / "split/last.pt")
    assert state["next_train_sample_id"] == 2
    model,cfg2 = restore_model(state)
    resumed = Trainer(model,cfg2)
    resumed.resume(state)
    resumed.train()
    assert resumed.next_train_sample_id == full.next_train_sample_id == 6
    assert resumed.global_step == full.global_step == 3
    assert resumed.scheduler.state_dict() == full.scheduler.state_dict()
    for key,value in full_state.items():
        torch.testing.assert_close(value,resumed.model.state_dict()[key],rtol=0,atol=0)
    assert torch.equal(full_rng["torch"],capture_rng()["torch"])
    for path in (tmp_path / "split/best.pt", tmp_path / "split/last.pt"):
        assert path.exists() and not path.with_suffix(".pt.tmp").exists()

@pytest.mark.parametrize("name", ["grt","rmt"])
def test_shared_trainer_eval_partition_and_final_save(tmp_path,name):
    cfg = tiny_run_config(tmp_path,name,steps=1)
    logger = Logger(tmp_path,cfg)
    model = create_model(cfg.model)
    trainer = Trainer(model,cfg,logger)
    trainer.train()
    assert trainer.global_step == 1
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()
    assert "val/loss" in (tmp_path / "metrics.jsonl").read_text()
    rng = torch.get_rng_state().clone()
    a = evaluate(model,make_loader(cfg.data,"test",4,3,1))
    b = evaluate(model,make_loader(cfg.data,"test",4,3,2))
    assert model.training and torch.equal(rng,torch.get_rng_state())
    for k in ("loss","token_accuracy","exact_match"):
        assert a[k] == pytest.approx(b[k],abs=1e-6)
    perf = measure_performance(model,next(iter(make_loader(cfg.data,"test",4,2,2))),warmup=0,iterations=1)
    assert perf["input_tokens_per_sec"] > 0 and perf["peak_allocated_gpu_memory_bytes"] is None
    assert model.training and torch.equal(rng,torch.get_rng_state())

def test_accumulation_uses_effective_token_count(tmp_path):
    cfg = tiny_run_config(tmp_path / "batch", "rmt", steps=1, batch_size=4)
    cfg.model.alu.dropout = 0.0
    seed_all(42)
    baseline = Trainer(create_model(cfg.model),cfg)
    baseline.train()
    accumulated_cfg = copy.deepcopy(cfg)
    accumulated_cfg.run.output_dir = str(tmp_path / "accum")
    accumulated_cfg.training.batch_size = 2
    accumulated_cfg.training.grad_accum_steps = 2
    seed_all(42)
    accumulated = Trainer(create_model(accumulated_cfg.model),accumulated_cfg)
    accumulated.train()
    assert baseline.next_train_sample_id == accumulated.next_train_sample_id == 4
    for a,b in zip(baseline.model.parameters(),accumulated.model.parameters()):
        torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-6)

def test_weight_decay_uses_module_types():
    model = create_model(tiny_model_config())
    groups = optimizer_groups(model,0.1)
    exempt = {id(p) for p in groups[1]["params"]}
    assert id(model.alu.layers[0].norm1.weight) in exempt
    assert id(model.router.pool_attn.in_proj_bias) in exempt
    assert id(model.embedding.weight) not in exempt
    assert len([p for g in groups for p in g["params"]]) == len(list(model.parameters()))

@pytest.mark.parametrize("name", ["grt","rmt"])
def test_fixed_batch_overfit_smoke(name):
    seed_all(9)
    cfg = tiny_model_config(name,dropout=0.0)
    model = create_model(cfg)
    x = torch.tensor([[4,5,6,7,8,9,10,11,12,13,14,15,3,3,3,3]])
    y = torch.full_like(x,-100)
    y[:,-4:] = torch.tensor([4,5,6,7])
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.02,weight_decay=0.0)
    initial = masked_ce(model(x).logits,y).item()
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_ce(model(x).logits,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
    model.eval()
    final = masked_ce(model(x).logits,y).item()
    assert final < initial * 0.1
    assert torch.equal(model(x).logits.argmax(-1)[:,-4:],y[:,-4:])

def test_legacy_checkpoint_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": {}},path)
    with pytest.raises(ValueError,match="migration"):
        load_checkpoint(path)

def test_unsupported_precision():
    with pytest.raises(ValueError,match="CUDA"):
        check_precision(torch.device("cpu"),"fp16")

def test_periodic_last_checkpoint_survives_interruption(tmp_path):
    class InterruptedLogger:
        def log(self, step, payload):
            if step == 3:
                raise RuntimeError("simulated interruption before normal shutdown")
    cfg = tiny_run_config(tmp_path, "rmt", steps=3)
    cfg.checkpoint.every_steps = 2
    trainer = Trainer(create_model(cfg.model), cfg, InterruptedLogger())
    with pytest.raises(RuntimeError, match="interruption"):
        trainer.train()
    last = load_checkpoint(tmp_path / "last.pt")
    periodic = load_checkpoint(tmp_path / "step_000002.pt")
    assert last["global_step"] == periodic["global_step"] == 2
    assert last["next_train_sample_id"] == 4
    for key, value in last["model"].items():
        torch.testing.assert_close(value, periodic["model"][key], rtol=0, atol=0)

def test_nonfinite_gradient_stops_and_writes_diagnostics(tmp_path):
    cfg = tiny_run_config(tmp_path,"rmt",steps=1)
    model = create_model(cfg.model)
    hook = model.embedding.weight.register_hook(lambda grad: torch.full_like(grad,float("nan")))
    trainer = Trainer(model,cfg)
    with pytest.raises(FloatingPointError,match="gradient"):
        trainer.train()
    hook.remove()
    assert trainer.global_step == 0 and (tmp_path / "failure.json").exists()

def test_overflow_skip_does_not_advance_optimizer_or_scheduler(tmp_path):
    class SimulatedOverflow:
        def __init__(self):
            self.value = 8.0
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def is_enabled(self):
            return True
        def get_scale(self):
            return self.value
        def step(self, optimizer):
            pass
        def update(self):
            self.value /= 2
    cfg = tiny_run_config(tmp_path, "rmt", steps=1)
    trainer = Trainer(create_model(cfg.model), cfg, Logger(tmp_path,cfg))
    parameters = [p.detach().clone() for p in trainer.model.parameters()]
    scheduler = trainer.scheduler.state_dict().copy()
    trainer.scaler = SimulatedOverflow()
    with pytest.raises(FloatingPointError, match="overflow"):
        trainer.train()
    assert trainer.global_step == 0 and trainer.scheduler.state_dict() == scheduler
    assert "overflow_skip" in (tmp_path / "metrics.jsonl").read_text()
    for before, after in zip(parameters, trainer.model.parameters()):
        torch.testing.assert_close(before,after,rtol=0,atol=0)
