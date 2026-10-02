import pytest
import torch
from grt.config import (GRTConfig, ModelConfig, ALUConfig, DataConfig, TrainingConfig,
                        EvaluationConfig, RTLAConfig, RunConfig, RouterConfig)

@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)

def tiny_model_config(name="grt", dropout=0.1, segment_len=8):
    return ModelConfig(name=name, vocab_size=32, segment_len=segment_len, num_registers=2, d_model=8,
                       alu=ALUConfig(num_layers=1, nhead=2, d_ff=16, dropout=dropout),
                       router=RouterConfig(pool_nhead=2, mlp_hidden=12) if name == "grt" else None)

def tiny_run_config(path, name="grt", *, steps=4, batch_size=2, accum=1):
    cfg = GRTConfig(model=tiny_model_config(name, segment_len=128),
                    data=DataConfig(task="copy", target_len=20, validation_samples=3, test_samples=3),
                    training=TrainingConfig(batch_size=batch_size, grad_accum_steps=accum, max_steps=steps, warmup_steps=0),
                    evaluation=EvaluationConfig(batch_size=2, every_steps=2, warmup=0, iterations=1),
                    rtla=RTLAConfig(enabled=False), run=RunConfig(output_dir=str(path)))
    # Training fixtures use the real benchmark token space and task positions.
    cfg.model.vocab_size = 1024
    return cfg
