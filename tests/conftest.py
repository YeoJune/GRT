import pytest
import torch
from grt.config import load_config


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_config(path, model="rmt"):
    cfg = load_config(
        ["configs/base.yaml", f"configs/{model}.yaml", "configs/cpu.yaml"],
        output_dir=path,
    )
    if model == "grt":
        cfg.model.router.mlp_hidden = 32
    return cfg
