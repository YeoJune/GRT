"""Versioned, atomic checkpoints and RNG-preserving analysis contexts."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import os
import random
import numpy as np
import torch
from grt.config import SPEC_VERSION, GRTConfig, build_config

SCHEMA_VERSION = 1

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG devices do not match this environment")
        torch.cuda.set_rng_state_all(state["cuda"])

@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)

@contextmanager
def evaluation_context(model):
    modes = [(module, module.training) for module in model.modules()]
    with preserve_rng():
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            for module, mode in modes:
                module.training = mode

def save_checkpoint(path, model, cfg, optimizer, scheduler, scaler, progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": SCHEMA_VERSION, "spec_version": SPEC_VERSION,
             "config": asdict(cfg), "model_config": asdict(cfg.model),
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
             "rng": capture_rng(), **progress}
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def load_checkpoint(path):
    # Our checkpoints contain Python/NumPy RNG state. Load only trusted local runs.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != SCHEMA_VERSION or state.get("spec_version") != SPEC_VERSION:
        raise ValueError("Unsupported legacy checkpoint; explicit migration is required")
    return state

def restore_model(state, device="cpu"):
    from grt.models.factory import create_model
    cfg = build_config(GRTConfig, state["config"])
    model = create_model(cfg.model).to(device)
    model.load_state_dict(state["model"])
    return model, cfg
