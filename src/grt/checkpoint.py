"""Checkpoint/RNG utilities and explicit read-only import of native author_rmt/1."""

from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np
import torch
from grt.config import Config, build_config, legacy_config

CHECKPOINT_FORMAT = "memory_experiment/2"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    local_cuda = (
        torch.cuda.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            [torch.cuda.get_rng_state()]
            if local_cuda
            else torch.cuda.get_rng_state_all()
        )
        if torch.cuda.is_available()
        else [],
    }
    if local_cuda:
        state["cuda_device"] = torch.cuda.current_device()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and "cuda_device" in state:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG device mismatch")
        torch.cuda.set_rng_state(state["cuda"][0], torch.cuda.current_device())
    elif state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("CUDA RNG device mismatch")
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
    modes = [(m, m.training) for m in model.modules()]
    with preserve_rng(), torch.no_grad():
        try:
            model.eval()
            yield
        finally:
            for m, mode in modes:
                m.training = mode


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    # Locally produced experiment files also contain Python/NumPy RNG state.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") not in (CHECKPOINT_FORMAT, "author_rmt/1"):
        raise ValueError("Unsupported checkpoint format")
    return state


def restore_model(state, device="cpu"):
    cfg = (
        legacy_config(state["config"])
        if state["format"] == "author_rmt/1"
        else build_config(Config, state["config"])
    )
    from grt.models.factory import create_model

    model = create_model(cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model, cfg
