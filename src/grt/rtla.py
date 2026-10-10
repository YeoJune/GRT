from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import json
import numpy as np
import torch
from grt.checkpoint import evaluation_context
from grt.config import EXPERIMENT_VERSION
from grt.data import make_loader, move_batch

if TYPE_CHECKING:
    pass


@dataclass
class TraceBuffer:
    read_active: bool = False
    records: dict = field(
        default_factory=lambda: {
            key: []
            for key in (
                "r_gates",
                "w_gates",
                "s_norms",
                "update_distances",
                "attn_weights",
            )
        }
    )

    def record(self, read, write, state, candidate, attention):
        with torch.no_grad():
            values = {
                "r_gates": read.float().squeeze(-1).mean(0),
                "w_gates": write.float().squeeze(-1).mean(0),
                "s_norms": state.float().norm(dim=-1).mean(0),
                "update_distances": (candidate.float() - state.float())
                .norm(dim=-1)
                .mean(0),
                "attn_weights": attention.float().mean(0),
            }
            for key, value in values.items():
                self.records[key].append(value.detach().cpu())

    def to_numpy(self):
        if not self.records["w_gates"]:
            raise ValueError("Empty trace")
        return {
            key: torch.stack(values).numpy().astype(np.float32)
            for key, values in self.records.items()
        }

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if any(
            path.with_suffix(suffix).exists() for suffix in (".npz", ".json", ".png")
        ):
            raise FileExistsError(f"Trace already exists: {path}")
        meta = {
            **metadata,
            "read_active": self.read_active,
            "experiment_version": EXPERIMENT_VERSION,
        }
        np.savez_compressed(
            path, **self.to_numpy(), metadata=np.array(json.dumps(meta))
        )
        path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )


def analyze(model, batch, path, metadata):
    if not callable(getattr(model, "forward_with_trace", None)):
        raise ValueError("Trace analysis requires a model with forward_with_trace")
    device = next(model.parameters()).device
    batch = move_batch(batch, device)
    with evaluation_context(model):
        _, trace = model.forward_with_trace(
            **{
                k: batch[k]
                for k in ("input_ids", "attention_mask", "labels", "labels_mask")
            }
        )
    trace.save(
        path,
        {
            **metadata,
            "N": model.cfg.segment_len,
            "M": model.cfg.num_registers,
            "D": model.cfg.d_model,
            "trace_scope": "committed fact updates only; query prefix recomputation excluded",
        },
    )
    from grt.plots import plot_trace

    panel = Path(path).with_suffix(".png")
    plot_trace(trace.to_numpy(), panel, trace.read_active)
    return trace, panel


def log_trace(model, cfg, dataset, device, stage_dir, step):
    from torch.utils.data import Subset

    count = min(cfg.rtla.batch_size, len(dataset))
    path = stage_dir / "rtla" / f"step_{step:06d}.npz"
    if path.exists():
        return {}
    with evaluation_context(model):
        batch = next(iter(make_loader(cfg, Subset(dataset, range(count)), vary=False)))
        trace, _ = analyze(model, batch, path, {"step": step, "split": "validation"})
    arrays = trace.to_numpy()
    return {
        "grt/write_mean": float(arrays["w_gates"].mean()),
        "grt/register_norm": float(arrays["s_norms"].mean()),
        "grt/update_distance": float(arrays["update_distances"].mean()),
    }
