from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import torch
from grt.checkpoint import evaluation_context
from grt.config import EXPERIMENT_VERSION
from grt.data import make_loader, move_batch

@dataclass
class TraceBuffer:
    read_active: bool = False
    per_example: bool = False
    records: dict = field(
        default_factory=lambda: {
            key: []
            for key in (
                "r_gates",
                "w_gates",
                "s_norms",
                "candidate_distances",
                "update_distances",
                "relative_updates",
                "slot_cosines",
                "write_pool_attention",
            )
        }
    )

    def record(self, read, write, state, candidate, attention):
        with torch.no_grad():
            state, candidate, write = state.float(), candidate.float(), write.float()
            norms = state.norm(dim=-1)
            delta = candidate - state
            update = (write * delta).norm(dim=-1)
            unit = torch.nn.functional.normalize(state, dim=-1)
            cosine = unit @ unit.transpose(-1, -2)
            slots = state.shape[1]
            off_diagonal = ~torch.eye(slots, dtype=torch.bool, device=state.device)
            values = {
                "r_gates": read.float().squeeze(-1).mean(0),
                "w_gates": write.float().squeeze(-1).mean(0),
                "s_norms": norms.mean(0),
                "candidate_distances": delta.norm(dim=-1).mean(0),
                "update_distances": update.mean(0),
                "relative_updates": (update / (norms+candidate.norm(dim=-1)).clamp_min(1e-8)).mean(0),
                "slot_cosines": cosine[:, off_diagonal].mean() if slots > 1 else state.new_tensor(0.),
                "write_pool_attention": attention.float().mean(0),
            }
            if self.per_example:
                values["w_gates_per_example"] = write.squeeze(-1)
            for key, value in values.items():
                self.records.setdefault(key, []).append(value.detach().cpu())

    def to_numpy(self):
        if not self.records["w_gates"]:
            raise ValueError("Empty trace")
        arrays = {}
        for key, values in self.records.items():
            if key == 'write_pool_attention':
                width = max(value.numel() for value in values)
                values = [torch.nn.functional.pad(value, (0, width-value.numel()), value=float('nan'))
                          for value in values]
            arrays[key] = torch.stack(values).numpy().astype(np.float32)
        return arrays

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
            "trace_scope": model.trace_scope,
        },
    )
    from grt.plots import plot_trace

    panel = Path(path).with_suffix(".png")
    plot_trace(trace.to_numpy(), panel, trace.read_active)
    return trace, panel


def log_trace(model, cfg, dataset, device, stage_dir, step):
    from torch.utils.data import Subset

    if cfg.model.name != 'grt':
        return {}
    count = min(cfg.rtla.batch_size, len(dataset))
    path = stage_dir / "rtla" / f"step_{step:06d}.npz"
    if path.exists():
        return {}
    with evaluation_context(model):
        batch = next(iter(make_loader(cfg, Subset(dataset, range(count)), vary=False)))
        trace, _ = analyze(model, batch, path, {"step": step, "split": "validation"})
    arrays = trace.to_numpy()
    return trace_metrics(arrays, trace.read_active)


def trace_metrics(arrays, read_active):
    metrics = {
        "grt/write_mean": float(arrays["w_gates"].mean()),
        "grt/register_norm": float(arrays["s_norms"].mean()),
        "grt/update_distance": float(arrays["update_distances"].mean()),
        "grt/candidate_distance": float(arrays["candidate_distances"].mean()),
        "grt/relative_update": float(arrays["relative_updates"].mean()),
        "grt/slot_cosine": float(arrays["slot_cosines"].mean()),
    }
    if read_active:
        metrics['grt/read_mean'] = float(arrays['r_gates'].mean())
    return metrics
