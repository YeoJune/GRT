from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import json
import numpy as np
import torch
from grt.checkpoint import evaluation_context
from grt.config import SPEC_VERSION
from grt.data import make_loader, move_batch
from grt.evaluator import autocast_context

if TYPE_CHECKING:
    from grt.models.grt import GRTModel

@dataclass
class TraceBuffer:
    read_active: bool = False
    records: dict = field(default_factory=lambda: {key: [] for key in
                       ("r_gates", "w_gates", "s_norms", "update_distances", "attn_weights")})

    def record(self, read, write, state, candidate, attention):
        with torch.no_grad():
            values = {"r_gates": read.float().squeeze(-1).mean(0),
                      "w_gates": write.float().squeeze(-1).mean(0),
                      "s_norms": state.float().norm(dim=-1).mean(0),
                      "update_distances": (candidate.float()-state.float()).norm(dim=-1).mean(0),
                      "attn_weights": attention.float().mean(0)}
            for key, value in values.items():
                self.records[key].append(value.detach().cpu())

    def to_numpy(self):
        if not self.records["w_gates"]:
            raise ValueError("Empty trace")
        return {key: torch.stack(values).numpy().astype(np.float32) for key, values in self.records.items()}

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if any(path.with_suffix(suffix).exists() for suffix in (".npz", ".json", ".png")):
            raise FileExistsError(f"Trace already exists: {path}")
        meta = {**metadata, "read_active": self.read_active, "spec_version": SPEC_VERSION,
                "legacy_pred_error": "Renamed update_distances; candidate is not a delta"}
        np.savez_compressed(path, **self.to_numpy(), metadata=np.array(json.dumps(meta)))
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def analyze(model: "GRTModel", batch, path, metadata, precision="fp32"):
    if not callable(getattr(model, "forward_with_trace", None)):
        raise ValueError("RTLA requires a GRT checkpoint")
    device = next(model.parameters()).device
    batch = move_batch(batch, device)
    with evaluation_context(model), autocast_context(device, precision):
        _, trace = model.forward_with_trace(batch["input_ids"], batch["attention_mask"])
    cfg = model.cfg
    trace.save(path, {**metadata, "N": cfg.segment_len, "M": cfg.num_registers,
                      "D": cfg.d_model, "T": batch["input_ids"].shape[1] // cfg.segment_len})
    from grt.plots import plot_trace
    panel = Path(path).with_suffix(".png")
    plot_trace(trace.to_numpy(), panel, trace.read_active)
    return trace, panel

def make_callback(cfg, logger=None):
    def on_evaluate(model, global_step):
        loader = make_loader(cfg.data, "validation", cfg.data.train_segments,
                             cfg.rtla.batch_size, cfg.rtla.batch_size)
        path = Path(cfg.run.output_dir) / "rtla" / f"step_{global_step:06d}_{cfg.data.task}_T{cfg.data.train_segments}.npz"
        trace, panel = analyze(model, next(iter(loader)), path,
                               {"sample_ids": list(range(cfg.rtla.batch_size)), "split": "validation",
                                "task": cfg.data.task, "global_step": global_step}, cfg.training.mixed_precision)
        arrays = trace.to_numpy()
        payload = {"rtla/mean_w": float(arrays["w_gates"].mean()),
                   "rtla/mean_r": float(arrays["r_gates"].mean())}
        # Batch media with the scalar payload in the logger's single step upload.
        if logger:
            logger.queue_trace(global_step, arrays, panel)
        return payload
    return on_evaluate
