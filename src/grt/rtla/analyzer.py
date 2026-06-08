from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from grt.model.grt import GRTModel, TraceBuffer
from grt.rtla.plots import make_figure


class RegisterAnalyzer:
    def __init__(self, model: GRTModel) -> None:
        self.model = model
        self._hook_handles: list = []

    def attach_hooks(self, model: GRTModel | None = None) -> None:
        target = model or self.model
        self._hook_handles.clear()

        def _noop_hook(*args, **kwargs):
            return None

        self._hook_handles.append(target.router.mlp_r.register_forward_hook(lambda *args: None))
        self._hook_handles.append(target.router.mlp_w.register_forward_hook(lambda *args: None))
        self._hook_handles.append(target.router.pool_attn.register_forward_hook(lambda *args: None))

    def run_trace(self, input_ids: Tensor) -> TraceBuffer:
        with torch.no_grad():
            out = self.model(input_ids, return_trace=True)
        if out.trace is None:
            raise RuntimeError("RegisterAnalyzer: trace was not produced")
        return out.trace

    def save_trace(self, trace: TraceBuffer, path: str) -> None:
        trace.save(path, batch_idx=0)

    def plot(self, trace_path: str, output_dir: str = ".") -> plt.Figure:
        data = np.load(trace_path)
        fig = make_figure(data)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "rtla_panels.png"), dpi=150, bbox_inches="tight")
        return fig
