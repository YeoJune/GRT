from __future__ import annotations

import matplotlib.pyplot as plt
import wandb

from grt.config import RTLAConfig
from grt.model.grt import TraceBuffer
from grt.rtla.analyzer import RegisterAnalyzer
from grt.rtla.plots import make_figure


class RTLAUploader:
    def __init__(self, analyzer: RegisterAnalyzer, cfg: RTLAConfig) -> None:
        self.analyzer = analyzer
        self.cfg = cfg

    def upload(self, trace: TraceBuffer, step: int, trace_path: str) -> None:
        arrays = trace.to_numpy(batch_idx=0)
        stats = trace.gate_stats(batch_idx=0)
        log_dict: dict[str, object] = {}

        if self.cfg.upload_panels:
            fig = make_figure(arrays)
            log_dict["rtla/panels"] = wandb.Image(fig, caption=f"RTLA panels @ step {step}")
            plt.close(fig)

        if self.cfg.upload_table:
            table = wandb.Table(columns=["slot_id", "mean_w", "mean_r", "mean_norm", "mean_pred_err", "tier"])
            for idx in range(len(stats["slot_id"])):
                table.add_data(
                    int(stats["slot_id"][idx]),
                    float(stats["mean_w"][idx]),
                    float(stats["mean_r"][idx]),
                    float(stats["mean_norm"][idx]),
                    float(stats["mean_pred_err"][idx]),
                    str(stats["tier"][idx]),
                )
            log_dict["rtla/gate_stats"] = table

        w_gates = arrays["w_gates"]
        log_dict.update(
            {
                "rtla/mean_w": float(w_gates.mean()),
                "rtla/mean_r": float(arrays["r_gates"].mean()),
                "rtla/w_std_across_slots": float(w_gates.mean(axis=0).std()),
                "rtla/dead_register_count": int((w_gates.mean(axis=0) < 0.05).sum()),
            }
        )

        wandb.log(log_dict, step=step)
