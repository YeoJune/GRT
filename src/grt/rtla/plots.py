from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def panel_a_rw_heatmap(ax, r_gates: np.ndarray, w_gates: np.ndarray) -> None:
    combined = np.concatenate([r_gates[None, ...], w_gates[None, ...]], axis=0)
    ax.imshow(combined.reshape(2 * combined.shape[1], combined.shape[2]), aspect="auto", interpolation="nearest")
    ax.set_title("Panel A - R/W Gates")
    ax.set_xlabel("Register Slot")
    ax.set_ylabel("Time / Gate")


def panel_b_lifecycle(ax, w_gates: np.ndarray, s_norms: np.ndarray) -> None:
    ax.plot(w_gates.mean(axis=1), label="mean W")
    ax.plot(s_norms.mean(axis=1), label="mean ||S||")
    ax.set_title("Panel B - Lifecycle")
    ax.set_xlabel("Timestep")
    ax.legend(loc="upper right")


def panel_c_token_map(ax, attn_weights: np.ndarray) -> None:
    ax.imshow(attn_weights, aspect="auto", interpolation="nearest")
    ax.set_title("Panel C - Attention Pooling")
    ax.set_xlabel("Token")
    ax.set_ylabel("Timestep")


def panel_d_pred_error(ax, pred_errors: np.ndarray) -> None:
    ax.imshow(pred_errors, aspect="auto", interpolation="nearest")
    ax.set_title("Panel D - Prediction Error")
    ax.set_xlabel("Register Slot")
    ax.set_ylabel("Timestep")


def make_figure(data: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    panel_a_rw_heatmap(axes[0, 0], data["r_gates"], data["w_gates"])
    panel_b_lifecycle(axes[0, 1], data["w_gates"], data["s_norms"])
    panel_c_token_map(axes[1, 0], data["attn_weights"])
    panel_d_pred_error(axes[1, 1], data["pred_errors"])
    fig.tight_layout()
    return fig
