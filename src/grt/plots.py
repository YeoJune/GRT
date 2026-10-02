from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_trace(arrays, path, read_active=False):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    heatmaps = [(axes[0,0], "r_gates", "A: Read gates" + (" (inactive read head)" if not read_active else ""), "Register"),
                (axes[0,1], "w_gates", "A: Write gates", "Register"),
                (axes[2,0], "attn_weights", "C: Pooling attention", "Token position"),
                (axes[2,1], "update_distances", "D: Candidate update distance", "Register")]
    for ax, key, title, xlabel in heatmaps:
        image = ax.imshow(arrays[key], aspect="auto", origin="lower")
        ax.set(title=title, xlabel=xlabel, ylabel="Segment timestep")
        fig.colorbar(image, ax=ax)
    time = np.arange(arrays["w_gates"].shape[0])
    slots = arrays["w_gates"].shape[1]
    for m in range(slots):
        for ax, key in ((axes[1,0], "w_gates"), (axes[1,1], "s_norms")):
            ax.plot(time, arrays[key][:,m], label=f"Slot {m}")
    axes[1,0].set(title="B: Write over time", xlabel="Segment timestep", ylabel="Write gate")
    axes[1,1].set(title="B: State norm before update", xlabel="Segment timestep", ylabel="L2 state norm")
    axes[1,1].legend(ncol=4, fontsize=6, loc="upper left")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
