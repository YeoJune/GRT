from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trace(arrays, path, read_active=False):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    heatmaps = [
        (
            axes[0, 0],
            "r_gates",
            "A: Read gates" + (" (inactive read head)" if not read_active else ""),
            "Register",
        ),
        (axes[0, 1], "w_gates", "A: Write gates", "Register"),
        (axes[2, 0], "write_pool_attention", "C: Write pooling attention", "Token position"),
        (axes[2, 1], "update_distances", "D: Actual state update norm", "Register"),
    ]
    for ax, key, title, xlabel in heatmaps:
        if key == 'r_gates' and not read_active:
            ax.text(0.5, 0.5, 'Read gate off\nReading is controlled by token attention',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        image = ax.imshow(arrays[key], aspect="auto", origin="lower")
        ax.set(title=title, xlabel=xlabel, ylabel="Segment timestep")
        fig.colorbar(image, ax=ax)
    time = np.arange(arrays["w_gates"].shape[0])
    slots = arrays["w_gates"].shape[1]
    for m in range(slots):
        for ax, key in ((axes[1, 0], "w_gates"), (axes[1, 1], "s_norms")):
            ax.plot(time, arrays[key][:, m], label=f"Slot {m}")
    axes[1, 0].set(
        title="B: Write over time", xlabel="Segment timestep", ylabel="Write gate"
    )
    axes[1, 1].set(
        title="B: State norm before update",
        xlabel="Segment timestep",
        ylabel="L2 state norm",
    )
    axes[1, 1].legend(ncol=4, fontsize=6, loc="upper left")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_memory_analysis(summary, arrays, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    counts = arrays['target_counts']
    for name in ('reset', 'swap'):
        delta = arrays[name+'_delta_nll']
        positions = np.divide(np.nansum(delta*counts, axis=0), counts.sum(0),
            out=np.full(counts.shape[1], np.nan), where=counts.sum(0)>0)
        segments = np.nansum(delta*counts, axis=1)/counts.sum(1)
        axes[0, 0].plot(np.arange(1, len(positions)+1), positions, label=name)
        axes[0, 1].plot(np.arange(len(segments)), segments, label=name)
    axes[0, 0].set(title='Memory effect by token position', xlabel='Position in segment')
    axes[0, 1].set(title='Memory effect by segment', xlabel='Measured segment')
    if summary['slot_delta_nll']:
        axes[1, 0].bar(np.arange(summary['num_registers']), summary['slot_delta_nll'])
    else:
        axes[1, 0].text(.5, .5, 'Slot analysis not requested', ha='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set(title='Reset one slot', xlabel='Slot')
    if summary['horizon']:
        values = [np.nan if x is None else x for x in summary['retention_delta_nll']]
        axes[1, 1].plot(np.arange(len(values)), values, marker='o')
    axes[1, 1].set(title='One state exchange: subsequent effect', xlabel='Segments after intervention (0=current)')
    for ax in axes.flat:
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_ylabel('Intervention NLL - normal NLL (nats/token)')
        ax.grid(alpha=.3)
    axes[0, 0].legend()
    axes[0, 1].legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_register_diagnostics(analysis, path):
    """A compact view; the JSON retains all masks, lags and per-example effects."""
    from grt.register_analysis import GROUP_NAMES
    windows = analysis['windows']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = [str(w['start_segment']) for w in windows]
    counts = np.array([[w['stability']['threshold_sensitivity']['0.01']['counts'][n]
                        for n in GROUP_NAMES] for w in windows])
    bottom = np.zeros(len(windows))
    for i, name in enumerate(GROUP_NAMES):
        axes[0, 0].bar(names, counts[:, i], bottom=bottom, label=name)
        bottom += counts[:, i]
    axes[0, 0].set(title='Gate types by window (threshold .01)', xlabel='Start segment', ylabel='Slots')
    axes[0, 0].legend()
    for w in windows:
        z = np.asarray(w['diagnostics']['gate_logits']['raw']).ravel()
        axes[0, 1].hist(z, bins=30, alpha=.4, label=str(w['start_segment']))
        spectrum_ = w['diagnostics']['representations']['context_centered']['mean_normalized_squared_singular_values']
        axes[1, 0].plot(np.cumsum(spectrum_), label=str(w['start_segment']))
    axes[0, 1].set(title='Write logits (before sigmoid)', xlabel='Logit')
    axes[0, 1].legend(title='Start')
    axes[1, 0].set(title='Context-centered state energy', xlabel='SVD dimension', ylabel='Cumulative energy')
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].legend(title='Start')
    for w in windows[1:]:
        for mode in ('reset', 'swap'):
            x, y = [0.], [0.]
            for size in sorted({p['size'] for p in analysis['intervention_plan'] if p['kind'] == 'random_fraction'}):
                selected = [p for p in analysis['intervention_plan'] if p['kind'] == 'random_fraction' and p['size'] == size]
                values = [w['interventions']['effects'][p['mask']][mode]['delta_nll_by_lag'][0] for p in selected]
                x.append(size/analysis['num_registers'])
                y.append(np.mean(values))
            if x[-1] < 1:
                key = analysis['intervention_plan'][0]['mask']
                x.append(1.)
                y.append(w['interventions']['effects'][key][mode]['delta_nll_by_lag'][0])
            axes[1, 1].plot(x, y, marker='o' if mode == 'reset' else 'x',
                            label=f'window {w["start_segment"]} {mode}')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set(title='Held-out random-group damage',
                   xlabel='Fraction of slots altered', ylabel='Current-segment delta NLL')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
