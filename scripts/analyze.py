"""Remember traces or paired WikiText GRT memory interventions."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grt.checkpoint import load_checkpoint, restore_model
from grt.data import make_dataset, make_loader, get_adapter
from grt.rtla import analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--split", choices=['validation', 'test'], default='test')
    p.add_argument("--segments", type=int)
    p.add_argument("--warmup-segments", type=int)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--slots", action='store_true', help='Also reset each memory slot separately')
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if a.batch_size is not None and a.batch_size <= 0:
        p.error("batch-size must be positive")
    state = load_checkpoint(a.checkpoint)
    model, cfg = restore_model(state, a.device)
    if a.batch_size is not None:
        cfg.rtla.batch_size = a.batch_size
    if cfg.model.name != 'grt':
        p.error('RTLA requires a GRT checkpoint')
    adapter = get_adapter(cfg.data.name)
    dataset = make_dataset(cfg, a.split, cfg.data.train_segments - 1)
    if callable(getattr(adapter, 'analyze_memory', None)):
        summary = adapter.analyze_memory(model, cfg, dataset, a.output_dir,
            dict(checkpoint=str(Path(a.checkpoint).resolve()), checkpoint_step=state['step'], split=a.split),
            segments=a.segments, warmup_segments=a.warmup_segments, horizon=a.horizon, slots=a.slots)
        print(summary)
        return
    cfg.evaluation.batch_size = cfg.rtla.batch_size
    batch = next(iter(make_loader(cfg, dataset, vary=False)))
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    analyze(
        model,
        batch,
        root / f"step_{state['step']:06d}.npz",
        {"step": state["step"], "checkpoint": str(Path(a.checkpoint).resolve())},
    )


if __name__ == "__main__":
    main()
