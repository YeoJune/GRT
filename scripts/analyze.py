"""Save committed GRT fact-update traces from the common dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grt.checkpoint import load_checkpoint, restore_model
from grt.data import make_dataset, make_loader
from grt.rtla import analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if a.batch_size <= 0:
        p.error("batch-size must be positive")
    state = load_checkpoint(a.checkpoint)
    model, cfg = restore_model(state, a.device)
    dataset = make_dataset(cfg, "test", cfg.data.train_segments - 1)
    cfg.evaluation.batch_size = a.batch_size
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
