"""Evaluate a current or native author_rmt/1 checkpoint on the common dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from grt.checkpoint import load_checkpoint, restore_model
from grt.config import validate_config
from grt.data import make_dataset
from grt.evaluator import evaluate
from grt.logger import write_result_files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument(
        "--samples", type=int, help="Explicit smaller test subset for a CPU check"
    )
    p.add_argument("--batch-size", type=int)
    a = p.parse_args()
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if a.device == "auto"
        else a.device
    )
    state = load_checkpoint(a.checkpoint)
    model, cfg = restore_model(state, device)
    if a.samples is not None:
        cfg.data.test_samples = a.samples
    if a.batch_size is not None:
        cfg.evaluation.batch_size = a.batch_size
    validate_config(cfg, require_output=False)
    pairs = cfg.data.train_segments - 1
    result = evaluate(model, cfg, make_dataset(cfg, "test", pairs), device, vary=False)
    root = Path(a.output_dir) if a.output_dir else Path(a.checkpoint).parent
    root.mkdir(parents=True, exist_ok=True)
    write_result_files(
        root,
        "evaluation",
        {
            "checkpoint": str(Path(a.checkpoint).resolve()),
            "checkpoint_step": state["step"],
            "num_pairs": pairs,
            **result,
        },
    )
    print(result)


if __name__ == "__main__":
    main()
