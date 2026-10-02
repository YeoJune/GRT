import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from grt.checkpoint import load_checkpoint, restore_model
from grt.config import validate_config
from grt.data import make_loader
from grt.rtla import analyze

def main(argv=None):
    parser = argparse.ArgumentParser(description="Save GRT RTLA batch-mean NPZ/PNG")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--segments", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    state = load_checkpoint(args.checkpoint)
    if state["config"]["model"]["name"] != "grt":
        parser.error("RTLA is supported only for GRT checkpoints")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, cfg = restore_model(state, device)
    validate_config(cfg)
    segments = args.segments if args.segments is not None else cfg.data.train_segments
    batch_size = args.batch_size if args.batch_size is not None else cfg.rtla.batch_size
    if segments not in cfg.data.eval_segments or batch_size <= 0 or args.start_id < 0:
        parser.error("Use configured eval lengths, positive batch size and nonnegative start-id")
    sample_count = cfg.data.test_samples if args.split == "test" else cfg.data.validation_samples
    if args.start_id + batch_size > sample_count:
        parser.error("Probe exceeds the configured split sample count")
    loader = make_loader(cfg.data, args.split, segments, batch_size, batch_size, args.start_id)
    root = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "rtla"
    path = root / f"step_{state['global_step']:06d}_{cfg.data.task}_T{segments}_{args.split}_id{args.start_id}_{Path(args.checkpoint).stem}.npz"
    analyze(model, next(iter(loader)), path,
            {"checkpoint": str(Path(args.checkpoint).resolve()), "global_step": state["global_step"],
             "task": cfg.data.task, "split": args.split, "sample_ids": list(range(args.start_id, args.start_id+batch_size))},
            cfg.training.mixed_precision)
    print(f"Trace saved: {path}")

if __name__ == "__main__":
    main()
