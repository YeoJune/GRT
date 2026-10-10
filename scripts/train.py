"""Train either model through the same dataset and curriculum interfaces."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from grt.config import load_config
from grt.trainer import run
from grt.distributed import DistributedContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if args.benchmark_steps < 0 or not 0 < args.memory_fraction < 1:
        parser.error("Invalid benchmark settings")
    cfg = load_config(args.config, output_dir=args.output_dir)
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        cfg.training.grad_accum_steps = args.grad_accum_steps
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    ctx = DistributedContext(device, initialize=True)
    try:
        if args.benchmark_steps:
            from grt.benchmark import benchmark

            benchmark(cfg, ctx, args.benchmark_steps, args.memory_fraction)
        else:
            run(cfg, ctx.device, ctx)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
