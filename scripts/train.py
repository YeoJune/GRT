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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="One resolved Config YAML")
    source.add_argument("--task", help='Task preset from configs/experiments.yaml')
    parser.add_argument('--model', choices=['rmt', 'grt'])
    parser.add_argument('--size')
    parser.add_argument('--tokenization')
    parser.add_argument('--budget')
    parser.add_argument('--runtime')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.FIELD=VALUE')
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    args = parser.parse_args()
    if args.benchmark_steps < 0:
        parser.error("Invalid benchmark settings")
    from grt.experiments import load_experiment, parse_overrides
    overrides = parse_overrides(args.set)
    if args.config:
        if any(value is not None for value in [args.model, args.size, args.tokenization, args.budget, args.runtime, args.steps]) or args.set:
            parser.error('Use --task for named presets/--steps/--set, or --config for a resolved YAML')
        cfg = load_config(args.config, output_dir=args.output_dir)
    else:
        cfg = load_experiment(task=args.task, model=args.model or 'rmt', size=args.size,
            tokenization=args.tokenization, budget=args.budget, runtime=args.runtime,
            steps=args.steps, overrides=overrides, output_dir=args.output_dir)
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

            benchmark(cfg, ctx, args.benchmark_steps)
        else:
            run(cfg, ctx.device, ctx)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
