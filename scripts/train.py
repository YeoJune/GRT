"""Assemble model capabilities here; trainer stays model-independent."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from grt.config import load_config, validate_config, save_config, GRTConfig, build_config
from grt.checkpoint import seed_all, load_checkpoint, restore_model
from grt.models.factory import create_model
from grt.trainer import Trainer
from grt.evaluator import evaluate_lengths, check_precision
from grt.logger import Logger, metadata, write_json

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train GRT/RMT synthetic memory benchmarks")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", nargs="+")
    source.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--mixed-precision", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    state = None
    if args.resume:
        if any(v is not None for v in (args.output_dir, args.max_steps, args.batch_size, args.grad_accum_steps, args.mixed_precision)):
            parser.error("--resume restores the saved config; execution overrides are not allowed")
        state = load_checkpoint(args.resume)
        cfg = build_config(GRTConfig, state["config"])
        # Colab/Kaggle checkpoints may be restored in a new filesystem location.
        cfg.run.output_dir = str(Path(args.resume).resolve().parent)
    else:
        cfg = load_config(args.config, output_dir=args.output_dir)
        for name in ("max_steps", "batch_size", "grad_accum_steps", "mixed_precision"):
            value = getattr(args, name)
            if value is not None:
                setattr(cfg.training, name, value)
    validate_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    check_precision(device, cfg.training.mixed_precision)
    directory = Path(cfg.run.output_dir)
    if state is None and directory.exists() and any(directory.iterdir()):
        parser.error("Run directory is nonempty; use --resume or a new --output-dir")
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.training.model_seed)
    model = create_model(cfg.model).to(device)
    save_config(cfg, directory / "resolved_config.yaml")
    run_meta = metadata(model, cfg)
    if state is not None:
        run_meta["resumed_from"] = str(Path(args.resume).resolve())
        write_json(directory / f"resume_metadata_step_{state['global_step']:06d}.json", run_meta)
    else:
        write_json(directory / "metadata.json", run_meta)
    logger = Logger(directory, cfg)
    callback = None
    if cfg.model.name == "grt" and cfg.rtla.enabled:
        from grt.rtla import make_callback
        callback = make_callback(cfg, logger)
    trainer = Trainer(model, cfg, logger, callback)
    try:
        if state is not None:
            trainer.resume(state)
        progress = trainer.train()
        selected = "converged.pt" if progress["converged"] else "best.pt"
        best_state = load_checkpoint(directory / selected)
        best_model, _ = restore_model(best_state, device)
        rows = evaluate_lengths(best_model, cfg, progress)
        write_json(directory / "evaluation.json", {"checkpoint": selected, "checkpoint_step": best_state["global_step"], "results": rows})
        print(f"Completed {progress['global_step']} optimizer updates; results: {directory}")
    finally:
        logger.finish()

if __name__ == "__main__":
    main()
