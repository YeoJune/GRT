import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from grt.checkpoint import load_checkpoint, restore_model
from grt.config import validate_config
from grt.evaluator import evaluate_lengths
from grt.logger import write_json

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate all checkpoint-configured test lengths")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    state = load_checkpoint(args.checkpoint)
    model, cfg = restore_model(state, device)
    validate_config(cfg)
    progress = state
    last_path = Path(args.checkpoint).parent / "last.pt"
    if last_path.exists():
        last = load_checkpoint(last_path)
        saved_config = {k: v for k, v in state["config"].items() if k != "run"}
        last_config = {k: v for k, v in last["config"].items() if k != "run"}
        if saved_config == last_config and last["global_step"] >= state["global_step"]:
            progress = last
    results = evaluate_lengths(model, cfg, progress)
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"checkpoint": str(Path(args.checkpoint).resolve()),
                        "checkpoint_step": state["global_step"], "results": results})
    print(f"Evaluation saved: {output}")

if __name__ == "__main__":
    main()
