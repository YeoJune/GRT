from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from grt.config import load_config
from grt.model.grt import GRTModel
from grt.rtla.analyzer import RegisterAnalyzer
from grt.training.trainer import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to run RTLA on; 'auto' picks cuda if available")
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    # determine device: allow user override
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = GRTModel(cfg.model).to(device).eval()
    load_checkpoint(args.checkpoint, model)
    analyzer = RegisterAnalyzer(model)

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    except Exception:
        tokenizer = None

    if tokenizer is None:
        input_ids = torch.randint(0, cfg.model.vocab_size, (1, cfg.model.segment_len * 4))
    else:
        encoded = tokenizer(args.input_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        remainder = input_ids.shape[1] % cfg.model.segment_len
        if remainder != 0:
            pad = cfg.model.segment_len - remainder
            pad_tensor = torch.full((1, pad), tokenizer.eos_token_id or 0)
            input_ids = torch.cat([input_ids, pad_tensor], dim=1)

    # move input_ids to device when calling model; we'll handle retries below
    input_ids_orig = input_ids

    # If using a tokenizer whose vocabulary is larger than the model embedding,
    # clamp token ids to the model vocab to avoid embedding index errors.
    try:
        max_id = int(input_ids_orig.max().item())
    except Exception:
        max_id = None
    model_vocab = cfg.model.vocab_size
    if max_id is not None and max_id >= model_vocab:
        print(
            f"Warning: tokenizer produced token id {max_id} >= model.vocab_size ({model_vocab}).\n"
            "Clamping token ids to the model vocabulary range to avoid IndexError.\n"
            "For accurate evaluation prefer using a tokenizer with matching vocab size or set `model.vocab_size` accordingly."
        )
        input_ids_orig = torch.clamp(input_ids_orig, 0, model_vocab - 1)

    # Try running trace on the selected device; if we hit a CUBLAS alloc error, retry on CPU.
    try:
        input_ids = input_ids_orig.to(device)
        trace = analyzer.run_trace(input_ids)
    except RuntimeError as e:
        msg = str(e).lower()
        if "cublas" in msg or "cublas_status_alloc_failed" in msg or "cublas_status_memory_error" in msg:
            print("CUDA CUBLAS allocation failed — retrying RTLA trace on CPU.")
            try:
                # clear CUDA cache and move model to CPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model.cpu()
                input_ids = input_ids_orig.cpu()
                trace = analyzer.run_trace(input_ids)
            except Exception as e2:
                print("Retry on CPU also failed:", e2)
                raise
        else:
            raise
    os.makedirs(cfg.rtla.output_dir, exist_ok=True)
    trace_path = os.path.join(cfg.rtla.output_dir, "trace_rtla.npz")
    analyzer.save_trace(trace, trace_path)
    analyzer.plot(trace_path, output_dir=cfg.rtla.output_dir)
    print(f"Saved trace to {trace_path}")


if __name__ == "__main__":
    main()
