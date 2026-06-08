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
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        input_ids = torch.randint(0, cfg.model.vocab_size, (1, cfg.model.segment_len * 4), device=device)
    else:
        encoded = tokenizer(args.input_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        remainder = input_ids.shape[1] % cfg.model.segment_len
        if remainder != 0:
            pad = cfg.model.segment_len - remainder
            pad_tensor = torch.full((1, pad), tokenizer.eos_token_id or 0, device=device)
            input_ids = torch.cat([input_ids, pad_tensor], dim=1)

    trace = analyzer.run_trace(input_ids)
    os.makedirs(cfg.rtla.output_dir, exist_ok=True)
    trace_path = os.path.join(cfg.rtla.output_dir, "trace_rtla.npz")
    analyzer.save_trace(trace, trace_path)
    analyzer.plot(trace_path, output_dir=cfg.rtla.output_dir)
    print(f"Saved trace to {trace_path}")


if __name__ == "__main__":
    main()
