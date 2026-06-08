from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from grt.config import load_config
from grt.model.grt import GRTModel
from grt.training import Trainer, build_dataloaders, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRTModel(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model)

    _, eval_loader = build_dataloaders(cfg, device=device)
    trainer = Trainer(model=model, cfg=cfg, eval_dataloader=eval_loader, device=device)
    metrics = trainer.evaluate()
    print(f"eval/loss={metrics['loss']:.4f} eval/ppl={metrics['ppl']:.4f} tokens/sec={metrics['tokens_per_sec']:.2f}")


if __name__ == "__main__":
    main()
