from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from grt.config import GRTConfig, load_config
from grt.logging import RTLAUploader, WandbLogger
from grt.model.grt import GRTModel
from grt.rtla.analyzer import RegisterAnalyzer
from grt.training import Trainer, build_dataloaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRTModel(cfg.model).to(device)
    analyzer = RegisterAnalyzer(model)
    wandb_logger = WandbLogger(cfg.wandb, full_cfg=cfg)
    rtla_uploader = RTLAUploader(analyzer, cfg.rtla) if cfg.rtla.enabled else None

    train_loader, eval_loader = build_dataloaders(cfg, device=device)
    trainer = Trainer(
        model=model,
        cfg=cfg,
        dataloader=train_loader,
        eval_dataloader=eval_loader,
        wandb_logger=wandb_logger,
        rtla_uploader=rtla_uploader,
        analyzer=analyzer,
        device=device,
    )
    trainer.train(max_steps=cfg.training.max_steps)
    wandb_logger.finish()


if __name__ == "__main__":
    main()
