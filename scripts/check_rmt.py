"""Bounded CPU convergence on original random data; not a generalization benchmark."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from grt.config import load_config
from grt.models.factory import create_model as make_model
from grt.data import forward_batch, make_dataset, make_loader
from grt.vendor.armt.data import make_collator
from grt.logger import metadata, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--size", choices=["reference", "small"], default="small")
    parser.add_argument("--fixture", choices=["random", "contrast"], default="random")
    parser.add_argument("--output-dir", default="runs/author_cpu_convergence")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Use a new output directory")
    torch.set_num_threads(2)
    torch.manual_seed(54)
    base = Path(__file__).resolve().parents[1]
    paths = [base / "configs/base.yaml", base / "configs/rmt.yaml"]
    if args.size == "small":
        paths.append(base / "configs/cpu.yaml")
    cfg = load_config(paths, output_dir=output)
    cfg.data.train_segments = 3
    model = make_model(cfg)
    rows = []
    loader = None
    iterator = None
    if args.fixture == "random":
        cfg.data.train_samples = 32
        cfg.training.batch_size = 8
        data = make_dataset(cfg, "train", 2)
        for i in range(len(data)):
            for query in (0, 1):
                row = dict(data[i])
                row["target_key_ind"] = torch.tensor(query)
                rows.append(row)
        loader = make_loader(cfg, data, training=True)
        iterator = iter(loader)
    else:
        for swap in (False, True):
            for reverse in (False, True):
                keys = torch.tensor([[1], [12]])
                values = torch.tensor([[7], [8]]) if swap else torch.tensor([[8], [7]])
                if reverse:
                    keys = keys.flip(0)
                    values = values.flip(0)
                for target in (0, 1):
                    rows.append(
                        {
                            "keys": keys,
                            "values": values,
                            "target_key_ind": torch.tensor(target),
                        }
                    )
    batch = make_collator(SimpleNamespace(vary_n_segments=False, value_size=1))(rows)
    provenance = metadata(model, cfg)
    provenance.update(
        fixture=args.fixture, closed_training_set=True, evaluated_queries=len(rows)
    )
    write_json(output / "metadata.json", provenance)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
    history = []
    started = time.perf_counter()
    converged = False
    for step in range(args.max_steps + 1):
        if step % 100 == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                loss = forward_batch(model, batch, "cpu").loss.item()
                generated = model.generate(
                    batch["input_ids_generate"],
                    attention_mask=batch["attention_mask_generate"],
                    max_new_tokens=2,
                    eos_token_id=102,
                    pad_token_id=0,
                )
                exact = (
                    generated[:, -2:]
                    .eq(batch["labels"][:, -2:])
                    .all(1)
                    .float()
                    .mean()
                    .item()
                    if generated.shape[1] >= 2
                    else 0.0
                )
            row = {
                "step": step,
                "loss": loss,
                "fixture_generated_exact_match": exact,
                "seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            converged = exact == 1.0 and loss < 0.1
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "size": args.size,
                        "fixture": args.fixture,
                        "examples": len(rows),
                        "closed_fixture": True,
                        "converged": converged,
                        "history": history,
                    },
                    indent=2,
                )
            )
            if converged:
                break
        if step == args.max_steps:
            break
        training_batch = batch
        if iterator is not None:
            try:
                training_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                training_batch = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        forward_batch(model, training_batch, "cpu").loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
