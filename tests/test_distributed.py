"""Exercise the production CLI with two CPU workers, including state replay."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]


def invoke(config, output, workers=2, benchmark=False):
    command = [sys.executable]
    if workers > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={workers}",
        ]
    command += [
        "scripts/train.py",
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--device",
        "cpu",
    ]
    if benchmark:
        command += ["--benchmark-steps", "1"]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HOME": "/tmp/grt-test-hf",
        "MPLCONFIGDIR": "/tmp/grt-test-mpl",
    }
    result = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout


def configuration(tmp_path, model, workers=2):
    from dataclasses import asdict
    from grt.config import CurriculumStage
    from conftest import tiny_config

    cfg = tiny_config(tmp_path, model)
    cfg.data.train_samples = 12  # Global batches 8 then 4: verify partial accumulation.
    cfg.data.validation_samples = cfg.data.test_samples = 5  # Unequal rank tails.
    cfg.training.batch_size = 2 if workers == 2 else 8
    cfg.training.grad_accum_steps = 2 if workers == 2 else 1
    cfg.training.global_batch_size = 8
    cfg.training.curriculum = [CurriculumStage(1, 1, 2), CurriculumStage(20, 2, 4)]
    cfg.evaluation.batch_size = 2 if workers == 2 else 4
    cfg.evaluation.pair_counts = [1, 20, 40]
    cfg.evaluation.generalization_samples = 5
    if model == "grt":
        cfg.model.router.mlp_hidden = 32
    path = tmp_path / f"{model}_{workers}.yaml"
    path.write_text(yaml.safe_dump(asdict(cfg)))
    return path


@pytest.mark.parametrize("model", ["rmt", "grt"])
def test_cli_distributed_matches_global_batch_and_replays_checkpoint(tmp_path, model):
    config = configuration(tmp_path, model)
    output = tmp_path / "distributed"
    invoke(config, output)
    summary = json.loads((output / "summary.json").read_text())
    assert [s["step"] for s in summary["stages"]] == [2, 4]
    assert [s["train_samples_seen"] for s in summary["stages"]] == [12, 24]
    assert json.loads((output / "evaluation.json").read_text())["samples"] == 5
    lengths = json.loads((output / "length_evaluation.json").read_text())
    assert [r["samples"] for r in lengths["results"]] == [5, 5, 5]
    stage = output / "stage_02_pairs20_key2"
    final = torch.load(stage / "last.pt", weights_only=False)
    assert final["world_size"] == 2 and len(final["rng_by_rank"]) == 2
    assert not any(n.startswith("module.") for n in final["model"])
    serial = tmp_path / "serial"
    invoke(configuration(tmp_path, model, workers=1), serial, workers=1)
    serial_final = torch.load(
        serial / "stage_02_pairs20_key2/last.pt", weights_only=False
    )
    for name, weight in final["model"].items():
        other = serial_final["model"][name]
        torch.testing.assert_close(weight, other, rtol=1e-4, atol=2e-5)
    distributed_loss = json.loads((output / "evaluation.json").read_text())["loss"]
    serial_loss = json.loads((serial / "evaluation.json").read_text())["loss"]
    assert abs(distributed_loss - serial_loss) < 2e-6
    # The earliest best (zero EM at this tiny budget) is a genuine distributed checkpoint.
    earlier = torch.load(stage / "best.pt", weights_only=False)
    assert earlier["step"] < final["step"]
    torch.save(earlier, stage / "last.pt")
    invoke(config, output)
    replayed = torch.load(stage / "last.pt", weights_only=False)
    for name, weight in final["model"].items():
        assert torch.equal(weight, replayed["model"][name]), name
    assert final["scheduler"] == replayed["scheduler"]
    for original, replay in zip(final["rng_by_rank"], replayed["rng_by_rank"]):
        assert torch.equal(original["torch"], replay["torch"])
    assert json.loads((output / "metadata.json").read_text())["global_batch_size"] == 8
    assert (output / "summary.txt").read_text() == (output / "summary.json").read_text()


@pytest.mark.parametrize("model", ["rmt", "grt"])
def test_cli_distributed_preflight_preserves_batch(tmp_path, model):
    config = configuration(tmp_path, model)
    output = tmp_path / "preflight"
    invoke(config, output, benchmark=True)
    report = json.loads((output / "benchmark.json").read_text())
    assert [r["num_pairs"] for r in report["stages"]] == [1, 20]
    for row in report["stages"]:
        assert (
            row["global_batch_size"]
            == row["microbatch_per_gpu"] * row["accumulation"] * 2
            == 8
        )
        assert row["maximum_step_seconds"] > 0
    assert report["calibrated_override"]["training"]["curriculum"][1]["key_size"] == 2


def test_cli_distributed_time_budget_synchronizes_stop(tmp_path):
    config = configuration(tmp_path, "rmt")
    raw = yaml.safe_load(config.read_text())
    raw["training"]["max_seconds"] = 0.001
    config.write_text(yaml.safe_dump(raw))
    output = tmp_path / "capped"
    invoke(config, output)
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["stages"]) == 1 and summary["stages"][0]["step"] == 0
    assert summary["stages"][0]["stop_reason"] == "time_budget"
    lengths = json.loads((output / "length_evaluation.json").read_text())
    assert [r["status"] for r in lengths["results"]] == [
        "evaluated",
        "incompatible",
        "incompatible",
    ]
