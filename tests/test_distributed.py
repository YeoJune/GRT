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
    from grt.config import load_config, CurriculumStage

    cfg = load_config(
        ["configs/base.yaml", f"configs/{model}.yaml", "configs/cpu.yaml"]
    )
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
        cfg.model.position_capacity = 5
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
        if name.endswith("self_attn.in_proj_bias"):
            # A common key bias cancels in attention softmax. Its near-zero fp32
            # gradients can yield different Adam roundoff when batch kernels differ.
            query, key, value = weight.chunk(3)
            oq, ok, ov = other.chunk(3)
            torch.testing.assert_close(key, ok, rtol=0, atol=1e-4)
            torch.testing.assert_close(query, oq, rtol=1e-4, atol=2e-5)
            torch.testing.assert_close(value, ov, rtol=1e-4, atol=2e-5)
        else:
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


def test_kaggle_presets_keep_original_global_batch_and_budget(tmp_path):
    from grt.config import load_config, validate_config

    for model in ["rmt", "grt"]:
        paths = ["configs/base.yaml", f"configs/{model}.yaml", "configs/kaggle.yaml"]
        if model == "grt":
            paths.append("configs/kaggle_grt.yaml")
        cfg = validate_config(load_config(paths, output_dir=tmp_path))
        assert [(s.num_pairs, s.key_size) for s in cfg.training.curriculum] == [
            (1, 1),
            (2, 1),
            (3, 1),
            (5, 1),
            (10, 1),
            (20, 2),
            (40, 2),
        ]
        assert [s.max_steps for s in cfg.training.curriculum] == [2000] + [10000] * 6
        assert all(
            s.batch_size * s.grad_accum_steps * 2 == 512
            for s in cfg.training.curriculum
        )
        assert (
            cfg.training.global_batch_size == 512 and cfg.training.max_seconds == 23400
        )
        assert not cfg.wandb.enabled and not cfg.training.stop_on_convergence


def test_notebook_watchdog_terminates_child_group_and_accounts_time(tmp_path):
    import signal
    import time

    nb = json.loads((REPO / "notebooks/kaggle.ipynb").read_text())
    source = "".join(nb["cells"][2]["source"])
    budget = {"started_at": time.time(), "used": {}, "executions": [], "active": None}
    namespace = dict(
        os=os,
        time=time,
        subprocess=subprocess,
        signal=signal,
        json=json,
        BUDGET_FILE=tmp_path / "execution.json",
        budget=budget,
        TOTAL_SECONDS=3600,
        MODEL_SECONDS=3600,
        REPO_DIR=tmp_path,
    )
    exec(compile(source, "kaggle_watchdog", "exec"), namespace)
    # Reserve 15s for termination, leaving one second to run a real child process.
    command = [
        sys.executable,
        "-u",
        "-c",
        'import time; print("started", flush=True); time.sleep(30)',
    ]
    assert not namespace["run_bounded"](command, "rmt", tmp_path / "console.txt", 16)
    assert "started" in (tmp_path / "console.txt").read_text()
    recorded = json.loads((tmp_path / "execution.json").read_text())
    assert recorded["active"] is None
    assert recorded["executions"][0]["status"] == "time_budget"
    assert 1 <= recorded["used"]["rmt"] < 5


@pytest.mark.parametrize(
    "available,total,expected",
    [
        (
            {"rmt": 25000, "grt": 25000},
            17000,
            2,
        ),  # Joint budget limits an otherwise affordable third stage.
        (
            {"rmt": 25000, "grt": 5000},
            54000,
            1,
        ),  # Slower model determines the common prefix.
    ],
)
def test_measured_plan_respects_joint_and_individual_budgets(
    available, total, expected
):
    from grt.benchmark import plan_curriculum

    def report(costs):
        return {
            "stages": [
                {
                    "num_pairs": i,
                    "key_size": 1,
                    "max_steps": 10000,
                    "estimated_training_seconds": cost,
                    "estimated_validation_seconds": 0,
                }
                for i, cost in enumerate(costs, 1)
            ]
        }

    reports = {"rmt": report([1000, 1000, 6000]), "grt": report([2000, 2000, 6000])}
    plan = plan_curriculum(reports, available, total)
    assert plan["stage_count"] == expected
    assert plan["pairs"] == list(range(1, expected + 1))
    assert plan["estimated_seconds"]["grt"] == expected * 2000


def test_measured_plan_rejects_different_task_budgets():
    from grt.benchmark import plan_curriculum

    rows = [
        {
            "num_pairs": 1,
            "key_size": 1,
            "max_steps": 2000,
            "estimated_training_seconds": 1,
            "estimated_validation_seconds": 0,
        }
    ]
    with pytest.raises(ValueError, match="same curriculum"):
        plan_curriculum(
            {
                "rmt": {"stages": rows},
                "grt": {"stages": [{**rows[0], "max_steps": 500}]},
            },
            {"rmt": 25000, "grt": 25000},
            54000,
        )


def test_latest_notebook_orchestration_runs_actual_cli_on_cpu(tmp_path):
    """Execute preflight→shared plan→both trainers→plots, replacing only GPU/scale."""
    import signal
    import time
    import shutil
    from dataclasses import asdict
    from grt.config import load_config, validate_config

    repo = tmp_path / "checkout"
    (repo / "configs").mkdir(parents=True)
    (repo / "scripts").symlink_to(REPO / "scripts", target_is_directory=True)
    cfg = load_config(["configs/base.yaml", "configs/rmt.yaml", "configs/cpu.yaml"])
    cfg.data.train_samples = 12
    cfg.data.validation_samples = cfg.data.test_samples = 5
    (repo / "configs/base.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    for model in ["rmt", "grt"]:
        raw = yaml.safe_load((REPO / f"configs/{model}.yaml").read_text())
        if model == "grt":
            raw["model"]["router"]["mlp_hidden"] = 32
        (repo / f"configs/{model}.yaml").write_text(yaml.safe_dump(raw))
    shutil.copy(REPO / "configs/kaggle_grt.yaml", repo / "configs/kaggle_grt.yaml")
    (repo / "configs/kaggle.yaml").write_text(
        yaml.safe_dump(
            {
                "training": {
                    "batch_size": 2,
                    "grad_accum_steps": 2,
                    "global_batch_size": 8,
                    "max_seconds": 23400,
                    "stop_on_convergence": False,
                    "curriculum": [
                        {"num_pairs": 1, "key_size": 1, "max_steps": 2},
                        {"num_pairs": 20, "key_size": 2, "max_steps": 4},
                    ],
                },
                "evaluation": {
                    "batch_size": 2,
                    "every_steps": 1,
                    "pair_counts": [1, 20, 40],
                    "generalization_samples": 5,
                },
            }
        )
    )
    output = tmp_path / "output"
    output.mkdir()
    settings = output / "settings.yaml"
    settings.write_text("wandb: {enabled: false}\n")
    budget = {"started_at": time.time(), "used": {}, "executions": [], "active": None}
    namespace = dict(
        os=os,
        time=time,
        subprocess=subprocess,
        signal=signal,
        json=json,
        yaml=yaml,
        sys=sys,
        torch=torch,
        Path=Path,
        REPO_DIR=repo,
        RUN_DIR=output,
        BUDGET_FILE=output / "execution.json",
        budget=budget,
        SETTINGS=settings,
        TOTAL_SECONDS=54000,
        MODEL_SECONDS=25200,
        MODELS=["rmt", "grt"],
        load_config=load_config,
        validate_config=validate_config,
    )
    nb = json.loads((REPO / "notebooks/kaggle.ipynb").read_text())
    exec(compile("".join(nb["cells"][2]["source"]), "watchdog", "exec"), namespace)
    source = "".join(nb["cells"][4]["source"]).replace(
        "'--device', 'cuda'", "'--device', 'cpu'"
    )
    exec(compile(source, "notebook_cpu_run", "exec"), namespace)
    exec(
        compile("".join(nb["cells"][5]["source"]), "notebook_cpu_plot", "exec"),
        namespace,
    )
    plan = json.loads((output / "experiment_plan.json").read_text())
    assert plan["pairs"] == [1, 20]
    assert [r["status"] for r in budget["executions"]] == ["completed"] * 4
    for model in ["rmt", "grt"]:
        assert (output / model / "convergence.png").stat().st_size > 0
        summary = json.loads((output / model / "summary.json").read_text())
        assert [r["step"] for r in summary["stages"]] == [2, 4]
