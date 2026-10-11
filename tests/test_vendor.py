import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from conftest import tiny_config
from grt import trainer as runner
from grt.vendor.armt.data import make_collator


def config(tmp_path):
    return tiny_config(tmp_path)


def test_pinned_author_sources():
    root = Path(runner.__file__).parent / "vendor/armt"
    manifest = json.loads((root / "SOURCE.json").read_text())
    for name, record in manifest["files"].items():
        assert (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            == record["vendored_sha256"]
        )
    # Byte hashes are portable; ast.dump formatting changed in Python 3.13.
    assert (
        hashlib.sha256((root / "data.py").read_bytes()).hexdigest()
        == manifest["data"]["vendored_sha256"]
    )


def test_current_loader_collates_one_effective_batch_before_accumulation(tmp_path):
    cfg = config(tmp_path)
    cfg.training.grad_accum_steps = 2
    data = runner.make_dataset(cfg, "train", 2)
    batch = next(iter(runner.make_loader(cfg, data, training=True)))
    assert len(batch["input_ids"]) == 8
    assert batch["input_ids"].shape == batch["labels_mask"].shape


def test_remember_collator_preserves_query_target_and_predictor_mask():
    rows = [
        {
            "keys": torch.tensor([[1], [12]]),
            "values": torch.tensor([[8], [7]]),
            "target_key_ind": torch.tensor(i),
        }
        for i in (0, 1)
    ]
    batch = make_collator(SimpleNamespace(vary_n_segments=False, value_size=1))(rows)
    assert batch["input_ids_generate"].tolist() == [
        [1, 100, 8, 102, 12, 100, 7, 102, 1, 101],
        [1, 100, 8, 102, 12, 100, 7, 102, 12, 101],
    ]
    assert torch.equal(batch["labels"], batch["input_ids"])
    assert batch["labels"][:, -2:].tolist() == [[8, 102], [7, 102]]
    assert batch["labels_mask"][:, :-1].nonzero().tolist() == [
        [0, 9],
        [0, 10],
        [1, 9],
        [1, 10],
    ]


def test_current_run_transfers_stage_weights_and_completed_resume_is_unchanged(
    tmp_path, monkeypatch
):
    cfg = config(tmp_path)
    original = runner.create_model
    loads = []

    def make_model(c):
        model = original(c)
        load = model.load_state_dict

        def record(state, *args, **kwargs):
            loads.append({k: v.clone() for k, v in state.items()})
            return load(state, *args, **kwargs)

        model.load_state_dict = record
        return model

    monkeypatch.setattr(runner, "create_model", make_model)
    rows = runner.run(cfg, torch.device("cpu"))
    assert len(rows) == 2 and all(r["step"] == 2 and r["completed"] for r in rows)
    first = torch.load(
        tmp_path / "stage_01_pairs1_key1" / "best.pt", weights_only=False
    )["model"]
    assert all(torch.equal(v, loads[0][k]) for k, v in first.items())
    checkpoint = tmp_path / "stage_02_pairs2_key1" / "last.pt"
    before = torch.load(checkpoint, weights_only=False)
    again = runner.run(cfg, torch.device("cpu"))
    after = torch.load(checkpoint, weights_only=False)
    assert all(torch.equal(v, after["model"][k]) for k, v in before["model"].items())
    assert [r["step"] for r in again] == [2, 2]
    assert json.loads((tmp_path / "evaluation.json").read_text())["num_pairs"] == 2
    for name in ("summary", "evaluation"):
        assert (tmp_path / f"{name}.txt").read_text() == (
            tmp_path / f"{name}.json"
        ).read_text()
    text = (tmp_path / "stage_02_pairs2_key1" / "metrics.txt").read_text()
    assert text.count("Step 1\n") == 1 and text.count("Step 2\n") == 1
    assert "Fixed-length validation:" in text and "peak VRAM=" in text


def test_incomplete_current_resume_replays_same_updates(tmp_path):
    cfg = config(tmp_path)
    cfg.training.curriculum = cfg.training.curriculum[:1]
    cfg.training.curriculum[0].max_steps = 4
    runner.run(cfg, torch.device("cpu"))
    directory = tmp_path / "stage_01_pairs1_key1"
    uninterrupted = torch.load(directory / "last.pt", weights_only=False)
    earlier = torch.load(directory / "best.pt", weights_only=False)
    assert earlier["step"] < 4
    torch.save(earlier, directory / "last.pt")
    runner.run(cfg, torch.device("cpu"))
    resumed = torch.load(directory / "last.pt", weights_only=False)
    assert all(
        torch.equal(v, resumed["model"][k]) for k, v in uninterrupted["model"].items()
    )


def test_pilot_budget_exhaustion_does_not_advance_unconverged_stage(
    tmp_path, monkeypatch
):
    cfg = config(tmp_path)
    cfg.training.stop_on_convergence = True
    monkeypatch.setattr(
        runner,
        "evaluate",
        lambda *args, **kwargs: {"loss": 1.0, "exact_match": 0.0, "samples": 4},
    )
    rows = runner.run(cfg, torch.device("cpu"))
    assert len(rows) == 1 and rows[0]["completed"] and not rows[0]["converged"]
    assert rows[0]["stop_reason"] == "update_budget"
    assert not (tmp_path / "stage_02_pairs2_key1").exists()
    assert (
        json.loads((tmp_path / "summary.json").read_text())["status"]
        == "stage_not_converged"
    )
    assert rows[0]["training_seconds"] > 0
    assert len(runner.run(cfg, torch.device("cpu"))) == 1
