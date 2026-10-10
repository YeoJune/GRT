import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from grt.config import load_config
from grt import trainer as runner
from grt.vendor.armt.data import make_collator


def config(tmp_path):
    return load_config(
        ["configs/base.yaml", "configs/rmt.yaml", "configs/cpu.yaml"],
        output_dir=tmp_path,
    )


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


def test_native_loader_collates_one_effective_batch_before_accumulation(tmp_path):
    cfg = config(tmp_path)
    cfg.training.grad_accum_steps = 2
    data = runner.make_dataset(cfg, "train", 2)
    batch = next(iter(runner.make_loader(cfg, data, training=True)))
    assert len(batch["input_ids"]) == 8
    assert batch["input_ids"].shape == batch["labels_mask"].shape


def test_author_native_loss_memory_gradients_and_generation_match_fixture(tmp_path):
    cfg = config(tmp_path)
    cfg.data.train_segments = 4
    model = runner.create_model(cfg)
    fixture = torch.load("tests/fixtures/neox_rmt_reference.pt", weights_only=False)
    backbone = {
        n.removeprefix("backbone."): v
        for n, v in fixture["state"].items()
        if n.startswith("backbone.")
    }
    model.memory_cell.model.load_state_dict(backbone, strict=False)
    model.memory_cell.memory.data.copy_(fixture["state"]["mem0"])
    ids = fixture["input_ids"]
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, -3:] = True
    output = model(
        ids, labels=ids, labels_mask=mask, attention_mask=torch.ones_like(ids)
    )
    torch.testing.assert_close(output.logits, fixture["logits"])
    torch.testing.assert_close(output.loss, fixture["loss"])
    output.loss.backward()
    for name, p in model.memory_cell.model.named_parameters():
        torch.testing.assert_close(
            p.grad, fixture["grads"]["backbone." + name], rtol=2e-5, atol=1e-6
        )
    torch.testing.assert_close(
        model.memory_cell.memory.grad, fixture["grads"]["mem0"], rtol=2e-5, atol=1e-6
    )
    memory = None
    with torch.no_grad():
        for index, seg in enumerate(ids.split(4, 1)):
            _, memory = model.memory_cell(seg, memory_state=memory)
            torch.testing.assert_close(memory, fixture["memories"][index])
        generated = model.generate(
            ids[:, :-2],
            attention_mask=torch.ones_like(ids[:, :-2]),
            max_new_tokens=2,
            eos_token_id=None,
            pad_token_id=0,
        )
    assert torch.equal(generated, fixture["generated"])


def test_native_collator_preserves_query_target_and_predictor_mask():
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


def test_reference_run_transfers_stage_weights_and_completed_resume_is_unchanged(
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


def test_incomplete_native_resume_replays_same_updates(tmp_path):
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


def test_colab_native_preset_and_code_cells(tmp_path):
    cfg = load_config(
        ["configs/base.yaml", "configs/rmt.yaml", "configs/t4.yaml"],
        output_dir=tmp_path,
    )
    assert cfg.model.name == "rmt"
    assert cfg.data.name == "remember"
    assert cfg.training.batch_size * cfg.training.grad_accum_steps == 2048
    assert cfg.training.stop_on_convergence and cfg.training.max_seconds is None
    assert [s.num_pairs for s in cfg.training.curriculum] == [1, 2]
    assert [s.max_steps for s in cfg.training.curriculum] == [500, 2500]
    assert [s.max_steps * 2048 for s in cfg.training.curriculum] == [
        2000 * 512,
        10000 * 512,
    ]
    assert (
        cfg.data.train_samples,
        cfg.data.validation_samples,
        cfg.data.test_samples,
    ) == (1000000, 1000, 10000)
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    assert [p.name for p in notebooks] == ["colab.ipynb", "template.ipynb"]
    for path in notebooks:
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")
