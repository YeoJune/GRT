import sys
from types import SimpleNamespace
import pytest
import torch
from grt import trainer
from grt.logger import Monitor
from conftest import tiny_config


class FakeRun:
    def __init__(self):
        self.logs = []
        self.summary = {}
        self.finished = False
        self.definitions = []

    def define_metric(self, name, **settings):
        torch.rand(3)
        self.definitions.append((name, settings))

    def log(self, payload, step):
        torch.rand(5)
        self.logs.append((step, payload))

    def finish(self, *, exit_code=0):
        torch.rand(5)
        self.finished = True
        self.exit_code = exit_code


def test_disabled_monitor_does_not_import_or_connect(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    cfg = tiny_config(tmp_path)
    monitor = Monitor(tmp_path, cfg)
    monitor.log(1, {"train/loss": 1.0})
    monitor.finish()
    assert monitor.run is None and not (tmp_path / "wandb_run.json").exists()


def test_enabled_monitor_preserves_training_rng_and_resumes_id(tmp_path, monkeypatch):
    calls = []
    runs = []

    def init(**kwargs):
        torch.rand(4)
        calls.append(kwargs)
        run = FakeRun()
        runs.append(run)
        return run

    fake = SimpleNamespace(init=init)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    a = tiny_config(tmp_path / "plain")
    b = tiny_config(tmp_path / "logged")
    b.wandb.enabled = True
    b.wandb.every_steps = 1
    trainer.run(a, torch.device("cpu"))
    trainer.run(b, torch.device("cpu"))
    for stage in ("stage_01_pairs1_key1", "stage_02_pairs2_key1"):
        left = torch.load(tmp_path / "plain" / stage / "last.pt", weights_only=False)
        right = torch.load(tmp_path / "logged" / stage / "last.pt", weights_only=False)
        assert all(torch.equal(v, right["model"][k]) for k, v in left["model"].items())
        assert torch.equal(left["rng_by_rank"][0]["torch"], right["rng_by_rank"][0]["torch"])
    assert [s for s, p in runs[0].logs] == [1, 2, 3, 4]
    assert [p['progress/num_pairs'] for s, p in runs[0].logs] == [1, 1, 2, 2]
    assert 'result/test_exact_match' in runs[0].summary and runs[0].finished
    trainer.run(b, torch.device("cpu"))
    assert calls[0]["id"] == calls[1]["id"]
    assert len(calls[0]["id"]) == 8
    assert calls[0]["resume"] == "allow"


def test_monitor_failure_leaves_readable_local_warning(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("offline test failure")

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=fail, util=SimpleNamespace(generate_id=lambda: "id")),
    )
    cfg = tiny_config(tmp_path)
    cfg.wandb.enabled = True
    with pytest.warns(RuntimeWarning, match="initialization failed"):
        monitor = Monitor(tmp_path, cfg)
    assert (
        monitor.run is None
        and "offline test failure" in (tmp_path / "warnings.txt").read_text()
    )
    import json
    status = json.loads((tmp_path / 'wandb_status.json').read_text())
    assert status['state'] == 'initialization_failed' and status['failures'] == 1


def test_training_exception_finishes_wandb_as_failed(tmp_path, monkeypatch):
    import json
    run = FakeRun()
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kwargs: run))
    cfg = tiny_config(tmp_path)
    cfg.wandb.enabled = True
    def crash(*args):
        raise RuntimeError('training crashed')
    monkeypatch.setattr(trainer, '_run', crash)
    with pytest.raises(RuntimeError, match='training crashed'):
        trainer.run(cfg, torch.device('cpu'))
    assert run.finished and run.exit_code == 1
    assert json.loads((tmp_path / 'wandb_status.json').read_text())['state'] == 'failed'
