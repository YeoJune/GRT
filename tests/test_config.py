from dataclasses import asdict
from pathlib import Path
import pytest
import yaml
from grt.config import GRTConfig, build_config, load_config, validate_config, save_config

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("name", ["grt", "rmt"])
@pytest.mark.parametrize("task", ["copy", "reverse", "passkey"])
@pytest.mark.parametrize("size", ["small", "large"])
def test_all_standard_configs(tmp_path, name, task, size):
    paths = [ROOT / "configs" / f"{p}.yaml" for p in ("base", name, task)]
    if size == "large":
        paths.append(ROOT / "configs/large.yaml")
    cfg = validate_config(load_config(paths, output_dir=tmp_path))
    save_config(cfg, tmp_path / "resolved.yaml")
    assert asdict(load_config(tmp_path / "resolved.yaml")) == asdict(cfg)

@pytest.mark.parametrize("raw", [
    {"training": {"batch_size": True}}, {"training": {"max_steps": "1"}},
    {"model": {"typo": 5}}, {"_base_": "base.yaml"},
    {"data": {"eval_segments": [4, "10"]}}, {"rtla": {"enabled": "false"}},
    {"training": {"lr": float("nan")}}, {"model": []},
])
def test_invalid_types_and_keys(raw):
    with pytest.raises(ValueError):
        build_config(GRTConfig, raw)

def test_task_and_model_validation(tmp_path):
    cfg = load_config([ROOT / "configs/base.yaml", ROOT / "configs/rmt.yaml", ROOT / "configs/copy.yaml"], output_dir=tmp_path)
    from grt.config import RegisterConfig
    cfg.model.register = RegisterConfig()
    with pytest.raises(ValueError, match="RMT"):
        validate_config(cfg)
    cfg.model.register = None
    cfg.data.num_facts = 4
    with pytest.raises(ValueError, match="num_facts"):
        validate_config(cfg)
    cfg.data.num_facts = None
    cfg.model.segment_len = 64
    with pytest.raises(ValueError, match="standard benchmark"):
        validate_config(cfg)

def test_recursive_merge_replaces_lists(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({"data": {"eval_segments": [4,10,20]}, "training": {"lr": 0.001}}))
    b.write_text(yaml.safe_dump({"data": {"eval_segments": [5,15,30]}, "training": {"batch_size": 2}}))
    cfg = load_config([a,b])
    assert cfg.data.eval_segments == [5,15,30]
    assert cfg.training.lr == 0.001 and cfg.training.batch_size == 2
