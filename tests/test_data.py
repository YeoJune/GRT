import torch
import pytest
from grt.data import (
    RememberAdapter,
    register_adapter,
    get_adapter,
    make_dataset,
    make_loader,
)
from grt.config import validate_config, build_config, Config
from conftest import tiny_config


def test_dataset_is_identical_for_both_models_and_preserves_rng(tmp_path):
    a = tiny_config(tmp_path, "rmt")
    b = tiny_config(tmp_path, "grt")
    rng = torch.get_rng_state()
    left = make_dataset(a, "train", 2)
    right = make_dataset(b, "train", 2)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(left.keys, right.keys) and torch.equal(left.values, right.values)
    assert not torch.equal(left.keys, make_dataset(a, "validation", 2).keys)


def test_alternate_adapter_dispatches_without_model_branch(tmp_path):
    class Probe(RememberAdapter):
        def generation_kwargs(self, cfg):
            return {"max_new_tokens": 3, "eos_token_id": None}

    register_adapter("test_probe", Probe())
    cfg = tiny_config(tmp_path)
    cfg.data.name = "test_probe"
    validate_config(cfg)
    batch = next(iter(make_loader(cfg, make_dataset(cfg, "train", 2), vary=False)))
    assert batch["input_ids"].shape == (4, 12)
    assert get_adapter(cfg.data.name).generation_kwargs(cfg)["max_new_tokens"] == 3
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_adapter("missing")
    with pytest.raises(ValueError, match="Unknown keys"):
        build_config(Config, {"training": {"typo": 1}})
