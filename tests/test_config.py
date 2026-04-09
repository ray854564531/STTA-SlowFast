import pytest
import sys
sys.path.insert(0, '..')

from utils.config import load_config


def test_load_base_config(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("train:\n  lr: 0.1\n  max_epochs: 100\n")
    cfg = load_config(str(cfg_file))
    assert cfg.train.lr == 0.1
    assert cfg.train.max_epochs == 100


def test_config_inheritance(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("train:\n  lr: 0.1\n  max_epochs: 100\nmodel:\n  dropout: 0.5\n")

    child = tmp_path / "child.yaml"
    child.write_text(f"_base_: {base}\ntrain:\n  max_epochs: 150\n")

    cfg = load_config(str(child))
    assert cfg.train.lr == 0.1          # inherited from base
    assert cfg.train.max_epochs == 150  # overridden by child
    assert cfg.model.dropout == 0.5    # inherited from base


def test_nested_override(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("model:\n  enable_tcw: true\n  enable_tch: true\n  enable_thw: true\n")

    child = tmp_path / "child.yaml"
    child.write_text(f"_base_: {base}\nmodel:\n  enable_tch: false\n  enable_thw: false\n")

    cfg = load_config(str(child))
    assert cfg.model.enable_tcw is True
    assert cfg.model.enable_tch is False
    assert cfg.model.enable_thw is False
