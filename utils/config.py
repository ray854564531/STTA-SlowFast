"""YAML config loading with _base_ inheritance support."""
import yaml
from pathlib import Path
from typing import Any


class Config:
    """Simple namespace-style config object supporting dot access."""

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override wins on conflicts."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> Config:
    """Load YAML config, resolving _base_ inheritance."""
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    base_path = raw.pop('_base_', None)
    if base_path is not None:
        base_path = Path(base_path)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_cfg = load_config(str(base_path)).to_dict()
        raw = _deep_merge(base_cfg, raw)

    return Config(raw)
