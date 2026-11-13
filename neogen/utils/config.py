"""Load and access NeoGen configuration."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


# Project root = two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | None = None) -> dict:
    """Load YAML config and resolve environment variables."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve any ${ENV_VAR} placeholders
    _resolve_env_vars(config)
    return config


def _resolve_env_vars(obj):
    """Recursively replace ${VAR} with os.environ.get(VAR, '')."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                obj[key] = os.environ.get(env_var, "")
            elif isinstance(value, (dict, list)):
                _resolve_env_vars(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                env_var = item[2:-1]
                obj[i] = os.environ.get(env_var, "")
            elif isinstance(item, (dict, list)):
                _resolve_env_vars(item)
