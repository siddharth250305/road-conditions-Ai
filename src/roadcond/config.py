"""YAML config loading. The YAML files are the single source of truth for class lists."""

from pathlib import Path
from typing import Any

import yaml

# Resolves to <repo>/configs/default.yaml; valid for a repo checkout or editable install.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, defaulting to configs/default.yaml."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as f:
        return yaml.safe_load(f)
