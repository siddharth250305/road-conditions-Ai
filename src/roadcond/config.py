"""YAML config loading. The YAML files are the single source of truth for class lists."""

from pathlib import Path
from typing import Any

import yaml

# Repo root; valid for a repo checkout or editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Make a path absolute; relative paths are taken from base (the repo root by default)."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else base / path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, defaulting to configs/default.yaml."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as f:
        return yaml.safe_load(f)
