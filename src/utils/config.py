"""Configuration management utilities using OmegaConf."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(config_paths: List[str], overrides: Optional[List[str]] = None) -> DictConfig:
    """Load and merge multiple YAML configs with optional CLI overrides.

    Args:
        config_paths: List of paths to YAML config files (merged in order).
        overrides: Optional list of dotlist overrides, e.g. ["train.epochs=15"].

    Returns:
        Merged DictConfig.
    """
    configs = []
    for path in config_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        configs.append(OmegaConf.load(str(path)))

    merged = OmegaConf.merge(*configs)

    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, override_conf)

    return merged


def parse_cli_configs() -> tuple[DictConfig, List[str]]:
    """Parse CLI arguments to extract --config paths and dotlist overrides.

    Returns:
        Tuple of (merged_config, list_of_config_paths).
    """
    args = sys.argv[1:]
    config_paths = []
    overrides = []

    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_paths.append(args[i + 1])
            i += 2
        elif "=" in args[i]:
            overrides.append(args[i])
            i += 1
        else:
            i += 1

    if not config_paths:
        raise ValueError("At least one --config path must be provided.")

    cfg = load_config(config_paths, overrides)
    return cfg, config_paths


def save_config(cfg: DictConfig, save_path: str | Path) -> None:
    """Save a resolved config to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        OmegaConf.save(cfg, f)


def get_nested(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """Safely get a nested config value using dot notation."""
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def config_to_dict(cfg: DictConfig) -> dict:
    """Convert OmegaConf config to a plain Python dict."""
    return OmegaConf.to_container(cfg, resolve=True)
