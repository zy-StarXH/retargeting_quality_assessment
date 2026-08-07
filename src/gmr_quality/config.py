from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = (
    "project",
    "paths",
    "input",
    "doctor",
    "robot",
    "thresholds",
    "contact_policies",
    "fidelity",
    "scoring",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the project YAML and resolve configured filesystem paths."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Configuration is missing sections: {', '.join(missing)}")

    paths = config["paths"]
    if not isinstance(paths, dict):
        raise ValueError("Configuration section 'paths' must be a mapping")

    for name, value in tuple(paths.items()):
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = config_path.parent / resolved
        paths[name] = str(resolved.resolve())

    config["_config_path"] = str(config_path)
    return config
