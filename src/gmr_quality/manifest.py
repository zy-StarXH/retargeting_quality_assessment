from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import yaml


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version() -> str:
    try:
        return importlib.metadata.version("gmr-quality-eval")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0+uninstalled"


def _git_output(path: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ("git", "-C", str(path), *args),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_state(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    top_level = _git_output(source, "rev-parse", "--show-toplevel")
    commit = _git_output(source, "rev-parse", "HEAD")
    status = _git_output(source, "status", "--porcelain")
    changes = tuple(line for line in (status or "").splitlines() if line)
    return {
        "requested_path": str(source),
        "repository_root": top_level,
        "commit": commit,
        "dirty": bool(changes) if status is not None else None,
        "change_count": len(changes) if status is not None else None,
    }


def runtime_info() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "mujoco_version": mujoco.__version__,
    }


def write_resolved_config(config: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            config,
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
    return destination


def write_run_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return destination
