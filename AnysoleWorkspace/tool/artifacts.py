#!/usr/bin/env python3
"""Small, dependency-free helpers for generated artifact contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def producer_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_artifact(
    root: Path,
    *,
    schema_version: str,
    producer: str,
    repository_root: Path,
    parameters: dict[str, Any] | None = None,
    source_artifacts: Iterable[str] = (),
    source_hashes: dict[str, str] | None = None,
    frame_id_min: int | None = None,
    frame_id_max: int | None = None,
    frame_count: int | None = None,
    consumers: Iterable[str] = (),
) -> Path:
    """Write an ``artifact.json`` atomically enough for local CLI use."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": schema_version,
        "producer": producer,
        "producer_commit": producer_commit(repository_root),
        "parameters": parameters or {},
        "source_artifacts": list(source_artifacts),
        "source_hashes": source_hashes or {},
        "frame_id_min": frame_id_min,
        "frame_id_max": frame_id_max,
        "frame_count": frame_count,
        "consumers": list(consumers),
    }
    output = root / "artifact.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output
