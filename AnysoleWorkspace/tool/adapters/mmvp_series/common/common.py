"""Shared helpers for the pressure_toolkit adapter: canonical paths, manifest/split
readers and per-session frame facts.

All shared/protocol trees are read-only here; every write lands under
``model-input://pressure_toolkit/v1``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from AnysoleWorkspace.tool.workspace import resolve_uri

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
MANIFEST = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
SPLITS = WORKSPACE / "protocol/splits/default/splits.csv"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
MMVP_ROOT = WORKSPACE / "shared/representations/tactile/mmvp_31x11/v1"
SAM31_INDEX = WORKSPACE / "shared/frontends/human_masks/sam31/v1/index.jsonl"
DEPTHPRO_ROOT = WORKSPACE / "shared/frontends/depthpro/v1"
RTMPOSE_ROOT = WORKSPACE / "shared/frontends/rtmpose_halpe26/v1"
CLIFF_ROOT = WORKSPACE / "shared/frontends/cliff_hr48/v1"
ADAPTER_ROOT = resolve_uri("model-input://pressure_toolkit/v1")

ADAPTER_VERSION = "v1"
CALIBRATION_DIR = WORKSPACE / "protocol/calibration"
TOOLKIT_ROOT = REPO_ROOT / "Baselines/pressure_tookit"


def manifest_rows() -> dict[str, dict]:
    """session_id -> manifest row (read-only)."""
    rows: dict[str, dict] = {}
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def split_sessions(split: str) -> list[str]:
    """Canonical split session ids (frozen 92/12/36)."""
    columns = ("train", "val", "test") if split == "all" else (split,)
    values: list[str] = []
    with SPLITS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                value = (row.get(column) or "").strip()
                if value:
                    values.append(value)
    return sorted(set(values))


def session_parts(row: dict) -> tuple[str, str, str]:
    """(date, subject, session) from a manifest row's raw video path."""
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2], row["session_id"]


def load_session_facts(date: str, subject: str, session: str) -> dict:
    """Read one shared facts session: frames + pressure_48 + session.json."""
    seq_dir = FACTS_ROOT / date / subject / session
    frames = dict(np.load(seq_dir / "frames.npz"))
    pressure = dict(np.load(seq_dir / "pressure_48.npz"))
    meta = json.loads((seq_dir / "session.json").read_text(encoding="utf-8"))
    return {"dir": seq_dir, "frames": frames, "pressure": pressure, "meta": meta}


def load_sam31_index() -> dict[str, list[dict]]:
    """session -> per-frame SAM3.1 human-mask index rows (frame_id ascending)."""
    index: dict[str, list[dict]] = {}
    if not SAM31_INDEX.is_file():
        return index
    with SAM31_INDEX.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index.setdefault(row["session"], []).append(row)
    for rows in index.values():
        rows.sort(key=lambda item: int(item["frame_id"]))
    return index


def adapter_artifact(source_artifacts: list[str], parameters: dict,
                     frame_count: int | None = None) -> dict:
    return {
        "schema_version": "model_input.pressure_toolkit.v1",
        "producer": "build_inputs.py",
        "parameters": parameters,
        "source_artifacts": source_artifacts,
        "source_hashes": {},
        "frame_id_min": 0,
        "frame_id_max": None if frame_count is None else frame_count - 1,
        "frame_count": frame_count,
        "consumers": ["pressure_toolkit"],
    }
