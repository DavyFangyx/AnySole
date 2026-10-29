"""Atomic state and compact-output helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
from typing import Any, Mapping

import numpy as np

from .config import RESULT_ROOT


SCHEMA_VERSION = 1
RESULT_SCHEMA = "tactile.sam3db-baseline.result"
STATE_SCHEMA = "tactile.sam3db-baseline.state"
_CAMERA_RE = re.compile(r"^(?:camera_|cam)?(\d+)$", re.IGNORECASE)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_bytes(path: str | Path, payload: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_json(path: str | Path, value: Any) -> Path:
    payload = (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    return atomic_bytes(path, payload)


def atomic_pickle(path: str | Path, value: Any) -> Path:
    return atomic_bytes(path, pickle.dumps(value, protocol=5))


def read_json(path: str | Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError("result.pkl must contain a dict")
    return value


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_result_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    expected = RESULT_ROOT.resolve()
    if root != expected:
        raise ValueError(f"result root must be exactly {expected}, got {root}")
    return root


def short_unit_id(unit: Mapping[str, Any]) -> str:
    method = str(unit["method"])
    aliases = {"method_1_offset": "m1", "method_2_projection": "m2"}
    if method not in aliases:
        raise ValueError(f"unsupported method {method!r}")
    camera_match = _CAMERA_RE.fullmatch(str(unit["camera"]))
    if camera_match is None:
        raise ValueError(f"invalid camera {unit['camera']!r}")
    session = str(unit["session"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError(f"unsafe session {session!r}")
    return f"{unit['date']}_{aliases[method]}_{session}_c{int(camera_match.group(1))}"


def unit_root(result_root: str | Path, unit: Mapping[str, Any]) -> Path:
    return Path(result_root).resolve() / short_unit_id(unit)


def validate_result_payload(value: Mapping[str, Any], frame_count: int) -> None:
    if value.get("schema_id") != RESULT_SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported result schema")
    arrays = value.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("result has no arrays mapping")
    shapes = {
        "frame": (frame_count,),
        "bbox_xyxy": (frame_count, 4),
        "global_orient": (frame_count, 3),
        "body_pose": (frame_count, 63),
        "betas": (frame_count, 10),
        "camera_translation": (frame_count, 3),
        "focal_length": (frame_count,),
        "similarity_scale": (frame_count,),
        "similarity_rotation": (frame_count, 3, 3),
        "similarity_translation": (frame_count, 3),
        "smpl_joints_2d": (frame_count, 24, 2),
        "alignment_rmse": (frame_count,),
    }
    for name, shape in shapes.items():
        array = np.asarray(arrays.get(name))
        if array.shape != shape:
            raise ValueError(f"{name} shape {array.shape} != {shape}")
        if name != "frame" and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")


def compact_unit_valid(result_root: str | Path, unit: Mapping[str, Any], config_hash: str) -> bool:
    root = unit_root(result_root, unit)
    expected = {"result.pkl", "overlay.mp4", "state.json"}
    try:
        if {path.name for path in root.iterdir()} != expected:
            return False
        state = read_json(root / "state.json")
        if not state or state.get("status") != "succeeded" or state.get("config_hash") != config_hash:
            return False
        if state.get("frame_count") != int(unit["frame_count"]):
            return False
        if sha256(root / "result.pkl") != state.get("result_sha256"):
            return False
        if sha256(root / "overlay.mp4") != state.get("overlay_sha256"):
            return False
        validate_result_payload(load_result(root / "result.pkl"), int(unit["frame_count"]))
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError):
        return False
    return True
