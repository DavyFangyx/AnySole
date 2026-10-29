"""Atomic, validated compact outputs for the GVHMR adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping

import numpy as np


RESULT_SCHEMA_ID = "tactile.gvhmr-baseline.result"
RESULT_SCHEMA_VERSION = 1
STATE_SCHEMA_ID = "tactile.gvhmr-baseline.unit-state"
STATE_SCHEMA_VERSION = 1
PUBLIC_ARTIFACTS = frozenset(
    {"result.npz", "result.pkl", "overlay.mp4", "state.json", "model_metadata.json", "evaluation_camera.json"}
)


class ResultError(ValueError):
    """Raised when a GVHMR output is incomplete, malformed, or non-finite."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(arrays: Mapping[str, Any], name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    if name not in arrays:
        raise ResultError(f"result is missing array {name!r}")
    value = np.asarray(arrays[name])
    if value.ndim != len(shape) or any(expected is not None and actual != expected for actual, expected in zip(value.shape, shape)):
        raise ResultError(f"result array {name!r} has shape {value.shape}, expected {shape}")
    if value.dtype.kind not in "iuf":
        raise ResultError(f"result array {name!r} must be numeric")
    if not np.isfinite(value).all():
        raise ResultError(f"result array {name!r} contains non-finite values")
    return value


def validate_result_arrays(arrays: Mapping[str, Any], expected_count: int | None = None) -> dict[str, np.ndarray]:
    """Validate compact native GVHMR and fitted neutral-SMPL arrays."""

    required_shapes: dict[str, tuple[int | None, ...]] = {
        "source_frame_numbers": (None,),
        "frame_indices": (None,),
        "physical_times_s": (None,),
        "bbox_xyxy_px": (None, 4),
        "pose": (None, 72),
        "betas": (None, 10),
        "trans": (None, 3),
        "pose_global": (None, 72),
        "betas_global": (None, 10),
        "trans_global": (None, 3),
        "native_pose_camera": (None, 66),
        "native_pose_global": (None, 66),
        "native_betas_camera": (None, 10),
        "native_betas_global": (None, 10),
        "native_trans_camera": (None, 3),
        "native_trans_global": (None, 3),
        "fit_vertex_rmse_mm": (None,),
        "fit_vertex_p95_mm": (None,),
        "fit_joint_rmse_mm": (None,),
        "fit_global_vertex_rmse_mm": (None,),
        "fit_global_vertex_p95_mm": (None,),
        "fit_global_joint_rmse_mm": (None,),
        "joints_camera": (None, 24, 3),
        "joints_global": (None, 24, 3),
        "K_fullimg": (None, 3, 3),
    }
    output = {name: _array(arrays, name, shape) for name, shape in required_shapes.items()}
    count = output["source_frame_numbers"].shape[0]
    if count == 0 or (expected_count is not None and count != expected_count):
        raise ResultError(f"result frame count {count} does not match expected {expected_count}")
    if any(value.shape[0] != count for value in output.values()):
        raise ResultError("result arrays have inconsistent frame counts")
    frame_numbers = output["source_frame_numbers"].astype(np.int64)
    if not np.array_equal(output["source_frame_numbers"], frame_numbers):
        raise ResultError("source_frame_numbers must contain integer-valued frame IDs")
    if np.any(np.diff(frame_numbers) <= 0):
        raise ResultError("source_frame_numbers must be strictly increasing")
    frame_indices = output["frame_indices"].astype(np.int64)
    if not np.array_equal(output["frame_indices"], frame_indices):
        raise ResultError("frame_indices must contain integer-valued indices")
    if np.any(frame_indices < 0) or np.any(np.diff(frame_indices) <= 0):
        raise ResultError("frame_indices must be non-negative and strictly increasing")
    if np.any(np.diff(output["physical_times_s"].astype(np.float64)) < 0.0):
        raise ResultError("physical_times_s must be non-decreasing")
    boxes = output["bbox_xyxy_px"]
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ResultError("bbox_xyxy_px contains a degenerate box")
    if np.any(output["K_fullimg"][:, 0, 0] <= 0) or np.any(output["K_fullimg"][:, 1, 1] <= 0):
        raise ResultError("K_fullimg contains a non-positive focal length")
    for name, value in output.items():
        if name in {"source_frame_numbers", "frame_indices"}:
            output[name] = value.astype(np.int64, copy=False)
        elif name == "physical_times_s":
            output[name] = value.astype(np.float64, copy=False)
        else:
            output[name] = value.astype(np.float32, copy=False)
    return output


def atomic_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_npz(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checked = validate_result_arrays(arrays)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **checked)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_pickle(path: str | Path, value: Any) -> Path:
    """Publish a Python-readable SMPL bundle alongside the canonical NPZ."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_result(path: str | Path, *, expected_count: int | None = None) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return validate_result_arrays(arrays, expected_count)


def read_state(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"cannot parse unit state {source}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_id") != STATE_SCHEMA_ID or value.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ResultError(f"unsupported unit state schema: {source}")
    if value.get("status") not in {"running", "succeeded", "failed"}:
        raise ResultError(f"invalid unit state status: {source}")
    if bool(value.get("terminal")) != (value.get("status") in {"succeeded", "failed"}):
        raise ResultError(f"unit state terminal flag disagrees with status: {source}")
    return value


__all__ = [
    "PUBLIC_ARTIFACTS",
    "RESULT_SCHEMA_ID",
    "RESULT_SCHEMA_VERSION",
    "ResultError",
    "STATE_SCHEMA_ID",
    "STATE_SCHEMA_VERSION",
    "atomic_json",
    "atomic_npz",
    "atomic_pickle",
    "load_result",
    "read_state",
    "sha256_file",
    "validate_result_arrays",
]
