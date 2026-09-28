"""AnySole's private adapter over the frozen public workspace contract.

This module is deliberately model-owned.  It is the only AnySole data code
which knows how a ``shared.session.v1`` artifact becomes AnySole inputs.
Shared facts remain read-only; labels and visual caches are written below the
AnySole-owned ``model_inputs`` tree.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from AnysoleWorkspace.tool.workspace import canonical_uri, resolve_uri


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "AnysoleWorkspace"
SHARED_SESSION_ROOT = WORKSPACE_ROOT / "shared/facts/sessions/cam3"
PROTOCOL_SPLIT = WORKSPACE_ROOT / "protocol/splits/default/splits.csv"
ADAPTER_VERSION = "adapter_v1"
ANYSOLE_INPUT_ROOT = WORKSPACE_ROOT / "model_inputs/AnySole" / ADAPTER_VERSION
LABEL_ROOT = ANYSOLE_INPUT_ROOT / "labels"
HRNET_ROOT = ANYSOLE_INPUT_ROOT / "visual/hrnet/cam3"
HMR_ROOT = ANYSOLE_INPUT_ROOT / "visual/hmr"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_session_dir(session_id: str, root: Path = SHARED_SESSION_ROOT) -> Path:
    """Return one public shared session, rejecting ambiguous identities."""
    matches = sorted(
        path.parent
        for path in Path(root).glob(f"*/*/{session_id}/session.json")
        if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(
            f"shared session {session_id!r} not found under {Path(root)}"
        )
    if len(matches) > 1:
        raise ValueError(f"session id is ambiguous in shared facts: {session_id}: {matches}")
    return matches[0]


def _check_frame_axis(session_dir: Path, frames: dict, pressure: dict) -> None:
    frame_id = np.asarray(frames["frame_id"])
    if frame_id.ndim != 1 or frame_id.dtype.kind not in "iu":
        raise ValueError(f"{session_dir}: frame_id must be a 1-D integer array")
    if not np.array_equal(frame_id, np.arange(len(frame_id), dtype=frame_id.dtype)):
        raise ValueError(f"{session_dir}: frame_id must be unique and monotonic from zero")
    n = len(frame_id)
    for key in ("visual_time_s", "mocap_time_s", "valid", "fake"):
        if np.asarray(frames[key]).shape != (n,):
            raise ValueError(f"{session_dir}: frames.{key} length does not match frame_id")
    if np.any(np.asarray(frames["fake"]) & np.asarray(frames["valid"])):
        # A fake frame is not a valid observed frame in the public contract.
        raise ValueError(f"{session_dir}: fake and valid masks overlap")
    if not np.all(np.diff(np.asarray(frames["visual_time_s"], dtype=float)) > 0):
        raise ValueError(f"{session_dir}: visual_time_s must be strictly increasing")
    if not np.all(np.diff(np.asarray(frames["mocap_time_s"], dtype=float)) > 0):
        raise ValueError(f"{session_dir}: mocap_time_s must be strictly increasing")
    if not np.array_equal(np.asarray(pressure["frame_id"]), frame_id):
        raise ValueError(f"{session_dir}: pressure frame_id differs from frames.npz")
    for key in ("left48", "right48", "valid", "fake"):
        value = np.asarray(pressure[key])
        if key in ("left48", "right48"):
            expected = (n, 48)
        else:
            expected = (n,)
        if value.shape != expected:
            raise ValueError(f"{session_dir}: pressure.{key} shape {value.shape} != {expected}")


def load_shared_session(session_id: str, root: Path = SHARED_SESSION_ROOT) -> dict:
    """Load and validate one shared session without reading any model tree."""
    session_dir = shared_session_dir(session_id, root)
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = dict(np.load(session_dir / "frames.npz", allow_pickle=True))
    pressure = dict(np.load(session_dir / "pressure_48.npz", allow_pickle=True))
    required_frames = {"frame_id", "visual_time_s", "mocap_time_s", "valid", "fake"}
    required_pressure = {"frame_id", "left48", "right48", "valid", "fake"}
    if not required_frames <= frames.keys():
        raise ValueError(f"{session_dir}: missing frame keys {required_frames - frames.keys()}")
    if not required_pressure <= pressure.keys():
        raise ValueError(f"{session_dir}: missing pressure keys {required_pressure - pressure.keys()}")
    _check_frame_axis(session_dir, frames, pressure)
    if str(meta.get("session_id")) != str(session_id):
        raise ValueError(f"{session_dir}: session.json identity mismatch")
    return {
        "session_id": str(session_id),
        "dir": session_dir,
        "meta": meta,
        "frames": frames,
        "pressure": pressure,
        "source_artifact": session_dir / "artifact.json",
    }


def shared_bvh_path(session: dict) -> Path:
    value = session["meta"].get("source_files", {}).get("bvh", "")
    path = resolve_uri(value, must_exist=True)
    return path


def shared_smpl_path(session: dict, roots: Iterable[Path] = ()) -> Path:
    value = session["meta"].get("source_files", {}).get("smpl", "")
    if value:
        try:
            return resolve_uri(value, must_exist=True)
        except (FileNotFoundError, ValueError):
            pass
    sid = session["session_id"]
    for root in roots:
        matches = sorted(Path(root).glob(f"**/{sid}/motion_neutral_smpl.npz"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"SMPL NPZ not found for {sid}; shared metadata={value!r}")


def model_input_root() -> Path:
    return ANYSOLE_INPUT_ROOT


def hrnet_cache_path(session_id: str) -> Path:
    return HRNET_ROOT / f"{session_id}.pt"


def hmr_cache_path(model: str, session_id: str) -> Path:
    return HMR_ROOT / str(model) / "cam3" / f"{session_id}.pt"


def label_path(session_id: str, method: str) -> Path:
    if not method or "/" in method or "\\" in method or method in {".", ".."}:
        raise ValueError(f"invalid AnySole contact method: {method!r}")
    return LABEL_ROOT / method / f"{session_id}.npz"


def load_contact_label(session: dict, method: str) -> np.ndarray:
    path = label_path(session["session_id"], method)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing private AnySole contact labels {path}. Run "
            f"`python -m anysole.data.contact_adapter --session {session['session_id']} "
            f"--methods {method}`."
        )
    data = dict(np.load(path, allow_pickle=True))
    if "contact" not in data or "frame_id" not in data:
        raise ValueError(f"{path}: expected contact and frame_id arrays")
    contact = np.asarray(data["contact"], dtype=np.float32)
    frame_id = np.asarray(data["frame_id"])
    expected = np.asarray(session["frames"]["frame_id"])
    if contact.shape != (len(expected), 2) or not np.array_equal(frame_id, expected):
        raise ValueError(f"{path}: label shape/frame axis does not match shared session")
    return contact


def write_label(path: Path, *, session: dict, method: str, contact: np.ndarray,
                parameters: dict) -> None:
    frame_id = np.asarray(session["frames"]["frame_id"], dtype=np.int64)
    contact = np.asarray(contact, dtype=np.float32)
    if contact.shape != (len(frame_id), 2):
        raise ValueError(f"contact shape {contact.shape} != {(len(frame_id), 2)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        frame_id=frame_id,
        contact=contact,
        valid=np.asarray(session["frames"]["valid"], dtype=np.uint8),
        fake=np.asarray(session["frames"]["fake"], dtype=np.uint8),
        method=np.asarray(method),
        adapter_version=np.asarray(ADAPTER_VERSION),
        parameters_json=np.asarray(json.dumps(parameters, sort_keys=True)),
        source_session=np.asarray(canonical_uri(session["dir"])),
        source_session_hash=np.asarray(
            sha256_file(session["source_artifact"]) if session["source_artifact"].is_file() else ""
        ),
    )
    os.replace(temporary, path)


def write_cache_provenance(cache_path: Path, *, session: dict, kind: str,
                           shape: tuple[int, ...], parameters: dict | None = None) -> None:
    payload = {
        "schema_version": "model_input.anysole.v1",
        "adapter_version": ADAPTER_VERSION,
        "producer": f"anysole.data.{kind}",
        "session_id": session["session_id"],
        "source_artifacts": [canonical_uri(session["dir"] / "frames.npz")],
        "source_hashes": {
            "session": sha256_file(session["source_artifact"])
            if session["source_artifact"].is_file() else "",
        },
        "frame_id_min": 0,
        "frame_id_max": int(shape[0] - 1),
        "frame_count": int(shape[0]),
        "shape": list(shape),
        "parameters": parameters or {},
        "consumers": ["AnySole"],
    }
    sidecar = Path(str(cache_path) + ".artifact.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
