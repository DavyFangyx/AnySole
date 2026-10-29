"""Bounded execution, publication, status and integrity audit for GVHMR units."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid
from typing import Any, Mapping

import cv2
import numpy as np

from .config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PYTHON,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SMPL_MODEL,
    DEFAULT_SMPLX_MODEL,
    DEFAULT_SMPLX_TO_SMPL,
    DEFAULT_VITPOSE_CHECKPOINT,
    DEFAULT_HMR2_CHECKPOINT,
    UPSTREAM_ROOT,
    WORKSPACE_ROOT,
)
from .io import (
    PUBLIC_ARTIFACTS,
    RESULT_SCHEMA_ID,
    RESULT_SCHEMA_VERSION,
    STATE_SCHEMA_ID,
    STATE_SCHEMA_VERSION,
    ResultError,
    atomic_json,
    atomic_npz,
    atomic_pickle,
    load_result,
    read_state,
    sha256_file,
)
from .visualize import render_overlay


class RuntimeError_(RuntimeError):
    """Raised when a bounded GVHMR run cannot be started or published."""


def safe_unit_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "__", str(value)).strip("._-")
    if not text:
        raise RuntimeError_(f"unit id has no safe filesystem characters: {value!r}")
    return text[:180]


def unit_output_dir(output_root: str | Path, unit: Mapping[str, Any]) -> Path:
    name = str(unit.get("public_name", unit["unit_id"]))
    if "public_name" in unit and (name != safe_unit_id(name) or len(name) > 180):
        raise RuntimeError_(f"unsafe public result directory name: {name!r}")
    return Path(output_root).expanduser().resolve() / safe_unit_id(name)


def write_inference_video(
    unit: Mapping[str, Any],
    dataset_root: str | Path,
    destination: str | Path,
    *,
    frame_limit: int,
    model_fps: float = 30.0,
) -> tuple[Path, list[dict[str, Any]]]:
    """Create the exact bounded frame stream consumed by official preprocessors."""

    frames = list(unit.get("frames", []))[:frame_limit]
    if len(frames) < 2:
        raise RuntimeError_("GVHMR requires at least two consecutive frames for its camera-motion feature")
    frame_numbers = [int(frame["frame"]) for frame in frames]
    if unit.get("input_kind") == "annotated_rgb_image_sequence" and any(
        right - left != 1 for left, right in zip(frame_numbers, frame_numbers[1:])
    ):
        raise RuntimeError_(
            "image-sequence frame IDs are not consecutive; resampling/interpolating dropped frames is not implemented"
        )
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if unit.get("input_kind") == "explicit_video":
        source = Path(unit["video_path"]).expanduser().resolve()
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError_(f"cannot open explicit input video: {source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), model_fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError_(f"cannot create staged GVHMR video: {path}")
        written = 0
        try:
            for _ in frames:
                ok, image = capture.read()
                if not ok:
                    break
                writer.write(image)
                written += 1
        finally:
            capture.release()
            writer.release()
        if written != len(frames):
            raise RuntimeError_(f"explicit video ended after {written}/{len(frames)} requested frames")
    elif unit.get("input_kind") == "annotated_rgb_image_sequence":
        root = Path(dataset_root).expanduser().resolve()
        first_path = root / frames[0]["relative_path"]
        first = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError_(f"cannot decode source RGB frame: {first_path}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), model_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError_(f"cannot create staged GVHMR video: {path}")
        written = 0
        try:
            for frame in frames:
                image_path = root / frame["relative_path"]
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (height, width):
                    raise RuntimeError_(f"cannot decode frame or geometry changes within unit: {image_path}")
                writer.write(image)
                written += 1
        finally:
            writer.release()
        if written != len(frames):
            raise RuntimeError_(f"wrote {written}/{len(frames)} requested RGB frames")
    else:
        raise RuntimeError_(f"unsupported unit input_kind: {unit.get('input_kind')!r}")

    check = cv2.VideoCapture(str(path))
    if not check.isOpened():
        raise RuntimeError_(f"staged GVHMR video is not readable: {path}")
    count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    check.release()
    expected_width = int(frames[0]["image_width"])
    expected_height = int(frames[0]["image_height"])
    if (count, width, height) != (len(frames), expected_width, expected_height):
        raise RuntimeError_(
            f"staged video verification mismatch {(count, width, height)} != "
            f"{(len(frames), expected_width, expected_height)}"
        )
    return path, frames


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _resume_action(previous: Mapping[str, Any] | None, *, retry: bool, force: bool) -> str:
    """Pure resume policy shared by CLI control and focused tests."""

    if previous is None:
        return "run"
    status = previous.get("status")
    if status == "succeeded" and not force:
        return "skip"
    if status in {"failed", "running"} and not (retry or force):
        return "refuse"
    return "run"


def _read_native_output(path: Path, expected_count: int) -> dict[str, np.ndarray]:
    required = {
        "native_pose_camera": (expected_count, 66),
        "native_pose_global": (expected_count, 66),
        "native_betas_camera": (expected_count, 10),
        "native_betas_global": (expected_count, 10),
        "native_trans_camera": (expected_count, 3),
        "native_trans_global": (expected_count, 3),
        "pose": (expected_count, 72),
        "betas": (expected_count, 10),
        "trans": (expected_count, 3),
        "pose_global": (expected_count, 72),
        "betas_global": (expected_count, 10),
        "trans_global": (expected_count, 3),
        "fit_vertex_rmse_mm": (expected_count,),
        "fit_vertex_p95_mm": (expected_count,),
        "fit_joint_rmse_mm": (expected_count,),
        "fit_global_vertex_rmse_mm": (expected_count,),
        "fit_global_vertex_p95_mm": (expected_count,),
        "fit_global_joint_rmse_mm": (expected_count,),
        "joints_camera": (expected_count, 24, 3),
        "joints_global": (expected_count, 24, 3),
        "K_fullimg": (expected_count, 3, 3),
    }
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    for name, shape in required.items():
        if name not in values or values[name].shape != shape or not np.isfinite(values[name]).all():
            raise RuntimeError_(f"worker output {name} invalid: {values.get(name, np.asarray([])).shape}, expected {shape}")
    return values


def run_unit(
    manifest: Mapping[str, Any],
    unit_id: str,
    *,
    max_frames: int,
    output_root: str | Path = DEFAULT_RESULT_ROOT,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    python_executable: str | Path = DEFAULT_PYTHON,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    smplx_model_path: str | Path = DEFAULT_SMPLX_MODEL,
    smpl_model_path: str | Path = DEFAULT_SMPL_MODEL,
    mapper_path: str | Path = DEFAULT_SMPLX_TO_SMPL,
    hmr2_checkpoint_path: str | Path = DEFAULT_HMR2_CHECKPOINT,
    vitpose_checkpoint_path: str | Path = DEFAULT_VITPOSE_CHECKPOINT,
    static_camera: bool = True,
    retry: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run one bounded unit with terminal state and atomic public artifacts."""

    if isinstance(max_frames, bool) or int(max_frames) != max_frames or max_frames < 2:
        raise RuntimeError_("max_frames must be an integer >= 2")
    if manifest.get("schema_id") != "tactile.gvhmr-baseline.inventory":
        raise RuntimeError_("manifest schema id is unsupported")
    matches = [unit for unit in manifest.get("units", []) if str(unit.get("unit_id")) == str(unit_id)]
    if len(matches) != 1:
        raise RuntimeError_(f"expected exactly one manifest unit with id {unit_id!r}, found {len(matches)}")
    unit = matches[0]
    selected_count = min(int(max_frames), len(unit.get("frames", [])))
    if selected_count < 2:
        raise RuntimeError_(f"unit {unit_id} contains fewer than two selected frames")
    destination = unit_output_dir(output_root, unit)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "state.json"
    previous = read_state(state_path)
    action = _resume_action(previous, retry=retry, force=force)
    if action == "skip":
        audit_unit(destination)
        return previous
    if action == "refuse":
        state_status = previous.get("status") if previous else "unknown"
        if state_status == "failed":
            raise RuntimeError_("unit has a terminal failed state; pass --retry to start a fresh attempt")
        raise RuntimeError_("unit state is still running/incomplete; inspect it, then pass --retry to rerun")

    started = {
        "schema_id": STATE_SCHEMA_ID,
        "schema_version": STATE_SCHEMA_VERSION,
        "unit_id": str(unit_id),
        "status": "running",
        "terminal": False,
        "started_utc": _utc_now(),
        "manifest_schema": manifest.get("schema_id"),
        "frame_count": selected_count,
        "max_frames_requested": int(max_frames),
        "input_kind": unit.get("input_kind"),
        "static_camera": bool(static_camera),
        "output_schema_id": RESULT_SCHEMA_ID,
        "output_schema_version": RESULT_SCHEMA_VERSION,
    }
    atomic_json(state_path, started)

    output_path = Path(output_root).expanduser().resolve()
    scratch_root = output_path / ".run" / "work"
    scratch: Path | None = None
    worker_log = ""
    try:
        scratch_root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f"{safe_unit_id(unit_id)}.", dir=scratch_root))
        video_path, frames = write_inference_video(
            unit, dataset_root, scratch / "target_input.mp4", frame_limit=selected_count
        )
        request = {
            "upstream_root": str(UPSTREAM_ROOT),
            "work_dir": str(scratch),
            "video_path": str(video_path),
            "frame_count": len(frames),
            "bbox_xyxy_px": [frame["bbox_xyxy_px"] for frame in frames],
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "smplx_model_path": str(Path(smplx_model_path).expanduser().resolve()),
            "smpl_model_path": str(Path(smpl_model_path).expanduser().resolve()),
            "mapper_path": str(Path(mapper_path).expanduser().resolve()),
            "hmr2_checkpoint_path": str(Path(hmr2_checkpoint_path).expanduser().resolve()),
            "vitpose_checkpoint_path": str(Path(vitpose_checkpoint_path).expanduser().resolve()),
            "static_camera": bool(static_camera),
        }
        request_path = scratch / "request.json"
        atomic_json(request_path, request)
        child_env = os.environ.copy()
        code_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = code_root + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        command = [
            str(Path(python_executable).expanduser().resolve()),
            "-m",
            "gvhmr_baseline.worker",
            "--request",
            str(request_path),
            "--output",
            str(scratch / "native_output.npz"),
        ]
        completed = subprocess.run(
            command,
            cwd=UPSTREAM_ROOT,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        worker_log = completed.stdout[-12_000:]
        if completed.returncode != 0:
            raise RuntimeError_(
                f"GVHMR worker exited {completed.returncode}; output tail:\n{worker_log[-4_000:]}"
            )
        native = _read_native_output(scratch / "native_output.npz", len(frames))
        max_vertex_rmse = max(
            float(native["fit_vertex_rmse_mm"].max()),
            float(native["fit_global_vertex_rmse_mm"].max()),
        )
        if max_vertex_rmse > 50.0:
            raise RuntimeError_(
                f"SMPL conversion quality gate failed before publication: max vertex RMSE "
                f"{max_vertex_rmse:.3f} mm exceeds 50 mm"
            )
        arrays = {
            **native,
            "source_frame_numbers": np.asarray([int(frame["frame"]) for frame in frames], dtype=np.int64),
            "frame_indices": np.asarray([int(frame["frame_index"]) for frame in frames], dtype=np.int64),
            "physical_times_s": np.asarray([float(frame["timestamp_seconds"]) for frame in frames], dtype=np.float64),
            "bbox_xyxy_px": np.asarray([frame["bbox_xyxy_px"] for frame in frames], dtype=np.float32),
        }
        atomic_npz(scratch / "result.npz", arrays)
        atomic_pickle(scratch / "result.pkl", {
            "schema_id": RESULT_SCHEMA_ID, "schema_version": RESULT_SCHEMA_VERSION,
            "unit_id": unit_id, "frame_count": len(frames),
            "coordinate_semantics": "camera and gravity-view/global; not checkerboard-aligned",
            "smpl_model": "neutral SMPL fitted to GVHMR 2024 SMPL-X output",
            **arrays,
        })
        overlay_metadata = render_overlay(
            video_path,
            scratch / "overlay.mp4",
            arrays["bbox_xyxy_px"],
            arrays["joints_camera"],
            arrays["K_fullimg"],
            fps=30.0,
            labels=[f"{unit_id} frame {frame['frame']}" for frame in frames],
        )
        overlay_metadata["path"] = "overlay.mp4"
        # Public artifacts are atomically replaced only after all inference,
        # conversion, schema validation and overlay verification succeed.
        os.replace(scratch / "result.npz", destination / "result.npz")
        os.replace(scratch / "result.pkl", destination / "result.pkl")
        os.replace(scratch / "overlay.mp4", destination / "overlay.mp4")
        metadata_path = scratch / "native_output.metadata.json"
        if metadata_path.is_file():
            os.replace(metadata_path, destination / "model_metadata.json")
        result_arrays = load_result(destination / "result.npz", expected_count=len(frames))
        state = {
            **started,
            "status": "succeeded",
            "terminal": True,
            "finished_utc": _utc_now(),
            "frame_count": len(frames),
            "source_frame_start": int(frames[0]["frame"]),
            "source_frame_end": int(frames[-1]["frame"]),
            "source_fps": unit.get("fps"),
            "model_fps": 30.0,
            "source_timestamps_preserved": True,
            "coordinate_semantics": {
                "native_camera": "GVHMR 2024 incam frame, metres; estimated pinhole camera; not checkerboard-aligned",
                "native_global": "GVHMR 2024 gravity-view/global gauge, metres; not checkerboard-aligned",
                "pose_betas_translation": "neutral-SMPL parameters fitted to official mapped SMPL-X vertices; pose is full 72D axis-angle",
                "world_metrics": "not computed until global gauge/axes are explicitly aligned and validated",
            },
            "fit_quality": {
                "max_camera_vertex_rmse_mm": float(result_arrays["fit_vertex_rmse_mm"].max()),
                "max_global_vertex_rmse_mm": float(result_arrays["fit_global_vertex_rmse_mm"].max()),
                "quality_gate_mm": 50.0,
            },
            "overlay": overlay_metadata,
            "artifacts": {
                "result.npz": _artifact_record(destination / "result.npz"),
                "result.pkl": _artifact_record(destination / "result.pkl"),
                "overlay.mp4": _artifact_record(destination / "overlay.mp4"),
                **({"model_metadata.json": _artifact_record(destination / "model_metadata.json")} if (destination / "model_metadata.json").is_file() else {}),
            },
            "worker_log_tail": worker_log[-4_000:],
        }
        atomic_json(state_path, state)
        return state
    except BaseException as exc:
        failed = {
            **started,
            "status": "failed",
            "terminal": True,
            "finished_utc": _utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "worker_log_tail": worker_log[-4_000:],
        }
        atomic_json(state_path, failed)
        raise
    finally:
        # Scratch belongs solely to this invocation and contains full-frame
        # videos/features; public artifacts and compact terminal state survive.
        import shutil

        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema_id") != "tactile.gvhmr-baseline.inventory":
        raise RuntimeError_(f"unsupported manifest schema at {source}")
    return value


def audit_unit(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    state = read_state(root / "state.json")
    if state is None:
        raise FileNotFoundError(root / "state.json")
    if state["status"] != "succeeded":
        raise RuntimeError_(f"unit is not succeeded: {state['status']}")
    checked: dict[str, Any] = {}
    artifacts = state.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise RuntimeError_("unit state artifacts field must be a mapping")
    required_artifacts = {"result.npz", "overlay.mp4"}
    if not required_artifacts.issubset(artifacts):
        raise RuntimeError_(f"unit state does not declare required artifacts: {sorted(required_artifacts - set(artifacts))}")
    undeclared = set(artifacts) - PUBLIC_ARTIFACTS
    if undeclared:
        raise RuntimeError_(f"unit state declares unsupported artifacts: {sorted(undeclared)}")
    if "state.json" in artifacts:
        raise RuntimeError_("state.json cannot declare itself as a hashed result artifact")
    for optional in ("result.pkl", "model_metadata.json", "evaluation_camera.json"):
        if (root / optional).is_file() and optional not in artifacts:
            raise RuntimeError_(f"optional output exists but is not declared in unit state: {optional}")
    for name in sorted(artifacts):
        path = root / name
        record = artifacts.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise RuntimeError_(f"required artifact is missing or unrecorded: {path}")
        actual = _artifact_record(path)
        if actual != record:
            raise RuntimeError_(f"artifact checksum/size mismatch: {path}")
        checked[name] = actual
    arrays = load_result(root / "result.npz", expected_count=int(state["frame_count"]))
    video = cv2.VideoCapture(str(root / "overlay.mp4"))
    if not video.isOpened():
        raise RuntimeError_(f"overlay cannot be opened: {root / 'overlay.mp4'}")
    overlay_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    if overlay_count != int(state["frame_count"]):
        raise RuntimeError_(f"overlay frame count {overlay_count} != {state['frame_count']}")
    return {"unit_id": state["unit_id"], "status": state["status"], "frame_count": len(arrays["pose"]), "artifacts": checked}


def list_status(manifest: Mapping[str, Any], output_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in manifest.get("units", []):
        directory = unit_output_dir(output_root, unit)
        state = read_state(directory / "state.json")
        rows.append({
            "unit_id": unit["unit_id"],
            "status": state["status"] if state else "not_started",
            "frame_count": state.get("frame_count") if state else None,
            "output_dir": str(directory),
        })
    return rows


def audit_manifest(manifest: Mapping[str, Any], output_root: str | Path) -> dict[str, Any]:
    audited: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for unit in manifest.get("units", []):
        directory = unit_output_dir(output_root, unit)
        try:
            audited.append(audit_unit(directory))
        except Exception as exc:
            failures.append({"unit_id": str(unit["unit_id"]), "error": f"{type(exc).__name__}: {exc}"})
    return {"unit_count": len(manifest.get("units", [])), "audited_count": len(audited), "failure_count": len(failures), "audited": audited, "failures": failures}


__all__ = [
    "RuntimeError_",
    "audit_manifest",
    "audit_unit",
    "list_status",
    "_resume_action",
    "read_manifest",
    "run_unit",
    "safe_unit_id",
    "unit_output_dir",
    "write_inference_video",
]
