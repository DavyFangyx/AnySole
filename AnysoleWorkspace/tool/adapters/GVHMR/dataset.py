"""Deterministic RGB and target-bbox manifests for GVHMR."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from wham_baseline.constraints import read_index_metadata
from wham_baseline.constraints import load_camera_calibration
from wham_baseline import dataset as wham_dataset
from wham_baseline.manifest_index import load_index


SCHEMA_ID = "tactile.gvhmr-baseline.inventory"
SCHEMA_VERSION = 1


class DatasetError(ValueError):
    """Raised when RGB images or target annotations violate the input contract."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera_number(camera: str) -> int:
    text = str(camera)
    if text.startswith("camera_"):
        text = text[len("camera_") :]
    if not text.isdigit() or int(text) <= 0:
        raise DatasetError(f"camera must be camera_<positive integer>, got {camera!r}")
    return int(text)


def validate_bbox(box: Any, width: int, height: int, *, where: str = "bbox") -> np.ndarray:
    """Check a raw-image xyxy box without silently clipping annotation data."""

    value = np.asarray(box, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise DatasetError(f"{where} must be four finite xyxy values")
    x1, y1, x2, y2 = value.tolist()
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise DatasetError(f"{where} is outside image bounds {width}x{height}: {value.tolist()}")
    return value.astype(np.float32)


def _target_boxes(index_path: str | Path, unit: Mapping[str, Any]) -> dict[int, list[float]]:
    source = Path(index_path).expanduser().resolve()
    if not source.is_file():
        raise DatasetError(f"annotation index is not a file: {source}")
    metadata = read_index_metadata(source)
    date = str(unit["date"])
    session = str(unit["session"])
    camera = _camera_number(str(unit["camera"]))
    frame_numbers = [frame.get("frame") for frame in unit["frames"]]
    if any(number is None for number in frame_numbers):
        raise DatasetError(f"unit has a frame without a source frame number: {unit.get('unit_id')}")
    requested = {int(number) for number in frame_numbers}
    if date not in metadata.get("dates", []):
        raise DatasetError(f"annotation index does not cover date {date}")
    result: dict[int, list[float]] = {}
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """SELECT frame,bbox_x1,bbox_y1,bbox_x2,bbox_y2 FROM annotations
               WHERE date=? AND session=? AND camera=?""",
            (date, session, camera),
        ).fetchall()
    for row in rows:
        number = int(row[0])
        if number in requested:
            result[number] = [float(item) for item in row[1:]]
    missing = sorted(requested - result.keys())
    if missing:
        raise DatasetError(f"missing target bbox annotations for {unit.get('unit_id')}: {missing[:8]}")
    if len(result) != len(requested):
        raise DatasetError(f"duplicate target bbox annotations for {unit.get('unit_id')}")
    return result


def build_manifest(
    dataset_root: str | Path,
    annotation_index: str | Path,
    *,
    dates: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
    cameras: Iterable[str] | None = None,
    max_units: int | None = None,
    max_frames_per_unit: int | None = None,
) -> dict[str, Any]:
    """Inventory selected image sequences and join every frame to its target bbox."""

    root = Path(dataset_root).expanduser().resolve()
    if max_units is not None and (isinstance(max_units, bool) or max_units <= 0):
        raise DatasetError("max_units must be a positive integer")
    if max_frames_per_unit is not None and (
        isinstance(max_frames_per_unit, bool) or max_frames_per_unit <= 0
    ):
        raise DatasetError("max_frames_per_unit must be a positive integer")
    selected_dates = None if dates is None else [str(item) for item in dates]
    selected_methods = None if methods is None else [str(item) for item in methods]
    requested_sessions = None if sessions is None else {str(item) for item in sessions}
    requested_cameras = None if cameras is None else {str(item) for item in cameras}
    if not root.is_dir():
        raise DatasetError(f"dataset root is not a directory: {root}")
    date_names = (
        sorted(set(selected_dates))
        if selected_dates is not None
        else sorted(
            entry.name
            for entry in root.iterdir()
            if entry.is_dir() and len(entry.name) == 8 and entry.name.isdigit()
        )
    )
    method_names = None if selected_methods is None else set(selected_methods)
    selected: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for date in date_names:
        rgb_root = root / date / "RGB"
        if not rgb_root.is_dir():
            continue
        methods = (
            [rgb_root / name for name in sorted(method_names)]
            if method_names is not None
            else wham_dataset._iter_directories(rgb_root)
        )
        for method_root in methods:
            if not method_root.is_dir():
                continue
            sessions_for_method = (
                [method_root / name for name in sorted(requested_sessions)]
                if requested_sessions is not None
                else wham_dataset._iter_directories(method_root)
            )
            for session_root in sessions_for_method:
                if not session_root.is_dir():
                    continue
                cameras_for_session = (
                    [session_root / name for name in sorted(requested_cameras)]
                    if requested_cameras is not None
                    else wham_dataset._iter_directories(session_root)
                )
                for camera_root in cameras_for_session:
                    if not camera_root.is_dir():
                        continue
                    raw_unit = wham_dataset._make_unit(
                        dataset_root=root,
                        date=date,
                        method=method_root.name,
                        session=session_root.name,
                        camera=camera_root.name,
                        camera_root=camera_root,
                        max_frames_per_unit=None,
                        issues=issues,
                        strict=True,
                    )
                    if raw_unit is None:
                        continue
                    selected.append(raw_unit.as_dict())
                    if max_units is not None and len(selected) >= max_units:
                        break
                if max_units is not None and len(selected) >= max_units:
                    break
            if max_units is not None and len(selected) >= max_units:
                break
        if max_units is not None and len(selected) >= max_units:
            break
    if not selected:
        raise DatasetError("selection produced no RGB image units")

    index = Path(annotation_index).expanduser().resolve()
    index_metadata = read_index_metadata(index)
    units: list[dict[str, Any]] = []
    total_frames = 0
    available_frames = 0
    for raw in selected:
        all_frames = [dict(frame) for frame in raw["frames"]]
        input_frames = all_frames if max_frames_per_unit is None else all_frames[:max_frames_per_unit]
        box_unit = dict(raw)
        box_unit["frames"] = input_frames
        boxes = _target_boxes(index, box_unit)
        frames: list[dict[str, Any]] = []
        for frame in input_frames:
            frame_number = frame.get("frame")
            if frame_number is None:
                raise DatasetError(f"unit {raw['unit_id']} contains a frame with no source frame ID")
            image_path = root / frame["relative_path"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3:
                raise DatasetError(f"cannot decode RGB frame: {image_path}")
            height, width = image.shape[:2]
            bbox = validate_bbox(boxes[int(frame_number)], width, height, where=f"{raw['unit_id']} frame {frame_number}")
            frame["bbox_xyxy_px"] = bbox.tolist()
            frame["image_width"] = int(width)
            frame["image_height"] = int(height)
            frames.append(frame)
        unit = dict(raw)
        unit["alignment_frames"] = [
            {
                "frame_index": int(frame["frame_index"]),
                "frame": frame["frame"],
                "filename": frame["filename"],
                "relative_path": frame["relative_path"],
                "timestamp_seconds": frame["timestamp_seconds"],
            }
            for frame in all_frames
        ]
        unit["frames"] = frames
        unit["frame_count"] = len(frames)
        unit["source_frame_count"] = len(all_frames)
        unit["input_kind"] = "annotated_rgb_image_sequence"
        unit["frame_start"] = frames[0]["frame"]
        unit["frame_end"] = frames[-1]["frame"]
        unit["bbox_coordinate_space"] = "raw_distorted_pixels_xyxy"
        unit["bbox_source"] = "date-local annotations/records.parquet target annotation"
        unit["bbox_count"] = len(frames)
        units.append(unit)
        total_frames += len(frames)
        available_frames += len(all_frames)

    filtered = (
        dates is not None
        or methods is not None
        or sessions is not None
        or cameras is not None
        or max_units is not None
        or max_frames_per_unit is not None
    )
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(root),
        "source_tree": "dated/RGB/<method>/<session>/<camera>",
        "input_kind": "annotated_rgb_image_sequence",
        "selection": {
            "dates": sorted(set(selected_dates)) if selected_dates is not None else None,
            "methods": sorted(set(selected_methods)) if selected_methods is not None else None,
            "sessions": sorted(requested_sessions) if requested_sessions is not None else None,
            "cameras": sorted(requested_cameras) if requested_cameras is not None else None,
            "max_units": max_units,
            "max_frames_per_unit": max_frames_per_unit,
        },
        "complete": not filtered,
        "annotation_index": {
            "path": str(index),
            "sha256": file_sha256(index),
            "dates": index_metadata.get("dates", []),
            "sources": index_metadata.get("sources", []),
        },
        "units": units,
        "summary": {
            "unit_count": len(units),
            "frame_count": total_frames,
            "available_frame_count": available_frames,
            "source_image_count": sum(int(unit.get("source_image_count", 0)) for unit in units),
            "issue_count": len(issues),
        },
        "issues": issues,
    }


def build_manifest_from_indexed_inventory(
    dataset_root: str | Path,
    inventory_path: str | Path,
    annotation_index: str | Path,
) -> dict[str, Any]:
    """Convert a verified complete WHAM inventory without decoding 1.4M JPEGs twice.

    Every source filename and target box is checked here. The inference video
    writer subsequently decodes every selected JPEG before model execution.
    """

    root = Path(dataset_root).expanduser().resolve()
    source = Path(inventory_path).expanduser().resolve()
    roi = Path(annotation_index).expanduser().resolve()
    index = load_index(source, verify_hash=True)
    metadata = read_index_metadata(roi)
    if Path(metadata.get("dataset_root", "")).resolve() != root:
        raise DatasetError("annotation index belongs to a different dataset root")
    dates = sorted(entry.name for entry in root.iterdir() if entry.is_dir() and len(entry.name) == 8 and entry.name.isdigit() and (entry / "RGB").is_dir())
    if dates != sorted(metadata.get("dates", [])):
        raise DatasetError("annotation dates differ from current RGB dates")
    for record in metadata.get("sources", []):
        path = Path(record["path"]).resolve()
        if path != root / record["date"] / "annotations" / "records.parquet" or file_sha256(path) != record["sha256"]:
            raise DatasetError(f"annotation source changed: {path}")
    if set(index["methods"]) != {"method_1_offset", "method_2_projection"}:
        raise DatasetError("indexed inventory must contain both expected RGB methods")

    units: list[dict[str, Any]] = []
    observed_dirs: set[str] = set()
    image_sizes: dict[tuple[str, str], tuple[int, int]] = {}
    with source.open("rb") as stream, sqlite3.connect(f"file:{roi}?mode=ro", uri=True) as connection:
        for row in index["units"]:
            line = stream.readline()
            if hashlib.sha256(line).hexdigest() != row["line_sha256"]:
                raise DatasetError(f"indexed inventory line changed: {row['unit_id']}")
            raw = json.loads(line)
            unit_id = str(raw["unit_id"])
            if unit_id != row["unit_id"] or raw.get("complete") is not True or raw.get("truncated") is not False:
                raise DatasetError(f"indexed unit is not complete: {unit_id}")
            if Path(raw["dataset_root"]).resolve() != root:
                raise DatasetError(f"indexed unit points to another dataset: {unit_id}")
            date, method, session, camera = (str(raw[key]) for key in ("date", "method", "session", "camera"))
            relative_dir = f"{date}/RGB/{method}/{session}/{camera}"
            if unit_id != f"{date}/{method}/{session}/{camera}" or raw["source_relpath"] != relative_dir:
                raise DatasetError(f"indexed unit path mismatch: {unit_id}")
            frames_raw = raw["frames"]
            if len(frames_raw) != row["frame_count"] or len(frames_raw) < 2:
                raise DatasetError(f"invalid frame count: {unit_id}")
            camera_root = root / relative_dir
            actual_names = {entry.name for entry in os.scandir(camera_root) if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png"))}
            expected_names = {frame["filename"] for frame in frames_raw}
            if actual_names != expected_names or len(expected_names) != len(frames_raw):
                raise DatasetError(f"source RGB files differ from indexed inventory: {unit_id}")
            observed_dirs.add(relative_dir)
            size_key = (date, camera)
            if size_key not in image_sizes:
                width, height = load_camera_calibration(root, date, camera)["image_size_px"]
                image_sizes[size_key] = (int(width), int(height))
            width, height = image_sizes[size_key]
            numbers = [int(frame["frame"]) for frame in frames_raw]
            if any(right - left != 1 for left, right in zip(numbers, numbers[1:])):
                raise DatasetError(f"GVHMR cannot ingest nonconsecutive source frames: {unit_id}")
            first, last = numbers[0], numbers[-1]
            rows = connection.execute(
                """SELECT frame,bbox_x1,bbox_y1,bbox_x2,bbox_y2 FROM annotations
                   WHERE date=? AND session=? AND camera=? AND frame BETWEEN ? AND ?""",
                (date, session, _camera_number(camera), first, last),
            ).fetchall()
            boxes = {int(item[0]): list(item[1:]) for item in rows}
            if set(boxes) != set(numbers):
                raise DatasetError(f"target annotations differ from RGB frames: {unit_id}")
            frames = []
            alignment_frames = []
            for frame in frames_raw:
                number = int(frame["frame"])
                if frame["relative_path"] != f"{relative_dir}/{frame['filename']}":
                    raise DatasetError(f"frame path mismatch: {unit_id} frame {number}")
                common = {
                    "frame_index": int(frame["frame_index"]), "frame": number,
                    "filename": frame["filename"], "relative_path": frame["relative_path"],
                    "timestamp_seconds": float(frame["timestamp_seconds"]),
                }
                alignment_frames.append(common)
                frames.append({**common, "size_bytes": int(frame["size_bytes"]),
                               "bbox_xyxy_px": validate_bbox(boxes[number], width, height, where=f"{unit_id} frame {number}").tolist(),
                               "image_width": width, "image_height": height})
            public_method = "m1" if method == "method_1_offset" else "m2"
            unit = {
                "unit_id": unit_id, "public_name": f"{date}_{public_method}_{session}_c{_camera_number(camera)}",
                "date": date, "method": method, "session": session, "camera": camera,
                "input_kind": "annotated_rgb_image_sequence", "source_relpath": relative_dir,
                "frames": frames, "alignment_frames": alignment_frames,
                "frame_count": len(frames), "source_frame_count": len(frames),
                "source_image_count": len(frames), "frame_start": first, "frame_end": last,
                "bbox_coordinate_space": "raw_distorted_pixels_xyxy",
                "bbox_source": "date-local annotations/records.parquet target annotation",
                "bbox_count": len(frames),
            }
            units.append(unit)
        if stream.read(1):
            raise DatasetError("indexed inventory has unindexed trailing bytes")

    current_dirs = {
        str(camera.relative_to(root))
        for date in dates
        for method in (root / date / "RGB").iterdir() if method.is_dir()
        for session in method.iterdir() if session.is_dir()
        for camera in session.iterdir() if camera.is_dir()
    }
    if current_dirs != observed_dirs:
        raise DatasetError(f"RGB unit directories differ from indexed inventory: missing={len(current_dirs - observed_dirs)}, stale={len(observed_dirs - current_dirs)}")
    if len(units) != index["unit_count"] or sum(unit["frame_count"] for unit in units) != index["frame_count"]:
        raise DatasetError("indexed inventory totals differ after conversion")
    return {
        "schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION,
        "dataset_root": str(root), "source_tree": "dated/RGB/<method>/<session>/<camera>",
        "input_kind": "annotated_rgb_image_sequence", "selection": {"dates": None, "methods": None, "sessions": None, "cameras": None, "max_units": None, "max_frames_per_unit": None},
        "complete": True,
        "source_inventory": {"path": str(source), "sha256": index["manifest_sha256"], "unit_count": index["unit_count"], "frame_count": index["frame_count"], "methods": index["methods"]},
        "annotation_index": {"path": str(roi), "sha256": file_sha256(roi), "dates": metadata["dates"], "sources": metadata["sources"]},
        "units": units,
        "summary": {"unit_count": len(units), "frame_count": index["frame_count"], "available_frame_count": index["frame_count"], "source_image_count": index["frame_count"], "issue_count": 0},
        "issues": [],
    }


def explicit_video_unit(
    video_path: str | Path,
    boxes: Any,
    *,
    fps: float,
    unit_id: str = "explicit_video",
) -> dict[str, Any]:
    """Describe an explicit video and a required per-frame target-box sidecar."""

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not np.isfinite(fps) or fps <= 0:
        raise DatasetError("fps must be finite and positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise DatasetError(f"cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise DatasetError(f"video has invalid decoded geometry/frame count: {(frame_count, width, height)}")
    values = np.asarray(boxes, dtype=np.float64)
    if values.shape != (frame_count, 4) or not np.isfinite(values).all():
        raise DatasetError(f"target boxes must have shape ({frame_count},4), got {values.shape}")
    checked = [validate_bbox(box, width, height, where=f"video frame {index}").tolist() for index, box in enumerate(values)]
    return {
        "unit_id": unit_id,
        "date": None,
        "method": "explicit_video",
        "session": path.stem,
        "camera": "camera_0",
        "input_kind": "explicit_video",
        "video_path": str(path),
        "fps": float(fps),
        "frame_count": frame_count,
        "available_frame_count": frame_count,
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "bbox_coordinate_space": "raw_video_pixels_xyxy",
        "bbox_source": "explicit per-frame target bbox sidecar",
        "frames": [
            {
                "frame_index": index,
                "frame": index,
                "timestamp_seconds": index / float(fps),
                "bbox_xyxy_px": box,
                "image_width": width,
                "image_height": height,
            }
            for index, box in enumerate(checked)
        ],
    }


__all__ = ["DatasetError", "SCHEMA_ID", "SCHEMA_VERSION", "build_manifest", "build_manifest_from_indexed_inventory", "explicit_video_unit", "file_sha256", "validate_bbox"]
