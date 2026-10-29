"""Persistent-GPU worker that publishes compact units."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from wham_baseline.dataset import load_manifest, manifest_units

from .config import DEFAULT_FPS
from .io import (
    RESULT_SCHEMA,
    SCHEMA_VERSION,
    STATE_SCHEMA,
    atomic_json,
    atomic_pickle,
    compact_unit_valid,
    now,
    sha256,
    short_unit_id,
    unit_root,
    validate_result_payload,
)
from .runtime import Engine, draw_overlay


def annotation_boxes(index_path: str | Path, unit: Mapping[str, Any]) -> list[np.ndarray]:
    camera = int(str(unit["camera"]).replace("camera_", "").replace("cam", ""))
    boxes: list[np.ndarray] = []
    with sqlite3.connect(f"file:{Path(index_path).resolve()}?mode=ro", uri=True) as connection:
        for frame in unit["frames"]:
            row = connection.execute(
                "SELECT bbox_x1,bbox_y1,bbox_x2,bbox_y2 FROM annotations "
                "WHERE date=? AND session=? AND camera=? AND frame=?",
                (str(unit["date"]), str(unit["session"]), camera, int(frame["frame"])),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"missing target bbox for {unit['unit_id']} frame {frame['frame']}"
                )
            boxes.append(np.asarray(row, dtype=np.float32))
    return boxes


def _video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    return writer


def run_unit(
    engine: Engine,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    *,
    result_root: str | Path,
    annotation_index: str | Path,
    config_hash: str,
    fps: float = DEFAULT_FPS,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(result_root).resolve()
    destination = unit_root(root, unit)
    if compact_unit_valid(root, unit, config_hash) and not force:
        return {"unit_id": unit["unit_id"], "status": "skipped", "frame_count": unit["frame_count"]}
    if destination.exists():
        if not force:
            raise RuntimeError(f"invalid existing output requires --force: {destination}")
        attempts = root / ".run" / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        os.replace(destination, attempts / f"{destination.name}-{int(time.time())}-{uuid.uuid4().hex[:6]}")

    stage = root / ".run" / "stage" / f"{destination.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    stage.mkdir(parents=True, exist_ok=False)
    started = now()
    started_clock = time.perf_counter()
    writer: cv2.VideoWriter | None = None
    try:
        boxes = annotation_boxes(annotation_index, unit)
        array_lists: dict[str, list[Any]] = {
            "frame": [], "bbox_xyxy": [], "global_orient": [], "body_pose": [],
            "betas": [], "camera_translation": [], "focal_length": [],
            "similarity_scale": [], "similarity_rotation": [],
            "similarity_translation": [], "smpl_joints_2d": [], "alignment_rmse": [],
        }
        dataset_root = Path(manifest["dataset_root"])
        width = height = None
        for index, (frame, bbox) in enumerate(zip(unit["frames"], boxes)):
            image_path = dataset_root / frame["relative_path"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot decode source image {image_path}")
            current_size = (image.shape[1], image.shape[0])
            if width is None:
                width, height = current_size
                writer = _video_writer(stage / "overlay.mp4", fps, current_size)
            elif current_size != (width, height):
                raise RuntimeError(f"source geometry changed within unit: {current_size} != {(width,height)}")
            output = engine.infer(image_path, bbox, current_size)
            array_lists["frame"].append(int(frame["frame"]))
            for name in array_lists:
                if name != "frame":
                    array_lists[name].append(output[name])
            writer.write(draw_overlay(image, output["smpl_joints_2d"], bbox, "Fast SAM3DB -> SMPL"))
            if (index + 1) % 25 == 0 or index + 1 == int(unit["frame_count"]):
                print(json.dumps({"unit": unit["unit_id"], "frames": index + 1,
                                  "total": unit["frame_count"]}), flush=True)
        if writer is None or width is None or height is None:
            raise RuntimeError("unit contains no frames")
        writer.release()
        writer = None
        arrays = {name: np.asarray(values) for name, values in array_lists.items()}
        arrays["frame"] = arrays["frame"].astype(np.int32)
        for name in arrays:
            if name != "frame":
                arrays[name] = arrays[name].astype(np.float32)
        result = {
            "schema_id": RESULT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "unit_id": unit["unit_id"],
            "config_hash": config_hash,
            "frame_count": int(unit["frame_count"]),
            "image_size": [int(width), int(height)],
            "fps": float(fps),
            "representation": {
                "model": "SMPL neutral",
                "global_orient": "axis-angle [F,3]",
                "body_pose": "axis-angle [F,21,3] flattened",
                "betas": "[F,10]",
                "projection": "per-frame MHR-camera similarity alignment",
            },
            "arrays": arrays,
        }
        validate_result_payload(result, int(unit["frame_count"]))
        atomic_pickle(stage / "result.pkl", result)
        capture = cv2.VideoCapture(str(stage / "overlay.mp4"))
        video_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        video_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        video_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if (video_frames, video_width, video_height) != (int(unit["frame_count"]), width, height):
            raise RuntimeError(
                f"overlay verification failed: {(video_frames,video_width,video_height)}"
            )
        state = {
            "schema_id": STATE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "terminal": True,
            "unit_id": unit["unit_id"],
            "short_unit_id": short_unit_id(unit),
            "config_hash": config_hash,
            "started_at": started,
            "completed_at": now(),
            "elapsed_seconds": round(time.perf_counter() - started_clock, 3),
            "frame_count": int(unit["frame_count"]),
            "fps": float(fps),
            "video": {"frames": video_frames, "width": video_width, "height": video_height,
                      "fps": video_fps},
            "result_sha256": sha256(stage / "result.pkl"),
            "overlay_sha256": sha256(stage / "overlay.mp4"),
        }
        atomic_json(stage / "state.json", state)
        if {path.name for path in stage.iterdir()} != {"result.pkl", "overlay.mp4", "state.json"}:
            raise RuntimeError("staged compact unit does not contain exactly three files")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
        if not compact_unit_valid(root, unit, config_hash):
            raise RuntimeError("published compact unit failed validation")
        return state
    except Exception as exc:
        if writer is not None:
            writer.release()
        failure = {
            "status": "failed", "terminal": True, "unit_id": unit.get("unit_id"),
            "at": now(), "error": f"{type(exc).__name__}: {exc}", "config_hash": config_hash,
        }
        atomic_json(root / ".run" / "failures" / f"{short_unit_id(unit)}.json", failure)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _parse_indices(value: str, count: int) -> list[int]:
    if value == "all":
        return list(range(count))
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(index < 0 or index >= count for index in result):
        raise ValueError("invalid unit indices")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--annotation-index", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    units = manifest_units(manifest)
    indices = _parse_indices(args.indices, len(units))
    engine = Engine()
    failures = 0
    try:
        for index in indices:
            try:
                run_unit(engine, manifest, units[index], result_root=args.result_root,
                         annotation_index=args.annotation_index, config_hash=args.config_hash,
                         fps=args.fps, force=args.force)
            except Exception as exc:
                failures += 1
                print(json.dumps({"unit": units[index]["unit_id"], "status": "failed",
                                  "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr, flush=True)
    finally:
        engine.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
