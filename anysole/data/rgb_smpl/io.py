"""Strict readers for the fixed local evaluation corpus."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Iterator

import joblib
import numpy as np

from .alignment import map_rgb_to_mocap


DATE_TO_MOCAP = {
    "20260804": "0804smpl",
    "20260807": "0807smpl",
    "20260808": "0808smpl",
    "20260810": "0810smpl",
}


@dataclass(frozen=True)
class Unit:
    unit_id: str
    date: str
    method: str
    session: str
    camera: str
    camera_index: int
    frame_numbers: np.ndarray
    physical_times: np.ndarray

    @property
    def short_id(self) -> str:
        method = {"method_1_offset": "m1", "method_2_projection": "m2"}[self.method]
        return f"{self.date}_{method}_{self.session}_c{self.camera_index}"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gt_inventory(mocap_root: str | Path) -> dict[tuple[str, str], Path]:
    root = Path(mocap_root)
    inventory: dict[tuple[str, str], Path] = {}
    for date, directory in DATE_TO_MOCAP.items():
        for path in sorted((root / directory / "mocap_ori_c3d").glob("S*/motion_neutral_smpl.npz")):
            key = date, path.parent.name
            if key in inventory:
                raise ValueError(f"duplicate GT' session: {key}")
            inventory[key] = path
    return inventory


def _unit_from_json(value: dict[str, Any]) -> Unit:
    frames = value.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"unit has no frames: {value.get('unit_id')}")
    numbers = np.asarray([frame["frame"] for frame in frames], dtype=np.int64)
    times = np.asarray([frame["timestamp_seconds"] for frame in frames], dtype=np.float64)
    if numbers.size != int(value["frame_count"]) or np.any(np.diff(numbers) <= 0):
        raise ValueError(f"invalid frame inventory: {value['unit_id']}")
    camera = str(value["camera"])
    camera_index = int(camera.removeprefix("camera_"))
    return Unit(
        unit_id=str(value["unit_id"]),
        date=str(value["date"]),
        method=str(value["method"]),
        session=str(value["session"]),
        camera=camera,
        camera_index=camera_index,
        frame_numbers=numbers,
        physical_times=times,
    )


def matched_units(manifest: str | Path, gt_keys: set[tuple[str, str]]) -> Iterator[Unit]:
    seen: set[str] = set()
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if (str(value["date"]), str(value["session"])) not in gt_keys:
                continue
            unit = _unit_from_json(value)
            if unit.unit_id in seen:
                raise ValueError(f"duplicate matched unit: {unit.unit_id}")
            seen.add(unit.unit_id)
            yield unit


def alignment_rows(dataset_root: str | Path, date: str, session: str) -> tuple[dict[str, str], dict[str, str]]:
    path = Path(dataset_root) / date / "RGB_MoCap_time_alignment" / f"{session}.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2 or rows[0].get("标记") != "start" or rows[1].get("标记") != "end":
        raise ValueError(f"alignment CSV must contain start/end rows: {path}")
    return rows[0], rows[1]


def unit_mocap_times(unit: Unit, dataset_root: str | Path) -> np.ndarray:
    start, end = alignment_rows(dataset_root, unit.date, unit.session)
    return map_rgb_to_mocap(
        unit.frame_numbers,
        unit.physical_times,
        visual_start_frame=int(start["视觉帧"]),
        visual_end_frame=int(end["视觉帧"]),
        mocap_start_time=float(start["动捕时间(s)"]),
        mocap_end_time=float(end["动捕时间(s)"]),
    )


def load_gt(path: str | Path) -> dict[str, np.ndarray]:
    archive = np.load(Path(path), allow_pickle=False)
    required = {
        "poses": (None, 72),
        "trans": (None, 3),
        "betas": (10,),
        "source_frame_times_s": (None,),
        "source_frame_numbers": (None,),
    }
    result: dict[str, np.ndarray] = {}
    for name, expected in required.items():
        value = np.asarray(archive[name])
        if len(expected) != value.ndim or any(want is not None and got != want for got, want in zip(value.shape, expected)):
            raise ValueError(f"GT' {name} shape {value.shape} does not match {expected}: {path}")
        if not np.isfinite(value).all():
            raise ValueError(f"GT' {name} contains non-finite values: {path}")
        result[name] = value
    count = result["poses"].shape[0]
    if result["trans"].shape[0] != count or result["source_frame_times_s"].shape[0] != count:
        raise ValueError(f"GT' sequence lengths differ: {path}")
    return result


def repaired_frames(gt_path: str | Path) -> set[int]:
    quality_path = Path(gt_path).with_name("motion_quality.json")
    value = json.loads(quality_path.read_text(encoding="utf-8"))
    intervals = value.get("joint_continuity", {}).get("repaired_intervals", [])
    repaired: set[int] = set()
    for interval in intervals:
        repaired.update(int(index) for index in interval.get("array_frames", []))
    return repaired


def load_wham(path: str | Path, unit: Unit) -> dict[str, np.ndarray]:
    value = joblib.load(Path(path))
    if not isinstance(value, dict) or set(value) != {0}:
        raise ValueError(f"WHAM result must contain exactly track 0: {path}")
    track = value[0]
    required = {"pose", "trans", "betas", "verts", "frame_ids"}
    if not required.issubset(track):
        raise ValueError(f"WHAM result fields missing: {required - set(track)}")
    frame_ids = np.asarray(track["frame_ids"], dtype=np.int64)
    if not np.array_equal(frame_ids, np.arange(unit.frame_numbers.size)):
        raise ValueError(f"WHAM frame_ids do not resolve exactly through manifest: {unit.unit_id}")
    result = {name: np.asarray(track[name]) for name in required}
    expected = {
        "pose": (frame_ids.size, 72),
        "trans": (frame_ids.size, 3),
        "betas": (frame_ids.size, 10),
        "verts": (frame_ids.size, 6890, 3),
    }
    for name, shape in expected.items():
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise ValueError(f"invalid WHAM {name} for {unit.unit_id}: {result[name].shape}")
    return result


def load_sam(path: str | Path, unit: Unit) -> dict[str, np.ndarray]:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    arrays = value.get("arrays") if isinstance(value, dict) else None
    if not isinstance(arrays, dict):
        raise ValueError(f"SAM3DB result has no arrays mapping: {path}")
    required = {
        "frame", "bbox_xyxy", "global_orient", "body_pose", "betas", "camera_translation",
        "focal_length",
        "similarity_scale", "similarity_rotation", "similarity_translation",
        "smpl_joints_2d",
    }
    if not required.issubset(arrays):
        raise ValueError(f"SAM3DB result fields missing: {required - set(arrays)}")
    result = {name: np.asarray(arrays[name]) for name in required}
    if not np.array_equal(result["frame"].astype(np.int64), unit.frame_numbers):
        raise ValueError(f"SAM3DB source frames differ from manifest: {unit.unit_id}")
    expected = {
        "bbox_xyxy": (unit.frame_numbers.size, 4),
        "global_orient": (unit.frame_numbers.size, 3),
        "body_pose": (unit.frame_numbers.size, 63),
        "betas": (unit.frame_numbers.size, 10),
        "camera_translation": (unit.frame_numbers.size, 3),
        "focal_length": (unit.frame_numbers.size,),
        "similarity_scale": (unit.frame_numbers.size,),
        "similarity_rotation": (unit.frame_numbers.size, 3, 3),
        "similarity_translation": (unit.frame_numbers.size, 3),
        "smpl_joints_2d": (unit.frame_numbers.size, 24, 2),
    }
    for name, shape in expected.items():
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise ValueError(f"invalid SAM3DB {name} for {unit.unit_id}: {result[name].shape}")
    image_size = np.asarray(value.get("image_size"), dtype=np.int64)
    if image_size.shape != (2,) or np.any(image_size <= 0):
        raise ValueError(f"invalid SAM3DB image size for {unit.unit_id}: {image_size}")
    result["image_size"] = image_size
    return result


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
