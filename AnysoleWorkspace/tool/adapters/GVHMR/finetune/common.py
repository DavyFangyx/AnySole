"""Shared data access for the GVHMR fine-tune path.

Every training item is tied to one package RGB unit (cam3 by default), the
canonical 40 Hz facts grid, and the date calibration.  The functions here are
deliberately dependency-light so label preparation and feature extraction can
run in the ``vb_gvhmr`` environment without importing ``anysole`` (which needs
``chumpy`` for the SMPL pickle).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
WORKSPACE = ROOT / "AnysoleWorkspace"
UPSTREAM = ROOT / "Baselines" / "GVHMR"
for entry in (str(ROOT), str(UPSTREAM)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

PROTOCOL = WORKSPACE / "protocol"
SPLITS_DEFAULT = PROTOCOL / "splits" / "default"
SESSION_MANIFEST = PROTOCOL / "manifests" / "session_manifest.jsonl"
UNIT_MANIFEST = (
    WORKSPACE
    / "work/visual_baselines_local/reference_results/3_Result/SAM3DB_baseline/.run/manifest.jsonl"
)
FRONTEND_BBOX = WORKSPACE / "shared/frontends/human_masks/sam31/v1"
FACTS = WORKSPACE / "shared/facts/sessions"
DEFAULT_OUTPUT = WORKSPACE / "work/gvhmr_finetune"

FOLD_SPLITS = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12.csv",
    "fS5S11S12": "splits_test_S5S11S12_val_S13.csv",
    "fS6S7": "splits_test_S6S7_val_S14.csv",
    "fS5S6S8": "splits_test_S5S6S8_val_S7.csv",
}
MODEL_FPS = 30.0
ALLOWED_GPUS = (1, 2, 3, 4)
_JPEG_RE = re.compile(r"^(\d+)_(\d{6}(?:\.\d+)?)\.jpe?g$", re.IGNORECASE)


class FinetuneDataError(ValueError):
    """Raised when an RGB unit, GT label or calibration violates the contract."""


def check_gpu(value: str | int) -> str:
    text = str(value).split(",")[0]
    if not text.isdigit() or int(text) not in ALLOWED_GPUS:
        raise ValueError(f"GPU must be one of {ALLOWED_GPUS}, got {value!r}")
    return text


def load_session_manifest() -> dict[str, dict]:
    rows = {}
    with SESSION_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["session_id"])] = row
    return rows


def load_units() -> dict[str, dict]:
    units = {}
    with UNIT_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            unit = json.loads(line)
            units[str(unit["unit_id"])] = unit
    return units


def load_unit(date: str, session: str, cam: int = 3) -> dict:
    unit_id = f"{date}/method_1_offset/{session}/camera_{cam}"
    units = load_units()
    if unit_id not in units:
        raise FinetuneDataError(f"unit absent from master manifest: {unit_id}")
    unit = units[unit_id]
    if unit.get("complete") not in (True, "true", 1):
        raise FinetuneDataError(f"unit is incomplete: {unit_id}")
    return unit


def fold_sessions(fold: str) -> dict[str, list[str]]:
    """Train/val/test session pools for one four-fold LSO split."""
    if fold not in FOLD_SPLITS:
        raise FinetuneDataError(f"unknown fold {fold!r}; expected one of {sorted(FOLD_SPLITS)}")
    path = SPLITS_DEFAULT / FOLD_SPLITS[fold]
    if not path.is_file():
        raise FinetuneDataError(f"fold split is missing: {path}")
    import csv

    values = {"train": [], "val": [], "test": []}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in values:
                value = (row.get(key) or "").strip()
                if value and value not in values[key]:
                    values[key].append(value)
    return values


def select_sessions(
    *,
    fold: str | None = None,
    split: str = "train,val",
    sessions: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Resolve a session list from fold/split or an explicit list."""
    if sessions:
        selected = [str(value).strip() for value in sessions if str(value).strip()]
    else:
        if fold is None:
            raise FinetuneDataError("either --fold or --sessions is required")
        pools = fold_sessions(fold)
        names = [name.strip() for name in split.split(",") if name.strip()]
        unknown = [name for name in names if name not in pools]
        if unknown:
            raise FinetuneDataError(f"unknown split(s) {unknown}")
        selected = []
        for name in names:
            selected.extend(pools[name])
    selected = sorted(dict.fromkeys(selected))
    if limit is not None:
        if limit <= 0:
            raise FinetuneDataError("limit must be positive")
        selected = selected[:limit]
    return selected


def resolve_uri(value: str | Path) -> Path:
    from AnysoleWorkspace.tool.workspace import resolve_uri as _resolve

    return _resolve(value)


def session_date_subject(row: Mapping[str, Any]) -> tuple[str, str]:
    recording = resolve_uri(row["video_path"])
    return str(recording.parts[-3]), str(recording.parts[-2])


def facts_frames(row: Mapping[str, Any], cam: int = 3) -> dict[str, np.ndarray]:
    date, subject = session_date_subject(row)
    path = FACTS / f"cam{cam}" / date / subject / str(row["session_id"]) / "frames.npz"
    if not path.is_file():
        raise FinetuneDataError(f"canonical facts frames are missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]).copy() for key in archive.files}
    n = len(data["frame_id"])
    if n != int(float(row["n_frames"])):
        raise FinetuneDataError(
            f"{row['session_id']}: facts length {n} != manifest n_frames {row['n_frames']}"
        )
    return data


def recording_start_day_s(row: Mapping[str, Any]) -> float:
    recording = resolve_uri(row["video_path"])
    meta = json.loads((recording / "meta.json").read_text(encoding="utf-8"))
    iso = meta.get("started_at_iso")
    if not iso:
        raise FinetuneDataError(f"{row['session_id']}: meta.json lacks started_at_iso")
    value = datetime.fromisoformat(str(iso))
    return value.hour * 3600.0 + value.minute * 60.0 + value.second + value.microsecond / 1e6


def _jpeg_wall_seconds(name: str) -> float:
    match = _JPEG_RE.match(name)
    if match is None:
        raise FinetuneDataError(f"unexpected RGB filename: {name}")
    clock = match.group(2)
    hour = int(clock[:2])
    minute = int(clock[2:4])
    second = float(clock[4:])
    return hour * 3600.0 + minute * 60.0 + second


def recording_frames(row: Mapping[str, Any], cam: int = 3) -> dict[float, Path]:
    recording = resolve_uri(row["video_path"])
    directory = recording / str(cam)
    frames = {}
    for path in sorted(directory.glob("*.jpg")):
        frames[_jpeg_wall_seconds(path.name)] = path
    if not frames:
        raise FinetuneDataError(f"no JPEG frames under {directory}")
    return frames


def unit_frame_paths(row: Mapping[str, Any], unit: Mapping[str, Any], cam: int = 3) -> list[Path]:
    """Recording JPEG for each unit frame, matched by wall-clock stamp."""
    available = recording_frames(row, cam)
    ordered = sorted(available)
    stamps = [float(record["timestamp_seconds"]) for record in unit["frames"]]
    paths: list[Path] = []
    for stamp in stamps:
        index = int(np.searchsorted(ordered, stamp))
        candidates = ordered[max(0, index - 1) : index + 2]
        if not candidates:
            raise FinetuneDataError(f"cannot match unit stamp {stamp} to a recording frame")
        best = min(candidates, key=lambda value: abs(value - stamp))
        if abs(best - stamp) > 0.02:
            raise FinetuneDataError(f"unit stamp {stamp} is {abs(best - stamp):.3f}s from any JPEG")
        paths.append(available[best])
    return paths


def canonical_indices(row: Mapping[str, Any], unit: Mapping[str, Any], cam: int = 3) -> np.ndarray:
    """Canonical facts index for each unit frame (nearest wall-clock stamp)."""
    frames = facts_frames(row, cam)
    visual = np.asarray(frames["visual_time_s"], dtype=np.float64)
    start = recording_start_day_s(row)
    stamps = np.asarray([float(record["timestamp_seconds"]) for record in unit["frames"]])
    rel = stamps - start
    index = np.searchsorted(visual, rel)
    index = np.clip(index, 1, len(visual) - 1)
    lower = visual[index - 1]
    upper = visual[index]
    nearest = np.where(np.abs(rel - lower) <= np.abs(rel - upper), index - 1, index)
    residual = np.abs(visual[nearest] - rel)
    if float(residual.max()) > 0.10:
        raise FinetuneDataError(
            f"{row['session_id']}: unit frame is {residual.max():.3f}s from the canonical grid"
        )
    return nearest.astype(np.int64)


def label_times(row: Mapping[str, Any], unit: Mapping[str, Any], cam: int = 3) -> np.ndarray:
    """Exact SMPL query time for each RGB frame, interpolated on the facts grid."""
    frames = facts_frames(row, cam)
    visual = np.asarray(frames["visual_time_s"], dtype=np.float64)
    mocap = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    start = recording_start_day_s(row)
    rel = np.asarray([float(record["timestamp_seconds"]) for record in unit["frames"]]) - start
    return np.interp(rel, visual, mocap).astype(np.float64)


def calibration(date: str) -> dict:
    path = PROTOCOL / "calibration" / f"{date}.json"
    if not path.is_file():
        raise FinetuneDataError(f"calibration is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def camera_transform(date: str, cam: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rigid transform from the mocap y-up GT world to GVHMR camera metres.

    ``camera_chain`` in the evaluation exporter maps model camera points into
    the mocap world through checkerboard world.  This is its inverse:

        p_cam = R_cam R_cal p_gt + (R_cam t_cal + t_cam) / 100

    The per-date ``*_native_to_mocap`` offsets are *not* applied: those absorb
    the pretrained model's depth bias and are refit at evaluation time for a
    fine-tuned checkpoint.
    """
    value = calibration(date)
    camera = value["cameras"][f"cam{cam}"]
    R_cam = np.asarray(camera["R"], dtype=np.float64)
    t_cam_cm = np.asarray(camera["t"], dtype=np.float64)
    source = value["mocap_raw_to_checkerboard_world"]
    R_cal = np.asarray(source["R"], dtype=np.float64)
    t_cal_cm = np.asarray(source["t"], dtype=np.float64)
    rotation = R_cam @ R_cal
    translation = (R_cam @ t_cal_cm + t_cam_cm) / 100.0
    return rotation, translation, value


def bboxes_for_unit(row: Mapping[str, Any], unit: Mapping[str, Any], cam: int = 3) -> np.ndarray:
    """Raw-image xyxy target box for every unit frame from the frontend cache."""
    date, subject = session_date_subject(row)
    path = FRONTEND_BBOX / date / subject / str(row["session_id"]) / "bbox.npz"
    if not path.is_file():
        raise FinetuneDataError(f"frontend target boxes are missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        visual = np.asarray(archive["visual_time_s"], dtype=np.float64)
        boxes = np.asarray(archive["bbox_xyxy_norm"], dtype=np.float64)
    start = recording_start_day_s(row)
    width, height = [int(value) for value in calibration(date)["image_size_px"]]
    result = []
    for record in unit["frames"]:
        stamp = float(record["timestamp_seconds"]) - start
        index = int(np.argmin(np.abs(visual - stamp)))
        if abs(visual[index] - stamp) > 0.10:
            raise FinetuneDataError(
                f"{row['session_id']}: bbox stamp {stamp:.3f}s is off the frontend grid"
            )
        box = boxes[index] * np.asarray([width, height, width, height], dtype=np.float64)
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise FinetuneDataError(f"{row['session_id']}: degenerate target box at frame {index}")
        result.append(box)
    return np.asarray(result, dtype=np.float32)


def resample_smpl(path: str | Path, query_t: np.ndarray) -> dict:
    """SMPL-24 archive resampled at query times (slerp rotations, linear trans)."""
    from scipy.spatial.transform import Rotation as SciRotation, Slerp

    with np.load(Path(path), allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    poses = np.asarray(data["poses"], dtype=np.float64).reshape(-1, 24, 3)
    trans = np.asarray(data["trans"], dtype=np.float64)
    betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)[:10]
    fps = float(np.asarray(data.get("mocap_frame_rate", 120.0)).reshape(-1)[0])
    if "source_frame_times_s" in data:
        source_t = np.asarray(data["source_frame_times_s"], dtype=np.float64).reshape(-1)
    else:
        source_t = np.arange(len(poses), dtype=np.float64) / fps
    if poses.shape[0] != trans.shape[0] or source_t.size != len(poses):
        raise FinetuneDataError(f"{path}: inconsistent SMPL archive lengths")
    if not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise FinetuneDataError(f"{path}: SMPL archive contains non-finite values")
    query = np.asarray(query_t, dtype=np.float64).reshape(-1)
    valid = (query >= source_t[0] - 1e-9) & (query <= source_t[-1] + 1e-9)
    clipped = np.clip(query, source_t[0], source_t[-1])
    rotations = np.empty((len(query), 24, 3), dtype=np.float64)
    for joint in range(24):
        rotations[:, joint] = Slerp(
            source_t, SciRotation.from_rotvec(poses[:, joint])
        )(clipped).as_rotvec()
    translation = np.stack(
        [np.interp(clipped, source_t, trans[:, axis]) for axis in range(3)], axis=1
    )
    poses_flat = rotations.reshape(len(query), 72)
    return {
        "poses": poses_flat.astype(np.float32),
        "trans": translation.astype(np.float32),
        "betas": betas.astype(np.float32),
        "query_valid": valid,
        "source_time_range_s": (float(source_t[0]), float(source_t[-1])),
        "source_fps": fps,
    }


def stage_video(frame_paths: list[Path], destination: Path, fps: float = MODEL_FPS) -> Path:
    """Materialise the exact frame stream consumed by the official extractors."""
    import subprocess

    destination.parent.mkdir(parents=True, exist_ok=True)
    frames_file = destination.with_suffix(".txt")
    frames_file.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in frame_paths), encoding="utf-8"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-r", f"{fps:g}", "-i", str(frames_file),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not destination.is_file():
        raise FinetuneDataError(
            f"cannot stage video {destination}: {completed.stderr[-800:]}"
        )
    return destination


def unit_frame_meta(unit: Mapping[str, Any], canonical_index: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "canonical_index": np.asarray(canonical_index, dtype=np.int64),
        "source_frame_number": np.asarray(
            [int(record["frame"]) for record in unit["frames"]], dtype=np.int64
        ),
        "timestamp_seconds": np.asarray(
            [float(record["timestamp_seconds"]) for record in unit["frames"]], dtype=np.float64
        ),
    }


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
