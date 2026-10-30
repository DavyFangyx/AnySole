#!/usr/bin/env python3
"""Project-native generation driver for WHAM / SAM3DB / GVHMR predictions.

Runs the migrated upstream code (Baselines/<Model>/) against the project's own
inputs — RGB recordings under the raw farm, target boxes from the workspace
human-masks frontend (bbox.npz, direct_csv_bbox) — and writes per-unit
``result.pkl``/``result.npz`` in the package's saved-prediction contract so the
existing export adapter consumes them unchanged (decision archive 11).

Nothing here reads or imports the transfer package; after consolidation the
package itself can be deleted.

Usage (touch_gait):
  python AnysoleWorkspace/tool/adapters/visual_baselines/generate.py \
      --model wham --session S13011 [--cam 3] [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = ROOT / "AnysoleWorkspace"
NATIVE_ROOT = WORKSPACE / "work" / "visual_baselines" / "reference_results" / "3_Result"
FRONTEND_BBOX = WORKSPACE / "shared" / "frontends" / "human_masks" / "sam31" / "v1"
WEIGHTS_ROOT = WORKSPACE / "assets" / "third_party" / "visual_baselines" / "weights"
BASELINES = ROOT / "Baselines"
WHAM_PYTHON = "/data/fangyuxuan/miniconda3/envs/wham/bin/python"
WHAM_CHECKPOINTS = WEIGHTS_ROOT / "third_party" / "WHAM" / "checkpoints"
GENERATION_ENV = {
    "WHAM_ROOT": str(BASELINES / "WHAM"),
    "PYTHONPATH": str(BASELINES / "WHAM"),
}
SCHEMA_ID = "target_bboxes_v1"
SCHEMA_VERSION = 1


def session_parts(row: dict) -> tuple[str, str]:
    from AnysoleWorkspace.tool.workspace import resolve_uri
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2]


def unit_entry(unit_manifest: Path, date: str, session: str, cam: int) -> dict:
    unit_id = f"{date}/method_1_offset/{session}/camera_{cam}"
    with unit_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value.get("unit_id") == unit_id:
                return value
    raise FileNotFoundError(f"unit not in manifest: {unit_id}")


def recording_frames(row: dict, cam: int) -> list[Path]:
    """All JPEG frames of the manifest recording for one camera, sorted."""
    from AnysoleWorkspace.tool.workspace import resolve_uri
    recording = resolve_uri(row["video_path"], must_exist=True)
    camera_dir = recording / str(cam)
    frames = sorted(camera_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no jpegs under {camera_dir}")
    return frames


def unit_frame_paths(row: dict, cam: int, unit: dict) -> list[Path]:
    """Frame files for one unit, by source frame number.

    Package frame numbers are 1-based into the recording (frame1_... is the
    recording's second JPEG in practice); the wall-clock stamps in the unit
    manifest disambiguate the exact offset, so match by stamp: each unit frame
    picks the nearest recording JPEG by its filename wall-clock.
    """
    frames = recording_frames(row, cam)
    stamps = {_jpeg_wall_seconds(path.name): path for path in frames}
    ordered = sorted(stamps)
    unit_stamps = [float(record["timestamp_seconds"]) for record in unit["frames"]]
    unit_paths = []
    for stamp in unit_stamps:
        index = int(np.searchsorted(ordered, stamp))
        best = min(ordered[max(0, index - 1):index + 2],
                   key=lambda value: abs(value - stamp))
        unit_paths.append(stamps[best])
    return unit_paths


def _jpeg_wall_seconds(name: str) -> float:
    """Seconds-of-day from a recording JPEG name like ``3_205513.956.jpg``."""
    clock = name.split("_")[-1].split(".")[0]
    hour = int(clock[:2]); minute = int(clock[2:4]); second = float(clock[4:])
    return hour * 3600.0 + minute * 60.0 + second


def bbox_sidecar(row: dict, cam: int, unit: dict, unit_root: Path,
                 image_size: tuple[int, int]) -> Path:
    """Target-box sidecar for the unit's frames from the workspace frontend.

    The frontend bbox.npz is on the canonical (cam3) frame grid; unit frames
    are mapped to it by wall clock via the recording start (same anchor as the
    export adapter).  Boxes are normalized -> raw distorted pixels.
    """
    from AnysoleWorkspace.tool.adapters.visual_baselines import common
    date, subject = session_parts(row)
    side = np.load(FRONTEND_BBOX / date / subject / row["session_id"] / "bbox.npz",
                   allow_pickle=False)
    visual = np.asarray(side["visual_time_s"], dtype=np.float64)
    boxes_norm = np.asarray(side["bbox_xyxy_norm"], dtype=np.float64)
    width, height = image_size
    start = common.recording_start_day_s(row)
    frames = []
    for video_index, record in enumerate(unit["frames"]):
        stamp = float(record["timestamp_seconds"]) - start
        canonical = int(np.argmin(np.abs(visual - stamp)))
        norm = boxes_norm[canonical]
        px = np.asarray([norm[0] * width, norm[1] * height,
                         norm[2] * width, norm[3] * height], dtype=float)
        if px[2] <= px[0] or px[3] <= px[1]:
            raise ValueError(f"degenerate bbox at unit frame {record['frame']}")
        frames.append({
            "video_frame_index": video_index,
            "source_frame": int(record["frame"]),
            "source_filename": record.get("filename"),
            "bbox_xyxy_px": [round(float(value), 4) for value in px],
            "segmentation_confidence": None,
            "mask_relpath": None,
        })
    payload = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "unit_id": unit["unit_id"],
        "date": date,
        "session": row["session_id"],
        "camera": f"camera_{cam}",
        "coordinate_space": "raw_distorted_pixels_xyxy",
        "frame_count": len(frames),
        "index": {"path": str(FRONTEND_BBOX), "sha256": "", "metadata": {}},
        "frames": frames,
    }
    destination = unit_root / "target_bboxes.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return destination


def build_sequence_video(frame_paths: list[Path], unit_root: Path,
                         fps: float = 30.0) -> Path:
    """Materialise the unit's frames as an ffmpeg CFR video (source-side shape)."""
    frames_file = unit_root / "frames.txt"
    frames_file.write_text("".join(f"file '{path}'\n" for path in frame_paths),
                           encoding="utf-8")
    video = unit_root / "sequence.mp4"
    command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(frames_file),
               "-vf", f"fps={fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               str(video)]
    completed = subprocess.run(command, check=False,
                               capture_output=True, text=True)
    if completed.returncode != 0 or not video.is_file():
        raise RuntimeError(f"video preparation failed: {completed.stderr[-400:]}")
    return video


def generate_wham(row: dict, cam: int, unit: dict, work_dir: Path, gpu: int) -> Path:
    """Drive Baselines/WHAM/demo.py and compact to the saved-prediction contract."""
    import joblib
    date, subject = session_parts(row)
    frames = unit_frame_paths(row, cam, unit)
    unit_root = work_dir / "wham" / f"{date}_{row['session_id']}_c{cam}"
    unit_root.mkdir(parents=True, exist_ok=True)
    video = build_sequence_video(frames, unit_root)
    calibration, _, _ = __import__(
        "AnysoleWorkspace.tool.adapters.visual_baselines.common",
        fromlist=["load_calibration"]).load_calibration(date)
    width, height = calibration["image_size_px"]
    sidecar = bbox_sidecar(row, cam, unit, unit_root, (int(width), int(height)))
    output = unit_root / "wham_output"
    command = [
        WHAM_PYTHON, str(BASELINES / "WHAM" / "demo.py"),
        "--video", str(video),
        "--output_pth", str(output),
        "--target-bboxes", str(sidecar),
        "--save_pkl",
        "--estimate_local_only",
    ]
    env = {**__import__("os").environ, **GENERATION_ENV,
           "CUDA_VISIBLE_DEVICES": str(gpu)}
    completed = subprocess.run(command, cwd=BASELINES / "WHAM", env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"WHAM demo failed with exit code {completed.returncode}")
    raw = joblib.load(output / "wham_output.pkl")
    return compact_wham(raw, len(unit["frames"]), unit_root)


def compact_wham(raw_results: dict, frames: int, unit_root: Path) -> Path:
    """wham_output.pkl -> package result.pkl contract (io.load_wham)."""
    import joblib
    # WHAM demo results: per-frame records; the exact key layout is inspected
    # and validated at smoke time — assert the contract instead of guessing.
    if not isinstance(raw_results, dict) or len(raw_results) != frames:
        raise ValueError(f"unexpected WHAM output shape: {len(raw_results)} frames")
    track = {}
    for name in ("pose", "trans", "betas", "verts"):
        values = [raw_results[index][name] for index in range(frames)]
        track[name] = np.asarray(values)
    track["frame_ids"] = np.arange(frames, dtype=np.int64)
    for name, shape in (("pose", (frames, 72)), ("trans", (frames, 3)),
                        ("betas", (frames, 10)), ("verts", (frames, 6890, 3))):
        if track[name].shape != shape:
            raise ValueError(f"WHAM compact: {name} shape {track[name].shape}")
    result = unit_root / "result.pkl"
    joblib.dump({0: track}, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("wham", "sam3db", "gvhmr"), default="wham")
    parser.add_argument("--session", required=True)
    parser.add_argument("--cam", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    manifest = {}
    for line in (WORKSPACE / "protocol/manifests" / "session_manifest.jsonl").open(
            encoding="utf-8"):
        row = json.loads(line)
        manifest[row["session_id"]] = row
    if args.session not in manifest:
        raise SystemExit(f"session not in manifest: {args.session}")
    row = manifest[args.session]
    date, subject = session_parts(row)
    unit_manifest = NATIVE_ROOT / "SAM3DB_baseline" / ".run" / "manifest.jsonl"
    unit = unit_entry(unit_manifest, date, args.session, args.cam)
    with tempfile.TemporaryDirectory(dir=WORKSPACE / "work") as tmp:
        result = generate_wham(row, args.cam, unit, Path(tmp), args.gpu)
        print(f"wrote {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
