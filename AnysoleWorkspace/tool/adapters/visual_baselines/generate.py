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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WORKSPACE = ROOT / "AnysoleWorkspace"
NATIVE_ROOT = WORKSPACE / "work" / "visual_baselines" / "reference_results" / "3_Result"
FRONTEND_BBOX = WORKSPACE / "shared" / "frontends" / "human_masks" / "sam31" / "v1"
WEIGHTS_ROOT = WORKSPACE / "assets" / "third_party" / "visual_baselines" / "weights"
BASELINES = ROOT / "Baselines"
WHAM_PYTHON = "/data/fangyuxuan/miniconda3/envs/wham/bin/python"
WHAM_CHECKPOINTS = WEIGHTS_ROOT / "third_party" / "WHAM" / "checkpoints"
GENERATION_ENV = {
    "WHAM_ROOT": str(BASELINES / "WHAM"),
    # vendored ViTPose 自带整套 mmpose，必须先于 site-packages 遮蔽 pip 版
    # （否则 ViT backbone 未注册：KeyError: ViT is not in the models registry）。
    "PYTHONPATH": str(BASELINES / "WHAM" / "third-party" / "ViTPose") + ":"
                  + str(BASELINES / "WHAM"),
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
    # 镜像源侧 build_video_command（WHAM runner.py:210）：输入侧 -r 声明速率，
    # 不得用输出侧 fps 滤镜（会复制帧：60->72@30fps）。
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "concat", "-safe", "0", "-r", f"{fps:g}",
               "-i", str(frames_file.resolve()), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", str(video.resolve())]
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
        "--target_bboxes", str(sidecar),
        "--save_pkl",
        "--estimate_local_only",
    ]
    env = {**__import__("os").environ, **GENERATION_ENV,
           "CUDA_VISIBLE_DEVICES": str(gpu)}
    completed = subprocess.run(command, cwd=BASELINES / "WHAM", env=env, check=False,
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"WHAM demo failed with exit code {completed.returncode}:\n"
                           f"{completed.stderr[-1200:]}")
    # demo.py 在 output_pth 下再建视频名目录（sequence.mp4 -> sequence/）。
    raw = joblib.load(output / "sequence" / "wham_output.pkl")
    return compact_wham(raw, len(unit["frames"]), unit_root)


def compact_wham(raw_results: dict, frames: int, unit_root: Path) -> Path:
    """wham_output.pkl -> package result.pkl contract (io.load_wham).

    The demo already emits the track-0 record with all contract fields; the
    compact step validates the exact shapes the strict reader expects.
    """
    import joblib
    if not isinstance(raw_results, dict) or set(raw_results) != {0}:
        raise ValueError(f"unexpected WHAM output tracks: {set(raw_results)}")
    track = raw_results[0]
    required = {"pose", "trans", "betas", "verts", "frame_ids"}
    if not required.issubset(track):
        raise ValueError(f"WHAM track fields missing: {required - set(track)}")
    for name, shape in (("pose", (frames, 72)), ("trans", (frames, 3)),
                        ("betas", (frames, 10)), ("verts", (frames, 6890, 3))):
        value = np.asarray(track[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"WHAM compact: {name} shape {value.shape}")
    frame_ids = np.asarray(track["frame_ids"], dtype=np.int64)
    if not np.array_equal(frame_ids, np.arange(frames)):
        raise ValueError("WHAM frame_ids do not resolve exactly")
    result = unit_root / "result.pkl"
    joblib.dump({0: track}, result)
    return result


def generate_sam3db(row: dict, cam: int, unit: dict, work_dir: Path, gpu: int) -> Path:
    """Drive the SAM3DB worker in the vb_tactile environment (torch 2.10 stack)."""
    date, subject = session_parts(row)
    unit_root = work_dir / "sam3db" / f"{date}_{row['session_id']}_c{cam}"
    unit_root.mkdir(parents=True, exist_ok=True)
    command = [
        "/data/fangyuxuan/miniconda3/envs/vb_tactile/bin/python",
        str(Path(__file__).resolve().parent / "_sam3db_worker.py"),
        "--session", row["session_id"],
        "--cam", str(cam),
        "--gpu", str(gpu),
        "--work-dir", str(unit_root),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"SAM3DB worker failed with exit code {completed.returncode}:\n"
                           f"{completed.stderr[-1200:]}")
    return unit_root / "result.pkl"


def generate_gvhmr(row: dict, cam: int, unit: dict, work_dir: Path, gpu: int) -> Path:
    """Drive the migrated GVHMR worker (run_request) in the vb_gvhmr environment."""
    date, subject = session_parts(row)
    frames = unit_frame_paths(row, cam, unit)
    unit_root = work_dir / "gvhmr" / f"{date}_{row['session_id']}_c{cam}"
    unit_root.mkdir(parents=True, exist_ok=True)
    video = build_sequence_video(frames, unit_root)
    calibration, _, _ = common_loader().load_calibration(date)
    width, height = [int(v) for v in calibration["image_size_px"]]
    side = np.load(FRONTEND_BBOX / date / subject / row["session_id"] / "bbox.npz",
                   allow_pickle=False)
    visual = np.asarray(side["visual_time_s"], dtype=np.float64)
    boxes_norm = np.asarray(side["bbox_xyxy_norm"], dtype=np.float64)
    start = common_loader().recording_start_day_s(row)
    boxes_px = []
    for record in unit["frames"]:
        canonical = int(np.argmin(np.abs(visual - (float(record["timestamp_seconds"]) - start))))
        norm = boxes_norm[canonical]
        boxes_px.append([norm[0] * width, norm[1] * height,
                         norm[2] * width, norm[3] * height])
    import AnysoleWorkspace.tool.adapters.GVHMR.config as gvhmr_config
    request = {
        "work_dir": str(unit_root),
        "upstream_root": str(gvhmr_config.UPSTREAM_ROOT),
        "checkpoint_path": str(gvhmr_config.DEFAULT_CHECKPOINT),
        "smplx_model_path": str(gvhmr_config.DEFAULT_SMPLX_MODEL),
        "smpl_model_path": str(gvhmr_config.DEFAULT_SMPL_MODEL),
        "mapper_path": str(gvhmr_config.DEFAULT_SMPLX_TO_SMPL),
        "hmr2_checkpoint_path": str(gvhmr_config.DEFAULT_HMR2_CHECKPOINT),
        "vitpose_checkpoint_path": str(gvhmr_config.DEFAULT_VITPOSE_CHECKPOINT),
        "video_path": str(video),
        "frame_count": len(frames),
        "bbox_xyxy_px": boxes_px,
        "cuda_device": gpu,
    }
    request_file = unit_root / "request.json"
    request_file.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    output = unit_root / "result.npz"
    command = [
        "/data/fangyuxuan/miniconda3/envs/vb_gvhmr/bin/python",
        "-m", "GVHMR.worker",
        "--request", str(request_file),
        "--output", str(output),
    ]
    import os as _os
    env = {**_os.environ, "PYTHONPATH": str(ROOT / "AnysoleWorkspace/tool/adapters")}
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"GVHMR worker failed with exit code {completed.returncode}:\n"
                           f"{completed.stderr[-1200:]}")
    # 源侧 worker 的 result.npz 契约包含单元帧号与时间戳（保存预测同款 25 键）；
    # 本地 worker 写 21 键，驱动补上导出适配器严格读取所需的两键。
    with np.load(output, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive}
    payload["source_frame_numbers"] = np.asarray(
        [frame["frame"] for frame in unit["frames"]], dtype=np.int64)
    payload["physical_times_s"] = np.asarray(
        [frame["timestamp_seconds"] for frame in unit["frames"]], dtype=np.float64)
    np.savez(output, **payload)
    return output


def common_loader():
    from AnysoleWorkspace.tool.adapters.visual_baselines import common as _common
    return _common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("wham", "sam3db", "gvhmr"), default="wham")
    parser.add_argument("--session", required=True)
    parser.add_argument("--cam", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--work-dir", default="", help="persistent work dir (default: temp)")
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
    tmp = Path(args.work_dir) if args.work_dir else None
    if tmp is None:
        context = tempfile.TemporaryDirectory(dir=WORKSPACE / "work")
        tmp = Path(context.name)
    else:
        tmp.mkdir(parents=True, exist_ok=True)
        context = None
    try:
        if args.model == "wham":
            result = generate_wham(row, args.cam, unit, tmp, args.gpu)
        elif args.model == "sam3db":
            result = generate_sam3db(row, args.cam, unit, tmp, args.gpu)
        elif args.model == "gvhmr":
            result = generate_gvhmr(row, args.cam, unit, tmp, args.gpu)
        else:
            raise SystemExit(f"unknown model: {args.model}")
        print(f"wrote {result}")
    finally:
        if context is not None:
            context.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
