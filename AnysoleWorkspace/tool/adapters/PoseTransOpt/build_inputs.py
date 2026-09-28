#!/usr/bin/env python3
"""Materialise the PoseTransOpt model_inputs tree (adapter_v1) and the
frame-id join manifest.

Per session:

    model-input://PoseTransOpt/adapter_v1/<date>/<subject>/<session>/
    ├── color/                  symlinks for the final joined frames only
    ├── keypoints/              RTMPose HALPE-26 sidecars with frame ids
    ├── CLIFF_results.npz       produced by run_cliff.py (single person)
    ├── pred_contact_smpl/      symlinks to FPP-Net work predictions
    ├── template_scene_rgbd.npy first-valid-frame DepthPro RGB-D template
    └── join_manifest.json      the alignment authority (see below)

The join manifest is the single alignment authority for the three streams
(keypoints / CLIFF pose / FPP pred_contact_smpl): an inner join on shared
frame ids, with per-stream missing-frame lists and reasons, fake/invalid
drops, contiguous segment boundaries, and short segments (<21 frames, the
Savitzky-Golay minimum) dropped with reasons.  Nothing downstream zips
streams by array position.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

INPUT_ROOT = WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
FPP_PREDICTIONS = WORKSPACE / "work/VP-MoCap/v1/fpp_predictions"
ADAPTER_VERSION = "adapter_v1"
MIN_SEGMENT = 21  # >= savgol window 17 plus the 2-frame end trims
TRIM_END = 2


def read_manifest() -> dict[str, dict]:
    rows = {}
    for line in (WORKSPACE / "protocol/manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def split_sessions(name: str) -> list[str]:
    cols = ("train", "val", "test") if name == "all" else (name,)
    result = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in cols)
    return sorted(set(x for x in result if x))


def session_parts(row: dict) -> tuple[str, str, str]:
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2], row["session_id"]


def link_one(source: Path, target: Path, force: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(f"refusing to replace existing adapter file: {target}")
        target.unlink()
    target.symlink_to(os.path.relpath(source, target.parent))


def segments_of(frames: np.ndarray) -> list[list[int]]:
    """Contiguous runs of sorted frame ids as inclusive [start, end] pairs."""
    if frames.size == 0:
        return []
    diff = np.diff(frames)
    breaks = np.flatnonzero(diff != 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(frames) - 1]])
    return [[int(frames[s]), int(frames[e])] for s, e in zip(starts, ends)]


def index_segments(count: int, segments: list[list[int]]) -> list[list[int]]:
    """Frame-id inclusive segments -> index-space half-open [start, end)."""
    result = []
    cursor = 0
    for start, end in segments:
        length = end - start + 1
        result.append([cursor, cursor + length])
        cursor += length
    assert cursor == count
    return result


def build_join(row: dict, force: bool) -> dict:
    date, subject, sid = session_parts(row)
    n = int(row["n_frames"])
    session_root = INPUT_ROOT / date / subject / sid
    session_root.mkdir(parents=True, exist_ok=True)

    frames = np.load(FACTS_ROOT / date / subject / sid / "frames.npz")
    shared_valid = np.asarray(frames["valid"], dtype=np.uint8).astype(bool)
    shared_fake = np.asarray(frames["fake"], dtype=np.uint8).astype(bool)

    # stream 1: CLIFF (single-person, mask-bbox)
    cliff_path = session_root / "CLIFF_results.npz"
    if not cliff_path.is_file():
        raise FileNotFoundError(
            f"{sid}: CLIFF_results.npz missing; run adapters/PoseTransOpt/run_cliff.py first")
    cliff = np.load(cliff_path, allow_pickle=False)
    cliff_frame = np.asarray(cliff["frame_id"], dtype=np.int64)
    cliff_valid = np.asarray(cliff["valid"], dtype=np.uint8).astype(bool)
    if len(cliff_frame) != n or not np.array_equal(
            cliff_frame, np.arange(n, dtype=np.int64)):
        raise ValueError(f"{sid}: CLIFF npz violates the one-row-per-frame contract")

    # stream 2: RTMPose keypoints
    kp_dir = session_root / "keypoints"
    kp_present = np.zeros(n, dtype=bool)
    kp_bad_frame = []
    for i in range(n):
        path = kp_dir / f"{i:06d}.npy"
        if not path.is_file():
            continue
        payload = np.load(path, allow_pickle=True).item()
        if int(payload.get("frame_id", -1)) != i:
            kp_bad_frame.append(i)
            continue
        kp_present[i] = True

    # stream 3: FPP pred_contact_smpl
    fpp_src = FPP_PREDICTIONS / date / subject / sid / "pred_contact_smpl"
    fpp_dir = session_root / "pred_contact_smpl"
    fpp_present = np.zeros(n, dtype=bool)
    fpp_bad_frame = []
    for i in range(n):
        path = fpp_src / f"{i:06d}.npy"
        if not path.is_file():
            continue
        payload = np.load(path, allow_pickle=True).item()
        if int(payload.get("frame_id", -1)) != i:
            fpp_bad_frame.append(i)
            continue
        fpp_present[i] = True

    # inner join on frame id; then drop fake/invalid frames with reasons
    join = shared_valid & ~shared_fake & cliff_valid & kp_present & fpp_present
    missing = {
        "keypoints": {
            "count": int((~kp_present).sum()),
            "frames": np.flatnonzero(~kp_present)[:100].tolist(),
            "frame_id_mismatch": kp_bad_frame[:100],
            "reason": "sidecar missing or stored frame id != filename",
        },
        "cliff": {
            "count": int((~cliff_valid).sum()),
            "frames": np.flatnonzero(~cliff_valid)[:100].tolist(),
            "reason": "mask invalid or time error > 20ms (no multi-person fallback)",
        },
        "fpp_contact": {
            "count": int((~fpp_present).sum()),
            "frames": np.flatnonzero(~fpp_present)[:100].tolist(),
            "frame_id_mismatch": fpp_bad_frame[:100],
            "reason": "FPP-Net prediction sidecar missing or stored frame id != filename",
        },
    }
    dropped_fake_invalid = {
        "count": int((shared_fake | ~shared_valid).sum()),
        "frames": np.flatnonzero(shared_fake | ~shared_valid).tolist(),
        "reason": "shared facts fake=1 or valid=0",
    }
    joined_frames = np.flatnonzero(join)
    all_segments = segments_of(joined_frames)
    short = [seg for seg in all_segments if seg[1] - seg[0] + 1 < MIN_SEGMENT]
    kept = [seg for seg in all_segments if seg[1] - seg[0] + 1 >= MIN_SEGMENT]
    # short segments are removed from the join entirely (their frames go to
    # the dropped report), so the index-space segments below always cover
    # the final joined array
    joined_frames = np.concatenate(
        [np.arange(s, e + 1, dtype=np.int64) for s, e in kept]) if kept \
        else np.asarray([], dtype=np.int64)
    segments = index_segments(len(joined_frames), kept)
    final_frames = np.concatenate(
        [np.arange(s + TRIM_END, e + 1 - TRIM_END, dtype=np.int64)
         for s, e in kept]) if kept else np.asarray([], dtype=np.int64)

    # color links for the final frames only (visualizer order == optimized order)
    rgb_dir = FACTS_ROOT / date / subject / sid / "rgb"
    color_dir = session_root / "color"
    color_dir.mkdir(parents=True, exist_ok=True)
    for frame in final_frames.tolist():
        source = rgb_dir / f"{frame:06d}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"{sid}: RGB frame missing: {source}")
        link_one(source, color_dir / f"{frame:06d}.jpg", force)

    # FPP prediction links into the PoseTransOpt tree (own-tree copy of the
    # input; PoseTransOpt never reads the FPP model_inputs tree)
    fpp_dir.mkdir(parents=True, exist_ok=True)
    for frame in joined_frames.tolist():
        link_one(fpp_src / f"{frame:06d}.npy", fpp_dir / f"{frame:06d}.npy", force)

    # camera delivery contract (D6): per-date cam3 intrinsics from the
    # protocol calibration json; the scheduler overrides task.focal_length
    # with cam3_fx at runtime (the MMVP.yaml value is only a fallback)
    calibration_json = WORKSPACE / "protocol" / "calibration" / f"{date}.json"
    camera = {"source": str(calibration_json)}
    if calibration_json.is_file():
        cam3_k = json.loads(calibration_json.read_text(encoding="utf-8"))["cameras"]["cam3"]["K"]
        camera.update({"cam3_fx": float(cam3_k[0][0]), "cam3_fy": float(cam3_k[1][1])})
    else:
        camera["error"] = "protocol calibration json missing for this date"

    manifest = {
        "adapter_version": ADAPTER_VERSION,
        "session_id": sid, "date": date, "subject": subject,
        "camera": camera,
        "n_frames": n,
        "streams": {
            "keypoints": {"count": int(kp_present.sum()), "total": n},
            "cliff": {"count": int(cliff_valid.sum()), "total": n},
            "fpp_contact": {"count": int(fpp_present.sum()), "total": n},
        },
        "cliff_npz": str(cliff_path),
        "joined_count": int(len(joined_frames)),
        "joined_frames": joined_frames.tolist(),
        # index-space half-open segments within joined_frames (consumed by
        # the PoseTransOpt dataset)
        "segments": segments,
        # frame-id inclusive segments for reporting
        "segments_frames": kept,
        "final_count": int(len(final_frames)),
        "final_frames": final_frames.tolist(),
        "dropped": {
            "fake_or_invalid": dropped_fake_invalid,
            "missing_per_stream": missing,
            "short_segments": {"count": len(short), "segments": short,
                               "reason": f"segment shorter than {MIN_SEGMENT} frames "
                                         "(Savitzky-Golay minimum)"},
        },
        "trim": {"frames_per_segment_end": TRIM_END,
                 "note": "upstream [2:-2] crop generalized per contiguous segment"},
    }
    (session_root / "join_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = read_manifest()
    sessions = [s for s in args.sessions.split(",") if s] if args.sessions else split_sessions(args.split)
    reports = [build_join(rows[sid], args.force) for sid in sessions]
    summary = [
        {"session_id": r["session_id"], "joined": r["joined_count"],
         "final": r["final_count"], "n_frames": r["n_frames"]}
        for r in reports]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
