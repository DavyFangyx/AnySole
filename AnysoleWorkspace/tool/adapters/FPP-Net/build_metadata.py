#!/usr/bin/env python3
"""Build the FPP-Net split tree and subject pixel-weight metadata (V3 fix).

The upstream FPP-Net loader expects two legacy files:

* ``dataset_split_temporal5.npy`` with ``train``/``val``/``test`` nested by
  ``date/subject/session`` and frame names;
* one ``sub_info.npy`` per date containing the subject pressure weight.

Both are derived from the canonical manifest/split and the public MMVP 31x11
representation (``shared://representations/tactile/mmvp_31x11/v1``), which
FPP-Net also consumes at training time.  No private 31x11 copy is written.

V3 pixel-weight contract (2026-09-27): the weight is the sum over the 31x11
insole mask cells of the standing frame's mapped insole, i.e. the exact
domain ``InsoleModule.sigmoidNorm`` divides by ``pixel_num`` (the mask cell
count).  The retired 4x12 96-cell sum is not used.  ``sub_info`` records the
weight unit, computation domain, standing frame id, standing-frame source
hash and the sigmoid saturation/dynamic-range report of the standing frame.

Only valid five-frame windows enter the split tree: FPP-Net reads
``frame-2..frame+2``, so the first/last two frames and any window touching a
``valid=0`` or ``fake=1`` shared frame are excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
MANIFEST = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
SPLITS = WORKSPACE / "protocol/splits/default/splits.csv"
FPP_ROOT = WORKSPACE / "model_inputs/FPP-Net/adapter_v1"
TACTILE_ROOT = WORKSPACE / "shared/representations/tactile/mmvp_31x11/v1"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
ESSENTIALS = REPO_ROOT / "Baselines/VP-MoCap/FPP-Net/essentials/insole2cont"
ADAPTER_VERSION = "adapter_v1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Baselines.utils.motion_io import load_motion  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402


def read_manifest() -> dict[str, dict]:
    rows = {}
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["session_id"]] = row
    return rows


def read_split() -> dict[str, str]:
    result = {}
    with SPLITS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for split in ("train", "val", "test"):
                session = (row.get(split) or "").strip()
                if session:
                    result[session] = split
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_bvh(value: str) -> Path:
    if str(value).startswith("raw://"):
        return resolve_uri(value, must_exist=True)
    path = Path(value).expanduser()
    if path.is_file():
        return path
    normalized = path.as_posix()
    marker = "/mocap_ori_bvh/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        matches = sorted((WORKSPACE / "raw/bvh").glob("*/mocap_ori_bvh/" + suffix))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"BVH not found: {value}")


def session_dir(row: dict) -> Path:
    recording = resolve_uri(row["video_path"], must_exist=True)
    date, subject = recording.parts[-3], recording.parts[-2]
    return FACTS_ROOT / date / subject / row["session_id"]


def session_date(row: dict) -> str:
    return resolve_uri(row["video_path"], must_exist=True).parts[-3]


def shared_flags(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """frame_id / valid / fake arrays from the shared facts session."""
    frames = np.load(session_dir(row) / "frames.npz")
    frame_id = np.asarray(frames["frame_id"], dtype=np.int64)
    valid = np.asarray(frames["valid"], dtype=np.uint8).astype(bool)
    fake = np.asarray(frames["fake"], dtype=np.uint8).astype(bool)
    if not np.array_equal(frame_id, np.arange(len(frame_id))):
        raise ValueError(f"non-contiguous frame ids for {row['session_id']}")
    return frame_id, valid, fake


def aligned_bvh_speed(row: dict, n: int) -> np.ndarray:
    bvh = resolve_bvh(row["bvh_path"])
    fps = float(row.get("target_fps") or 40.0)
    query_t = (
        float(row.get("visual_start_s") or 0.0)
        + np.arange(n, dtype=np.float64) / fps
        - float(row.get("offset_s") or 0.0)
    )
    joints = np.asarray(load_motion(bvh, query_t=query_t)["joints"], dtype=np.float32)
    if joints.shape[0] != n:
        raise ValueError(f"BVH alignment length mismatch for {row['session_id']}: {joints.shape} vs {n}")
    speed = np.zeros(n, dtype=np.float32)
    if n > 1:
        speed[1:] = np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(axis=1) * fps
        speed[0] = speed[1]
    return speed


def insole_mask() -> tuple[np.ndarray, np.ndarray]:
    mask_l = np.loadtxt(ESSENTIALS / "insoleMaskL.txt").astype(np.int32)
    mask_r = np.loadtxt(ESSENTIALS / "insoleMaskR.txt").astype(np.int32)
    return mask_l, mask_r


def choose_standing(row: dict, mask_l: np.ndarray, mask_r: np.ndarray) -> dict:
    """Select the standing frame and compute the V3 pixel weight.

    The weight is the sum over the mapped 31x11 insole mask cells (both
    feet) at the standing frame -- the same domain FPP-Net's sigmoidNorm
    divides by ``pixel_num``.
    """
    sid = row["session_id"]
    date = session_date(row)
    _, valid, fake = shared_flags(row)
    n = len(valid)
    speed = aligned_bvh_speed(row, n)
    candidates = np.flatnonzero(valid & ~fake)
    if candidates.size == 0:
        raise ValueError(f"no valid double-foot-contact frame for {sid}")
    frame = int(candidates[np.argmin(speed[candidates])])
    insole = np.load(
        TACTILE_ROOT / date / row["subject_id"] / sid / "insole"
        / f"{frame:06d}.npy")
    if insole.shape != (2, 31, 11):
        raise ValueError(f"bad public insole shape for {sid}: {insole.shape}")
    left_total = float(np.sum(insole[0][mask_l > 0]))
    right_total = float(np.sum(insole[1][mask_r > 0]))
    weight = left_total + right_total
    pixel_num = int(np.sum(mask_l > 0) + np.sum(mask_r > 0))
    mean_press = weight / pixel_num
    # sigmoid input / saturation report over the masked cells of the
    # standing frame (the exact normalisation FPP-Net applies at train time)
    masked = np.concatenate(
        [insole[0][mask_l > 0], insole[1][mask_r > 0]]).astype(np.float64)
    ratio = (masked - mean_press) / mean_press
    sigmoid = 1.0 / (1.0 + np.exp(-ratio))
    report = {
        "session_id": sid,
        "standing_frame": frame,
        "standing_frame_visual_time_s": float(frame / float(row.get("target_fps") or 40.0)),
        "speed_mps": float(speed[frame]),
        "candidate_count": int(candidates.size),
        "weight": weight,
        "pixel_num": pixel_num,
        "mean_press_per_pixel": mean_press,
        "left_total": left_total,
        "right_total": right_total,
        "sigmoid_input_min": float(ratio.min()),
        "sigmoid_input_median": float(np.median(ratio)),
        "sigmoid_input_max": float(ratio.max()),
        "sigmoid_saturation_gt095": float((sigmoid > 0.95).mean()),
        "sigmoid_saturation_lt005": float((sigmoid < 0.05).mean()),
        "sigmoid_dynamic_range_10_90": float(
            np.percentile(sigmoid, 90) - np.percentile(sigmoid, 10)),
    }
    return report


def sequence_frames(row: dict, window: int = 5) -> list[str]:
    _, valid, fake = shared_flags(row)
    n = len(valid)
    keep = valid & ~fake
    half = window // 2
    frames = []
    for frame in range(half, n - half):
        if all(keep[frame - half:frame + half + 1]):
            frames.append(f"{frame:06d}.npy")
    return frames


def build_metadata(force: bool = False) -> dict:
    manifest = read_manifest()
    split_of = read_split()
    selected = [row for sid, row in manifest.items() if sid in split_of]
    if not selected:
        raise RuntimeError("canonical split selected no manifest rows")

    mask_l, mask_r = insole_mask()

    # First action/trial per subject is the only source allowed to define the
    # subject pressure scale; later actions are not used to tune it.
    first_by_subject: dict[str, dict] = {}
    for row in sorted(selected, key=lambda r: (r["subject_id"], r.get("action", ""), r.get("trial", ""), r["session_id"])):
        first_by_subject.setdefault(row["subject_id"], row)

    sub_info_by_date: dict[str, dict] = defaultdict(dict)
    standing_reports = {}
    no_temporal_window_sessions = []
    for subject, row in sorted(first_by_subject.items()):
        report = choose_standing(row, mask_l, mask_r)
        date = session_date(row)
        sub_info_by_date[date][subject] = {
            "weight": report["weight"],
            "max_value": 255.0,
            "height": -1.0,
            # V3 provenance: unit, computation domain, standing frame id,
            # source hash of the standing-frame insole file.
            "weight_unit": "sum over mapped 31x11 insole mask cells "
                           "(both feet); divided by pixel_num inside "
                           "InsoleModule.sigmoidNorm",
            "weight_domain": "shared mmvp_31x11 insole mask; "
                             f"pixel_num={report['pixel_num']}",
            "standing_frame_id": report["standing_frame"],
            "source_hash": sha256_file(
                TACTILE_ROOT / date / row["subject_id"]
                / row["session_id"] / "insole"
                / f"{report['standing_frame']:06d}.npy"),
            "adapter_version": ADAPTER_VERSION,
        }
        standing_reports[subject] = {**report, "date": date}

    split_tree: dict[str, dict] = {
        "train": defaultdict(lambda: defaultdict(dict)),
        "val": defaultdict(lambda: defaultdict(dict)),
        "test": defaultdict(lambda: defaultdict(dict)),
    }
    for row in sorted(selected, key=lambda r: r["session_id"]):
        split = split_of[row["session_id"]]
        date = session_date(row)
        frames = sequence_frames(row)
        if not frames:
            no_temporal_window_sessions.append(row["session_id"])
        split_tree[split][date][row["subject_id"]][row["session_id"]] = frames
    # the frozen default split keeps val == test as identical session sets
    # (protocol/splits/default/splits.csv); mirror test into val so the
    # dataset's val phase addresses the same 36 sessions.
    split_tree["val"] = split_tree["test"]

    FPP_ROOT.mkdir(parents=True, exist_ok=True)
    sub_info_paths = {}
    for date, info in sub_info_by_date.items():
        out = FPP_ROOT / date / "sub_info.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not force:
            old = np.load(out, allow_pickle=True).item()
            if old != info:
                raise FileExistsError(f"refusing to replace existing sub_info: {out}; use --force")
        else:
            np.save(out, info)
        sub_info_paths[date] = str(out)

    tree = {
        "train": {date: {sub: dict(seqs) for sub, seqs in subjects.items()} for date, subjects in split_tree["train"].items()},
        "val": {date: {sub: dict(seqs) for sub, seqs in subjects.items()} for date, subjects in split_tree["val"].items()},
        "test": {date: {sub: dict(seqs) for sub, seqs in subjects.items()} for date, subjects in split_tree["test"].items()},
    }
    split_path = FPP_ROOT / "dataset_split_temporal5.npy"
    if split_path.exists() and not force:
        old = np.load(split_path, allow_pickle=True).item()
        if old != tree:
            raise FileExistsError(f"refusing to replace existing tv_fn: {split_path}; use --force")
    else:
        np.save(split_path, tree)

    report = {
        "adapter_version": ADAPTER_VERSION,
        "split_path": str(split_path),
        "sub_info_paths": sub_info_paths,
        "session_counts": {phase: sum(len(seqs) for subjects in dates.values() for seqs in subjects.values()) for phase, dates in tree.items()},
        "frame_counts": {phase: sum(len(frames) for subjects in dates.values() for seqs in subjects.values() for frames in seqs.values()) for phase, dates in tree.items()},
        "no_temporal5_window_sessions": sorted(no_temporal_window_sessions),
        "standing": standing_reports,
        "weight_unit": "raw shared mmvp 31x11 insole values summed over the "
                       "insole mask cells; no kg conversion",
        "sigmoid_saturation_summary": {
            "subjects_gt095_over_10pct": sum(
                1 for r in standing_reports.values()
                if r["sigmoid_saturation_gt095"] > 0.10),
            "subjects_lt005_over_10pct": sum(
                1 for r in standing_reports.values()
                if r["sigmoid_saturation_lt005"] > 0.10),
        },
    }
    report_path = FPP_ROOT / "fpp_metadata_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace generated metadata")
    args = parser.parse_args()
    report = build_metadata(force=args.force)
    print(json.dumps({k: report[k] for k in (
        "split_path", "sub_info_paths", "session_counts", "frame_counts",
        "no_temporal5_window_sessions", "sigmoid_saturation_summary")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
