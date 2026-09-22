#!/usr/bin/env python3
"""Build the split and subject metadata consumed by MMVP FPP-Net.

The upstream FPP-Net loader expects two legacy files:

* ``dataset_split_temporal5.npy`` with ``train``/``val``/``test`` nested by
  ``date/subject/session`` and frame names;
* one ``sub_info.npy`` per date containing the subject pressure weight.

This adapter derives both from the canonical manifest/split and the already
materialised 31x11 insole files.  It deliberately uses only valid five-frame
windows: FPP-Net reads ``frame-2..frame+2`` and therefore the first/last two
frames and any window touching a fake frame are excluded.

The standing frame is selected from a real BVH-aligned sequence frame where
both feet are in contact and the aligned BVH joint speed is minimal.  No 3-D
keypoints are written as an observation by this script; they are used only for
the FPP-Net subject calibration metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
MANIFEST = WORKSPACE / "manifests/session_manifest.jsonl"
SPLITS = WORKSPACE / "splits/default/splits.csv"
FPP_ROOT = WORKSPACE / "derived/VP-MoCap"
META_ROOT = WORKSPACE / "derived/baseline_tactile"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from results_display.script.utils.motion_io import load_motion  # noqa: E402


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


def resolve_bvh(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path
    normalized = path.as_posix()
    marker = "/mocap_ori_bvh/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        matches = sorted((WORKSPACE / "sources/raw").glob("*/mocap_ori_bvh/" + suffix))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"BVH not found: {value}")


def session_dir(row: dict) -> Path:
    pressure = Path(row["pressure_path"])
    parts = pressure.parts
    try:
        cam = parts.index("cam3")
    except ValueError as exc:
        raise ValueError(f"pressure_path is not a cam3 sequence: {pressure}") from exc
    return REPO_ROOT / Path(*parts[: cam + 4])


def valid_indices(row: dict, n: int) -> np.ndarray:
    raw = row.get("valid_frame_indices", "")
    if raw:
        try:
            values = json.loads(raw) if isinstance(raw, str) else raw
            return np.asarray([int(i) for i in values if 0 <= int(i) < n], dtype=np.int64)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    fake_raw = row.get("fake_frame_indices", "")
    fake = set()
    if fake_raw:
        try:
            fake = {int(i) for i in json.loads(fake_raw)}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return np.asarray([i for i in range(n) if i not in fake], dtype=np.int64)


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


def load_insole_sum(row: dict) -> tuple[np.ndarray, np.ndarray]:
    sid = row["session_id"]
    pressure = np.load(session_dir(row) / "pressure.npz")["pressure"].astype(np.float32)
    fake = np.load(session_dir(row) / "fake_mask.npy").astype(bool).reshape(-1)
    # Keep the exact frozen 4x12 crop used by generate_baseline_tactile.py.
    from AnysoleWorkspace.tool.generate_baseline_tactile import crop_cells

    cells = crop_cells(pressure)
    if len(fake) != len(cells):
        raise ValueError(f"fake mask length mismatch for {sid}")
    return cells.sum(axis=(1, 2, 3)), fake


def choose_standing(row: dict) -> tuple[int, float, dict]:
    sums, fake = load_insole_sum(row)
    n = len(sums)
    valid = valid_indices(row, n)
    contact_path = session_dir(row) / "contact_bvh_h.npy"
    if not contact_path.is_file():
        raise FileNotFoundError(f"missing BVH contact labels: {contact_path}")
    contact = np.asarray(np.load(contact_path), dtype=np.float32)
    if contact.shape[0] != n or contact.shape[1] < 8:
        raise ValueError(f"invalid contact_bvh_h shape for {row['session_id']}: {contact.shape}")
    speed = aligned_bvh_speed(row, n)
    candidates = valid[(~fake[valid]) & (contact[valid, 6] > 0.5) & (contact[valid, 7] > 0.5)]
    if candidates.size == 0:
        raise ValueError(f"no valid double-foot-contact frame for {row['session_id']}")
    frame = int(candidates[np.argmin(speed[candidates])])
    return frame, float(sums[frame]), {
        "session_id": row["session_id"],
        "frame": frame,
        "speed_mps": float(speed[frame]),
        "pressure_sum_0_255": float(sums[frame]),
        "candidate_count": int(candidates.size),
    }


def sequence_frames(row: dict, window: int = 5) -> list[str]:
    n = int(row["n_frames"])
    valid = set(valid_indices(row, n).tolist())
    half = window // 2
    frames = []
    for frame in range(half, n - half):
        if all(idx in valid for idx in range(frame - half, frame + half + 1)):
            frames.append(f"{frame:03d}.npy")
    return frames


def output_path_for_date(date: str) -> Path:
    return FPP_ROOT / date / "sub_info.npy"


def build_metadata(force: bool = False) -> dict:
    manifest = read_manifest()
    split_of = read_split()
    selected = [row for sid, row in manifest.items() if sid in split_of]
    if not selected:
        raise RuntimeError("canonical split selected no manifest rows")

    # First action/trial per subject is the only source allowed to define the
    # subject pressure scale; later actions are not used to tune it.
    first_by_subject: dict[str, dict] = {}
    for row in sorted(selected, key=lambda r: (r["subject_id"], r.get("action", ""), r.get("trial", ""), r["session_id"])):
        first_by_subject.setdefault(row["subject_id"], row)

    sub_info_by_date: dict[str, dict] = defaultdict(dict)
    selection_meta = {}
    no_temporal_window_sessions = []
    for subject, row in sorted(first_by_subject.items()):
        frame, weight, meta = choose_standing(row)
        date = Path(row["pressure_path"]).parts[Path(row["pressure_path"]).parts.index("cam3") + 1]
        sub_info_by_date[date][subject] = {
            "weight": weight,
            "max_value": 255.0,
            "height": -1.0,
        }
        selection_meta[subject] = {**meta, "date": date}

    split_tree: dict[str, dict] = {
        "train": defaultdict(lambda: defaultdict(dict)),
        "val": defaultdict(lambda: defaultdict(dict)),
        "test": defaultdict(lambda: defaultdict(dict)),
    }
    for row in sorted(selected, key=lambda r: r["session_id"]):
        split = split_of[row["session_id"]]
        date = Path(row["pressure_path"]).parts[Path(row["pressure_path"]).parts.index("cam3") + 1]
        frames = sequence_frames(row)
        if not frames:
            # Keep the assigned session in the audit tree so the global
            # train/val/test collection remains 92/12/36.  FPP-Net simply has
            # no usable temporal window for an all-fake/short session.
            no_temporal_window_sessions.append(row["session_id"])
        split_tree[split][date][row["subject_id"]][row["session_id"]] = frames

    FPP_ROOT.mkdir(parents=True, exist_ok=True)
    for date, info in sub_info_by_date.items():
        out = output_path_for_date(date)
        if out.exists() and not force:
            old = np.load(out, allow_pickle=True).item()
            if old != info:
                raise FileExistsError(f"refusing to replace existing sub_info: {out}; use --force")
        else:
            np.save(out, info)

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
        "split_path": str(split_path),
        "sub_info_paths": {date: str(output_path_for_date(date)) for date in sorted(sub_info_by_date)},
        "session_counts": {phase: sum(len(seqs) for subjects in dates.values() for seqs in subjects.values()) for phase, dates in tree.items()},
        "frame_counts": {phase: sum(len(frames) for subjects in dates.values() for seqs in subjects.values() for frames in seqs.values()) for phase, dates in tree.items()},
        "no_temporal5_window_sessions": sorted(no_temporal_window_sessions),
        "selection": selection_meta,
        "weight_unit": "raw insole values in the shared 0-255 contract; no kg conversion",
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
        "no_temporal5_window_sessions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
