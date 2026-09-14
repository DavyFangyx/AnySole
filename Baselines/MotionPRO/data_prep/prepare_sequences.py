"""Convert PressureWasher sessions into MotionPRO sequence folders."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

MOTIONPRO_ROOT = Path(__file__).resolve().parents[1]
if str(MOTIONPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONPRO_ROOT))

from lib.util.workspace import WORKSPACE_ROOT, resolve_path, sequence_root


GAIT_ROOT = MOTIONPRO_ROOT.parents[1]
PRESSURE_WASHER_ROOT = WORKSPACE_ROOT / "sources/PressureWasher"
FAKE_MARKED_ROOT = PRESSURE_WASHER_ROOT / "outputs" / "fake_marked" / "reconstruction_20260817_161459_fake_marked"
ENCODED_ROOT = PRESSURE_WASHER_ROOT / "outputs" / "encoded" / "reconstruction_20260817_161459_fake_marked_encoded"
ALIGN_DIR = PRESSURE_WASHER_ROOT / "outputs" / "AlignReviews_csv"
BVH_ROOT = WORKSPACE_ROOT / "sources/raw"
RAW_ROOT = WORKSPACE_ROOT / "sources/raw"
OUT_ROOT = WORKSPACE_ROOT / "derived/MotionPRO/sequences"
SPLIT_ROOT = WORKSPACE_ROOT / "splits/default"
TARGET_FPS = 40.0
PRESSURE_HW = (160, 120)
PRESSURE_CLIP = 1023.0
CONTACT_SUM_THRESH = 100.0
DEFAULT_CAM_ID = 3
QUALITY_CSV = (
    PRESSURE_WASHER_ROOT
    / "outputs"
    / "stats"
    / "pressure_stats_20260814_231054"
    / "overall"
    / "missing_pressure_objects.csv"
)
ALLOWED_QUALITY = {"A", "B"}
# contact[:, i] aligns with Loss.contactIds = [1, 2, 4, 5, 7, 8, 10, 11, 20, 21]
# Only left/right feet can be in contact; hips, knees, ankles, and hands stay 0.
LEFT_CONTACT = [6]
RIGHT_CONTACT = [7]
SESSION_RE = re.compile(r"_(S\d+)_(\d+)$")
JPG_RE = re.compile(r"^\d+_(\d{2})(\d{2})(\d{2})\.(\d{3})")
# BVH child-joint index (Hips=0) -> SMPL body_pose index. Keep all 23 BVH joints;
# the final unused SMPL body slot remains zero.
BVH_TO_SMPL_BODY = {
    15: 0, 19: 1, 1: 2, 16: 3, 20: 4, 2: 5, 17: 6, 21: 7, 3: 8, 18: 9, 22: 10,
    4: 21, 5: 11, 7: 12, 11: 13, 6: 14, 8: 15, 12: 16, 9: 17, 13: 18, 10: 19, 14: 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MotionPRO sequences from PressureWasher outputs.")
    parser.add_argument("--session", type=str, default=None, help="Only convert this session id, e.g. S10011.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true", help="Skip sessions whose output directory exists.")
    mode.add_argument("--force", "--overwrite", dest="force", action="store_true", help="Rebuild and overwrite existing sessions.")
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID, help="Raw camera folder under each rec dir, e.g. 3.")
    parser.add_argument("--out-root", type=str, default=None, help="Override the centralized sequence output root.")
    parser.add_argument(
        "--refresh-contact",
        action="store_true",
        help="Rewrite contact.npy for existing sequences without reconverting.",
    )
    return parser.parse_args()


def rec_to_sid(name: str) -> str | None:
    match = SESSION_RE.search(name)
    return None if match is None else match.group(1) + match.group(2)


def win_to_posix(path: str) -> str:
    return path.replace("\\", "/")


def load_summary() -> dict[str, dict[str, str]]:
    rows = {}
    with (ALIGN_DIR / "conversion_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["session_id"]] = row
    return rows


def load_quality_map() -> dict[str, str]:
    if not QUALITY_CSV.is_file():
        raise FileNotFoundError(f"Quality table not found: {QUALITY_CSV}")
    rows = {}
    with QUALITY_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            session_id = (row.get("dataset_id") or "").strip()
            quality = (row.get("quality_class") or "").strip().upper()
            if session_id:
                rows[session_id] = quality
    return rows


def quality_skip_reason(session_id: str, quality_map: dict[str, str]) -> str | None:
    quality = quality_map.get(session_id)
    if quality is None:
        return "missing_quality_class"
    if quality not in ALLOWED_QUALITY:
        return f"quality_{quality}"
    return None


def load_align_csv(session_id: str) -> dict[str, dict[str, str]]:
    path = ALIGN_DIR / f"{session_id}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["标记"]: row for row in csv.DictReader(handle)}
    if "start" not in rows or "end" not in rows:
        raise ValueError(f"Align CSV missing start/end: {path}")
    return rows


def load_pressure_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty pressure CSV: {path}")
        value_cols = [name for name in reader.fieldnames if name.isdigit()]
        t_us, fake, valid, values = [], [], [], []
        for row in reader:
            t_us.append(float(row["t_us"]))
            fake.append(int(float(row.get("fake", 0))))
            valid.append(int(float(row.get("valid_mask", 1))))
            values.append([float(row[col]) for col in value_cols])
    return {
        "t_us": np.asarray(t_us, dtype=np.float64),
        "fake": np.asarray(fake, dtype=np.uint8),
        "valid": np.asarray(valid, dtype=np.uint8),
        "values": np.asarray(values, dtype=np.float32) if values else np.zeros((0, 48), dtype=np.float32),
    }


def nearest_indices(source_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    if source_t.size == 0:
        raise ValueError("Empty source timeline")
    idx = np.clip(np.searchsorted(source_t, query_t), 1, source_t.size - 1)
    left = idx - 1
    pick_right = np.abs(source_t[idx] - query_t) < np.abs(source_t[left] - query_t)
    return np.where(pick_right, idx, left)


def interpolate_values(source_t: np.ndarray, values: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((query_t.size, 48), dtype=np.float32)
    out = np.empty((query_t.size, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(query_t, source_t, values[:, col])
    return out


def rasterize_feet(left48: np.ndarray, right48: np.ndarray) -> np.ndarray:
    height, width = PRESSURE_HW
    img = np.zeros((left48.shape[0], height, width), dtype=np.float32)
    left = np.clip(left48, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP * 255.0
    right = np.clip(right48, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP * 255.0
    left = left.reshape(-1, 4, 12)
    right = right.reshape(-1, 4, 12)
    left_block = np.repeat(np.repeat(left, 20, axis=1), 4, axis=2)
    right_block = np.repeat(np.repeat(right, 20, axis=1), 4, axis=2)
    img[:, 40:120, 6:54] = left_block
    img[:, 40:120, 66:114] = right_block
    return img


def contact_from_insoles(left48: np.ndarray, right48: np.ndarray) -> np.ndarray:
    contact = np.zeros((left48.shape[0], 10), dtype=np.float32)
    left_on = left48.sum(axis=1) > CONTACT_SUM_THRESH
    right_on = right48.sum(axis=1) > CONTACT_SUM_THRESH
    contact[np.ix_(left_on, LEFT_CONTACT)] = 1.0
    contact[np.ix_(right_on, RIGHT_CONTACT)] = 1.0
    return contact


def parse_bvh(path: Path) -> tuple[np.ndarray, float]:
    text = path.read_text(errors="replace").splitlines()
    motion_idx = next(i for i, line in enumerate(text) if line.startswith("MOTION"))
    n_frames = int(text[motion_idx + 1].split(":")[1])
    frame_time = float(text[motion_idx + 2].split(":")[1])
    motion = np.asarray(
        [list(map(float, line.split())) for line in text[motion_idx + 3: motion_idx + 3 + n_frames]],
        dtype=np.float64,
    )
    if motion.shape[1] != 72:
        raise ValueError(f"Unexpected BVH channel count {motion.shape[1]} in {path}")
    return motion, frame_time


def euler_yxz_to_aa(eulers_deg: np.ndarray) -> np.ndarray:
    rot = SciRotation.from_euler("YXZ", eulers_deg, degrees=True)
    return rot.as_rotvec().astype(np.float32)


def interp_motion(motion: np.ndarray, frame_time: float, query_t: np.ndarray) -> np.ndarray:
    src_t = np.arange(motion.shape[0], dtype=np.float64) * frame_time
    out = np.empty((query_t.size, motion.shape[1]), dtype=np.float64)
    for col in range(motion.shape[1]):
        out[:, col] = np.interp(query_t, src_t, motion[:, col])
    return out


def motion_to_smpl(sampled: np.ndarray) -> dict[str, np.ndarray]:
    transl = (sampled[:, :3] / 100.0).astype(np.float32)
    global_orient = euler_yxz_to_aa(sampled[:, 3:6])
    joint_eulers = sampled[:, 6:].reshape(sampled.shape[0], 22, 3)
    body_aa = np.zeros((sampled.shape[0], 23, 3), dtype=np.float32)
    for bvh_idx, smpl_idx in BVH_TO_SMPL_BODY.items():
        body_aa[:, smpl_idx] = euler_yxz_to_aa(joint_eulers[:, bvh_idx - 1])
    return {
        "betas": np.zeros((10,), dtype=np.float32),
        "global_orient": global_orient,
        "body_pose": body_aa.reshape(sampled.shape[0], 69),
        "transl": transl,
    }


def resolve_bvh(session_id: str, summary_row: dict[str, str], start_row: dict[str, str]) -> Path:
    selected = summary_row.get("selected_bvh") or ""
    selected_name = Path(win_to_posix(selected)).name if selected else ""
    date = str(summary_row.get("date") or start_row.get("日期") or "")
    session_dir = BVH_ROOT / date / "mocap_ori_bvh" / session_id
    candidates = sorted(session_dir.glob("*.bvh"))
    if not candidates:
        raise FileNotFoundError(f"No BVH for {session_id}")
    if selected_name:
        for path in candidates:
            if path.name == selected_name:
                return path
    for needle in ("Skeleton1", "Skeleton0", "order", "position"):
        for path in candidates:
            if needle.lower() in path.name.lower():
                return path
    return candidates[0]


def list_cam_frames(cam_dir: Path) -> list[Path]:
    return sorted(list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.png")))


def frame_times(frames: list[Path], rec_dir: Path) -> np.ndarray:
    start = None
    meta_path = rec_dir / "meta.json"
    if meta_path.is_file():
        started = json.loads(meta_path.read_text()).get("started_at_iso")
        if started:
            hh, mm, rest = started.split("T")[1].split(":")
            start = int(hh) * 3600 + int(mm) * 60 + float(rest)
    times = []
    for idx, path in enumerate(frames):
        match = JPG_RE.match(path.name)
        if match is None or start is None:
            times.append(idx / TARGET_FPS)
            continue
        clock = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3)) + int(match.group(4)) / 1000.0
        times.append(clock - start)
    return np.asarray(times, dtype=np.float64)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_splits(kept: list[tuple[str, Path]]) -> None:
    print(
        f"converted {len(kept)} sequences. Write splits with "
        "data_prep/make_splits.py --cam-id 3 [--ood S11,S10] [--exclude S11,S10]"
    )


def list_sequence_dirs(out_root: Path) -> list[Path]:
    return [path.parent for path in sorted(out_root.glob("*/*/*/align_meta.json"))]


def list_kept_sequences(out_root: Path) -> list[tuple[str, Path]]:
    return [(seq_dir.parent.name, seq_dir) for seq_dir in list_sequence_dirs(out_root)]


def fake_marked_dir_from_meta(meta: dict) -> Path:
    return FAKE_MARKED_ROOT / meta["date"] / meta["subject"] / meta["rec_name"]


def contact_from_meta(meta: dict) -> np.ndarray:
    fake_marked_dir = fake_marked_dir_from_meta(meta)
    left = load_pressure_csv(fake_marked_dir / "pressure_left.csv")
    right = load_pressure_csv(fake_marked_dir / "pressure_right.csv")
    n = int(meta["n_frames"])
    t_grid = float(meta["visual_start_s"]) + np.arange(n, dtype=np.float64) / TARGET_FPS
    t_grid_us = t_grid * 1e6
    left48 = interpolate_values(left["t_us"], left["values"], t_grid_us)
    right48 = interpolate_values(right["t_us"], right["values"], t_grid_us)
    return contact_from_insoles(left48, right48)


def refresh_contacts(args: argparse.Namespace, quality_map: dict[str, str]) -> int:
    seq_dirs = list_sequence_dirs(args.out_root)
    if args.session:
        seq_dirs = [seq_dir for seq_dir in seq_dirs if seq_dir.name == args.session]
        if not seq_dirs:
            print(f"no existing sequence for {args.session}")
            return 1
    n_ok = 0
    n_removed = 0
    for seq_dir in seq_dirs:
        reason = quality_skip_reason(seq_dir.name, quality_map)
        if reason is not None:
            shutil.rmtree(seq_dir)
            n_removed += 1
            print(f"removed {seq_dir} ({reason})")
            continue
        meta = json.loads((seq_dir / "align_meta.json").read_text())
        contact = contact_from_meta(meta)
        np.save(seq_dir / "contact.npy", contact)
        n_ok += 1
        print(f"refreshed contact {seq_dir.name} -> {seq_dir} T={contact.shape[0]}")
    if args.session is None:
        write_splits(list_kept_sequences(args.out_root))
    print(f"refreshed {n_ok} contact files, removed {n_removed} C/D sequences")
    return 0



def convert_session(session_id: str, rec_dir_name: str, rel_dir: Path, summary_row: dict[str, str], out_root: Path, overwrite: bool, cam_id: int) -> tuple[str, Path | None, str]:
    align = load_align_csv(session_id)
    start = align["start"]
    end = align["end"]
    visual_start = float(start["视觉时间(s)"])
    visual_end = float(end["视觉时间(s)"])
    offset = float(start["偏移量(s)"])
    if visual_end <= visual_start:
        return "skip", None, "invalid_visual_range"

    fake_marked_dir = FAKE_MARKED_ROOT / rel_dir
    left_csv = fake_marked_dir / "pressure_left.csv"
    right_csv = fake_marked_dir / "pressure_right.csv"
    if not left_csv.is_file() or not right_csv.is_file():
        return "skip", None, "missing_pressure_csv"

    left = load_pressure_csv(left_csv)
    right = load_pressure_csv(right_csv)
    if left["t_us"].size == 0 or right["t_us"].size == 0:
        return "skip", None, "empty_foot"

    rec_dir = RAW_ROOT / rel_dir
    cam_dir = rec_dir / str(cam_id)
    if not cam_dir.is_dir():
        return "skip", None, f"missing_cam{cam_id}"
    frames = list_cam_frames(cam_dir)
    if not frames:
        return "skip", None, f"missing_cam{cam_id}"
    cam_t = frame_times(frames, rec_dir)
    bvh_path = resolve_bvh(session_id, summary_row, start)
    motion, frame_time = parse_bvh(bvh_path)

    n = int(math.floor((visual_end - visual_start) * TARGET_FPS + 1e-9)) + 1
    if n <= 0:
        return "skip", None, "empty_grid"
    t_grid = visual_start + np.arange(n, dtype=np.float64) / TARGET_FPS
    t_grid_us = t_grid * 1e6
    t_mocap = t_grid - offset

    left_idx = nearest_indices(left["t_us"], t_grid_us)
    right_idx = nearest_indices(right["t_us"], t_grid_us)
    left48 = interpolate_values(left["t_us"], left["values"], t_grid_us)
    right48 = interpolate_values(right["t_us"], right["values"], t_grid_us)
    pressure = rasterize_feet(left48, right48)
    contact = contact_from_insoles(left48, right48)
    fake_mask = np.maximum.reduce(
        [
            left["fake"][left_idx],
            right["fake"][right_idx],
            1 - left["valid"][left_idx],
            1 - right["valid"][right_idx],
        ]
    ).astype(np.uint8)

    sampled = interp_motion(motion, frame_time, t_mocap)
    smpl = motion_to_smpl(sampled)
    cam_idx = nearest_indices(cam_t, t_grid)

    subject = rel_dir.parts[1]
    date = rel_dir.parts[0]
    out_dir = out_root / date / subject / session_id
    if out_dir.exists():
        if not overwrite:
            return "skip", out_dir, "exists"
        shutil.rmtree(out_dir)
    color_dir = out_dir / "color"
    color_dir.mkdir(parents=True, exist_ok=True)
    for i, src_i in enumerate(cam_idx):
        src = frames[int(src_i)]
        link = color_dir / f"{i:06d}{src.suffix.lower()}"
        os.symlink(os.path.relpath(src, link.parent), link)

    np.savez_compressed(out_dir / "pressure.npz", pressure=pressure.astype(np.float32))
    np.save(out_dir / "smpl.npy", smpl, allow_pickle=True)
    np.save(out_dir / "fake_mask.npy", fake_mask)
    np.save(out_dir / "contact.npy", contact)
    meta = {
        "session_id": session_id,
        "date": date,
        "subject": subject,
        "rec_dir": str(rec_dir),
        "rec_name": rec_dir_name,
        "offset_s": offset,
        "visual_start_s": visual_start,
        "visual_end_s": visual_end,
        "mocap_start_s": float(start["动捕时间(s)"]),
        "mocap_end_s": float(end["动捕时间(s)"]),
        "n_frames": int(n),
        "target_fps": TARGET_FPS,
        "bvh_path": str(bvh_path),
        "cam_id": int(cam_id),
        "cam_dir": str(cam_dir),
        "fake_rate": float(fake_mask.mean()),
        "pressure_min": float(pressure.min()),
        "pressure_max": float(pressure.max()),
    }
    (out_dir / "align_meta.json").write_text(json.dumps({k: json_ready(v) for k, v in meta.items()}, indent=2))
    return "ok", out_dir, "converted"


def resolve_out_root(args: argparse.Namespace) -> Path:
    if args.out_root is not None:
        return Path(resolve_path(args.out_root, MOTIONPRO_ROOT))
    return sequence_root(args.cam_id)


def main() -> int:
    args = parse_args()
    args.out_root = resolve_out_root(args)
    quality_map = load_quality_map()
    if args.refresh_contact:
        return refresh_contacts(args, quality_map)
    summary = load_summary()
    recs = []
    for manifest in sorted(ENCODED_ROOT.rglob("reconstruction_manifest.csv")):
        rec_dir = manifest.parent
        rel = rec_dir.relative_to(ENCODED_ROOT)
        session_id = rec_to_sid(rec_dir.name)
        if session_id is not None:
            recs.append((session_id, rec_dir.name, rel))

    skip_rows = []
    kept = []
    n_ok = 0
    for session_id, rec_name, rel in recs:
        if args.session and session_id != args.session:
            continue
        reason = quality_skip_reason(session_id, quality_map)
        if reason is not None:
            skip_rows.append((session_id, rec_name, reason))
            continue
        row = summary.get(session_id)
        if row is None:
            skip_rows.append((session_id, rec_name, "missing_summary"))
            continue
        if row.get("status") != "通过":
            skip_rows.append((session_id, rec_name, f"status_{row.get('status')}"))
            continue
        try:
            status, out_dir, reason = convert_session(session_id, rec_name, rel, row, args.out_root, args.force, args.cam_id)
        except Exception as exc:
            skip_rows.append((session_id, rec_name, f"error:{exc}"))
            traceback.print_exc()
            continue
        if status != "ok":
            skip_rows.append((session_id, rec_name, reason))
            continue
        n_ok += 1
        kept.append((rel.parts[1], out_dir))
        n_frames = json.loads((out_dir / "align_meta.json").read_text())["n_frames"]
        print(f"converted {session_id} -> {out_dir} T={n_frames}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    skip_path = args.out_root / "skip_report.csv"
    with skip_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session_id", "rec_name", "reason"])
        writer.writerows(skip_rows)
    if args.session is None:
        write_splits(kept)
    print(f"converted {n_ok} sessions, skipped {len(skip_rows)}")
    print(f"skip report: {skip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
