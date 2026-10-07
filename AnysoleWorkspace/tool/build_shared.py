#!/usr/bin/env python3
"""Promote raw facts into ``shared/facts`` and frozen MMVP representation.

This builder never reads ``derived/`` or ``model_inputs/``.  The MMVP mapping
is the already audited 4x12 -> 31x11 conversion and is kept here as a public,
versioned representation rather than duplicated below either model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AnysoleWorkspace.tool.artifacts import sha256_file, write_artifact  # noqa: E402
from AnysoleWorkspace.tool.workspace import canonical_uri, resolve_uri  # noqa: E402

WORKSPACE = ROOT / "AnysoleWorkspace"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions"
MMVP_ROOT = WORKSPACE / "shared/representations/tactile/mmvp_31x11/v1"
MANIFEST = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
ESSENTIALS = ROOT / "Baselines/VP-MoCap/FPP-Net/essentials/insole2cont"
LAYOUT = WORKSPACE / "assets/foot_sensor_layout"


def read_manifest() -> dict[str, dict]:
    return {
        row["session_id"]: row
        for row in (json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines())
        if row.get("session_id")
    }


def source_csvs(pressure_dir: Path) -> tuple[Path, Path]:
    left = pressure_dir / "pressure_left.csv"
    right = pressure_dir / "pressure_right.csv"
    if not left.is_file() or not right.is_file():
        raise FileNotFoundError(f"missing final PressureWasher CSVs under {pressure_dir}")
    return left, right


def read_pressure(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_us, raw_time_us, values, valid, fake).

    ``time_us`` is the shipped (synthetic uniform-40 Hz) timeline;
    ``raw_time_us`` carries the reconstructed physical wall-clock time of each
    row (``raw_t_us``, computed by mark-fake from the per-segment manifest
    anchors) with NaN where the row must never be selected (bridge/fake rows).
    Raw times live on the same wall clock as the cam3 video clock (S2 ruling,
    Baselines/决策/05_R1-R3_风险裁定.md §三 证据 C) and are used for
    absolute-time resampling.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty pressure CSV: {path}")
    channels = [str(i) for i in range(1, 49)]
    missing = [name for name in channels if name not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing pressure columns {missing}")
    time = np.asarray([float(row["t_us"]) for row in rows], dtype=np.float64)
    values = np.asarray([[float(row[name]) for name in channels] for row in rows], dtype=np.float32)
    valid = np.asarray([int(float(row.get("valid_mask", 1))) for row in rows], dtype=np.uint8)
    fake = np.asarray([int(float(row.get("fake", 0))) for row in rows], dtype=np.uint8)
    raw_time = np.full(len(rows), np.nan, dtype=np.float64)
    if "raw_t_us" in rows[0]:
        raw_time[:] = [float(row["raw_t_us"]) if row.get("raw_t_us", "").strip() else np.nan for row in rows]
    if not np.isfinite(values).all() or not np.isfinite(time).all():
        raise ValueError(f"non-finite pressure source: {path}")
    return time, raw_time, values, valid, fake


def day_seconds_from_iso(value: str) -> float:
    """Local day-seconds from an ISO timestamp (same wall clock as jpeg names)."""
    _, hhmmss = value.strip().split("T", 1) if "T" in value else value.strip().split(" ", 1)
    hh, mm, rest = hhmmss.split(":")
    ss = float(rest)
    return int(hh) * 3600 + int(mm) * 60 + ss


def day_seconds_from_jpeg_name(name: str) -> float:
    """Local day-seconds from a cam3 frame name ``3_HHMMSS.mmm.jpg``."""
    core = name.split("_")[-1][:-4]  # strip ".jpg"
    return int(core[0:2]) * 3600 + int(core[2:4]) * 60 + float(core[4:])


def nearest_indices(times_s: np.ndarray, targets_s: np.ndarray) -> np.ndarray:
    """Index of the nearest element of ``times_s`` (ascending) for each target."""
    idx = np.clip(np.searchsorted(times_s, targets_s), 1, len(times_s) - 1)
    left = idx - 1
    return np.where(np.abs(times_s[left] - targets_s) <= np.abs(times_s[idx] - targets_s), left, idx)


def resample_pressure(left: tuple, right: tuple, n_frames: int, *,
                      video_start_day_s: float | None = None,
                      pressure_epoch_day_s: float | None = None,
                      fps: float = 40.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Map each video frame to the nearest pressure sample.

    Absolute mode (default when the wall-clock anchors are available and the
    CSVs carry raw ``source_t_us``): video frame ``i`` has wall time
    ``video_start_day_s + i/fps`` (cam3 jpeg clock); pressure row ``j`` has
    wall time ``pressure_epoch_day_s + raw_t_us[j]/1e6`` (``started_at_iso`` +
    raw ``time.monotonic`` bridge -- the same clock, S2 §三 证据 C).  Values and
    quality flags are taken from the nearest row; no interpolation.  Video
    frames outside the source span clamp to the edge row and are counted.

    Legacy mode (no anchors or no raw timestamps): the previous normalized
    grid, with nearest-row values instead of linear interpolation.
    """
    target = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    absolute = (video_start_day_s is not None and pressure_epoch_day_s is not None
                and np.isfinite(left[1]).any() and np.isfinite(right[1]).any())
    out = {}

    def map_foot(pack: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        time, raw_time, values, valid, fake = pack
        if absolute:
            # Map onto real raw samples only: rows without a raw timestamp
            # (bridge/interpolated rows) are never selected as a source row.
            has_raw = np.isfinite(raw_time)
            raw_rows = np.flatnonzero(has_raw)
            t_sub = raw_time[raw_rows] / 1e6 + pressure_epoch_day_s
            targets = video_start_day_s + np.arange(n_frames, dtype=np.float64) / fps
            idx = raw_rows[nearest_indices(t_sub, targets)]
            n_clamp_start = int((targets < t_sub[0]).sum())
            n_clamp_end = int((targets > t_sub[-1]).sum())
        else:
            idx = np.rint(target * max(len(time) - 1, 0)).astype(int)
            n_clamp_start = n_clamp_end = 0
        return values[idx], valid[idx], fake[idx], n_clamp_start, n_clamp_end

    left48, left_valid, left_fake, lc0, lc1 = map_foot(left)
    right48, right_valid, right_fake, rc0, rc1 = map_foot(right)
    fake = np.maximum(left_fake, right_fake).astype(np.uint8)
    valid = np.minimum(left_valid, right_valid).astype(np.uint8)
    valid = np.minimum(valid, 1 - fake).astype(np.uint8)
    out.update(
        alignment_method="nearest_values_absolute_t_us" if absolute else "nearest_values_normalized_t_us",
        absolute=absolute,
        video_frame0_wall_s=video_start_day_s, pressure_epoch_wall_s=pressure_epoch_day_s,
        out_of_span_frames_left=(lc0, lc1), out_of_span_frames_right=(rc0, rc1),
    )
    if absolute:
        frame_times = video_start_day_s + np.arange(n_frames, dtype=np.float64) / fps
    else:
        frame_times = target
    return left48, right48, valid, fake, frame_times, out


def sole_positions(foot: str) -> np.ndarray:
    name = {"L": "leftfoot", "R": "rightfoot"}[foot]
    dots = np.genfromtxt(LAYOUT / f"{name}_dots.csv", delimiter=",", skip_header=1, usecols=(1, 2)).astype(np.float64)
    ids = np.loadtxt(ESSENTIALS / ("footL_ids.txt" if foot == "L" else "footR_ids.txt")).astype(int)
    vertices = load_template()[ids]
    xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
    zmin, zmax = vertices[:, 2].min(), vertices[:, 2].max()
    xn = (dots[:, 0] - dots[:, 0].min()) / max(np.ptp(dots[:, 0]), 1e-9)
    yn = (dots[:, 1] - dots[:, 1].min()) / max(np.ptp(dots[:, 1]), 1e-9)
    if foot == "R":
        xn = 1.0 - xn
    return np.stack((xmin + xn * (xmax - xmin), zmax - yn * (zmax - zmin)), axis=1)


_TEMPLATE: np.ndarray | None = None


def load_template() -> np.ndarray:
    global _TEMPLATE
    if _TEMPLATE is not None:
        return _TEMPLATE
    vertices = []
    with (ESSENTIALS / "smpl_template.obj").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    _TEMPLATE = np.asarray(vertices, dtype=np.float64)
    return _TEMPLATE


def mmvp_map(foot: str) -> tuple[np.ndarray, np.ndarray]:
    mapping = np.load(ESSENTIALS / ("insole2smplL.npy" if foot == "L" else "insole2smplR.npy"), allow_pickle=True).item()
    template = load_template()
    positions = np.full((31, 11, 2), np.nan, dtype=np.float64)
    for vertex, indices in mapping.items():
        if len(indices[0]) == 0:
            continue
        for row, col in zip(indices[0], indices[1]):
            positions[int(row), int(col)] = template[int(vertex)][[0, 2]]
    cells = sole_positions(foot)
    mask = np.isfinite(positions[..., 0])
    available = positions[mask]
    distances = ((available[:, None, :] - cells[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    indices = np.full((31, 11), -1, dtype=np.int64)
    indices[mask] = nearest
    return indices, mask


def build_one(sid: str, row: dict, *, force: bool = False) -> Path:
    recording = resolve_uri(row["video_path"], must_exist=True)
    pressure_dir = resolve_uri(row["pressure_path"], must_exist=True)
    left_csv, right_csv = source_csvs(pressure_dir)
    left_uri = row["pressure_path"].rstrip("/") + "/pressure_left.csv"
    right_uri = row["pressure_path"].rstrip("/") + "/pressure_right.csv"
    left_raw, right_raw = read_pressure(left_csv), read_pressure(right_csv)
    n = int(row["n_frames"])
    rgb_source = recording / "3"
    images = sorted(rgb_source.glob("*.jpg"))
    if not images:
        raise ValueError(f"{sid}: raw RGB has no extracted frames")
    meta_path = recording / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    video_start_day_s = day_seconds_from_jpeg_name(images[0].name)
    pressure_epoch_day_s = day_seconds_from_iso(meta["started_at_iso"]) if meta.get("started_at_iso") else None
    left48, right48, valid, fake, frame_times, resample_meta = resample_pressure(
        left_raw, right_raw, n,
        video_start_day_s=video_start_day_s,
        pressure_epoch_day_s=pressure_epoch_day_s,
        fps=float(row["target_fps"]))
    date, subject = recording.parts[-3], recording.parts[-2]
    session_root = FACTS_ROOT / row["camera"] / date / subject / sid
    if session_root.exists() and not force and all((session_root / name).is_file() for name in ("session.json", "frames.npz", "pressure_48.npz")):
        return session_root
    session_root.mkdir(parents=True, exist_ok=True)
    # A few source recordings have an aligned session length one or more
    # frames longer than the extracted JPEG directory.  Preserve the public
    # frame count and provenance by repeating the final source frame; adapters
    # must not infer a new time axis from this compatibility padding.
    selected_images = images[:n] + [images[-1]] * max(0, n - len(images))
    rgb_dir = session_root / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    for index, image in enumerate(selected_images):
        link = rgb_dir / f"{index:06d}.jpg"
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(os.path.relpath(image, link.parent))

    frame_id = np.arange(n, dtype=np.int64)
    visual = float(row.get("visual_start_s", 0.0)) + frame_id.astype(np.float64) / float(row["target_fps"])
    mocap = float(row.get("mocap_start_s", 0.0)) + frame_id.astype(np.float64) / float(row["target_fps"])
    np.savez_compressed(session_root / "frames.npz", frame_id=frame_id,
                        visual_time_s=visual.astype(np.float64), mocap_time_s=mocap.astype(np.float64),
                        valid=valid.astype(np.uint8), fake=fake.astype(np.uint8))
    np.savez_compressed(
        session_root / "pressure_48.npz", frame_id=frame_id,
        left48=left48.astype(np.float32), right48=right48.astype(np.float32),
        valid=valid.astype(np.uint8), fake=fake.astype(np.uint8),
        unit=np.asarray("source_sensor_units"), value_min=np.asarray(min(left48.min(), right48.min()), dtype=np.float32),
        value_max=np.asarray(max(left48.max(), right48.max()), dtype=np.float32),
        source_file_left=np.asarray(left_uri), source_file_right=np.asarray(right_uri),
        source_time_column=np.asarray("t_us"), alignment_method=np.asarray(resample_meta["alignment_method"]),
    )
    bvh = resolve_uri(row["bvh_path"], must_exist=True)
    smpl = resolve_uri(row["smpl_path"], must_exist=True) if row.get("smpl_path") else None
    meta_path = recording / "meta.json"
    source_files = {
        "rgb": row["video_path"], "bvh": row["bvh_path"], "smpl": row.get("smpl_path", ""),
        "left_source": left_uri, "right_source": right_uri,
    }
    source_hashes = {
        "left_source": sha256_file(left_csv), "right_source": sha256_file(right_csv),
        "bvh": sha256_file(bvh), "rgb_meta": sha256_file(meta_path) if meta_path.is_file() else "",
        "rgb_frame_listing": __import__("hashlib").sha256("\n".join(p.name for p in selected_images).encode()).hexdigest(),
    }
    if smpl and smpl.is_file():
        source_hashes["smpl"] = sha256_file(smpl)
    session = {
        "schema_version": "shared.session.v1", "session_id": sid, "date": date, "subject": subject,
        "camera": row["camera"], "fps": float(row["target_fps"]), "offset_s": float(row.get("offset_s", 0.0)),
        "frame_count": n, "quality": row.get("quality", ""), "valid_count": int(valid.sum()), "fake_count": int(fake.sum()),
        "source_files": source_files, "source_hashes": source_hashes,
        "pressure_provenance": {
            "layout": "4x12 per foot; 96 cells per frame across both feet", "value_columns": [str(i) for i in range(1, 49)],
            "time_column": "t_us", "quality_columns": ["valid_mask", "fake"],
            "left_source": left_uri, "right_source": right_uri,
            "shared_alignment": (
                "values and quality flags nearest-neighbour mapped by absolute wall-clock time "
                "(video frames: first cam3 jpeg name + i/fps; pressure rows: started_at_iso + raw_t_us column; "
                "same clock per S2 ruling Baselines/决策/05_R1-R3_风险裁定.md §三 证据 C); "
                "rows without a raw timestamp are never selected; out-of-span video frames clamp to the edge sample"
                if resample_meta["absolute"] else
                "legacy normalized grid, nearest-row values (no selectable raw_t_us rows available)"
            ),
            "video_frame0_wall_s": resample_meta["video_frame0_wall_s"],
            "pressure_epoch_wall_s": resample_meta["pressure_epoch_wall_s"],
            "out_of_span_frames_left": list(resample_meta["out_of_span_frames_left"]),
            "out_of_span_frames_right": list(resample_meta["out_of_span_frames_right"]),
            "left_hash": source_hashes["left_source"], "right_hash": source_hashes["right_source"],
        },
    }
    (session_root / "session.json").write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_artifact(session_root, schema_version="shared.session.v1", producer="build_shared.py", repository_root=ROOT,
                   parameters={"camera": row["camera"], "fps": row["target_fps"], "pressure_alignment": resample_meta["alignment_method"]},
                   source_artifacts=[row["video_path"], row["bvh_path"], row.get("smpl_path", ""), left_uri, right_uri],
                   source_hashes=source_hashes, frame_id_min=0, frame_id_max=n - 1, frame_count=n,
                   consumers=["model_inputs/<model>"])
    return session_root


def build_mmvp(sid: str, session_root: Path, *, force: bool = False) -> Path:
    pressure = np.load(session_root / "pressure_48.npz", allow_pickle=True)
    # Representation contract omits camera because this is the single frozen
    # camera-3 public conversion; facts retain the explicit camera component.
    target = MMVP_ROOT / session_root.parts[-3] / session_root.parts[-2] / sid
    if target.exists() and not force:
        if not (target / "frame_id.npy").is_file():
            np.save(target / "frame_id.npy", pressure["frame_id"].astype(np.int64))
        return target
    target.mkdir(parents=True, exist_ok=True)
    out = target / "insole"
    out.mkdir(exist_ok=True)
    np.save(target / "frame_id.npy", pressure["frame_id"].astype(np.int64))
    maps = {foot: mmvp_map(foot) for foot in ("L", "R")}
    for index in range(len(pressure["frame_id"])):
        feet = []
        for foot, key in (("L", "left48"), ("R", "right48")):
            indices, mask = maps[foot]
            grid = np.zeros((31, 11), dtype=np.float32)
            grid[mask] = pressure[key][index][indices[mask]]
            feet.append(grid)
        np.save(out / f"{index:06d}.npy", np.stack(feet, axis=0))
    source_hash = sha256_file(session_root / "pressure_48.npz")
    write_artifact(target, schema_version="shared.representation.mmvp_31x11.v1", producer="build_shared.py", repository_root=ROOT,
                   parameters={"mapping": "audited_4x12_to_31x11_nearest_cell", "layout": "31x11 per foot", "mask_outside": 0},
                   source_artifacts=[canonical_uri(session_root / "pressure_48.npz")], source_hashes={"pressure_48": source_hash},
                   frame_id_min=0, frame_id_max=len(pressure["frame_id"]) - 1, frame_count=len(pressure["frame_id"]),
                   consumers=["pressure_toolkit", "FPP-Net"])
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default="", help="comma-separated IDs; default is all eligible manifest sessions")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--facts-only", action="store_true")
    parser.add_argument("--mmvp-only", action="store_true")
    args = parser.parse_args()
    if args.facts_only and args.mmvp_only:
        raise SystemExit("--facts-only and --mmvp-only are mutually exclusive")
    rows = read_manifest()
    wanted = {value.strip() for value in args.sessions.split(",") if value.strip()} or set(rows)
    partial = bool(args.sessions)
    built: list[str] = []
    for sid in sorted(wanted):
        if sid not in rows:
            raise SystemExit(f"{sid} not found in {MANIFEST}")
        if rows[sid].get("eligible_anysole") != "1":
            print(f"skip {sid}: not eligible ({rows[sid].get('eligibility_reason')})")
            continue
        row = rows[sid]
        recording = resolve_uri(row["video_path"], must_exist=True)
        date, subject = recording.parts[-3], recording.parts[-2]
        session_root = FACTS_ROOT / row["camera"] / date / subject / sid
        if not args.mmvp_only:
            session_root = build_one(sid, row, force=args.force)
        if args.mmvp_only:
            if not session_root.is_dir():
                raise SystemExit(f"facts missing for MMVP build: {session_root}")
            build_mmvp(sid, session_root, force=args.force)
        print(f"built {sid}")
        built.append(sid)
    # 部分构建（--sessions）只更新 session 级 artifact，不得覆写根 artifact
    # 的全局统计（08 全局审阅：根 artifact 曾因 C1 单 session 冒烟被覆写成
    # session_count=1，与实际 140 个 session 不符）。
    if partial:
        print(f"partial build ({len(built)} sessions); root artifacts not rewritten")
        return 0
    write_artifact(FACTS_ROOT, schema_version="shared.facts.v1", producer="build_shared.py", repository_root=ROOT,
                   parameters={"session_count": len(built)}, source_artifacts=["protocol://manifests/session_manifest.jsonl"],
                   consumers=["model_inputs/<model>"])
    if args.mmvp_only:
        write_artifact(MMVP_ROOT, schema_version="shared.representations.v1", producer="build_shared.py", repository_root=ROOT,
                       parameters={"representation": "tactile/mmvp_31x11/v1"}, source_artifacts=["shared://facts/sessions"],
                       consumers=["pressure_toolkit", "FPP-Net"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
