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


def read_pressure(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if not np.isfinite(values).all() or not np.isfinite(time).all():
        raise ValueError(f"non-finite pressure source: {path}")
    return time, values, valid, fake


def resample_pressure(left: tuple, right: tuple, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_t, left_values, left_valid, left_fake = left
    right_t, right_values, right_valid, right_fake = right
    target = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)

    def interp(times: np.ndarray, values: np.ndarray) -> np.ndarray:
        x = (times - times[0]) / max(times[-1] - times[0], 1.0)
        return np.stack([np.interp(target, x, values[:, col]) for col in range(values.shape[1])], axis=1).astype(np.float32)

    left48, right48 = interp(left_t, left_values), interp(right_t, right_values)
    left_index = np.rint(target * max(len(left_t) - 1, 0)).astype(int)
    right_index = np.rint(target * max(len(right_t) - 1, 0)).astype(int)
    fake = np.maximum(left_fake[left_index], right_fake[right_index]).astype(np.uint8)
    valid = np.minimum(left_valid[left_index], right_valid[right_index]).astype(np.uint8)
    valid = np.minimum(valid, 1 - fake).astype(np.uint8)
    return left48, right48, valid, fake, target


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
    left48, right48, valid, fake, normalized_pressure_time = resample_pressure(left_raw, right_raw, n)
    date, subject = recording.parts[-3], recording.parts[-2]
    session_root = FACTS_ROOT / row["camera"] / date / subject / sid
    if session_root.exists() and not force and all((session_root / name).is_file() for name in ("session.json", "frames.npz", "pressure_48.npz")):
        return session_root
    session_root.mkdir(parents=True, exist_ok=True)
    rgb_source = recording / "3"
    images = sorted(rgb_source.glob("*.jpg"))
    if not images:
        raise ValueError(f"{sid}: raw RGB has no extracted frames")
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
        source_time_column=np.asarray("t_us"), alignment_method=np.asarray("linear_values_nearest_quality_on_normalized_t_us"),
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
            "shared_alignment": "values linearly resampled to the shared frame count over normalized source t_us; valid/fake nearest-neighbour",
            "left_hash": source_hashes["left_source"], "right_hash": source_hashes["right_source"],
        },
    }
    (session_root / "session.json").write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_artifact(session_root, schema_version="shared.session.v1", producer="build_shared.py", repository_root=ROOT,
                   parameters={"camera": row["camera"], "fps": row["target_fps"], "pressure_alignment": session["pressure_provenance"]["shared_alignment"]},
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
