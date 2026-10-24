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
TOOL = Path(__file__).resolve().parent
if str(TOOL) not in sys.path:
    sys.path.insert(0, str(TOOL))

from AnysoleWorkspace.tool.artifacts import sha256_file, write_artifact  # noqa: E402
from AnysoleWorkspace.tool.workspace import canonical_uri, resolve_uri  # noqa: E402
from tactile_pipeline.align import nearest_indices, resample_quality  # noqa: E402

WORKSPACE = ROOT / "AnysoleWorkspace"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions"
MMVP_ROOT = WORKSPACE / "shared/representations/tactile/mmvp_31x11/v1"
MANIFEST = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
ESSENTIALS = ROOT / "Baselines/VP-MoCap/FPP-Net/essentials/insole2cont"
LAYOUT = WORKSPACE / "hf-raw" / "config" / "foot_sensor_layout"
# 触觉值正典（2026-10-22 生产替换）：tactile_pipeline 的清洗产物 dataset
# npz（processed48 + wall_time_day_s，与 jpeg 文件名同一日秒墙钟）。
# 2026-10-24 定妆照：行级质量包（{side}_{t_us,raw_t_us,valid,fake}）已由生产
# 端烤进 npz 本身，消费端不再依赖 quality_rows 生产中间件。
TACTILE_DATA_ROOT = WORKSPACE / "hf-raw" / "pressure"
OFFSET_TABLE_CSV = WORKSPACE / "protocol" / "calibration" / "offset_correction.csv"

_OFFSET_TABLE: dict[str, float] | None = None


def corrected_lag(sid: str) -> float:
    """B-class ``lag_eff_frames`` from the offset correction table (frames);
    0.0 for A/C/D (no effective correction; D 的 lag_eff 仅供排序参考)。

    Facts 的 mocap 轴 = manifest 原始对齐 − lag·dt（10-22 固化，见
    AnysoleWorkspace/protocol/calibration/offset_README.md）；重建时在此应用同一修正。
    """
    global _OFFSET_TABLE
    if _OFFSET_TABLE is None:
        import csv
        _OFFSET_TABLE = {}
        with OFFSET_TABLE_CSV.open(encoding="utf-8-sig", newline="") as handle:
            for r in csv.DictReader(handle):
                _OFFSET_TABLE[r["session_id"]] = (
                    float(r["lag_eff_frames"]) if r["conf_class"] == "B" else 0.0)
    return _OFFSET_TABLE.get(sid, 0.0)


def manifest_for(cam: int) -> Path:
    """Manifest source for a camera: the main manifest for cam3 (frozen
    public baseline), the per-camera manifest for cams 1/2/4
    (四相机决策记录 §4-1: ``session_manifest_cam{N}.jsonl``)."""
    if cam == 3:
        return MANIFEST
    path = WORKSPACE / "protocol/manifests" / f"session_manifest_cam{cam}.jsonl"
    if not path.is_file():
        raise SystemExit(
            f"missing per-camera manifest {path}; run build_manifest.py --cam 1,2,3,4 first")
    return path


def read_manifest(cam: int = 3) -> dict[str, dict]:
    return {
        row["session_id"]: row
        for row in (json.loads(line) for line in manifest_for(cam).read_text(encoding="utf-8").splitlines())
        if row.get("session_id")
    }


def v3_value_pack(date: str, subject: str, sid: str, side: str) -> dict[str, np.ndarray]:
    """Canonical pressure values + absolute wall clock + row quality pack from
    the tactile pipeline's dataset npz (fault-cell burst repair + isolated
    impulse + airborne baseline + supported step, no smoothing).

    ``wall_time_day_s`` lives on the same day-second wall clock as the JPEG
    filenames (started_at_iso + native receive clock), so it maps onto video
    frames directly without an epoch anchor.  The ``{side}_*`` quality arrays
    are baked into the npz by the producer (2026-10-24); sessions without a
    producer-side quality record lack these keys.
    """
    from tactile_pipeline import pressure_io
    p = pressure_io.load_pressure(TACTILE_DATA_ROOT / "data", sid, side)
    out = {"values": np.asarray(p["processed48"], dtype=np.float32),
           "wall": np.asarray(p["wall_time_day_s"], dtype=np.float64)}
    if f"{side}_t_us" in p:
        out["quality"] = (np.asarray(p[f"{side}_t_us"]), np.asarray(p[f"{side}_raw_t_us"]),
                          np.asarray(p[f"{side}_valid"]), np.asarray(p[f"{side}_fake"]))
    else:
        out["quality"] = None
    return out


def day_seconds_from_iso(value: str) -> float:
    """Local day-seconds from an ISO timestamp (same wall clock as jpeg names)."""
    _, hhmmss = value.strip().split("T", 1) if "T" in value else value.strip().split(" ", 1)
    hh, mm, rest = hhmmss.split(":")
    ss = float(rest)
    return int(hh) * 3600 + int(mm) * 60 + ss


def day_seconds_from_jpeg_name(name: str) -> float:
    """Local day-seconds from a frame name ``<cam>_HHMMSS.mmm.jpg``.

    Camera-prefix agnostic: only the timestamp tail is parsed, and the four
    cameras share one wall clock (四相机决策记录 D1: JPEG 文件名共钟).
    """
    core = name.split("_")[-1][:-4]  # strip ".jpg"
    return int(core[0:2]) * 3600 + int(core[2:4]) * 60 + float(core[4:])


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


def build_one(sid: str, row: dict, cam: int, *, force: bool = False) -> Path:
    recording = resolve_uri(row["video_path"], must_exist=True)
    date, subject = recording.parts[-3], recording.parts[-2]
    left_pack = v3_value_pack(date, subject, sid, "left")
    right_pack = v3_value_pack(date, subject, sid, "right")
    left_values, left_wall = left_pack["values"], left_pack["wall"]
    right_values, right_wall = right_pack["values"], right_pack["wall"]
    left_uri = (f"AnysoleWorkspace/hf-raw/pressure/"
                f"data/dataset/{date}/{subject}/{sid}/left.npz")
    right_uri = (f"AnysoleWorkspace/hf-raw/pressure/"
                 f"data/dataset/{date}/{subject}/{sid}/right.npz")
    if left_pack["quality"] is None or right_pack["quality"] is None:
        raise ValueError(f"{sid}: npz lacks producer-side quality rows (valid/fake pack)")
    left_quality, right_quality = left_pack["quality"], right_pack["quality"]
    n = int(row["n_frames"])
    rgb_source = recording / str(cam)
    images = sorted(rgb_source.glob("*.jpg"))
    if not images:
        raise ValueError(f"{sid}: raw RGB has no extracted frames")
    meta_path = recording / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    video_start_day_s = day_seconds_from_jpeg_name(images[0].name)
    pressure_epoch_day_s = day_seconds_from_iso(meta["started_at_iso"]) if meta.get("started_at_iso") else None
    fps = float(row["target_fps"])
    targets = video_start_day_s + np.arange(n, dtype=np.float64) / fps
    left48 = left_values[nearest_indices(left_wall, targets)]
    right48 = right_values[nearest_indices(right_wall, targets)]
    valid, fake, _frame_times, resample_meta = resample_quality(
        left_quality, right_quality, n,
        video_start_day_s=video_start_day_s,
        pressure_epoch_day_s=pressure_epoch_day_s,
        fps=fps)
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
    # 与 10-22 固化同一运算顺序：先算原始帧数组、再整体减 lag·dt（位精确复现生产）
    lag_eff = corrected_lag(sid)
    mocap = (float(row.get("mocap_start_s", 0.0))
             + frame_id.astype(np.float64) / float(row["target_fps"])) - lag_eff * 0.025
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
        source_time_column=np.asarray("wall_time_day_s"), alignment_method=np.asarray(resample_meta["alignment_method"]),
    )
    bvh = resolve_uri(row["bvh_path"], must_exist=True)
    smpl = resolve_uri(row["smpl_path"], must_exist=True) if row.get("smpl_path") else None
    meta_path = recording / "meta.json"
    value_csv_left = TACTILE_DATA_ROOT / "data" / "dataset" / date / subject / sid / "left.npz"
    value_csv_right = TACTILE_DATA_ROOT / "data" / "dataset" / date / subject / sid / "right.npz"
    quality_uri = (f"hfraw://pressure/data/quality_rows/"
                   f"{date}/{subject}/{recording.name}.npz")
    source_files = {
        "rgb": row["video_path"], "bvh": row["bvh_path"], "smpl": row.get("smpl_path", ""),
        "left_source": left_uri, "right_source": right_uri,
        "quality_source": quality_uri,
    }
    source_hashes = {
        "left_source": sha256_file(value_csv_left), "right_source": sha256_file(value_csv_right),
        "bvh": sha256_file(bvh), "rgb_meta": sha256_file(meta_path) if meta_path.is_file() else "",
        "rgb_frame_listing": __import__("hashlib").sha256("\n".join(p.name for p in selected_images).encode()).hexdigest(),
    }
    if smpl and smpl.is_file():
        source_hashes["smpl"] = sha256_file(smpl)
    session = {
        "schema_version": "shared.session.v1", "session_id": sid, "date": date, "subject": subject,
        "camera": row["camera"], "fps": float(row["target_fps"]),
        "offset_s": float(row.get("offset_s", 0.0)) + lag_eff * 0.025,
        "frame_count": n, "quality": row.get("quality", ""), "valid_count": int(valid.sum()), "fake_count": int(fake.sum()),
        "source_files": source_files, "source_hashes": source_hashes,
        "pressure_provenance": {
            "layout": "4x12 per foot; 96 cells per frame across both feet", "value_columns": [str(i) for i in range(1, 49)],
            "time_column": "t_us", "quality_columns": ["valid_mask", "fake"],
            "left_source": left_uri, "right_source": right_uri,
            "quality_source": quality_uri,
            "value_processing": (
                "tactile_pressure_handoff_v3: fault-cell burst repair + isolated impulse "
                "+ airborne baseline + supported step (no smoothing)"
            ),
            "shared_alignment": (
                "pressure values nearest-neighbour mapped from v3 handoff processed rows by "
                "absolute wall-clock time (row wall = v3 wall_time_day_s = started_at_iso + "
                f"native receive clock; frame wall = first cam{cam} jpeg name + i/fps; same "
                "clock per S2 ruling Baselines/决策/05_R1-R3_风险裁定.md §三 证据 C); quality "
                "flags (valid/fake) nearest-neighbour mapped from the pipeline quality_rows "
                "marks (started_at_iso + raw_t_us); rows without a raw timestamp are never "
                "selected; out-of-span video frames clamp to the edge sample"
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
                   source_artifacts=[row["video_path"], row["bvh_path"], row.get("smpl_path", ""),
                                     left_uri, right_uri, quality_uri],
                   source_hashes=source_hashes, frame_id_min=0, frame_id_max=n - 1, frame_count=n,
                   consumers=["model_inputs/<model>"])
    return session_root


def build_mmvp(sid: str, session_root: Path, *, force: bool = False) -> Path:
    pressure = np.load(session_root / "pressure_48.npz", allow_pickle=True)
    # Representation contract omits camera because this is the single frozen
    # camera-3 public conversion (四相机决策记录 §4-3: T 流内容相机无关且要求
    # 跨相机一致，维持单棵 cam3 源，per-camera facts 构建跳过本步); facts
    # retain the explicit camera component.
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


def _cam_list(value: str) -> tuple[int, ...]:
    if value.strip() == "all":
        return (1, 2, 3, 4)
    try:
        cams = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError:
        raise SystemExit(f"--cam expects a comma list or 'all', got {value!r}")
    unknown = set(cams) - {1, 2, 3, 4}
    if unknown:
        raise SystemExit(f"unknown camera(s): {sorted(unknown)}")
    if not cams:
        raise SystemExit("--cam list is empty")
    return cams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default="", help="comma-separated IDs; default is all eligible manifest sessions")
    parser.add_argument("--cam", default="3",
                        help="camera(s) to build shared facts for: 'all' or comma list (1,2,3,4), default 3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--facts-only", action="store_true")
    parser.add_argument("--mmvp-only", action="store_true")
    args = parser.parse_args()
    if args.facts_only and args.mmvp_only:
        raise SystemExit("--facts-only and --mmvp-only are mutually exclusive")
    cams = _cam_list(args.cam)
    if args.mmvp_only and cams != (3,):
        raise SystemExit(
            "--mmvp-only 只支持 cam3：MMVP 31x11 表示层是相机无关的单棵 cam3 源"
            "（四相机决策记录 §4-3），per-camera facts 构建跳过 mmvp 步")
    wanted = {value.strip() for value in args.sessions.split(",") if value.strip()}
    partial = bool(args.sessions)
    built_by_cam: dict[int, list[str]] = {}
    for cam in cams:
        rows = read_manifest(cam)
        wanted_cam = wanted or set(rows)
        built: list[str] = []
        for sid in sorted(wanted_cam):
            if sid not in rows:
                raise SystemExit(f"{sid} not found in {manifest_for(cam)}")
            if rows[sid].get("eligible_anysole") != "1":
                print(f"skip {sid} (cam{cam}): not eligible ({rows[sid].get('eligibility_reason')})")
                continue
            row = rows[sid]
            recording = resolve_uri(row["video_path"], must_exist=True)
            date, subject = recording.parts[-3], recording.parts[-2]
            session_root = FACTS_ROOT / row["camera"] / date / subject / sid
            if not args.mmvp_only:
                session_root = build_one(sid, row, cam, force=args.force)
            if args.mmvp_only:
                if not session_root.is_dir():
                    raise SystemExit(f"facts missing for MMVP build: {session_root}")
                build_mmvp(sid, session_root, force=args.force)
            print(f"built {sid} (cam{cam})")
            built.append(sid)
        built_by_cam[cam] = built
    # 部分构建（--sessions）只更新 session 级 artifact，不得覆写根 artifact
    # 的全局统计（08 全局审阅：根 artifact 曾因 C1 单 session 冒烟被覆写成
    # session_count=1，与实际 140 个 session 不符）。
    if partial:
        print(f"partial build ({len(built_by_cam[cams[0]])} sessions); root artifacts not rewritten")
        return 0
    for cam in cams:
        built = built_by_cam[cam]
        if cam == 3:
            write_artifact(FACTS_ROOT, schema_version="shared.facts.v1", producer="build_shared.py", repository_root=ROOT,
                           parameters={"session_count": len(built)}, source_artifacts=["protocol://manifests/session_manifest.jsonl"],
                           consumers=["model_inputs/<model>"])
        else:
            write_artifact(FACTS_ROOT / f"cam{cam}", schema_version="shared.facts.v1", producer="build_shared.py",
                           repository_root=ROOT, parameters={"session_count": len(built), "camera": f"cam{cam}"},
                           source_artifacts=[canonical_uri(manifest_for(cam))],
                           consumers=["model_inputs/<model>"])
    if args.mmvp_only:
        write_artifact(MMVP_ROOT, schema_version="shared.representations.v1", producer="build_shared.py", repository_root=ROOT,
                       parameters={"representation": "tactile/mmvp_31x11/v1"}, source_artifacts=["shared://facts/sessions"],
                       consumers=["pressure_toolkit", "FPP-Net"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
