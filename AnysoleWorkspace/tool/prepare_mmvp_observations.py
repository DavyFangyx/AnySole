#!/usr/bin/env python3
"""Prepare the shared MMVP directory layout without regenerating insole data.

This adapter only links the canonical RGB frames and derives calibration/floor
files from the date-level calibration summary.  The existing 31x11 insole
files under ``derived/pressure_tookit`` and ``derived/VP-MoCap`` are treated as
read-only inputs.  Depth, RTMPose and CLIFF remain explicit front-end stages;
the generated report records their presence and never substitutes GT data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
MANIFEST = WORKSPACE / "manifests/session_manifest.jsonl"
SPLITS = WORKSPACE / "splits/default/splits.csv"
SEQ_ROOT = WORKSPACE / "derived/MotionPRO/sequences/cam3"
TOOLKIT_ROOT = WORKSPACE / "derived/pressure_tookit"
FPP_ROOT = WORKSPACE / "derived/VP-MoCap"


def manifest_rows() -> dict[str, dict]:
    rows = {}
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["session_id"]] = row
    return rows


def split_sessions(split: str) -> list[str]:
    columns = ("train", "val", "test") if split == "all" else (split,)
    result = []
    with SPLITS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in columns)
    return sorted({value for value in result if value})


def sequence_info(row: dict) -> tuple[str, str, Path]:
    parts = Path(row["pressure_path"]).parts
    cam = parts.index("cam3")
    date, subject, session = parts[cam + 1 : cam + 4]
    return date, subject, SEQ_ROOT / date / subject / session


def link_one(source: Path, target: Path, force: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(f"refusing to replace existing adapter file: {target}")
        target.unlink()
    target.symlink_to(os.path.relpath(source, target.parent))


def link_color(seq_dir: Path, output_dir: Path, force: bool) -> int:
    frames = sorted(p for p in (seq_dir / "color").iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not frames:
        raise FileNotFoundError(f"no canonical RGB frames: {seq_dir / 'color'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(frames):
        # Keep a stable numeric frame index independent of the original camera
        # timestamp stem; all MMVP consumers sort these names.
        link_one(source, output_dir / f"{index:03d}{source.suffix.lower()}", force)
    return len(frames)


def load_calibration(date: str) -> dict:
    path = WORKSPACE / "calibration" / f"{date}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing date calibration: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    camera = data.get("cameras", {}).get("cam3")
    if not isinstance(camera, dict) or any(key not in camera for key in ("K", "D", "R", "t")):
        raise ValueError(f"calibration {path} has no complete cam3 K/D/R/t")
    K = np.asarray(camera["K"], dtype=np.float64)
    R = np.asarray(camera["R"], dtype=np.float64)
    t_mm = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    if K.shape != (3, 3) or R.shape != (3, 3):
        raise ValueError(f"invalid cam3 matrix shapes in {path}: K={K.shape}, R={R.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-3):
        raise ValueError(f"cam3 R is not orthonormal: {path}")
    return {"source": path, "K": K, "D": np.asarray(camera["D"], dtype=np.float64), "R": R, "t_mm": t_mm}


def write_calibration_and_floor(cal: dict, toolkit_session: Path, toolkit_root: Path, date: str, subject: str) -> None:
    K, R, t_mm = cal["K"], cal["R"], cal["t_mm"]
    calibration = {
        # DepthPro is run on the RGB image, so its depth camera is explicitly
        # the same calibrated camera.  d2c is therefore identity, not a guess.
        "color_Intr": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])},
        "depth_Intr": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])},
        "d2c": np.eye(4, dtype=np.float64),
        "distortion": cal["D"],
        "source_calibration": str(cal["source"]),
        "depth_source": "DepthPro on cam3 RGB; depth and color share the cam3 frame",
    }
    np.save(toolkit_session / "calibration.npy", calibration)

    # The published calibration defines X_camera = R X_checkerboard + t and
    # checkerboard coordinates use z=0 as the calibration plane.  Map that
    # plane to the toolkit's y-up floor coordinates [x, z, y].
    board_from_camera = np.eye(4, dtype=np.float64)
    board_from_camera[:3, :3] = R.T
    board_from_camera[:3, 3] = -R.T @ (t_mm / 1000.0)
    board_to_floor = np.array([[1., 0., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 0., 1.]])
    depth2floor = board_to_floor @ board_from_camera
    normal_camera = R.T @ np.array([0., 0., 1.])
    normal_camera /= max(np.linalg.norm(normal_camera), 1e-12)
    floor = {
        "trans": depth2floor[:3, 3],
        "normal": normal_camera,
        "depth2floor": depth2floor,
        "source_calibration": str(cal["source"]),
        "plane_definition": "published checkerboard world z=0 mapped to floor y=0",
    }
    floor_dir = toolkit_root / "annotations" / date / "floor_info"
    floor_dir.mkdir(parents=True, exist_ok=True)
    np.save(floor_dir / f"floor_{subject}.npy", floor)


def finish_depth(toolkit_session: Path, fpp_session: Path, make_mask: bool, make_scene: bool) -> None:
    depth_files = sorted((toolkit_session / "depth").glob("*.png"))
    if not depth_files:
        return
    if make_mask:
        mask_dir = toolkit_session / "depth_mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for depth_path in depth_files:
            depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
            mask = ((depth >= 400) & (depth <= 5000)).astype(np.uint8) * 255
            Image.fromarray(mask, mode="L").save(mask_dir / depth_path.name)
    if make_scene:
        color_files = sorted(p for p in (toolkit_session / "color").iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not color_files:
            raise FileNotFoundError(f"no color frame for template scene: {toolkit_session}")
        depth = np.asarray(Image.open(depth_files[0]), dtype=np.float32) / 1000.0
        rgb = np.asarray(Image.open(color_files[0]).convert("RGB"))
        np.save(fpp_session / "template_scene_rgbd.npy", {"rgb": rgb, "depth": depth})


def prepare_session(row: dict, force: bool, make_depth_mask: bool, make_scene: bool) -> dict:
    date, subject, seq_dir = sequence_info(row)
    toolkit_session = TOOLKIT_ROOT / "images" / date / subject / row["session_id"]
    fpp_session = FPP_ROOT / date / subject / row["session_id"]
    toolkit_session.mkdir(parents=True, exist_ok=True)
    fpp_session.mkdir(parents=True, exist_ok=True)
    n_color = link_color(seq_dir, toolkit_session / "color", force)
    link_color(seq_dir, fpp_session / "color", force)
    cal = load_calibration(date)
    write_calibration_and_floor(cal, toolkit_session, TOOLKIT_ROOT, date, subject)
    np.save(fpp_session / "calibration.npy", np.load(toolkit_session / "calibration.npy", allow_pickle=True).item())
    (TOOLKIT_ROOT / "input" / subject / row["session_id"] / "keypoints").mkdir(parents=True, exist_ok=True)
    (fpp_session / "keypoints").mkdir(parents=True, exist_ok=True)
    finish_depth(toolkit_session, fpp_session, make_depth_mask, make_scene)
    insole_toolkit = toolkit_session / "insole"
    insole_fpp = fpp_session / "insole"
    report = {
        "session_id": row["session_id"], "date": date, "subject": subject,
        "n_manifest": int(row["n_frames"]), "n_color": n_color,
        "n_insole_toolkit": len(list(insole_toolkit.glob("*.npy"))),
        "n_insole_fpp": len(list(insole_fpp.glob("*.npy"))),
        "calibration": str(toolkit_session / "calibration.npy"),
        "floor": str(TOOLKIT_ROOT / "annotations" / date / "floor_info" / f"floor_{subject}.npy"),
        "depth": len(list((toolkit_session / "depth").glob("*.png"))),
        "keypoints_toolkit": len(list((TOOLKIT_ROOT / "input" / subject / row["session_id"] / "keypoints").glob("*.npy"))),
        "keypoints_fpp": len(list((fpp_session / "keypoints").glob("*.npy"))),
        "cliff": (fpp_session / "CLIFF_results.npz").is_file(),
        "template_scene_rgbd": (fpp_session / "template_scene_rgbd.npy").is_file(),
    }
    report_path = fpp_session / "adapter_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--make-depth-mask", action="store_true",
                        help="derive 0.4-5.0m uint8 masks from existing DepthPro PNGs")
    parser.add_argument("--make-template-scene", action="store_true",
                        help="write PoseTransOpt template_scene_rgbd.npy from first RGB-D frame")
    args = parser.parse_args()
    rows = manifest_rows()
    sessions = [s for s in args.sessions.split(",") if s] if args.sessions else split_sessions(args.split)
    reports = [prepare_session(rows[sid], args.force, args.make_depth_mask, args.make_template_scene)
               for sid in sessions]
    print(json.dumps({"sessions": len(reports), "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
