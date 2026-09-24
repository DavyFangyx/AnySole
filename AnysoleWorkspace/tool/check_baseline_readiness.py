#!/usr/bin/env python3
"""Report baseline data/dependency readiness for the canonical session union."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
RESULTS = ROOT / "results"


def assigned_sessions() -> tuple[dict[str, dict], list[str]]:
    manifest = {}
    for line in (WORKSPACE / "manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            manifest[row["session_id"]] = row
    ids = []
    with (WORKSPACE / "splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for split in ("train", "val", "test"):
                value = (row.get(split) or "").strip()
                if value:
                    ids.append(value)
    return manifest, sorted(set(ids))


def session_paths(row: dict, sid: str) -> tuple[str, str, int, dict[str, Path]]:
    parts = Path(row["pressure_path"]).parts
    cam = parts.index("cam3")
    date, subject, n = parts[cam + 1], parts[cam + 2], int(row["n_frames"])
    seq = WORKSPACE / "derived/MotionPRO/sequences/cam3" / date / subject / sid
    return date, subject, n, {
        "motion_color": seq / "color",
        "motion_pressure": seq / "pressure.npz",
        "motion_smpl": seq / "smpl.npy",
        "motion_feature": seq / "feature_hrnet.pth",
        "motion_bbox": seq / "bbox.npy",
        "insole_tool": WORKSPACE / "derived/pressure_tookit/images" / date / subject / sid / "insole",
        "insole_fpp": WORKSPACE / "derived/VP-MoCap" / date / subject / sid / "insole",
        "mmvp_color_tool": WORKSPACE / "derived/pressure_tookit/images" / date / subject / sid / "color",
        "mmvp_color_fpp": WORKSPACE / "derived/VP-MoCap" / date / subject / sid / "color",
        "calibration": WORKSPACE / "derived/pressure_tookit/images" / date / subject / sid / "calibration.npy",
        "floor": WORKSPACE / "derived/pressure_tookit/annotations" / date / "floor_info" / f"floor_{subject}.npy",
        "depth": WORKSPACE / "derived/pressure_tookit/images" / date / subject / sid / "depth",
        "depth_mask": WORKSPACE / "derived/pressure_tookit/images" / date / subject / sid / "depth_mask",
        "keypoints_tool": WORKSPACE / "derived/pressure_tookit/input" / subject / sid / "keypoints",
        "keypoints_fpp": WORKSPACE / "derived/VP-MoCap" / date / subject / sid / "keypoints",
        "cliff": WORKSPACE / "derived/VP-MoCap" / date / subject / sid / "CLIFF_results.npz",
        "scene": WORKSPACE / "derived/VP-MoCap" / date / subject / sid / "template_scene_rgbd.npy",
    }


def present(path: Path, n: int | None = None, suffix: str | None = None) -> bool:
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    files = [p for p in path.iterdir() if p.is_file() and (suffix is None or p.suffix == suffix)]
    return n is None or len(files) == n


def fpp_expected_counts(split: str) -> dict[str, int]:
    """Return the exact temporal-window center count used by FPP-Net."""
    path = WORKSPACE / "derived" / "VP-MoCap" / "dataset_split_temporal5.npy"
    if not path.is_file():
        return {}
    payload = np.load(path, allow_pickle=True).item()
    phases = ("train", "val", "test") if split == "all" else (split,)
    result: dict[str, int] = {}
    for phase in phases:
        for date, subjects in payload.get(phase, {}).items():
            for subject, sessions in subjects.items():
                for session, frames in sessions.items():
                    result[session] = result.get(session, 0) + len(frames)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    args = parser.parse_args()
    manifest, all_ids = assigned_sessions()
    if args.split != "all":
        selected = []
        with (WORKSPACE / "splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = (row.get(args.split) or "").strip()
                if value:
                    selected.append(value)
        all_ids = sorted(set(selected))

    checks = [
        ("motion_color", "motion_color", None),
        ("motion_pressure", "motion_pressure", None),
        ("motion_smpl", "motion_smpl", None),
        ("motion_feature", "motion_feature", None),
        ("motion_bbox", "motion_bbox", None),
        ("insole_tool", "insole_tool", ".npy"),
        ("insole_fpp", "insole_fpp", ".npy"),
        ("mmvp_color_tool", "mmvp_color_tool", None),
        ("mmvp_color_fpp", "mmvp_color_fpp", None),
        ("calibration", "calibration", None),
        ("floor", "floor", None),
        ("depth", "depth", ".png"),
        ("depth_mask", "depth_mask", ".png"),
        ("keypoints_tool", "keypoints_tool", ".npy"),
        ("keypoints_fpp", "keypoints_fpp", ".npy"),
        ("cliff", "cliff", None),
        ("scene", "scene", None),
    ]
    report = {"split": args.split, "session_count": len(all_ids), "checks": {}, "missing": {}}
    for label, key, suffix in checks:
        ok_ids, missing = [], []
        for sid in all_ids:
            _, _, n, paths = session_paths(manifest[sid], sid)
            expected = n if key in {"motion_color", "insole_tool", "insole_fpp", "mmvp_color_tool", "mmvp_color_fpp", "depth", "depth_mask", "keypoints_tool", "keypoints_fpp"} else None
            if present(paths[key], expected, suffix):
                ok_ids.append(sid)
            else:
                missing.append(sid)
        report["checks"][label] = {"ready": len(ok_ids), "total": len(all_ids)}
        report["missing"][label] = missing

    report["dependencies"] = {
        "smpl_neutral": (WORKSPACE / "dependencies/smpl/SMPL_NEUTRAL.pkl").is_file(),
        "depthpro": (WORKSPACE / "dependencies/pressure_tookit/depthpro/model.safetensors").is_file(),
        "rtmpose_checkpoint": (WORKSPACE / "dependencies/rtmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth").is_file(),
        "cliff_checkpoint": (WORKSPACE / "dependencies/MotionPRO/cliff_ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt").is_file(),
        "yolov3_code": (WORKSPACE / "dependencies/CLIFF/lib/pytorch_yolo_v3_master/darknet.py").is_file(),
        "yolov3_weights": (WORKSPACE / "dependencies/CLIFF/data/ckpt/yolov3.weights").is_file(),
        "fpp_checkpoint": (ROOT / "Baselines/VP-MoCap/FPP-Net/checkpoints/tempKPSMPL_series5_mlp/latest").is_file(),
    }

    # Front-end readiness and model-output readiness are deliberately
    # reported separately.  A complete RGB/depth/CLIFF chain does not imply
    # that a trained FPP model or a fitted baseline output exists.
    artifact_checks = {
        "fpp_contact": [],
        "pressure_native_fit": [],
        "posetransopt_native_fit": [],
        "unified_motionpro": [],
        "unified_pressure_toolkit": [],
        "unified_vp_mocap": [],
    }
    fpp_expected = fpp_expected_counts(args.split)
    report["fpp_temporal_windows"] = {
        sid: int(fpp_expected.get(sid, 0)) for sid in all_ids
    }
    for sid in all_ids:
        date, subject, n, _ = session_paths(manifest[sid], sid)
        fpp_dir = WORKSPACE / "derived/VP-MoCap" / date / subject / sid / "pred_contact_smpl"
        fpp_count = len(list(fpp_dir.glob("*.npy"))) if fpp_dir.is_dir() else 0
        expected_fpp = int(fpp_expected.get(sid, 0))
        if expected_fpp > 0 and fpp_count == expected_fpp:
            artifact_checks["fpp_contact"].append(sid)

        pressure_dir = RESULTS / "offline/pressure_toolkit/results" / date / subject / sid
        pressure_count = len(list(pressure_dir.glob("smpl_*.npz"))) if pressure_dir.is_dir() else 0
        artifact_checks["pressure_native_fit"].append(sid) if pressure_count == n else None

        pose_path = WORKSPACE / "derived/VP-MoCap" / date / subject / sid / "opt_results/opt_result.pth"
        if pose_path.is_file():
            artifact_checks["posetransopt_native_fit"].append(sid)

        for label, model_dir in (
            ("unified_motionpro", "MotionPRO"),
            ("unified_pressure_toolkit", "pressure_tookit"),
            ("unified_vp_mocap", "VP-MoCap"),
        ):
            if (RESULTS / model_dir / "predictions/eval_motion" / f"{sid}.npz").is_file():
                artifact_checks[label].append(sid)
    report["artifacts"] = {
        label: {"ready": len(ids), "total": len(all_ids), "missing": sorted(set(all_ids) - set(ids))}
        for label, ids in artifact_checks.items()
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
