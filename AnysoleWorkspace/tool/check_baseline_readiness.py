#!/usr/bin/env python3
"""Report baseline data/dependency readiness for the canonical session union."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
RESULTS = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402


def assigned_sessions() -> tuple[dict[str, dict], list[str]]:
    manifest = {}
    for line in (WORKSPACE / "protocol/manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            manifest[row["session_id"]] = row
    ids = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for split in ("train", "val", "test"):
                value = (row.get(split) or "").strip()
                if value:
                    ids.append(value)
    return manifest, sorted(set(ids))


def session_paths(row: dict, sid: str) -> tuple[str, str, int, dict[str, Path]]:
    recording = resolve_uri(row["video_path"], must_exist=True)
    date, subject, n = recording.parts[-3], recording.parts[-2], int(row["n_frames"])
    seq = WORKSPACE / "shared/facts/sessions/cam3" / date / subject / sid
    # v2 布局：SMPL 来源记录在 shared session.json 的 source_files 里；
    # MMVP 31×11 是 FPP-Net/pressure_toolkit 共用的唯一公共表示。
    session_meta = json.loads((seq / "session.json").read_text(encoding="utf-8"))
    smpl = resolve_uri(session_meta["source_files"]["smpl"], must_exist=False)
    mmvp = WORKSPACE / "shared/representations/tactile/mmvp_31x11/v1" / date / subject / sid
    toolkit = WORKSPACE / "model_inputs/pressure_toolkit/v1"
    fpp_adapter = WORKSPACE / "model_inputs/FPP-Net/adapter_v1"
    pto_adapter = WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1"
    return date, subject, n, {
        "motion_color": seq / "rgb",
        "motion_pressure": seq / "pressure_48.npz",
        "motion_smpl": smpl,
        "insole_tool": toolkit / "images" / date / subject / sid / "insole",
        "insole_fpp": mmvp / "insole",
        "mmvp_color_tool": toolkit / "images" / date / subject / sid / "color",
        "mmvp_color_fpp": pto_adapter / date / subject / sid / "color",
        "calibration": toolkit / "images" / date / subject / sid / "calibration.npy",
        "floor": toolkit / "annotations" / date / "floor_info" / f"floor_{subject}.npy",
        "depth": toolkit / "images" / date / subject / sid / "depth",
        "depth_mask": toolkit / "images" / date / subject / sid / "depth_mask",
        "keypoints_tool": toolkit / "input" / subject / sid / "keypoints",
        "keypoints_fpp": fpp_adapter / date / subject / sid / "keypoints",
        "cliff": pto_adapter / date / subject / sid / "CLIFF_results.npz",
        "scene": pto_adapter / date / subject / sid / "template_scene_rgbd.npy",
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
    path = WORKSPACE / "model_inputs/FPP-Net/adapter_v1/dataset_split_temporal5.npy"
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
        with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = (row.get(args.split) or "").strip()
                if value:
                    selected.append(value)
        all_ids = sorted(set(selected))

    checks = [
        ("motion_color", "motion_color", None),
        ("motion_pressure", "motion_pressure", None),
        ("motion_smpl", "motion_smpl", None),
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
            # mmvp_color_fpp 只含 join 后的帧（join_manifest 为对齐权威），
            # 帧数必然 <= n，因此只做存在性检查。
            expected = n if key in {"motion_color", "insole_tool", "insole_fpp", "mmvp_color_tool", "depth", "depth_mask", "keypoints_tool", "keypoints_fpp"} else None
            if present(paths[key], expected, suffix):
                ok_ids.append(sid)
            else:
                missing.append(sid)
        report["checks"][label] = {"ready": len(ok_ids), "total": len(all_ids)}
        report["missing"][label] = missing

    report["dependencies"] = {
        "smpl_neutral": (WORKSPACE / "assets/third_party/smpl/SMPL_NEUTRAL.pkl").is_file(),
        "depthpro": (WORKSPACE / "assets/third_party/pressure_toolkit/depthpro/model.safetensors").is_file(),
        "rtmpose_checkpoint": (WORKSPACE / "assets/third_party/rtmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth").is_file(),
        "cliff_checkpoint": (WORKSPACE / "assets/third_party/MotionPRO/cliff_ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt").is_file(),
        "yolov3_code": (WORKSPACE / "assets/third_party/CLIFF/lib/pytorch_yolo_v3_master/darknet.py").is_file(),
        "yolov3_weights": (WORKSPACE / "assets/third_party/CLIFF/data/ckpt/yolov3.weights").is_file(),
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
        "unified_fpp_v2t": [],
    }
    fpp_expected = fpp_expected_counts(args.split)
    report["fpp_temporal_windows"] = {
        sid: int(fpp_expected.get(sid, 0)) for sid in all_ids
    }
    # 评估整改任务 02：原生拟合结果按 canonical workspace fitting 根检查；
    # 迁移期保留旧 results 树作为回退位置。
    pressure_roots = [
        WORKSPACE / "work/pressure_toolkit/fitting/results",
        RESULTS / "baselines/pressure_toolkit/results",
        RESULTS / "offline/pressure_toolkit/results",
    ]
    for sid in all_ids:
        date, subject, n, _ = session_paths(manifest[sid], sid)
        fpp_dir = WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1" / date / subject / sid / "pred_contact_smpl"
        fpp_count = len(list(fpp_dir.glob("*.npy"))) if fpp_dir.is_dir() else 0
        expected_fpp = int(fpp_expected.get(sid, 0))
        if expected_fpp > 0 and fpp_count == expected_fpp:
            artifact_checks["fpp_contact"].append(sid)

        pressure_dir = next(
            (root / date / subject / sid
             for root in pressure_roots if (root / date / subject / sid).is_dir()),
            None,
        )
        pressure_count = len(list(pressure_dir.glob("smpl_*.npz"))) if pressure_dir else 0
        artifact_checks["pressure_native_fit"].append(sid) if pressure_count == n else None

        pose_path = WORKSPACE / "work/VP-MoCap/v1/pose_optimization" / date / subject / sid / "opt_result.pth"
        if pose_path.is_file():
            artifact_checks["posetransopt_native_fit"].append(sid)

        for label, model_dir in (
            ("unified_motionpro", "baselines/MotionPRO"),
            ("unified_pressure_toolkit", "baselines/pressure_toolkit"),
            ("unified_vp_mocap", "baselines/VP-MoCap"),
        ):
            if (RESULTS / model_dir / "predictions/eval_motion" / f"{sid}.npz").is_file():
                artifact_checks[label].append(sid)
        if (RESULTS / "baselines/FPP-Net/predictions/v2t" / f"{sid}.npz").is_file():
            artifact_checks["unified_fpp_v2t"].append(sid)
    report["artifacts"] = {
        label: {"ready": len(ids), "total": len(all_ids), "missing": sorted(set(all_ids) - set(ids))}
        for label, ids in artifact_checks.items()
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
