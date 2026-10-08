#!/usr/bin/env python3
"""PoseTransOpt 原生系 → mocap 世界系变换（04_E §3.5 漂移点 m 执行，2026-10-08）。

PoseTransOpt 原生输出在 cam3 相机锚定系：x 右、y 上、z 前、原点 = 相机中心
（`visualize_utils.py:24` Camera.rotation=diag(-1,-1,1) 把该系投成 u=-x/z, v=-y/z）。
与标定针孔系（y 下）的关系：``X_cam3 = diag(-1,-1,1) @ X_native``。

修复口径：
- 旋转：标定链 ``Rm.T @ Rc.T @ diag(-1,-1,1)``（无自由参数，见
  ``calibration.cam3_pinhole_to_mocap_world``）；
- 平移：标定链平移实测偏差 ~0.42m（mocap_raw_to_checkerboard_world 的 tm 侧），
  逐日期用该日期全部 PoseTransOpt work outputs 与 smpl GT 骨盆均值最小二乘拟合 b，
  结果落盘 ``protocol/calibration/<date>.posetransopt_native_to_mocap.json``。

消费端：``export_baseline_motion.py::export_vp_mocap`` 与
``posetransopt/export_v2m.py``（两个写点同源同改）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.adapters.mmvp_series.calibration.calibration import (  # noqa: E402
    cam3_pinhole_to_mocap_world,
)
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

WORK_ROOT = WORKSPACE / "work/VP-MoCap/v1/pose_optimization"
CALIBRATION_DIR = WORKSPACE / "protocol/calibration"
MANIFEST = WORKSPACE / "protocol/manifests/session_manifest.jsonl"

# PoseTransOpt native (x 右, y 上, z 前, 原点=相机中心) -> 标定针孔系 (y 下)。
NATIVE_TO_CAM3 = np.diag([-1.0, -1.0, 1.0])


def _manifest_rows() -> dict[str, dict]:
    rows = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def _gt_pelvis_mocap(row: dict) -> np.ndarray:
    """Session 网格上的 GT 骨盆轨迹（mocap 世界系 y-up m），即 load_smpl 的 trans_m。"""
    from anysole.data.smpl_io import load_smpl

    n = int(float(row["n_frames"]))
    fps = float(row.get("target_fps") or 40.0)
    query_t = (float(row["visual_start_s"]) + np.arange(n, dtype=np.float64) / fps
               - float(row["offset_s"]))
    motion = load_smpl(resolve_uri(row["smpl_path"]), query_t=query_t)
    return np.asarray(motion["trans_m"], dtype=np.float64)


def fit_artifact(date: str, force: bool = False) -> dict:
    """拟合 <date> 的 PoseTransOpt 原生系 → mocap 世界系变换并落盘 artifact。"""
    cal_json = CALIBRATION_DIR / f"{date}.json"
    if not cal_json.is_file():
        raise FileNotFoundError(f"calibration missing: {cal_json}")
    chain = cam3_pinhole_to_mocap_world(cal_json)
    A = chain["A"] @ NATIVE_TO_CAM3
    rows = _manifest_rows()

    pairs = []  # (native_mean, gt_mean) per session, mocap y-up m
    sessions = []
    work_dir = WORK_ROOT / date
    if not work_dir.is_dir():
        raise FileNotFoundError(f"no PoseTransOpt work outputs for date: {work_dir}")
    for opt_path in sorted(work_dir.glob("*/*/opt_result.pth")):
        sid = opt_path.parent.name
        if sid not in rows:
            continue
        row = rows[sid]
        import torch
        data = torch.load(opt_path, map_location="cpu")
        trans = data["trans"].detach().cpu().numpy().astype(np.float64)
        frame_ids = np.asarray(data.get("frame_ids", np.arange(len(trans))), dtype=np.int64).reshape(-1)
        n = int(float(row["n_frames"]))
        keep = (frame_ids >= 0) & (frame_ids < n)
        if not keep.any():
            continue
        gt = _gt_pelvis_mocap(row)
        native_mean = trans[keep].mean(axis=0)
        gt_mean = gt[frame_ids[keep]].mean(axis=0)
        pairs.append((native_mean, gt_mean))
        sessions.append(sid)
    if not pairs:
        raise RuntimeError(f"no fittable sessions for date {date} under {work_dir}")

    P = np.stack([p[0] for p in pairs])
    G = np.stack([p[1] for p in pairs])
    b = (G - (A @ P.T).T).mean(axis=0)
    resid = (A @ P.T).T + b - G
    rmse_mm = float(np.sqrt((resid ** 2).mean()) * 1000.0)
    artifact = {
        "schema": "posetransopt_native_to_mocap.v1",
        "date": date,
        "A": A.tolist(),
        "b": b.tolist(),
        "method": ("rotation from calibration chain Rm.T @ Rc.T @ diag(-1,-1,1); "
                   "translation least-squares fitted over the date's PoseTransOpt "
                   "work outputs vs smpl GT pelvis (session means)"),
        "n_sessions": len(sessions),
        "sessions": sessions,
        "residual_rmse_mm": rmse_mm,
        "residual_max_mm": float(np.abs(resid).max() * 1000.0),
        "calibration_source": str(cal_json),
        "created": "2026-10-08",
        "drift_point": "Baselines/决策/04_E_评估收尾.md §3.5 漂移点 m",
    }
    out = CALIBRATION_DIR / f"{date}.posetransopt_native_to_mocap.json"
    if out.is_file() and not force:
        raise FileExistsError(f"{out} exists; pass --force to refit")
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}  n_sessions={len(sessions)} rmse={rmse_mm:.1f}mm max={artifact['residual_max_mm']:.1f}mm")
    return artifact


def load_artifact(date: str) -> dict:
    """读取 <date> 的变换 artifact；缺失时报错并提示重跑本脚本。"""
    out = CALIBRATION_DIR / f"{date}.posetransopt_native_to_mocap.json"
    if not out.is_file():
        raise FileNotFoundError(
            f"{out} missing; run python {Path(__file__).relative_to(REPO_ROOT)} --date {date} first")
    artifact = json.loads(out.read_text(encoding="utf-8"))
    return {
        "A": np.asarray(artifact["A"], dtype=np.float64),
        "b": np.asarray(artifact["b"], dtype=np.float64),
        "source": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="e.g. 20260810 (protocol/calibration/<date>.json)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fit_artifact(args.date, force=args.force)


if __name__ == "__main__":
    main()
