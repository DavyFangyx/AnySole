#!/usr/bin/env python3
"""pressure_tookit 原生系 → mocap 世界系变换（04_E §3.5 漂移点 n 执行，2026-10-08）。

pressure_tookit 原生输出在 cam3 相机锚定系：x 右、y 上（地面归一化 [0,1,0]）、
**z 指向相机**（SMPLify 式负深度；与 PoseTransOpt 的 z 前向相反）。与标定针孔系
（y 下、z 前）的关系：``X_cam3 = diag(1,-1,-1) @ X_native``——48 种轴/符号约定
实测最优（36 session RMSE 335mm，与经验全刚性拟合 326mm 相当）。

修复口径：
- 旋转：标定链 ``Rm.T @ Rc.T @ diag(1,-1,-1)``（无自由参数）；
- 平移：逐日期用该日期全部 pressure_tookit 拟合产物（smpl_*.npz transl）与
  smpl GT 骨盆均值最小二乘拟合 b，落盘
  ``protocol/calibration/<date>.pressure_toolkit_native_to_mocap.json``；
- **已知局限（登记）**：DepthPro 单目深度的逐 session 尺度漂移使绝对深度不可靠
  （拟合残差 ~0.33m、沿深度轴、session 内稳定/session 间漂移），刚性变换只能
  修正坐标系、不能消除该前端误差——属已知偏差 #1 家族，维持现状。

消费端：``export_baseline_motion.py::export_pressure``。
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

FITTING_ROOT = WORKSPACE / "work/pressure_toolkit/fitting"
CALIBRATION_DIR = WORKSPACE / "protocol/calibration"
MANIFEST = WORKSPACE / "protocol/manifests/session_manifest.jsonl"

# pressure_tookit native (x 右, y 上[地面归一化], z 指向相机) -> 标定针孔系 (y 下, z 前)。
NATIVE_TO_CAM3 = np.diag([1.0, -1.0, -1.0])


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
    """拟合 <date> 的 pressure_tookit 原生系 → mocap 世界系变换并落盘 artifact。"""
    cal_json = CALIBRATION_DIR / f"{date}.json"
    if not cal_json.is_file():
        raise FileNotFoundError(f"calibration missing: {cal_json}")
    chain = cam3_pinhole_to_mocap_world(cal_json)
    A = chain["A"] @ NATIVE_TO_CAM3
    rows = _manifest_rows()

    pairs = []
    sessions = []
    results_root = FITTING_ROOT / "fitting" / "results" / date
    if not results_root.is_dir():
        raise FileNotFoundError(f"no pressure_tookit fitting outputs for date: {results_root}")
    for sid_dir in sorted(results_root.glob("*/*")):
        sid = sid_dir.name
        if sid not in rows:
            continue
        frame_paths = sorted(sid_dir.glob("smpl_*.npz"))
        if not frame_paths:
            continue
        row = rows[sid]
        n = int(float(row["n_frames"]))
        transls, ids = [], []
        for path in frame_paths:
            frame = int(path.stem.split("_")[-1])
            if 0 <= frame < n:
                data = dict(np.load(path, allow_pickle=True))
                transls.append(np.asarray(data["transl"], dtype=np.float64).reshape(3))
                ids.append(frame)
        if not transls:
            continue
        transls = np.stack(transls)
        ids = np.asarray(ids, dtype=np.int64)
        gt = _gt_pelvis_mocap(row)
        pairs.append((transls.mean(axis=0), gt[ids].mean(axis=0)))
        sessions.append(sid)
    if not pairs:
        raise RuntimeError(f"no fittable sessions for date {date} under {results_root}")

    P = np.stack([p[0] for p in pairs])
    G = np.stack([p[1] for p in pairs])
    b = (G - (A @ P.T).T).mean(axis=0)
    resid = (A @ P.T).T + b - G
    rmse_mm = float(np.sqrt((resid ** 2).mean()) * 1000.0)
    artifact = {
        "schema": "pressure_toolkit_native_to_mocap.v1",
        "date": date,
        "A": A.tolist(),
        "b": b.tolist(),
        "method": ("rotation from calibration chain Rm.T @ Rc.T @ diag(1,-1,-1) "
                   "(native z points toward the camera); translation least-squares "
                   "fitted over the date's fitting outputs vs smpl GT pelvis "
                   "(session means)"),
        "n_sessions": len(sessions),
        "sessions": sessions,
        "residual_rmse_mm": rmse_mm,
        "residual_max_mm": float(np.abs(resid).max() * 1000.0),
        "known_limitation": ("DepthPro per-session depth-scale drift: residual is "
                             "systematic along the depth axis, stable within a "
                             "session, drifting across sessions; the rigid transform "
                             "fixes the frame, not the front-end depth error "
                             "(known deviation #1 family)"),
        "calibration_source": str(cal_json),
        "created": "2026-10-08",
        "drift_point": "Baselines/决策/04_E_评估收尾.md §3.5 漂移点 n",
    }
    out = CALIBRATION_DIR / f"{date}.pressure_toolkit_native_to_mocap.json"
    if out.is_file() and not force:
        raise FileExistsError(f"{out} exists; pass --force to refit")
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}  n_sessions={len(sessions)} rmse={rmse_mm:.1f}mm max={artifact['residual_max_mm']:.1f}mm")
    return artifact


def load_artifact(date: str) -> dict:
    """读取 <date> 的变换 artifact；缺失时报错并提示重跑本脚本。"""
    out = CALIBRATION_DIR / f"{date}.pressure_toolkit_native_to_mocap.json"
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
