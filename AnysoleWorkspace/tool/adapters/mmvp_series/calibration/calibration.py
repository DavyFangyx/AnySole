"""cam3 标定 → 上游 calibration.npy 契约。

DepthPro 在 RGB 图上估计深度，深度相机即 cam3 彩色相机，d2c 为单位阵。
棋盘格标定只用于相机坐标变换（K/D/R/t 记录为 provenance），不参与地面定义。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from AnysoleWorkspace.tool.adapters.mmvp_series.common.common import (
    CALIBRATION_DIR,
)


def load_calibration(protocol_json: Path, cam: int = 3) -> dict:
    """读取 protocol/calibration/<date>.json 的 cam{N} 标定（默认 cam3）。"""
    data = json.loads(protocol_json.read_text(encoding="utf-8"))
    camera = data.get("cameras", {}).get(f"cam{cam}")
    if not isinstance(camera, dict) or any(
            key not in camera for key in ("K", "D", "R", "t")):
        raise ValueError(f"calibration {protocol_json} has no complete cam{cam} K/D/R/t")
    K = np.asarray(camera["K"], dtype=np.float64)
    R = np.asarray(camera["R"], dtype=np.float64)
    t_mm = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    if K.shape != (3, 3) or R.shape != (3, 3):
        raise ValueError(
            f"invalid cam{cam} matrix shapes in {protocol_json}: K={K.shape}, R={R.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-3):
        raise ValueError(f"cam{cam} R is not orthonormal: {protocol_json}")
    return {
        "source": protocol_json,
        "K": K,
        "D": np.asarray(camera["D"], dtype=np.float64),
        "R": R,
        "t_mm": t_mm,
    }


def build_calibration_npy(cal: dict, cam: int = 3) -> dict:
    """上游 loadCalibrationFromNpy 期望的 dict：color_Intr/depth_Intr/d2c。"""
    K = cal["K"]
    return {
        "color_Intr": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        },
        "depth_Intr": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        },
        "d2c": np.eye(4, dtype=np.float64),
        "distortion": cal["D"],
        "source_calibration": str(cal["source"]),
        "depth_source": f"DepthPro on cam{cam} RGB; depth and color share the cam{cam} frame",
        "checkerboard_used_for_floor": False,
    }


def cam3_pinhole_to_mocap_world(protocol_json: Path) -> dict:
    """cam3 针孔相机系 → mocap 世界系刚体变换（p_mocap[m] = A @ p_cam3[m] + b）。

    标定链（单位 cm，口径与证据见 `Baselines/决策/05_R1-R3_风险裁定.md` §三）：
        X_checkerboard[cm] = Rm @ (100 * J_mocap[m]) + tm   # mocap_raw_to_checkerboard_world
        X_cam3[cm]          = Rc @ X_checkerboard + tc      # world_to_camera
    反解 J_mocap。2026-10-08 漂移点 m/n 执行（MMVP 相机系导出修复）时新增。
    """
    data = json.loads(protocol_json.read_text(encoding="utf-8"))
    Rc = np.asarray(data["cameras"]["cam3"]["R"], dtype=np.float64)
    tc = np.asarray(data["cameras"]["cam3"]["t"], dtype=np.float64).reshape(3)
    mc = data["mocap_raw_to_checkerboard_world"]
    Rm = np.asarray(mc["R"], dtype=np.float64)
    tm = np.asarray(mc["t"], dtype=np.float64).reshape(3)
    if not np.allclose(Rc.T @ Rc, np.eye(3), atol=2e-3):
        raise ValueError(f"cam3 R is not orthonormal: {protocol_json}")
    A = Rm.T @ Rc.T
    b = -Rm.T @ (Rc.T @ tc + tm) / 100.0
    return {"A": A, "b": b, "source": str(protocol_json)}
