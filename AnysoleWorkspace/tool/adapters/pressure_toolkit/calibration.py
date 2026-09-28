"""cam3 标定 → 上游 calibration.npy 契约。

DepthPro 在 RGB 图上估计深度，深度相机即 cam3 彩色相机，d2c 为单位阵。
棋盘格标定只用于相机坐标变换（K/D/R/t 记录为 provenance），不参与地面定义。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from AnysoleWorkspace.tool.adapters.pressure_toolkit.common import (
    CALIBRATION_DIR,
)


def load_calibration(protocol_json: Path) -> dict:
    """读取 protocol/calibration/<date>.json 的 cam3 标定。"""
    data = json.loads(protocol_json.read_text(encoding="utf-8"))
    camera = data.get("cameras", {}).get("cam3")
    if not isinstance(camera, dict) or any(
            key not in camera for key in ("K", "D", "R", "t")):
        raise ValueError(f"calibration {protocol_json} has no complete cam3 K/D/R/t")
    K = np.asarray(camera["K"], dtype=np.float64)
    R = np.asarray(camera["R"], dtype=np.float64)
    t_mm = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    if K.shape != (3, 3) or R.shape != (3, 3):
        raise ValueError(
            f"invalid cam3 matrix shapes in {protocol_json}: K={K.shape}, R={R.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-3):
        raise ValueError(f"cam3 R is not orthonormal: {protocol_json}")
    return {
        "source": protocol_json,
        "K": K,
        "D": np.asarray(camera["D"], dtype=np.float64),
        "R": R,
        "t_mm": t_mm,
    }


def build_calibration_npy(cal: dict) -> dict:
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
        "depth_source": "DepthPro on cam3 RGB; depth and color share the cam3 frame",
        "checkerboard_used_for_floor": False,
    }
