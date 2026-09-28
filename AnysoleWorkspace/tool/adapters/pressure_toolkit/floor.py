"""行走地面估计：复用上游 ``calculateFloorNormal`` 的深度地面路径。

地面契约（06 任务书 P1）：

- 地面来自行走场景深度帧（DepthPro），算法为上游
  ``Baselines/pressure_tookit/lib/utils/depth_utils.calculateFloorNormal``；
- 棋盘格标定只用于相机坐标变换，不定义 floor y=0；
- floor artifact 记录选择帧、深度源、算法、normal/trans/matrix 和质量残差。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from AnysoleWorkspace.tool.adapters.pressure_toolkit import common

# 上游深度地面估计实现（坐标契约以其为准）。
if str(common.TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(common.TOOLKIT_ROOT))
from lib.utils.depth_utils import (  # noqa: E402
    calculateFloorNormal,
    depth2PointCloud,
)

TARGET_NORMAL = np.array([0.0, 1.0, 0.0])


@contextlib.contextmanager
def _cwd(path: str):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def slice_points(depth_png: Path, xy: tuple[int, int, int, int],
                 fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """ROI 深度切片点云 (N,3)，过滤非有限点。"""
    import cv2

    depth_map = cv2.imread(str(depth_png), -1).astype(np.float32) / 1000.0
    point_cloud = depth2PointCloud(depth_map, fx, fy, cx, cy)
    slices = point_cloud[xy[0]:xy[1], xy[2]:xy[3], :].reshape(-1, 3)
    return slices[np.isfinite(slices).all(axis=1)]


def signed_floor_distance_m(depth2floor: np.ndarray,
                            points: np.ndarray) -> np.ndarray:
    """点到平面变换后 y 分量 = 到该平面 y=0 的有符号距离（m）。"""
    return transform_points(depth2floor, points)[:, 1]


def transform_points(depth2floor: np.ndarray, points: np.ndarray) -> np.ndarray:
    """把深度相机系点云经 depth2floor 变换到 floor 系（y 向上）。"""
    pts = (depth2floor[:3, :3] @ points.T + depth2floor[:3, 3].reshape(3, 1)).T
    return pts


def plane_residual_mm(depth2floor: np.ndarray, slices: np.ndarray) -> float:
    """质量残差：切片点到估计地面 y=0 的 RMS 距离（mm）。"""
    return float(np.sqrt(np.mean(signed_floor_distance_m(depth2floor, slices) ** 2)) * 1000.0)


def estimate_floor(depth_png: Path, xy: tuple[int, int, int, int],
                   fx: float, fy: float, cx: float, cy: float) -> dict:
    """调用上游 calculateFloorNormal 估计地面。

    Returns normal/trans/depth2floor/residual。上游函数会在 CWD 写
    ``debug/depth_slice_rot.obj``，这里把 CWD 限制在临时目录并清理。
    """
    with tempfile.TemporaryDirectory(prefix="floor_debug_") as tmp:
        # 上游函数无条件写 debug/depth_slice_rot.obj（相对 CWD），先建目录
        os.makedirs(os.path.join(tmp, "debug"), exist_ok=True)
        with _cwd(tmp):
            normal, trans, depth2floor = calculateFloorNormal(
                str(depth_png), xy, fx, fy, cx, cy, save_path=None)
    slices = slice_points(depth_png, xy, fx, fy, cx, cy)
    return {
        "normal": np.asarray(normal, dtype=np.float64),
        "trans": np.asarray(trans, dtype=np.float64),
        "depth2floor": np.asarray(depth2floor, dtype=np.float64),
        "residual_mm": plane_residual_mm(depth2floor, slices),
        "n_points": int(slices.shape[0]),
    }


def write_floor_artifact(adapter_root: Path, date: str, subject: str,
                         estimate: dict, *, depth_png: Path, xy: tuple,
                         depth_source: str, depth_hash: str,
                         frame_id: int) -> Path:
    """写 ``annotations/<date>/floor_info/floor_<subject>.npy`` + 审计 json。

    npy 键为上游 RgbdCamera 契约：trans / normal / depth2floor。
    """
    floor_dir = adapter_root / "annotations" / date / "floor_info"
    floor_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trans": estimate["trans"],
        "normal": estimate["normal"],
        "depth2floor": estimate["depth2floor"],
    }
    floor_path = floor_dir / f"floor_{subject}.npy"
    np.save(floor_path, payload)
    artifact = {
        "schema_version": "pressure_toolkit.floor.v1",
        "date": date,
        "subject": subject,
        "frame_id": int(frame_id),
        "depth_source": depth_source,
        "depth_file": str(depth_png),
        "depth_hash": depth_hash,
        "algorithm": "upstream calculateFloorNormal "
                     "(depth2PointCloud + plane fit + normalizeFloor to [0,1,0])",
        "roi_rows": [int(xy[0]), int(xy[1])],
        "roi_cols": [int(xy[2]), int(xy[3])],
        "target_normal": TARGET_NORMAL.tolist(),
        "normal": estimate["normal"].tolist(),
        "trans": estimate["trans"].tolist(),
        "depth2floor": estimate["depth2floor"].tolist(),
        "residual_mm": estimate["residual_mm"],
        "n_points": estimate["n_points"],
        "checkerboard_used_for_floor": False,
        "checkerboard_role": "camera coordinate transform only (calibration.npy)",
    }
    artifact_path = floor_dir / f"floor_artifact_{date}.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return floor_path


def checkerboard_floor_mapping(calibration_path: Path) -> dict:
    """仅审计用：旧适配器把棋盘格平面当行走地面的错误映射。

    正式流程不使用；D_Test6 用它证明约 1.45m 的偏移来源。
    """
    from AnysoleWorkspace.tool.adapters.pressure_toolkit.calibration import (
        load_calibration,
    )

    cal = load_calibration(calibration_path)
    R, t_mm = cal["R"], cal["t_mm"]
    board_from_camera = np.eye(4, dtype=np.float64)
    board_from_camera[:3, :3] = R.T
    board_from_camera[:3, 3] = -R.T @ (t_mm / 1000.0)
    board_to_floor = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    depth2floor = board_to_floor @ board_from_camera
    normal = R.T @ np.array([0.0, 0.0, 1.0])
    normal /= max(np.linalg.norm(normal), 1e-12)
    return {
        "normal": normal,
        "trans": depth2floor[:3, 3],
        "depth2floor": depth2floor,
        "plane_definition": "published checkerboard world z=0 mapped to floor y=0",
    }


def floor_audit(estimate: dict, checkerboard: dict, slices: np.ndarray) -> dict:
    """估计地面 vs 棋盘格平面的审计量（D_Test6）。

    - 估计平面上的切片 RMS 残差（mm）；
    - 同一批切片点到棋盘格平面的平均有符号距离（m，期望约 1.45）；
    - 两平面法向夹角（deg）。
    """
    est_d = signed_floor_distance_m(estimate["depth2floor"], slices)
    chk_d = signed_floor_distance_m(checkerboard["depth2floor"], slices)
    cos_angle = float(
        np.clip(np.dot(estimate["normal"], checkerboard["normal"]), -1.0, 1.0))
    return {
        "estimated_plane_rms_mm": float(np.sqrt(np.mean(est_d ** 2)) * 1000.0),
        "checkerboard_plane_mean_offset_m": float(np.mean(chk_d)),
        "checkerboard_plane_rms_mm": float(np.sqrt(np.mean(chk_d ** 2)) * 1000.0),
        "normal_angle_deg": float(np.degrees(np.arccos(cos_angle))),
    }
