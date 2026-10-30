"""阶段 1 评估共享工具：弱透视世界抬升 + 项目 SMPL-24 协议转换。

抬升口径（2026-10-17 数值验证）：
- J_cam = J_model + (tx,ty,Z)，Z = 2·f_norm_true/s（模型 s 深度），
  光轴偏移校正 tx ← tx − 2·cx_norm/s（模型弱透视 camera_center=0 = crop 中心）；
- J_world = A @ J_cam + b（A,b 契约 = p_cam[m]，cam3_pinhole_to_mocap_world 对每相机泛化）；
- 世界轨迹残差（ATE ~0.3m）为弱透视逐帧 s 噪声的已知局限（GT-2D 相机拟合实测引入
  0.43m 深度偏差已弃用，见 eval_zero.py 注释），TTA 阶段用 L_2D 精修。

协议转换：项目协议 = SMPL 原生 24 关节（anysole/types.py JOINT_NAMES 与 SMPL 原生序一致），
由 HMR2.0 SMPL 层的 J_regressor 从 6890 顶点回归得到。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmr2_tta.paths import calibration_json  # noqa: E402


def load_lift_chain(date: str, cam: int) -> dict:
    """标定链的 A,b（p_cam[m] → p_mocap[m]）+ K（fx/fy/cx/cy，世界抬升用）。"""
    data = json.loads(calibration_json(date).read_text(encoding="utf-8"))
    Rc = np.asarray(data["cameras"][f"cam{cam}"]["R"], dtype=np.float64)
    tc = np.asarray(data["cameras"][f"cam{cam}"]["t"], dtype=np.float64).reshape(3, 1)
    mc = data["mocap_raw_to_checkerboard_world"]
    Rm = np.asarray(mc["R"], dtype=np.float64)
    tm = np.asarray(mc["t"], dtype=np.float64).reshape(3, 1)
    K = np.asarray(data["cameras"][f"cam{cam}"]["K"], dtype=np.float64)
    A = Rm.T @ Rc.T
    b = -Rm.T @ (Rc.T @ tc + tm) / 100.0
    return {"A": A, "b": b, "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2])}


def lift_to_world(joints_model: np.ndarray, pred_cam: np.ndarray, patch_w: np.ndarray,
                  lift: dict, center: np.ndarray | None = None) -> np.ndarray:
    """模型口径抬升：s 深度 + 光轴校正（center 给 crop 中心全分辨率像素）。"""
    t = weak_persp_t(pred_cam, patch_w, lift, center)
    return lift_to_world_t(joints_model, t, lift)


def weak_persp_t(pred_cam: np.ndarray, patch_w: np.ndarray, lift: dict,
                 center: np.ndarray | None = None) -> np.ndarray:
    """模型弱透视 → 相机系平移 t=(tx,ty,Z)[m]。

    x_norm = (X+tx)·s/2（模型，camera_center=0）与 x_norm = f_norm·X_true/Z + cx_norm
    （真实）对同一图像点相等 ⇒ X_true = (X+tx) − 2·cx_norm/s；Z = 2·f_norm_true/s。
    """
    s = np.maximum(pred_cam[:, 0], 1e-6)
    tx, ty = pred_cam[:, 1].copy(), pred_cam[:, 2].copy()
    if center is not None:
        tx = tx - 2.0 * (lift["cx"] - center[:, 0]) / patch_w / s
        ty = ty - 2.0 * (lift["cy"] - center[:, 1]) / patch_w / s
    z = 2.0 * (lift["fx"] / patch_w) / s
    return np.stack([tx, ty, z], axis=1).astype(np.float64)


def lift_to_world_t(joints_model: np.ndarray, t_cam: np.ndarray, lift: dict) -> np.ndarray:
    """(N,J,3) 模型空间关节 + (N,3) 相机系平移 t[m] → 世界系 (N,J,3)。"""
    j_cam = joints_model + t_cam[:, None, :]
    return np.einsum("ij,nkj->nki", lift["A"], j_cam) + lift["b"].ravel()


def to_project_protocol(smpl_layer, vertices: torch.Tensor) -> np.ndarray:
    """HMR2.0 顶点 → 项目 SMPL-24 协议（原生 24 关节，米）。"""
    reg = smpl_layer.J_regressor.cpu()  # (24, 6890)
    joints = torch.einsum("ij,bjk->bik", reg, vertices).detach().cpu().numpy()
    return joints
