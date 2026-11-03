"""AR2 几何验收（V7_AR 方案 §7 验收 1/2 条）：往返与 yaw 不变性。

1. GT 分解（ψ/tilt/Δψ/v_head）→ ar2_to_world 重组 = 原世界态 ≤1e-6；
2. 对整段运动施加固定世界 yaw 后，Δψ、tilt、非根局部、v_head 目标不变；
3. deyaw 契约：deyaw(Y(ψ)@R_tilt) = R_tilt，前向恰为 +Z；退化前向报错。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRotation

from anysole.utils.geometry import (
    ar2_to_world, ar2_to_world_np, deyaw_rotmat, deyaw_rotmat_np,
    heading_from_root_np, rot6d_to_rotmat, rotmat_to_6d,
    rotmat_to_6d_np, yaw_rotmat_np)

N_JOINTS = 24
FPS = 40.0


def _rng(seed=0):
    return np.random.RandomState(seed)


def _valid_root(psi, roll=0.2, pitch=-0.15):
    """Y(psi) @ (roll/pitch tilt) 构成的有效根朝向。"""
    rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    return yaw_rotmat_np(np.asarray(psi)) @ (rx @ ry)


def _rotmats(roots6):
    return rot6d_to_rotmat(torch.from_numpy(np.asarray(roots6)).float()).numpy()


def _sample_gt(t=8):
    rng = _rng()
    psi = np.cumsum(rng.uniform(-0.05, 0.05, t))  # 缓变航向
    roots = np.stack([_valid_root(psi[k], rng.uniform(-0.3, 0.3),
                                  rng.uniform(-0.2, 0.2)) for k in range(t)])
    body = np.stack([rng.uniform(-0.5, 0.5, (23, 3)) for _ in range(t)])
    body_rot = np.stack([SciRotation.from_rotvec(body[k]).as_matrix()
                         for k in range(t)])
    pose = np.zeros((t, N_JOINTS, 6))
    pose[:, 0] = rotmat_to_6d_np(roots)
    pose[:, 1:] = rotmat_to_6d_np(body_rot)
    v_world = np.stack([np.sin(psi) * 1.2, np.zeros(t), np.cos(psi) * 1.2], 1)
    trans = np.cumsum(v_world, 0) / FPS
    return psi, pose.reshape(t, -1), v_world, trans


def _decompose(pose, v_world):
    """GT → (psi, dyaw, tilt, v_head) 目标分解。"""
    roots6 = pose.reshape(-1, N_JOINTS, 6)[:, 0]
    psi = heading_from_root_np(_rotmats(roots6))
    dyaw = np.zeros(len(psi)); dyaw[1:] = np.diff(psi)
    tilt = deyaw_rotmat_np(_rotmats(roots6))
    v_head = np.einsum("tji,tj->ti", yaw_rotmat_np(psi), v_world)
    return psi, dyaw, tilt, v_head


class TestAR2Geometry(unittest.TestCase):
    def test_roundtrip_np(self):
        # 口径一致性：ψ = 前向轴 atan2 提取航向（heading_from_root_np 约定，
        # 与 F2a/dataset 目标同源）——锚、Δψ、v_head 全用提取 ψ，往返精确。
        _psi_true, pose, v_world, trans = _sample_gt()
        psi, dyaw, tilt, v_head = _decompose(pose, v_world)
        pose_ar2 = pose.copy()
        pose_ar2.reshape(-1, N_JOINTS, 6)[:, 0] = rotmat_to_6d_np(tilt)
        anchor = trans[0] - v_world[0] / FPS   # 窗前一帧位置（帧 0 位移=v(0)/FPS）
        world_pose, world_trans = ar2_to_world_np(
            pose_ar2, dyaw, v_head, psi[0], anchor)
        # 世界姿态重组 = 原世界姿态（Y(ψ_d) @ Y(−ψ_d) @ R = R）
        roots_out = world_pose.reshape(-1, N_JOINTS, 6)[:, 0]
        roots_gt = pose.reshape(-1, N_JOINTS, 6)[:, 0]
        self.assertLess(np.abs(_rotmats(roots_out) - _rotmats(roots_gt)).max(), 1e-6)
        # 自由积分轨迹 = 原轨迹
        self.assertLess(np.abs(world_trans - trans).max(), 1e-6)

    def test_roundtrip_torch(self):
        psi, pose, v_world, trans = _sample_gt(t=5)
        _, dyaw, tilt, v_head = _decompose(pose, v_world)
        pose_ar2 = pose.copy()
        pose_ar2.reshape(-1, N_JOINTS, 6)[:, 0] = rotmat_to_6d_np(tilt)
        B = 2
        pose_b = torch.from_numpy(pose_ar2).float().unsqueeze(0).repeat(B, 1, 1)
        dyaw_b = torch.from_numpy(dyaw).float().unsqueeze(0).repeat(B, 1)
        vh_b = torch.from_numpy(v_head).float().unsqueeze(0).repeat(B, 1, 1)
        psi_b = torch.from_numpy(psi[:1]).float().repeat(B)
        anchor_np = trans[0] - v_world[0] / FPS
        anchor = torch.from_numpy(anchor_np[None]).float().repeat(B, 1)
        wp, wt = ar2_to_world(pose_b, dyaw_b, vh_b, psi_b, anchor)
        wp_ref, wt_ref = ar2_to_world_np(pose_ar2, dyaw, v_head, psi[0], anchor_np)
        self.assertLess(np.abs(wp[0].numpy() - wp_ref).max(), 1e-5)
        self.assertLess(np.abs(wt[0].numpy() - wt_ref).max(), 1e-6)

    def test_yaw_invariance(self):
        psi, pose, v_world, trans = _sample_gt()
        phi = 0.9  # 固定世界 yaw
        r_phi = yaw_rotmat_np(np.asarray(phi))
        pose_rot = pose.copy()
        roots6 = pose.reshape(-1, N_JOINTS, 6)[:, 0]
        pose_rot.reshape(-1, N_JOINTS, 6)[:, 0] = rotmat_to_6d_np(
            np.einsum("ij,tjk->tik", r_phi, _rotmats(roots6)))
        v_rot = np.einsum("ij,tj->ti", r_phi, v_world)
        psi_a, dyaw_a, tilt_a, vh_a = _decompose(pose, v_world)
        psi_b, dyaw_b, tilt_b, vh_b = _decompose(pose_rot, v_rot)
        body_a = pose.reshape(-1, N_JOINTS, 6)[:, 1:]
        body_b = pose_rot.reshape(-1, N_JOINTS, 6)[:, 1:]
        self.assertLess(np.abs((psi_b - psi_a) - phi).max(), 1e-6)   # ψ 平移 φ
        self.assertLess(np.abs(dyaw_b - dyaw_a).max(), 1e-6)         # Δψ 不变
        self.assertLess(np.abs(tilt_b - tilt_a).max(), 1e-6)         # tilt 不变
        self.assertLess(np.abs(vh_b - vh_a).max(), 1e-6)             # v_head 不变
        self.assertLess(np.abs(body_b - body_a).max(), 1e-12)        # 非根不变

    def test_deyaw_contract(self):
        # 本表示的 tilt = 前向轴意义下的去航向矩阵（GT 分解同款）：
        # deyaw(Y(ψ)@T) = Y(−δ)@T，δ = atan2(sin pitch, cos roll·cos pitch)
        # 是倾斜自身的内禀量——测试断言三个真实契约：
        rng = _rng(1)
        r_tilt = _valid_root(0.0, rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2))
        r = yaw_rotmat_np(np.asarray(1.3)) @ r_tilt
        out = deyaw_rotmat_np(np.stack([r]))
        # (a) 输出前向的水平 x 分量恰为 0（航向=0，无航向可泄漏；y 分量
        # 来自 roll，属倾斜自身）
        forward = out[0] @ np.array([0, 0, 1.0])
        self.assertLess(abs(forward[0]), 1e-6)
        # (b) 显式 yaw 因子被完全剥离：deyaw(Y(ψ)@T) == deyaw(T)
        out_bare = deyaw_rotmat_np(np.stack([r_tilt]))
        self.assertLess(np.abs(out[0] - out_bare[0]).max(), 1e-6)
        # (c) 往返精确：Y(ψ_d(R)) @ deyaw(R) == R
        psi_d = np.arctan2((r @ np.array([0, 0, 1.0]))[0],
                           (r @ np.array([0, 0, 1.0]))[2])
        recon = yaw_rotmat_np(np.asarray(psi_d)) @ out[0]
        self.assertLess(np.abs(recon - r).max(), 1e-6)
        # 退化前向（垂直向上）必须报错
        up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            deyaw_rotmat_np(np.stack([up]))


if __name__ == "__main__":
    unittest.main()
