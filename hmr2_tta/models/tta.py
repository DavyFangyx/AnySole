"""论文 TTA 复现（视觉 θ 支，2026-10-30 收口版，无标定）：per-view learned
SMPL token 优化 + virtual view（《Monocular Models Are Strong Learners for
Multi-View Human Mesh Recovery》）。

2026-10-30 收口 + 正确性修正（用户评审六项 + 无标定裁定）：
1. L_2D 目标 = RTMPose-Halpe26 检测器（44 槽 conf 掩码），非 GT 投影；
2. 无标定（用户裁定：TTA 本体即无标定方法）：零外部标定——投影 = 模型
   自带弱透视相机（deccam 读出，s/tx/ty），无任何标定链；
3. L_mesh 各视角先骨盆对齐再各自去朝向（J @ R^T，修正原 R 方向错误）后
   比较（消除 44-45cm 伪分歧）；θ 一致不含 global_orient（视角相关朝向）；
4. 优化协议对齐论文：200 步、lr 0.1、20 步 warmup、梯度裁剪 0.1；
5. virtual view = 独立优化变量（Θ0）；eval 侧另报 init 聚合臂（不迭代对照）。

评估口径：GT 相机系 44 关节（骨盆对齐 + Procrustes，PA 对朝向免疫）。
"""

from __future__ import annotations

import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

PELVIS = 0   # SMPL-24 原生骨盆索引（统一读出 SMPL-24，与 AnySole 协议一致）
# Halpe-26 → SMPL-24 监督子集（与 AnySole TTA 同一映射表，21 点）
S24_BODY = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 12, 0]
J_EXTRA = [24, 29, 32, 30, 33, 31, 34]  # 鼻/L大趾/R大趾/L小趾/R小趾/L跟/R跟


def quat_median_filter(rotmats: np.ndarray, delta: float = 0.26) -> np.ndarray:
    """(V,3,3) rotmat → 1-δ 滤波均值 rotmat（四元数中位，剔除测地距离 > δ 的视角）。"""
    quats = R.from_matrix(rotmats).as_quat()  # (V,4)
    med_q = np.median(quats, axis=0)
    med_q = med_q / np.linalg.norm(med_q)
    dists = np.arccos(np.clip(np.abs(quats @ med_q), -1, 1))
    keep = dists <= delta
    if keep.sum() == 0:
        keep = np.ones(len(quats), dtype=bool)
    mean = R.from_quat(quats[keep].mean(axis=0)).as_matrix()
    return mean


def init_virtual_view(poses_6d: torch.Tensor, betas: torch.Tensor,
                      cams: torch.Tensor, delta: float = 0.26,
                      world_rot: torch.Tensor | None = None) -> dict:
    """(V,144) 6D + (V,10) β + (V,3) cam → virtual view Θ0（1-δ 滤波）。

    world_rot=(V,3,3) 时 global_orient 先旋到世界系再滤波（校准模式，论文口径）。
    """
    from hmr2.utils.geometry import rot6d_to_rotmat
    # 官方 rot6d_to_rotmat 对 (V,144) 输入返回 (V*24,3,3)，view 回 (V,24,3,3)
    rot = rot6d_to_rotmat(poses_6d).view(-1, 24, 3, 3).detach()
    rot_np = rot.cpu().numpy()
    if world_rot is not None:
        A = world_rot.detach().cpu().numpy()
        rot_np[:, 0] = np.einsum("vij,vjk->vik", A, rot_np[:, 0])
    out = np.empty((24, 3, 3))
    for j in range(24):
        out[j] = quat_median_filter(rot_np[:, j], delta)
    rot0 = torch.from_numpy(out.astype(np.float32))
    pose0 = rot_to_6d(rot0)  # (24,6) → (144,)
    return {"pose_6d": pose0.flatten().float(),
            "betas": betas.detach().mean(0).float(),
            "cam": cams.detach().median(0).values.float()}


def rot_to_6d(rot: torch.Tensor) -> torch.Tensor:
    """(J,3,3) rotmat → (J,6) 6D（与官方 rot6d_to_rotmat 的第一二列口径一致）。"""
    a1, a2 = rot[..., 0], rot[..., 1]
    return torch.cat([a1, a2], dim=-1)


class TTAFrameSet:
    """单帧集合（V 视角）的 TTA 优化器（校准/免校准，视觉 θ 支）。"""

    def __init__(self, model, tokens: torch.Tensor, init_theta: dict,
                 det2d: torch.Tensor, centers: np.ndarray, patch_ws: np.ndarray,
                 lambda_theta: float = 1.0, lambda_beta: float = 1.0,
                 lambda_mesh: float = 1.0, lambda_2d: float = 0.3,
                 lambda_reg: float = 1e-2):
        """无标定（用户裁定：TTA 本体即无标定方法）。tokens (V,1024)；
        init_theta = {pose_6d (V,144), betas (V,10), cam (V,3)}；det2d
        (V,44,3) 归一化 crop 坐标 + conf（Halpe-26 映射子集 conf>0，其余 0）。
        投影 = 模型自带弱透视相机（deccam 读出），零外部标定。
        """
        self.model = model
        self.head = model.smpl_head
        # SMPL-24 原生层（不走 44 关节包装器；与 AnySole 协议同一身体模型文件）
        from hmr2_tta.paths import SMPL_MODEL_PATH
        import smplx
        self.smpl = smplx.create(str(SMPL_MODEL_PATH), "smpl", gender="neutral",
                                 batch_size=1, num_betas=10).to(tokens.device)
        self.l_theta, self.l_beta, self.l_mesh, self.l_reg = (
            lambda_theta, lambda_beta, lambda_mesh, lambda_reg)
        self.l_2d = lambda_2d  # 论文附录 λ2d=0.3（同一归一化 crop 坐标空间）
        v = len(tokens)
        dev = tokens.device
        self.tokens = tokens.detach().clone().requires_grad_(True)
        self.det2d = torch.from_numpy(det2d).to(dev)
        self.init = init_theta
        self.centers = torch.from_numpy(centers.astype(np.float32)).to(dev)
        self.patch_ws = torch.from_numpy(patch_ws.astype(np.float32)).to(dev)
        virt = init_virtual_view(init_theta["pose_6d"], init_theta["betas"],
                                 init_theta["cam"])
        self.vpose = virt["pose_6d"].to(dev).requires_grad_(True)   # (144,)
        self.vbetas = virt["betas"].to(dev).requires_grad_(True)    # (10,)
        self.vcam = virt["cam"].to(dev).requires_grad_(True)        # (3,)
        # global_orient 一致权重 = 0（视角相关朝向不进 L_con）
        self.theta_mask = torch.ones(24, 1, 1, device=dev)
        self.theta_mask[0] = 0.0

    def readout(self, token: torch.Tensor):
        """token (B,1024) → (pose_6d (B,144), betas (B,10), cam (B,3)) 残差叠加均值参数。"""
        pose = self.head.decpose(token) + self.head.init_body_pose
        betas = self.head.decshape(token) + self.head.init_betas
        cam = self.head.deccam(token) + self.head.init_cam
        return pose, betas, cam

    def _smpl_forward(self, rotmat: torch.Tensor, betas: torch.Tensor):
        """rotmat (B,24,3,3) → joints (B,45,3)（[:24]=SMPL-24 原生，
        [24:35]=smplx 附加关节 鼻/趾/跟）, vertices (B,6890,3)。"""
        out = self.smpl(global_orient=rotmat[:, :1], body_pose=rotmat[:, 1:],
                        betas=betas, pose2rot=False)
        return out.joints, out.vertices

    def project_weak(self, pts: torch.Tensor, cam: torch.Tensor):
        """无标定：模型自带弱透视投影。pts (V,24,3) 模型空间（根锚定）
        → 归一化 crop：u = s·pts[:2] + [tx,ty]。
        """
        s = cam[:, 0].unsqueeze(-1).unsqueeze(-1)
        txy = cam[:, 1:].unsqueeze(1)
        return s * pts[..., :2] + txy

    def step_loss(self):
        from hmr2.utils.geometry import rot6d_to_rotmat
        v = len(self.tokens)
        pose_i, betas_i, cam_i = self.readout(self.tokens)          # (V,...)
        rot_i = rot6d_to_rotmat(pose_i).view(v, 24, 3, 3)
        rot0 = rot6d_to_rotmat(self.vpose.unsqueeze(0)).view(24, 3, 3)
        joints_i, _ = self._smpl_forward(rot_i, betas_i)            # (V,44,3)
        joints0, _ = self._smpl_forward(rot0.unsqueeze(0),
                                        self.vbetas.unsqueeze(0))
        joints0 = joints0[0]

        # L_2D：21 点监督子集（与 AnySole 线同一映射表）vs 检测器；
        # 无标定：模型自带弱透视投影
        pts = torch.cat([joints_i[:, S24_BODY], joints_i[:, J_EXTRA]], dim=1)
        proj = self.project_weak(pts, cam_i)
        conf = self.det2d[..., 2].unsqueeze(-1)
        l2d = (conf * (proj - self.det2d[..., :2]).pow(2)).mean()

        # L_con（星形，修正 3）：θ 一致（校准：global_orient 先旋世界系；
        # 免校准：global_orient 不进）+ β + 骨盆对齐后的网格
        # θ 一致：global_orient 不进（视角相关朝向）
        l_theta = ((self.theta_mask[0] * (rot_i[:, 0] - rot0[0]).pow(2)).sum() +
                   (self.theta_mask[1:] * (rot_i[:, 1:] - rot0[1:]).pow(2)).sum()) / (v * 24)
        l_beta = (betas_i - self.vbetas).pow(2).mean()
        j_i_rel = joints_i[:, :24] - joints_i[:, PELVIS:PELVIS + 1]    # SMPL-24 骨盆对齐
        j0_rel = joints0[:24] - joints0[PELVIS:PELVIS + 1]
        # 各自去朝向（修正方向：J @ R^T）
        j_i = torch.einsum("vkj,vij->vki", j_i_rel,
                           rot_i[:, 0].transpose(-1, -2))
        j0 = torch.einsum("kj,ij->ki", j0_rel, rot0[0].transpose(-1, -2))
        l_mesh = (j_i - j0).pow(2).mean()
        lcon = self.l_theta * l_theta + self.l_beta * l_beta + self.l_mesh * l_mesh

        # L_reg：锚定初值（姿态/β/cam [+t 校准模式]）
        lreg = ((pose_i - self.init["pose_6d"]).pow(2).mean() +
                (betas_i - self.init["betas"]).pow(2).mean() +
                (cam_i - self.init["cam"]).pow(2).mean())
        return self.l_2d * l2d + lcon + self.l_reg * lreg, (l2d, lcon, lreg)

    def optimize(self, steps: int = 200, lr: float = 0.1, warmup: int = 20,
                 clip: float = 0.1):
        """修正 4：论文协议——200 步、lr 0.1、20 步 warmup、梯度裁剪 0.1。"""
        params = [self.tokens, self.vpose, self.vbetas, self.vcam]
        opt = torch.optim.Adam(params, lr=lr)
        hist = []
        for step in range(steps):
            opt.param_groups[0]["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))
            opt.zero_grad()
            loss, parts = self.step_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            hist.append((loss.item(),) + tuple(p.item() for p in parts))
        return np.array(hist)

    def result(self):
        """返回 per-view 精修 Θ_i 与 virtual view Θ0（detach，numpy）。"""
        from hmr2.utils.geometry import rot6d_to_rotmat
        pose_i, betas_i, cam_i = self.readout(self.tokens)
        with torch.no_grad():
            rot_i = rot6d_to_rotmat(pose_i).view(-1, 24, 3, 3).cpu().numpy()
            joints_i, _ = self._smpl_forward(
                rot6d_to_rotmat(pose_i).view(-1, 24, 3, 3), betas_i)
            rot0 = rot6d_to_rotmat(self.vpose.unsqueeze(0)).view(24, 3, 3)
            joints0, _ = self._smpl_forward(rot0.unsqueeze(0),
                                            self.vbetas.unsqueeze(0))
        return {"pose_rot_i": rot_i, "betas_i": betas_i.detach().cpu().numpy(),
                "cam_i": cam_i.detach().cpu().numpy(),
                "joints_i": joints_i.detach().cpu().numpy(),

                "pose_rot0": rot0.unsqueeze(0).detach().cpu().numpy(),
                "betas0": self.vbetas.detach().cpu().numpy(),
                "cam0": self.vcam.detach().cpu().numpy(),
                "joints0": joints0.detach().cpu().numpy()}
