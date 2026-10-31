"""E3 线 1「外部标定多视角基线」：EasyMocap 原生管线（RTMPose 检测 + 真值外参 + SMPL 拟合）。

流程（实验计划 E3 口径「每视角 2D 关键点 + 真值外参三角化 + SMPL 拟合」）：
RTMPose-Halpe26 检测缓存（推理时可得，与 TTA 各线 L_2D 证据同源）→
OpenPose body25 关节映射（25 点全为实检测点，无近似）→ 真值外参
（protocol/calibration 全链，世界系 m）下 EasyMocap 原生三角化
（simple_recon_person + 重投影清洗 50px）→ smpl_from_keypoints3d2d 多阶段
SMPL 拟合（k3d + k2d + 时序平滑 + 归零正则，上游默认权重）→ 协议 FK 关节
导出（fk_pose6d_np + GT offsets/parents，官方契约；含模板平移 trans 校正）。
n=1 已删除（2026-10-31 用户裁定：原生管线要求 ≥2 视角）。纯视觉、无训练、
无 T——E3 表 4「外部标定多视角基线（真值外参）」行。

上游：z_note/HMR2.0/EasyMocap（git submodule @ e6006fd，只读 0 改动）；
资产：AnysoleWorkspace/assets/third_party/easymocap/{SMPL_NEUTRAL.pkl 软链,
J_regressor_body25.npy}。EasyMocap 自带 2D 检测器不启用（RTMPose 检测注入），
三角化与 SMPL 拟合全部走原生函数。

输出缓存：results/baselines/HMR2.0/predictions/easymocap/{sid}_n{n}_{camstr}.npz
（joints (T,24,3) SMPL 原生 24 关节、世界系 m、frame_id——与 calib_baseline
同契约；存在即跳过，断点续跑）。评估复用 calib_baseline 同口径函数。

用法（touch_gait env）：
  python anysole_tta/easymocap_baseline.py --split test --n 2,3,4 --device cuda:2
  （拟合落缓存 {sid}_n{n}_{camstr}.npz、存在即跳过；跑完自动聚合组合间
  统计写入 metrics/e3_multiview.json 供表 4 导出——--no-aggregate 关闭，
  --aggregate 单独重聚合）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根
from hmr2_tta.paths import RESULTS_ROOT, calibration_json  # noqa: E402
from hmr2_tta.data.assemble_npz import load_calibration_chain  # noqa: E402
from anysole_tta.calib_baseline import (TEST_SESSIONS, VAL_SESSIONS,  # noqa: E402
                                        _date_of, _eval, load_gt_joints)

EMC_ROOT = Path("/data/fangyuxuan/projects/gait/z_note/HMR2.0/EasyMocap")
sys.path.insert(0, str(EMC_ROOT))
from easymocap.dataset import CONFIG  # noqa: E402
from easymocap.mytools.file_utils import get_bbox_from_pose  # noqa: E402
from easymocap.mytools.reconstruction import simple_recon_person  # noqa: E402
from easymocap.pipeline.basic import smpl_from_keypoints3d2d  # noqa: E402
from easymocap.smplmodel.body_model import SMPLlayer  # noqa: E402
from easymocap.smplmodel.body_param import check_keypoints  # noqa: E402

ASSETS = Path("/data/fangyuxuan/projects/gait/AnysoleWorkspace/"
              "assets/third_party/easymocap")
OUT_ROOT = RESULTS_ROOT / "predictions/easymocap"
CONF_THR = 0.3   # 上游 --thres2d 默认
MAX_REPRO = 50   # 上游 --MAX_REPRO_ERROR 默认（px）
CHUNK = 16       # n=1 GMM 拟合批大小（calib_baseline 同款）
# body25(OpenPose 序) 下标 → Halpe-26 下标（25 点全为实检测点）
BODY25_FROM_HALPE = [0, 18, 6, 8, 10, 5, 7, 9, 19, 12, 14, 16, 11, 13, 15,
                     2, 1, 4, 3, 20, 22, 24, 21, 23, 25]


def load_rtmpose_kp(sid: str, cam: int) -> tuple[np.ndarray, np.ndarray]:
    """RTMPose-Halpe26 检测缓存 → kp25 (T,25,3) body25 序 px+conf（畸变像素空间）。"""
    d = np.load(RESULTS_ROOT / "data/rtmpose26" / f"cam{cam}" / f"{sid}.npz")
    kp = d["kp"][:, BODY25_FROM_HALPE].astype(np.float64)       # (T,25,2)
    conf = d["score"][:, BODY25_FROM_HALPE].astype(np.float64)  # (T,25)
    return np.concatenate([kp, conf[..., None]], axis=-1), d["frame_id"]


def undistort_kp(kp: np.ndarray, cal: dict) -> np.ndarray:
    """(T,25,3) 畸变 px → 无畸变 px（cv2.undistortPoints P=K；conf 不变）。"""
    u = cv2.undistortPoints(kp[..., :2].reshape(-1, 1, 2), cal["K"], cal["D"],
                            P=cal["K"])
    out = kp.copy()
    out[..., :2] = u.reshape(kp.shape[0], kp.shape[1], 2)
    return out


def build_pall(cals: list[dict]) -> np.ndarray:
    """标定链 → EasyMocap 投影矩阵 (V,3,4)（世界系 m；X_cam = R@X_w + T）。"""
    Ps = []
    for cal in cals:
        R = cal["Rc"] @ cal["Rm"]
        T = (cal["Rc"] @ cal["tm"] + cal["tc"]) / 100.0        # cm → m
        Ps.append(cal["K"] @ np.hstack([R, T]))
    return np.stack(Ps)


def triangulate(kp2d: np.ndarray, Pall: np.ndarray) -> np.ndarray:
    """(V,T,25,3) 无畸变观测 → kp3ds (T,25,4) 世界系 m。

    上游 simple_recon_person 逐帧 + mv1p 同款重投影清洗：
    残差 > MAX_REPRO px 的观测置零后重三角化。
    """
    v, t = kp2d.shape[:2]
    kp2d = kp2d.copy()
    out = np.zeros((t, 25, 4), dtype=np.float64)
    for i in range(t):
        kp3d, kpts_repro = simple_recon_person(kp2d[:, i], Pall)
        conf = (kp3d[None, :, -1:] > 0) * (kp2d[:, i, :, -1:] > 0)
        dist = np.sqrt((((kpts_repro[..., :2] - kp2d[:, i, :, :2])
                         * conf) ** 2).sum(-1))
        bad_v, bad_j = np.where(dist > MAX_REPRO)
        if bad_v.size:
            kp2d[bad_v, i, bad_j, -1] = 0.0
            kp3d, _ = simple_recon_person(kp2d[:, i], Pall)
        out[i] = kp3d
    return out


def build_model(device):
    """SMPL 层 + body25 回归器（est = body25 关节，与检测 GT 同定义）。"""
    return SMPLlayer(str(ASSETS), model_type="smpl", gender="neutral",
                     device=device,
                     regressor_path=str(ASSETS / "J_regressor_body25.npy"))


def _fit_args() -> argparse.Namespace:
    return argparse.Namespace(model="smpl", gender="neutral", verbose=False,
                              robust3d=False, opts={})


def _bboxes(kp2d: np.ndarray) -> np.ndarray:
    """(V,T,25,3) → (T,V,5)（上游 get_bbox_from_pose：有效检测点外包框 +10%）。"""
    v, t = kp2d.shape[:2]
    return np.stack([[get_bbox_from_pose(kp2d[c, i])
                      for c in range(v)] for i in range(t)]).astype(np.float32)


def fit_mv(body_model, kp2d: np.ndarray, Pall: np.ndarray) -> dict:
    """n≥2：三角化 + smpl_from_keypoints3d2d（上游多阶段拟合，默认权重）。"""
    kp2d = check_keypoints(kp2d.copy(), min_conf=CONF_THR)   # (V,T,25,3)
    kp3ds = triangulate(kp2d, Pall)
    kp3ds = check_keypoints(kp3ds, 1, min_conf=CONF_THR)     # (T,25,4)
    kp2ds = kp2d.transpose(1, 0, 2, 3).astype(np.float32)    # (T,V,25,3)
    bboxes = _bboxes(kp2d)
    return smpl_from_keypoints3d2d(body_model, kp3ds.astype(np.float32),
                                   kp2ds, bboxes, Pall.astype(np.float32),
                                   CONFIG["body25"], _fit_args())


def _init_pnp(body_model, kp: np.ndarray,
              cal: dict) -> tuple[np.ndarray, np.ndarray]:
    """n=1 逐帧 Rh/Th 初始化：旋转 = PnP（均值姿态 body25 骨架 + 检测 2D），
    深度 = 骨盆射线×站立骨盆高度平面（y=0.9m 先验，非 GT；实测 x_w 精确
    投影回检测像素，PnP tvec 深度对部分相机视角退化至 ~9m 不可用）。

    时序 warm-start（上一帧 rvec 作初值）保持翻转一致性；首帧比较
    R 与 R·Ry(π) 两解的重投影残差取优（均值骨架对称的前后歧义，各候选
    配自己的 Th = x_w − R@mean_j）。Th 计及 R@mean_j：该 pkl 骨盆不在
    原点（native 关节 0 ≈ y=−0.22；calib_baseline 因 Rh=0 才可省略此项）。
    可见点 <6 帧沿用最近有效帧。
    """
    t = len(kp)
    with torch.no_grad():
        mean_j = body_model(poses=np.zeros((1, 72)), shapes=np.zeros((1, 10)),
                            Rh=np.zeros((1, 3)), Th=np.zeros((1, 3)),
                            return_verts=False, return_tensor=False,
                            return_smpl_joints=True)[0, 0]        # native 骨盆
        j3d = body_model(poses=np.zeros((1, 72)), shapes=np.zeros((1, 10)),
                         Rh=np.zeros((1, 3)), Th=np.zeros((1, 3)),
                         return_verts=False, return_tensor=False,
                         return_smpl_joints=False)[0]           # (25,3) body25
    K = cal["K"].astype(np.float64)
    Rcw = cal["Rc"] @ cal["Rm"]
    Tcw = (cal["Rc"] @ cal["tm"] + cal["tc"]).reshape(3) / 100.0    # m
    Rcw_inv = Rcw.T
    # 骨盆射线 × y=0.9 站立平面（深度锚）
    pel_px = kp[:, 8, :2]
    k_inv = np.linalg.inv(cal["K"])
    d_cam = np.einsum("ij,nj->ni", k_inv,
                      np.concatenate([pel_px, np.ones((t, 1))], -1))
    d_w = d_cam @ Rcw
    c_w = (cal["Rm"].T @ (-cal["Rc"].T @ cal["tc"].reshape(-1)
                          - cal["tm"].reshape(-1))) / 100.0
    d1 = np.where(np.abs(d_w[:, 1]) > 0.05, d_w[:, 1], 0.05)
    lam = np.clip((0.9 - c_w[1]) / d1, 0.2, 8.0)
    x_w = c_w[None] + lam[:, None] * d_w                     # (T,3) 世界系
    Rh = np.zeros((t, 3))
    Th = np.zeros((t, 3))
    rvec_prev, tvec_prev, prev_i = None, None, None

    def _residual(R_w: np.ndarray, Th_w: np.ndarray, i: int) -> float:
        """候选世界位姿 → 相机投影重投影残差（含世界→相机链）。"""
        m = kp[i, :, 2] >= CONF_THR
        Xc = (Rcw @ (R_w @ j3d[m].T + Th_w[:, None])).T + Tcw    # (m,3)
        px = (K @ Xc.T)[:2] / (K @ Xc.T)[2:3]
        return float(np.mean((px.T - kp[i, m, :2]) ** 2))

    for i in range(t):
        m = kp[i, :, 2] >= CONF_THR
        if m.sum() < 6:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            j3d[m].astype(np.float64), kp[i, m, :2].astype(np.float64), K,
            None, rvec=rvec_prev, tvec=tvec_prev,
            useExtrinsicGuess=rvec_prev is not None,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        R_pnp = cv2.Rodrigues(rvec)[0]
        R_w = Rcw_inv @ R_pnp
        Th_i = x_w[i] - R_w @ mean_j
        if rvec_prev is None:
            R_flip = R_w @ np.diag([-1.0, 1.0, -1.0])
            Th_flip = x_w[i] - R_flip @ mean_j
            if _residual(R_flip, Th_flip, i) < _residual(R_w, Th_i, i):
                R_w, Th_i = R_flip, Th_flip
        Rh[i] = cv2.Rodrigues(R_w)[0].ravel()
        Th[i] = Th_i
        rvec_prev = cv2.Rodrigues(Rcw @ R_w)[0]   # 与所选朝向一致
        tvec_prev = tvec
        prev_i = i
    for i in range(t):
        if Rh[i].any() or Th[i].any():
            continue
        j = prev_i if prev_i is not None else 0
        Rh[i] = Rh[j]
        Th[i] = Th[j]
    return Rh.astype(np.float64), Th.astype(np.float64)


def fit_1v(body_model, kp2d: np.ndarray, Pall: np.ndarray, cal: dict,
           device) -> dict:
    """n=1：无三角化，GMM 姿态先验 CHUNK 批 LBFGS 拟合（检测 2D 重投影）。

    上游 n=1 生产路径 = SPIN 学习初始化 + 轻优化（1v1p），SPIN 权重不可下载；
    上游纯优化路径无姿态先验（仅归零正则），实测塌缩。兜底 =
    calib_baseline.fit_2d 同款口径、CHUNK=16 批（每批 1200 参数；全序列
    LBFGS 24k 参数实测发散）：变量 global_orient/body_pose/trans，Rh/trans
    初值 = _init_pnp（均值姿态骨架 + 检测 2D 的 PnP 求解，透视一致），
    损失 = Σ conf·‖proj−kp2d‖²(px²) + 0.1·GMM(69 维身体姿态)。模型 =
    EasyMocap 身体模型（body25 输出与检测 1:1）；投影 = 无畸变 pinhole
    （Pall，与 kp2d 无畸变口径一致）。
    """
    from anysole_tta.calib_baseline import _prior_loss, build_prior
    kp = kp2d[0]                                             # (T,25,3)
    t = len(kp)
    prior = build_prior(device)
    w = (kp[:, :, 2] >= CONF_THR).astype(np.float32)
    kpx = torch.from_numpy(kp[:, :, :2].astype(np.float32)).to(device)
    mask = torch.from_numpy(w).to(device)
    P = torch.from_numpy(Pall[0].astype(np.float32)).to(device)
    rh0, th0 = _init_pnp(body_model, kp, cal)
    Rh = np.zeros((t, 3))
    Th = np.zeros((t, 3))
    Poses = np.zeros((t, 72))
    ones = torch.ones(CHUNK, 25, 1, device=device)
    zeros3 = torch.zeros(CHUNK, 3, device=device)
    for start in range(0, t, CHUNK):
        if t - start < CHUNK:
            start = t - CHUNK            # 末块与前块重叠，适配固定 batch
        sl = slice(start, start + CHUNK)
        global_orient = torch.tensor(rh0[sl], dtype=torch.float32,
                                     device=device, requires_grad=True)
        body_pose = torch.zeros(CHUNK, 69, device=device, requires_grad=True)
        trans = torch.tensor(th0[sl], dtype=torch.float32, device=device,
                             requires_grad=True)
        opt = torch.optim.LBFGS([global_orient, body_pose, trans],
                                max_iter=100, tolerance_grad=1e-9,
                                tolerance_change=1e-11,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            poses = torch.cat([zeros3, body_pose], dim=1)    # (C,72) 根槽恒等
            j = body_model(poses=poses,
                           shapes=torch.zeros(CHUNK, 10, device=device),
                           Rh=global_orient, Th=trans, return_verts=False,
                           return_tensor=True,
                           return_smpl_joints=False)         # (C,25,3)
            jh = torch.cat([j, ones], dim=-1)                # (C,25,4)
            cam = torch.einsum("ab,cjb->cja", P, jh)         # (C,25,3)
            proj = cam[..., :2] / cam[..., 2:3].clamp(min=1e-6)
            s = torch.where(mask[sl] > 0, (proj - kpx[sl]).pow(2).sum(-1),
                            torch.zeros(CHUNK, 25, device=device))
            # Geman-McClure 稳健项（σ²=1e4 px²）：检测离群点平滑抑制，
            # 硬钳制（>100px 梯度断崖）实测引发强 wolfe 大跳步飞出
            data = (1e4 * s / (1e4 + s)).sum() / mask[sl].sum().clamp(min=1)
            loss = data + 0.1 * _prior_loss(prior, body_pose)
            loss.backward()
            return loss

        opt.step(closure)
        Rh[sl] = global_orient.detach().cpu().numpy()
        Th[sl] = trans.detach().cpu().numpy()
        Poses[sl, 3:] = body_pose.detach().cpu().numpy()
    return {"poses": Poses, "shapes": np.zeros((1, 10)),
            "Rh": Rh, "Th": Th}


_fk_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _protocol_fk_assets(sid: str) -> tuple[np.ndarray, np.ndarray]:
    """GT 协议 offsets/parents（load_smpl，会话级缓存）。"""
    if sid not in _fk_cache:
        from anysole.data.smpl_io import load_smpl
        from hmr2_tta.paths import facts_session_dir, resolve_uri
        from hmr2_tta.data.assemble_npz import load_manifest_rows
        rows = load_manifest_rows(1)
        facts = np.load(facts_session_dir(sid, 3) / "frames.npz")
        smpl = load_smpl(resolve_uri(rows[sid]["smpl_path"]),
                         query_t=facts["mocap_time_s"])
        _fk_cache[sid] = (np.asarray(smpl["offsets_m"], dtype=np.float64),
                          np.asarray(smpl["parents"], dtype=np.int64))
    return _fk_cache[sid]


def export_joints(body_model, params: dict, sid: str) -> np.ndarray:
    """拟合参数 → 协议 FK 关节 (T,24,3) 世界系 m。

    官方导出契约（决策/11 W1）：pred = fk_pose6d_np + GT offsets/parents
    （协议 FK 关节，与 protocol_gt 关节定义一致）——2026-10-31 用户审查
    修正：此前导出模型回归关节（regressed），与契约不符。
    全局旋转 = R(Rh)·R(poses 根槽)、身体 = poses[3:]、trans = Th。
    """
    from anysole.utils.geometry import fk_pose6d_np, rotmat_to_6d_np
    s = _template_shift(body_model)                    # pkl 模板平移（协议系原点差）
    poses = np.asarray(params["poses"], dtype=np.float64)
    root = np.stack([cv2.Rodrigues(v)[0]
                     for v in np.asarray(params["Rh"], dtype=np.float64)])
    root_slot = np.stack([cv2.Rodrigues(v)[0]
                          for v in poses[:, :3].reshape(-1, 3)])  # (T,3,3)
    body = np.stack([cv2.Rodrigues(v)[0]
                     for v in poses[:, 3:].reshape(-1, 3)])
    body = body.reshape(len(root), 23, 3, 3)
    R_total = root @ root_slot                       # R(Rh)·R(poses 根槽)
    rotmats = np.concatenate([R_total[:, None], body], axis=1)   # (T,24,3,3)
    pose_6d = rotmat_to_6d_np(rotmats)                 # (T,24,6)
    offsets, parents = _protocol_fk_assets(sid)
    th_c = np.asarray(params["Th"], dtype=np.float64) \
        + np.einsum("tij,j->ti", R_total, s)           # 拟合系骨盆 → 协议系 trans
    joints = fk_pose6d_np(pose_6d, th_c, offsets, parents)
    return joints.astype(np.float32)


def _template_shift(body_model) -> np.ndarray:
    """pkl 模板相对协议系原点的平移（均值姿态 native 骨盆位置，模型级缓存）。"""
    if not hasattr(body_model, "_emc_shift"):
        with torch.no_grad():
            body_model._emc_shift = body_model(
                poses=np.zeros((1, 72)), shapes=np.zeros((1, 10)),
                Rh=np.zeros((1, 3)), Th=np.zeros((1, 3)),
                return_verts=False, return_tensor=False,
                return_smpl_joints=True)[0, 0].astype(np.float64)
    return body_model._emc_shift


def _eval3(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """PA-MPJPE / MPJPE / GlobalMPJPE（mm）——官方评估层同款函数。

    MPJPE = pelvis_align + mean_point_error、PA-MPJPE = pa_mpjpe、
    GlobalMPJPE = 原始世界系 mean_point_error（README_metrics 指标族表）。
    """
    from anysole.utils.metrics import (mean_point_error, pa_mpjpe,
                                       pelvis_align)
    p, _ = pelvis_align(pred)
    g, _ = pelvis_align(gt)
    mpjpe = float(mean_point_error(p, g, scale=1000).mean())
    pa = float(pa_mpjpe(pred, gt).mean())
    gmpjpe = float(mean_point_error(pred, gt, scale=1000).mean())
    return pa, mpjpe, gmpjpe


def run_session(sid: str, cams: list[int], device) -> Path:
    n = len(cams)
    camstr = "".join(str(c) for c in cams)
    out_path = OUT_ROOT / f"{sid}_n{n}_{camstr}.npz"
    if out_path.exists():
        print(f"[skip] {sid} n={n} {camstr}")
        return out_path
    date = _date_of(sid)
    cals = [load_calibration_chain(calibration_json(date), cam) for cam in cams]
    kp2ds, frame_ids = [], None
    for i, cam in enumerate(cams):
        kp, fid = load_rtmpose_kp(sid, cam)
        if frame_ids is None:
            frame_ids = fid
        else:
            assert np.array_equal(frame_ids, fid), f"{sid} cam{cam} 帧轴不一致"
        kp2ds.append(undistort_kp(kp, cals[i]))
    kp2d = np.stack(kp2ds)                                # (V,T,25,3) 无畸变
    Pall = build_pall(cals)                               # (V,3,4) 世界系 m
    body_model = build_model(device)
    if n >= 2:
        params = fit_mv(body_model, kp2d, Pall)
    else:
        params = fit_1v(body_model, kp2d, Pall, cals[0], device)
    joints = export_joints(body_model, params, sid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, joints=joints,
                        frame_id=frame_ids.astype(np.int64),
                        Rh=params["Rh"], Th=params["Th"],
                        poses=params["poses"], shapes=params["shapes"])
    gt, _ = load_gt_joints(sid)
    pa, mpjpe, gmpjpe = _eval3(joints, gt)
    print(f"[ok] {sid} n={n} {camstr}: PA {pa:.1f} MPJPE {mpjpe:.1f} "
          f"GMPJPE {gmpjpe:.1f}")
    return out_path


def aggregate(split: str | None = None) -> None:
    """全量缓存 → E3 组合间聚合，写入 metrics/e3_multiview.json 的 easymocap 键。

    口径（2026-10-30 用户裁定，与 e_results_export._combo_stats 一致）：
    每组合取会话均值（per-session MPJPE/GMPJPE = _eval 同款），再跨组合
    mean±std；n=4 单一组合 std=None。split 过滤会话集（test = 表 4 正式
    口径，val = 校验，None = 全部）。导出脚本（纯 python 可跑）只读此键，
    不依赖本驱动重算。
    """
    combos = [(1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4),
              (3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4)]
    gts: dict[str, np.ndarray] = {}
    out: dict[str, dict] = {}
    for n in (2, 3, 4):
        p_means, m_means, g_means = [], [], []
        for combo in [c for c in combos if len(c) == n]:
            camstr = "".join(str(c) for c in combo)
            p_vals, m_vals, g_vals = [], [], []
            for f in sorted(OUT_ROOT.glob(f"*_n{n}_{camstr}.npz")):
                sid = f.name.split("_")[0]
                if split == "test" and sid not in TEST_SESSIONS:
                    continue
                if split == "val" and sid not in VAL_SESSIONS:
                    continue
                if sid not in gts:
                    gts[sid] = load_gt_joints(sid)[0]
                d = np.load(f)
                pa, m, g = _eval3(d["joints"], gts[sid])
                p_vals.append(pa)
                m_vals.append(m)
                g_vals.append(g)
            if m_vals:
                p_means.append(float(np.mean(p_vals)))
                m_means.append(float(np.mean(m_vals)))
                g_means.append(float(np.mean(g_vals)))
        if not m_means:
            continue
        out[str(n)] = {
            "pa_mpjpe_mean": float(np.mean(p_means)),
            "pa_mpjpe_std": float(np.std(p_means)) if len(p_means) > 1 else None,
            "mpjpe_mean": float(np.mean(m_means)),
            "mpjpe_std": float(np.std(m_means)) if len(m_means) > 1 else None,
            "gmpjpe_mean": float(np.mean(g_means)),
            "gmpjpe_std": float(np.std(g_means)) if len(g_means) > 1 else None}
        p, ps = out[str(n)]["pa_mpjpe_mean"], out[str(n)]["pa_mpjpe_std"]
        m, ms = out[str(n)]["mpjpe_mean"], out[str(n)]["mpjpe_std"]
        g, gs = out[str(n)]["gmpjpe_mean"], out[str(n)]["gmpjpe_std"]
        print(f"[aggregate] n={n}: PA {p:.3f}"
              f"±{ps if ps is None else round(ps, 3)}  MPJPE {m:.3f}"
              f"±{ms if ms is None else round(ms, 3)}  GMPJPE {g:.3f}"
              f"±{gs if gs is None else round(gs, 3)}")
    metrics = RESULTS_ROOT / "metrics" / "e3_multiview.json"
    d = json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else {}
    key = {"test": "easymocap", "val": "easymocap_val"}.get(split, "easymocap_all")
    d[key] = out
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(d, indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print(f"[aggregate] → {metrics}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=None, choices=("val", "test"),
                    help="缺省时用 --sessions")
    ap.add_argument("--sessions", default=None, help="逗号分隔")
    ap.add_argument("--n", default="2,3,4", help="视角数（逗号分隔；n=1 已删除——"
                    "2026-10-31 用户裁定：原生管线要求 ≥2 视角，单视角 GMM 兜底"
                    "不代表标定方法能力，不列入 E3 表 4）")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--aggregate", action="store_true",
                    help="不拟合：只读缓存算 E3 组合间聚合写 e3_multiview.json")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="拟合后不自动聚合（缺省：拟合跑完自动执行聚合）")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.split)
        return

    sids = (args.sessions.split(",") if args.sessions else
            (VAL_SESSIONS if args.split == "val" else TEST_SESSIONS))
    ns = sorted({int(x) for x in args.n.split(",")})
    combos = [(1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4),
              (3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4)]
    for sid in sids:
        for n in ns:
            for combo in [c for c in combos if len(c) == n]:
                run_session(sid, list(combo), args.device)
    if not args.no_aggregate:
        aggregate(args.split)


if __name__ == "__main__":
    main()
