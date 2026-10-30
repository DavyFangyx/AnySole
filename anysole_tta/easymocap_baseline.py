"""E3 线 1「外部标定多视角基线」：EasyMocap 原生管线（RTMPose 检测 + 真值外参 + SMPL 拟合）。

流程（实验计划 E3 口径「每视角 2D 关键点 + 真值外参三角化 + SMPL 拟合」）：
RTMPose-Halpe26 检测缓存（推理时可得，与 TTA 各线 L_2D 证据同源）→
OpenPose body25 关节映射（25 点全为实检测点，无近似）→ 真值外参
（protocol/calibration 全链，世界系 m）下 EasyMocap 原生三角化
（simple_recon_person + 重投影清洗 50px）→ smpl_from_keypoints3d2d 多阶段
SMPL 拟合（k3d + k2d + 时序平滑 + GMM 先验，上游默认权重）；n=1 无三角化，
走 multi_stage_optimize2d 纯 2D 拟合（上游 1v1p 同路径，k2d 主项权重）。
纯视觉、无训练、无 T——E3 表 4「外部标定多视角基线（真值外参）」行。

上游：z_note/HMR2.0/EasyMocap（git submodule @ e6006fd，只读 0 改动）；
资产：AnysoleWorkspace/assets/third_party/easymocap/{SMPL_NEUTRAL.pkl 软链,
J_regressor_body25.npy}。EasyMocap 自带 2D 检测器不启用（RTMPose 检测注入），
三角化与 SMPL 拟合全部走原生函数。

输出缓存：results/baselines/HMR2.0/predictions/easymocap/{sid}_n{n}_{camstr}.npz
（joints (T,24,3) SMPL 原生 24 关节、世界系 m、frame_id——与 calib_baseline
同契约；存在即跳过，断点续跑）。评估复用 calib_baseline 同口径函数。

用法（touch_gait env）：
  python anysole_tta/easymocap_baseline.py --sessions S13013 --n 1,4 --device cuda:7
  python anysole_tta/easymocap_baseline.py --split val --n 1,2,3,4 --device cuda:7
"""
from __future__ import annotations

import argparse
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
from easymocap.mytools.reconstruction import projectN3, simple_recon_person  # noqa: E402
from easymocap.pipeline.basic import (multi_stage_optimize2d,  # noqa: E402
                                      smpl_from_keypoints3d2d)
from easymocap.smplmodel.body_model import SMPLlayer  # noqa: E402
from easymocap.smplmodel.body_param import check_keypoints  # noqa: E402

ASSETS = Path("/data/fangyuxuan/projects/gait/AnysoleWorkspace/"
              "assets/third_party/easymocap")
OUT_ROOT = RESULTS_ROOT / "predictions/easymocap"
CONF_THR = 0.3   # 上游 --thres2d 默认
MAX_REPRO = 50   # 上游 --MAX_REPRO_ERROR 默认（px）
# body25(OpenPose 序) 下标 → Halpe-26 下标（25 点全为实检测点）
BODY25_FROM_HALPE = [0, 18, 6, 8, 10, 5, 7, 9, 19, 12, 14, 16, 11, 13, 15,
                     2, 1, 4, 3, 20, 22, 24, 21, 23, 25]
WEIGHT_2D = {"k2d": 1.0, "reg_poses": 1e-3, "reg_poses_zero": 1e-2,
             "smooth_body": 5.0, "smooth_poses": 1.0, "smooth_Rh": 1.0,
             "init_poses": 1e-2}  # n=1 纯 2D（k2d 主项，上游 1v1p 同构）


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


def _init_trans_1v(body_model, kp: np.ndarray, cal: dict) -> np.ndarray:
    """n=1 trans 初始化（零初始化在深度歧义下落入坏局部极小）。

    骨盆射线（body25 MidHip=8，畸变前像素→世界方向）与站立骨盆高度平面
    （y=0.9m 先验，非 GT）求交——与 calib_baseline.fit_2d 同款；骨盆
    conf<阈值帧沿用最近有效帧。均值姿态骨盆 = 模型零参数 forward 原生关节 0。
    """
    t = len(kp)
    with torch.no_grad():
        mean_j = body_model(poses=np.zeros((1, 72)), shapes=np.zeros((1, 10)),
                            Rh=np.zeros((1, 3)), Th=np.zeros((1, 3)),
                            return_verts=False, return_tensor=False,
                            return_smpl_joints=True)[0, 0]
    pel_px = kp[:, 8, :2]
    pel_ok = kp[:, 8, 2] >= CONF_THR
    R = cal["Rc"] @ cal["Rm"]
    c_w = (cal["Rm"].T @ (-cal["Rc"].T @ cal["tc"].reshape(-1)
                          - cal["tm"].reshape(-1))) / 100.0
    k_inv = np.linalg.inv(cal["K"])
    d_cam = np.einsum("ij,nj->ni", k_inv,
                      np.concatenate([pel_px, np.ones((t, 1))], -1))
    d_w = d_cam @ R                                        # 世界射线（未归一）
    d1 = np.where(np.abs(d_w[:, 1]) > 0.05, d_w[:, 1], 0.05)
    lam = np.clip((0.9 - c_w[1]) / d1, 0.2, 8.0)
    x_w = c_w[None] + lam[:, None] * d_w                   # (T,3) 世界系
    trans0 = (x_w - mean_j).astype(np.float64)
    if not pel_ok.all():
        ok_idx = np.where(pel_ok)[0]
        for i in np.where(~pel_ok)[0]:
            j = ok_idx[np.abs(ok_idx - i).argmin()] if len(ok_idx) else 0
            trans0[i] = trans0[j]
    return trans0


def fit_1v(body_model, kp2d: np.ndarray, Pall: np.ndarray, cal: dict) -> dict:
    """n=1：无三角化，multi_stage_optimize2d 纯 2D（上游 1v1p 同路径）。

    trans 初始化 = _init_trans_1v（骨盆射线×站立平面）；优化器与权重全上游。
    """
    kp2d = check_keypoints(kp2d.copy(), min_conf=CONF_THR)
    t = kp2d.shape[1]
    params = body_model.init_params(nFrames=t)
    params["Th"] = _init_trans_1v(body_model, kp2d[0], cal)
    kp2ds = kp2d.transpose(1, 0, 2, 3).astype(np.float32)
    bboxes = _bboxes(kp2d)
    return multi_stage_optimize2d(body_model, params, kp2ds, bboxes,
                                  Pall.astype(np.float32), weight=WEIGHT_2D,
                                  args=_fit_args())


def export_joints(body_model, params: dict) -> np.ndarray:
    """拟合参数 → SMPL 原生 24 关节 (T,24,3) 世界系 m。

    模型自身回归关节（SMPL_NEUTRAL.pkl J_regressor），与 calib_baseline 的
    smplx 层回归关节同族同口径；GT 端为协议 FK 关节（两基线同规则）。
    """
    joints = body_model(poses=params["poses"], shapes=params["shapes"],
                        Rh=params["Rh"], Th=params["Th"],
                        return_verts=False, return_tensor=False,
                        return_smpl_joints=True)
    return joints[:, :24].astype(np.float32)


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
        params = fit_1v(body_model, kp2d, Pall, cals[0])
    joints = export_joints(body_model, params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, joints=joints,
                        frame_id=frame_ids.astype(np.int64))
    gt, _ = load_gt_joints(sid)
    mpjpe, gmpjpe = _eval(joints, gt)
    print(f"[ok] {sid} n={n} {camstr}: MPJPE {mpjpe:.1f} GMPJPE {gmpjpe:.1f}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=None, choices=("val", "test"),
                    help="缺省时用 --sessions")
    ap.add_argument("--sessions", default=None, help="逗号分隔")
    ap.add_argument("--n", default="1,2,3,4", help="视角数（逗号分隔，默认全部）")
    ap.add_argument("--device", default="cuda:7")
    args = ap.parse_args()

    sids = (args.sessions.split(",") if args.sessions else
            (VAL_SESSIONS if args.split == "val" else TEST_SESSIONS))
    ns = sorted({int(x) for x in args.n.split(",")})
    combos = [(1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4),
              (3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4)]
    for sid in sids:
        for n in ns:
            for combo in [c for c in combos if len(c) == n]:
                run_session(sid, list(combo), args.device)


if __name__ == "__main__":
    main()
