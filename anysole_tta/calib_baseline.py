"""E3/E7 标定多视角基线（真值外参参考上限）。

流程（实验计划 E3 口径）：每视角 2D 关键点（GT SMPL-24 投影，prep_kp2d24
缓存）+ 真值外参三角化（n≥2）+ SMPL 拟合（GMM 先验，SMPLify 式）；n=1 时
单视角 2D 重投影拟合。纯视觉、无训练、无 T——E3 表 4 参考上限行与 E7
多视角阶段基线共用此产线。

输出缓存：results/baselines/HMR2.0/predictions/calib_baseline/
{sid}_n{n}_{camstr}.npz（joints (T,24,3) 世界系 m、frame_id、j3d、conf）
存在即跳过（断点续跑）。评估由 eval_multiview.py 读缓存完成。

用法（touch_gait env）：
  python anysole_tta/calib_baseline.py --sessions S13013 --n 1,4 --device cuda:7
  python anysole_tta/calib_baseline.py --split val --n 1,2,3,4 --device cuda:7
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根
from hmr2_tta.paths import (PROJECT_ROOT, RESULTS_ROOT, calibration_json,  # noqa: E402
                             facts_session_dir, resolve_uri)
from hmr2_tta.data.assemble_npz import (load_calibration_chain,  # noqa: E402
                                         load_manifest_rows)
from anysole.data.smpl_io import load_smpl  # noqa: E402

SMPL_MODEL = PROJECT_ROOT / "AnysoleWorkspace/assets/third_party/smpl/SMPL_NEUTRAL.pkl"
GMM_PATH = (PROJECT_ROOT / "anysole/ReferenceWorks/motion-diffusion-model/"
            "visualize/joints2smpl/smpl_models/gmm_08.pkl")
_MD_ROOT = PROJECT_ROOT / "anysole/ReferenceWorks/motion-diffusion-model"
OUT_ROOT = RESULTS_ROOT / "predictions/calib_baseline"
CONF_THR = 0.5
N_JOINTS = 24
CHUNK = 16
FIT_ITERS = 100
VAL_SESSIONS = ["S12011", "S12012", "S12013", "S12021", "S12042", "S12061",
                "S12062", "S12073", "S12101", "S12102", "S12103", "S12113"]
TEST_SESSIONS = ["S13011", "S13013", "S13021", "S13023", "S13061", "S13062",
                 "S13063", "S13071", "S13072", "S13073", "S13082", "S13092",
                 "S13093", "S13101", "S13102", "S13103", "S13112", "S13113",
                 "S14011", "S14012", "S14013", "S14022", "S14033", "S14061",
                 "S14062", "S14063", "S14071", "S14072", "S14073", "S14091",
                 "S14092", "S14093", "S14102", "S14103", "S14111", "S14112"]


def load_kp2d24(sid: str, cam: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(RESULTS_ROOT / "data/kp2d24" / f"cam{cam}" / f"{sid}.npz")
    return d["kp2d"], d["frame_id"]


def load_gt_joints(sid: str) -> tuple[np.ndarray, np.ndarray]:
    """facts + load_smpl + 项目 FK → GT 世界关节 (N,24,3)[m]（prep_kp2d24 同路径）。"""
    from anysole.utils.geometry import fk_pose6d_np
    rows = load_manifest_rows(1)
    row = rows[sid]
    facts = np.load(facts_session_dir(sid, 3) / "frames.npz")
    smpl = load_smpl(resolve_uri(row["smpl_path"]), query_t=facts["mocap_time_s"])
    n = len(facts["frame_id"])
    pose = smpl["pose_6d"].reshape(n, 24, 6)
    joints = fk_pose6d_np(pose, smpl["trans_m"], smpl["offsets_m"], smpl["parents"])
    return joints, facts["frame_id"].astype(np.int64)


def triangulate_joints(kp2ds: np.ndarray, confs: np.ndarray,
                       cals: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(V,T,24,2)/(V,T,24) → 世界系 (T,24,3) + 有效视图数 (T,24)。

    各视图射线 DLT 最小二乘（pinhole 近似，忽略畸变；run_tta.triangulate_pelvis
    的 24 关节推广）。有效视图 <2 → NaN（下游置零权重并插值）。
    """
    v, t = kp2ds.shape[:2]
    a_sum = np.zeros((t, N_JOINTS, 3, 3), dtype=np.float64)
    b_sum = np.zeros((t, N_JOINTS, 3), dtype=np.float64)
    n_view = np.zeros((t, N_JOINTS), dtype=np.int64)
    for i in range(v):
        cal = cals[i]
        ok = confs[i] >= CONF_THR
        px = np.concatenate([kp2ds[i], np.ones((t, N_JOINTS, 1))], axis=-1)
        d_cam = px @ np.linalg.inv(cal["K"]).T      # (T,24,3)：K^{-1}·[u,v,1]
        d_w = d_cam @ (cal["Rc"] @ cal["Rm"])       # 相机射线 → 世界方向（链的逆）
        c_w = (cal["Rm"].T @ (-cal["Rc"].T @ cal["tc"].ravel()
                              - cal["tm"].ravel())) / 100.0
        d_w = d_w / np.linalg.norm(d_w, axis=-1, keepdims=True).clip(min=1e-9)
        p_mat = np.eye(3) - np.einsum("tij,tik->tijk", d_w, d_w)
        w = ok.astype(np.float64)
        a_sum += p_mat * w[..., None, None]
        b_sum += np.einsum("tijk,k->tij", p_mat, c_w) * w[..., None]
        n_view += ok.astype(np.int64)
    j3d = np.full((t, N_JOINTS, 3), np.nan, dtype=np.float64)
    good = n_view >= 2
    if good.any():
        a = a_sum[good]
        b = b_sum[good]
        j3d[good] = np.linalg.solve(a, b)
    return j3d, n_view


def build_smpl_layer(device, batch_size: int):
    import smplx
    return smplx.create(str(SMPL_MODEL), "smpl", gender="neutral",
                        batch_size=batch_size, num_betas=10).to(device)


def build_prior(device):
    if str(_MD_ROOT) not in sys.path:
        sys.path.insert(0, str(_MD_ROOT))
    from visualize.joints2smpl.src.prior import MaxMixturePrior
    return MaxMixturePrior(prior_folder=str(GMM_PATH.parent), num_gaussians=8,
                           dtype=torch.float32).to(device)


def _prior_loss(prior, body_aa: torch.Tensor) -> torch.Tensor:
    pose = body_aa.reshape(-1, 69)
    betas = torch.zeros(pose.shape[0], 10, device=pose.device)
    out = prior(pose, betas)
    if out.ndim == 0:
        return out
    return out.mean()


def fit_3d(j3d: np.ndarray, conf: np.ndarray, layer, prior, device,
           iters: int = FIT_ITERS) -> np.ndarray:
    """(T,24,3) 三角化目标 + (T,24) 有效视图数 → 拟合 SMPL 关节 (T,24,3) 世界系。

    chunk LBFGS：变量 global_orient/body_pose 轴角 + trans；损失 =
    Σ conf·‖J+trans−j3d‖² (m²) + w_prior·GMM。逐帧 trans 初值 = 目标 − 均值姿态关节。
    """
    t = len(j3d)
    out = np.full((t, N_JOINTS, 3), np.nan, dtype=np.float32)
    w = (conf >= 2).astype(np.float32)  # 有效视图 ≥2 的关节参与拟合
    j3d_i = np.nan_to_num(j3d, nan=0.0).astype(np.float32)
    with torch.no_grad():
        mean_j = layer(global_orient=torch.zeros(CHUNK, 3, device=device),
                       body_pose=torch.zeros(CHUNK, 69, device=device),
                       betas=torch.zeros(CHUNK, 10, device=device)).joints[0, :N_JOINTS]
    mj = mean_j
    for start in range(0, t, CHUNK):
        if t - start < CHUNK:
            start = t - CHUNK  # 末块与前块重叠，适配 smplx 固定 batch_size
        sl = slice(start, start + CHUNK)
        target = torch.from_numpy(j3d_i[sl]).to(device)
        mask = torch.from_numpy(w[sl]).to(device)
        n_chunk = target.shape[0]
        trans0 = ((target - mj[None]) * mask[..., None]).sum(1) / \
            mask.sum(1, keepdim=True).clamp(min=1)
        trans0 = trans0.contiguous()
        global_orient = torch.zeros(n_chunk, 3, device=device, requires_grad=True)
        body_pose = torch.zeros(n_chunk, 69, device=device, requires_grad=True)
        trans = trans0.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([global_orient, body_pose, trans],
                                max_iter=iters, tolerance_grad=1e-9,
                                tolerance_change=1e-11, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            j = layer(global_orient=global_orient, body_pose=body_pose,
                      betas=torch.zeros(n_chunk, 10, device=device)).joints[:, :N_JOINTS]
            data = (mask[..., None]
                    * ((j + trans[:, None] - target) * 1000.0).pow(2)).sum() / \
                mask.sum().clamp(min=1)
            prior_loss = _prior_loss(prior, body_pose)
            loss = data + 0.01 * prior_loss
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            j = layer(global_orient=global_orient, body_pose=body_pose,
                      betas=torch.zeros(n_chunk, 10, device=device)).joints[:, :N_JOINTS]
            out[start:start + n_chunk] = (j + trans[:, None]).cpu().numpy()
    return out


def fit_2d(kp2d: np.ndarray, conf: np.ndarray, cal_t: dict, layer, prior,
           device, iters: int = FIT_ITERS) -> np.ndarray:
    """(T,24,3)px+conf 单视角 → 2D 重投影拟合（n=1 用）。

    损失 = Σ conf·‖proj(J+trans)−kp2d‖² (px²) + w_prior·GMM；trans 初值 =
    骨盆射线与站立骨盆高度平面（y=0.9m 先验，非 GT）求交。
    """
    from anysole_tta.run_tta import project_world_to_cam_torch
    t = len(kp2d)
    out = np.full((t, N_JOINTS, 3), np.nan, dtype=np.float32)
    w = (conf >= CONF_THR).astype(np.float32)
    kpx = torch.from_numpy(kp2d[..., :2].astype(np.float32)).to(device)
    mask = torch.from_numpy(w).to(device)
    with torch.no_grad():
        mean_j = layer(global_orient=torch.zeros(CHUNK, 3, device=device),
                       body_pose=torch.zeros(CHUNK, 69, device=device),
                       betas=torch.zeros(CHUNK, 10, device=device)).joints[0, :N_JOINTS]
    for start in range(0, t, CHUNK):
        if t - start < CHUNK:
            start = t - CHUNK  # 末块与前块重叠，适配 smplx 固定 batch_size
        sl = slice(start, start + CHUNK)
        n_chunk = CHUNK
        # 骨盆射线与站立骨盆高度平面（y=0.9m 先验，非 GT）求交 → trans 初值
        # （骨盆 conf<阈值帧沿用最近有效帧）
        pel_px = torch.from_numpy(kp2d[sl, 0, :2].astype(np.float32)).to(device)
        pel_ok = conf[sl, 0] >= CONF_THR
        k_inv = torch.linalg.inv(cal_t["K"])
        d_cam = torch.einsum("ij,nj->ni", k_inv,
                             torch.cat([pel_px, torch.ones(n_chunk, 1, device=device)], -1))
        d_w = d_cam @ (cal_t["Rc"] @ cal_t["Rm"])          # 世界射线方向（未归一）
        c_w = (cal_t["Rm"].T @ (-cal_t["Rc"].T @ cal_t["tc"].reshape(-1)
                                - cal_t["tm"].reshape(-1))) / 100.0
        d1 = torch.where(d_w[:, 1].abs() > 0.05, d_w[:, 1],
                         torch.full_like(d_w[:, 1], 0.05))
        lam = (0.9 - c_w[1]) / d1
        lam = lam.clamp(min=0.2, max=8.0)
        x_w = c_w[None] + lam[:, None] * d_w                # (B,3) 世界系
        trans0 = (x_w - mean_j[0:1]).contiguous()
        if not pel_ok.all():
            ok_idx = np.where(pel_ok)[0]
            for i in np.where(~pel_ok)[0]:
                j = ok_idx[np.abs(ok_idx - i).argmin()] if len(ok_idx) else 0
                trans0[i] = trans0[j]
        global_orient = torch.zeros(n_chunk, 3, device=device, requires_grad=True)
        body_pose = torch.zeros(n_chunk, 69, device=device, requires_grad=True)
        trans = trans0.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([global_orient, body_pose, trans],
                                max_iter=iters, tolerance_grad=1e-9,
                                tolerance_change=1e-11, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            j = layer(global_orient=global_orient, body_pose=body_pose,
                      betas=torch.zeros(n_chunk, 10, device=device)).joints[:, :N_JOINTS]
            world = j + trans[:, None]
            proj = project_world_to_cam_torch(world, cal_t, device)
            # 掩码外关节可能位于相机平面后（投影溢出/NaN）——只在掩码内求和，
            # 并对极端重投影误差钳制（坏初始化帧防爆炸）
            err2 = torch.where(mask[start:start + n_chunk] > 0,
                               (proj - kpx[start:start + n_chunk]).pow(2).sum(-1),
                               torch.zeros_like(mask[start:start + n_chunk]))
            err2 = err2.clamp(max=1e4)
            data = err2.sum() / mask[start:start + n_chunk].sum().clamp(min=1)
            loss = data + 0.1 * _prior_loss(prior, body_pose)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            j = layer(global_orient=global_orient, body_pose=body_pose,
                      betas=torch.zeros(n_chunk, 10, device=device)).joints[:, :N_JOINTS]
            out[start:start + n_chunk] = (j + trans[:, None]).cpu().numpy()
    return out


def run_session(sid: str, cams: list[int], device) -> Path:
    n = len(cams)
    camstr = "".join(str(c) for c in cams)
    out_path = OUT_ROOT / f"{sid}_n{n}_{camstr}.npz"
    if out_path.exists():
        print(f"[skip] {sid} n={n} {camstr}")
        return out_path
    date = _date_of(sid)
    kp2ds, confs = [], []
    for cam in cams:
        kp, frame_id = load_kp2d24(sid, cam)
        kp2ds.append(kp[..., :2])
        confs.append(kp[..., 2])
    kp2ds = np.stack(kp2ds)  # (V,T,24,2)
    confs = np.stack(confs)  # (V,T,24)
    layer = build_smpl_layer(device, CHUNK)
    prior = build_prior(device)
    if n >= 2:
        cals = [load_calibration_chain(calibration_json(date), cam) for cam in cams]
        j3d, n_view = triangulate_joints(kp2ds, confs, cals)
        joints = fit_3d(j3d, n_view.astype(np.float32), layer, prior, device)
        # 三角化缺帧线性插值（fit 已对 NaN 置零权重，插值仅回填输出）
        joints = _interp_missing(joints)
    else:
        from anysole_tta.run_tta import cal_tensors
        cal = load_calibration_chain(calibration_json(date), cams[0])
        cal_t = cal_tensors([cal], device)[0]
        joints = fit_2d(kp2ds[0], confs[0], cal_t, layer, prior, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, joints=joints.astype(np.float32),
                        frame_id=frame_id.astype(np.int64))
    gt, _ = load_gt_joints(sid)
    mpjpe, gmpjpe = _eval(joints, gt)
    print(f"[ok] {sid} n={n} {camstr}: MPJPE {mpjpe:.1f} GMPJPE {gmpjpe:.1f}")
    return out_path


def _interp_missing(joints: np.ndarray) -> np.ndarray:
    out = joints.copy()
    for j in range(N_JOINTS):
        col = out[:, j]
        bad = ~np.isfinite(col).all(axis=-1)
        if bad.all() or not bad.any():
            continue
        idx = np.arange(len(col))
        for k in range(3):
            col[bad, k] = np.interp(idx[bad], idx[~bad], col[~bad, k])
    return out


def _eval(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    from anysole.utils.metrics import mean_point_error, pelvis_align
    p, _ = pelvis_align(pred)
    g, _ = pelvis_align(gt)
    mpjpe = float(mean_point_error(p, g, scale=1000).mean())
    gmpjpe = float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000)
    return mpjpe, gmpjpe


def _date_of(sid: str) -> str:
    rows = load_manifest_rows(1)
    return re.search(r"(\d{8})", rows[sid]["video_path"]).group(1)


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
