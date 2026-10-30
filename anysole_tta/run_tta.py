"""AnySole V3_Bv1 多视角 TTA（第一阶段：原方案 + TTA，网格读出版，无标定）。

设计规则（2026-10-26 裁定；2026-10-30 无标定裁定修订）：
- 优化只使用推理时可得信息；GT 只进独立评估，不进初始化/优化/停止/选择；
- 输入条件 = 纯视觉 CONFIG_V（2026-10-30 用户裁定：本线 = AnySole +
  无标定**纯视觉** TTA，代码级固定 CONFIG_V，不依赖运行参数；带触觉的
  VT 线后续单独实现）；
- 优化变量 = per-view h（pose head 部位表示：decoder 输出、_unembed 前，
  论文 token_out 的对应位置；梯度只经冻结的 _unembed/SMPL·FK 层，不回传
  decoder/fusion/编码器）+ 逐视角弱透视相机叶子（R 6D、s、txy）+ 共享
  β 叶子 (10,) init 零（论文 Θ0 的 β 槽位；协议拟合 betas 是标注 GT
  不进优化）；
- **无标定（2026-10-30 用户裁定：TTA 本体即无标定方法，E3 实验设计
  = 无标定 TTA vs 有标定基线）**：不读任何外部标定链。逐视角相机叶子
  init = 模型自身 init 3D ↔ 检测 2D 的 PnP 式最小二乘（init_weak_cam，
  仅用模型输出与检测器）；L_2D 投影 = 逐视角弱透视 u = s·(R@X)[:2] + txy；
- L_2D 预测链 = 论文口径：θ → SMPL(θ, β) → 网格读出（24 SMPL 关节 +
  7 smplx 附加关节）→ 弱透视投影（可微）。监督子集 = Halpe-26 检测器
  有对应标签的 21 点（14 身体关节 + 7 表面点；头顶点映射待验证，
  暂不参与损失）；
- L_con = θ 星形到 1-δ 滤波 virtual view（纯模型输出）——virtual view 为
  每轮聚合锚（简化设计），非论文的独立 Θ0 优化变量，不声称完整复现；
- L_reg = h/姿态/相机/β 锚定；比较三臂：① 单视角 init ② 多视角直接聚合
  （不迭代）③ 多视角 TTA（per-view 精修 + 精修后聚合）。三臂评估均用
  kp_gt（FK 24 关节，GT 只进评估——该口径衡量原评估骨架下的姿态精度，
  不证明优化后的 β/网格/相机准确）。

优化协议（2026-10-26）：Adam + 20 步线性 warmup + 全局梯度裁剪 0.1，
总 200 步；每窗在 0/30/50/100/150/200 步记录三臂与各项损失值；
--grad-diag 在首窗第 0 步输出各损失项的有效梯度范数（损失尺度审计）。
超参在验证集（--split val）选定后跑测试集（--split test）。

用法（touch_gait env）：
  python anysole_tta/run_tta.py --model V7_Cv5 --split val --lr 0.01 --device cuda:7
  python anysole_tta/run_tta.py --model V7_Cv5 --wins 10,11 --dump-kp /tmp/tta_kp  # E7
输出：results/baselines/HMR2.0/metrics/anysole_tta_stage1_{model}_{split}_lr{lr}.json

模型表（TTA 驱动接入，2026-10-27）：
- V3_Bv1：叶子 = h（pose head 部位表示，h_override 注入，model_v2）
- V7_Cv1..Cv5：叶子 = z（V7MotionHead decoder 输出 = 根+9 部位 query，
  z_override 注入，out_parts/root_head 前，model_v7）——V7C 消融组 TTA 管线
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根（anysole/anysole_tta/hmr2_tta）
from hmr2_tta.paths import RESULTS_ROOT  # noqa: E402
from anysole.registry import get_builder  # noqa: E402
from anysole.types import CONFIG_V, CONFIG_VT  # noqa: E402

TTA_MODELS = {
    "V3_Bv1": dict(leaf="h", contact="joint_and",
                   variant="tw20_st20_lr0.0001_lp3_lt1_lk1_la0.05_ls0.01_sf0.1_wu0.05_gc5_ep740_bs256_sd1"),
    "V7_Cv1": dict(leaf="z", contact="f6_soft_time_accel_state_smpl24_v1",
                   variant="tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1"),
    "V7_Cv2": dict(leaf="z", contact="f6_soft_time_accel_state_smpl24_v1",
                   variant="tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1"),
    "V7_Cv3": dict(leaf="z", contact="f6_soft_time_accel_state_smpl24_v1",
                   variant="tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1"),
    "V7_Cv4": dict(leaf="z", contact="f6_soft_time_accel_state_smpl24_v1",
                   variant="tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1"),
    "V7_Cv5": dict(leaf="z", contact="f6_soft_time_accel_state_smpl24_v1",
                   variant="tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1"),
}
SMPL_MODEL = Path("/data/fangyuxuan/projects/gait/AnysoleWorkspace/assets/"
                  "third_party/smpl/SMPL_NEUTRAL.pkl")
MANIFEST = Path("/data/fangyuxuan/projects/gait/AnysoleWorkspace/protocol/"
                "manifests/session_manifest.jsonl")

# ---- 网格读出与 Halpe-26 监督子集（21 点）----
# smplx SMPLLayer joints：[:24] = SMPL native；[24:35] = [鼻, 右眼, 左眼,
# 右耳, 左耳, L大趾, L小趾, L跟, R大趾, R小趾, R跟]。监督子集序与 pts 拼接序
# 一致：14 身体关节（H26 5-16 肩肘腕髋膝踝 + 18 颈 + 19 骨盆）→ S24_BODY，
# 7 表面点（0 鼻 + 20/21 大趾 + 22/23 小趾 + 24/25 跟）→ J_EXTRA。
# H26 17（头顶）映射待验证（模板顶点 3689 偏中线 4.3cm、低于最高点 1.55cm），
# 暂不参与损失。
S24_BODY = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 12, 0]
J_EXTRA = [24, 29, 32, 30, 33, 31, 34]          # 鼻/L大趾/R大趾/L小趾/R小趾/L跟/R跟
H26_ORDER = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19,
             0, 20, 21, 22, 23, 24, 25]
CONF_THR = 0.5
IMG_W, IMG_H = 1624.0, 1240.0
# 损失尺度审计（2026-10-26，S12011 首窗 init）：
#   L_2D（px²）：val ≈ 5500（RMS≈74px）；‖∂l2d/∂h‖ ≈ 226（λ=1 时）
#   L_con（rad²）：val ≈ 0.019（RMS 元素 ≈ 0.14 ≈ 8°）；‖∂lcon/∂h‖ ≈ 6.5e-4
# 平衡（值与梯度同比）：λ2d·5500 ≈ 0.019 → λ2d ≈ 3.5e-6；λ2d·226 ≈ 6.5e-4
#   → λ2d ≈ 2.9e-6。取 λ2d = 3e-6（px²）、λcon = 1.0（rad²）。
# L_reg 为锚定项（init 梯度为零，随优化增长），λreg = 1e-2 按保护角色设。
LAMBDA_2D, LAMBDA_CON, LAMBDA_REG = 3e-6, 1.0, 1e-2
RECORD_STEPS = (0, 30, 50, 100, 150, 200)
VAL_SESSIONS = ["S12011", "S12012", "S12013", "S12021", "S12042", "S12061",
                "S12062", "S12073", "S12101", "S12102", "S12103", "S12113"]
TEST_SESSIONS = ["S13011", "S13013", "S13021", "S13023", "S13061", "S13062",
                 "S13063", "S13071", "S13072", "S13073", "S13082", "S13092",
                 "S13093", "S13101", "S13102", "S13103", "S13112", "S13113",
                 "S14011", "S14012", "S14013", "S14022", "S14033", "S14061",
                 "S14062", "S14063", "S14071", "S14072", "S14073", "S14091",
                 "S14092", "S14093", "S14102", "S14103", "S14111", "S14112"]


def session_date(sid: str) -> str:
    rows = {json.loads(l)["session_id"]: json.loads(l)
            for l in MANIFEST.read_text().splitlines() if l.strip()}
    return re.search(r"(\d{8})", rows[sid]["video_path"]).group(1)


def infer_checkpoint_path(model_name, which):
    from anysole.registry import infer_anysole_paths
    cfg = TTA_MODELS[model_name]
    return infer_anysole_paths(model_name, contact=cfg["contact"], variant=cfg["variant"], which=which)["ckpt"]


def build_model(model_name, device, which: str = "last", checkpoint=None, seed: int = 1):
    from anysole.registry import get_builder, infer_anysole_paths
    cfg = TTA_MODELS[model_name]
    variant = cfg["variant"]
    if seed != 1:
        # variant 尾 sd1 → sdN（多种子 ckpt 目录；2026-10-29）
        variant = variant[: -len("sd1")] + ("sd%d" % seed)
    paths = infer_anysole_paths(model_name, contact=cfg["contact"],
                                variant=variant, which=which)
    ckpt = torch.load(checkpoint or paths["ckpt"], map_location="cpu")
    saved = ckpt["config"]
    entry = get_builder(model_name)
    build_cfg = dict(saved)
    for k in ("model_name", "epochs", "lr"):
        build_cfg.pop(k, None)
    part_joints = build_cfg.pop("part_joints", None)
    model = entry.build(build_cfg, part_joints=part_joints).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.requires_grad_(False)
    return model, saved


def build_smpl_layer(device):
    """中性 SMPL 层（smplx，batch=tw=20，可微 LBS）。

    forward 传 pose2rot=False：直接吃旋转矩阵（x0_hat rot6d→rotmat 全程可微；
    smpl_io 的轴角用法只是输入口径不同，模型文件与关节读出同一份）。
    """
    import smplx
    return smplx.create(str(SMPL_MODEL), "smpl", gender="neutral",
                        batch_size=20, num_betas=10).to(device)


def quat_median_filter(rotmats: np.ndarray, delta: float = 0.26) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    quats = R.from_matrix(rotmats).as_quat()
    med_q = np.median(quats, axis=0)
    med_q = med_q / np.linalg.norm(med_q)
    dists = np.arccos(np.clip(np.abs(quats @ med_q), -1, 1))
    keep = dists <= delta
    if keep.sum() == 0:
        keep = np.ones(len(quats), dtype=bool)
    return R.from_quat(quats[keep].mean(axis=0)).as_matrix()


def virtual_pose(rots: torch.Tensor) -> torch.Tensor:
    """(V,tw,24,3,3) → 1-δ 滤波聚合 (tw,24,3,3)（detach，星形锚）。"""
    v, tw = rots.shape[:2]
    rots_np = rots.detach().cpu().numpy()
    out = np.empty((tw, 24, 3, 3))
    for t in range(tw):
        for j in range(24):
            out[t, j] = quat_median_filter(rots_np[:, t, j])
    return torch.from_numpy(out.astype(np.float32)).to(rots.device)


def project_world_to_cam_torch(joints_world: torch.Tensor, cal_t: dict,
                               dev) -> torch.Tensor:
    """(T,J,3)[m] 世界系 → (T,J,2) px（OpenCV 畸变模型，可微；
    与 assemble_npz.project_world_to_cam 同口径）。"""
    t, j, _ = joints_world.shape
    x_cb = torch.einsum("ij,nj->ni", cal_t["Rm"], 100.0 * joints_world.reshape(-1, 3)) \
        + cal_t["tm"].reshape(1, 3)
    x_cam = torch.einsum("ij,nj->ni", cal_t["Rc"], x_cb) + cal_t["tc"].reshape(1, 3)
    x = x_cam[:, 0] / x_cam[:, 2]
    y = x_cam[:, 1] / x_cam[:, 2]
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = cal_t["D"]
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = cal_t["K"][0, 0] * xd + cal_t["K"][0, 2]
    v = cal_t["K"][1, 1] * yd + cal_t["K"][1, 2]
    return torch.stack([u, v], -1).reshape(t, j, 2)


def init_weak_cam(joints_world: np.ndarray, det_px: np.ndarray,
                  mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,21,3) 骨盆锚定世界关节 + (T,21,2) 检测 px + (T,21) 掩码
    → (R6 (T,6), s (T,1), txy (T,2))：逐帧弱透视相机最小二乘 init。

    模型 u ≈ s·(R@X)[:2] + txy。零外部标定——仅模型自身 init 3D 与
    检测 2D 的对应。M2 = ucᵀXc(XcᵀXc)⁻¹ ≈ s·R[:2]（最小二乘的行即
    s·r1、s·r2）——**直接取 M2 行恢复旋转**（正交化 + 叉乘补第三行；
    2026-10-30 修复：此前用 SVD 行空间基，基在平面内可任意旋转，
    与真实 r1/r2 差一个 2D 正交变换——合成数据旋转误差 180°）。
    6D 口径 = R 前两列（与 rot6d_to_rotmat 一致）。
    """
    T = joints_world.shape[0]
    R6 = np.zeros((T, 6))
    S = np.zeros((T, 1))
    TXY = np.zeros((T, 2))
    for t in range(T):
        m = mask[t] > 0.5
        if m.sum() < 4:
            m = np.ones(m.shape[0], dtype=bool)
        X = joints_world[t, m]
        u = det_px[t, m]
        Xc = X - X.mean(0)
        uc = u - u.mean(0)
        M2 = (uc.T @ Xc) @ np.linalg.inv(Xc.T @ Xc + 1e-8 * np.eye(3))  # (2,3) ≈ s·R[:2]
        m1, m2 = M2[0], M2[1]
        r1 = m1 / np.linalg.norm(m1)
        r2_raw = m2 - (m2 @ r1) * r1
        r2 = r2_raw / np.linalg.norm(r2_raw)
        r3 = np.cross(r1, r2)                       # det=+1 由构造保证
        R = np.stack([r1, r2, r3])
        s = float(np.clip((np.linalg.norm(m1) + np.linalg.norm(r2_raw)) / 2.0,
                          1e-3, 1e4))
        txy = u.mean(0) - s * (R @ X.mean(0))[:2]
        R6[t] = np.concatenate([R[:, 0], R[:, 1]])
        S[t, 0] = s
        TXY[t] = txy
    return R6, S, TXY


def pa_mpjpe_np(pred: np.ndarray, ref: np.ndarray) -> float:
    """(T,J,3) 骨盆对齐 + Procrustes → mm（项目口径，逐帧均值）。"""
    from anysole.utils.metrics import pelvis_align, pa_mpjpe
    p, _ = pelvis_align(pred)
    r, _ = pelvis_align(ref)
    return float(pa_mpjpe(p, r).mean())


def mpjpe_np(pred: np.ndarray, ref: np.ndarray) -> float:
    """(T,J,3) 骨盆对齐 MPJPE → mm（eval_multiview.metrics 同口径）。"""
    from anysole.utils.metrics import mean_point_error, pelvis_align
    p, _ = pelvis_align(pred)
    g, _ = pelvis_align(ref)
    return float(mean_point_error(p, g, scale=1000).mean())


def gmpjpe_np(pred: np.ndarray, ref: np.ndarray) -> float:
    """(T,J,3) 世界系 GlobalMPJPE → mm（eval_multiview.metrics 同口径）。"""
    return float(np.linalg.norm(pred - ref, axis=-1).mean() * 1000)


def cal_tensors(cals: list, device) -> list:
    out = []
    for cal in cals:
        out.append({k: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device)
                    if k != "D" else tuple(float(x) for x in cal["D"])
                    for k, v in cal.items() if k in ("K", "D", "Rc", "tc", "Rm", "tm")})
    return out


def load_detector(sid: str, cam: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(RESULTS_ROOT / "data/rtmpose26" / f"cam{cam}" / f"{sid}.npz")
    return d["kp"], d["score"]


def run_session(model, smpl_layer, leaf, sid, n_windows, steps, warmup, clip, lr,
                grad_diag, device, wins=None, dump_dir=None,
                config_value=CONFIG_V, win_frac=None, contact_method="f6_soft",
                cams=(1, 2, 3, 4), calib: bool = False):
    from anysole.data.dataset import AnySoleDataset, collate_windows
    from anysole.train import condition_inputs, move_batch
    from anysole.utils.geometry import fk_pose6d, rot6d_to_rotmat

    date = session_date(sid)
    dsets = {cam: AnySoleDataset("val" if sid.startswith("S12") else "test",
                                 session_ids=[sid], cam_id=cam, stride=20, contact_method=contact_method)
             for cam in cams}
    det = {cam: load_detector(sid, cam) for cam in cams}

    starts = None
    items = {}
    for cam, ds in dsets.items():
        its = [ds[i] for i in range(len(ds))]
        s = {int(it["frame_start"].reshape(-1)[0]) for it in its}
        starts = s if starts is None else starts & s
        items[cam] = {int(it["frame_start"].reshape(-1)[0]): it for it in its}
    starts = sorted(starts)
    if wins is not None:
        # E7 用：按窗口序号（0 起）选窗——窗口网格 = stride 20 非重叠，
        # 序号 k ↔ 帧 [k*20, (k+1)*20)
        starts = [starts[i] for i in wins if 0 <= i < len(starts)]
    elif win_frac is not None:
        # E7 五阶段时间轴：取窗口起始帧占比在 [f0, f1) 的窗口
        # （帧级占比，与 e7_driver 的逐帧阶段规则一致）
        n_frames = len(det[cams[0]][0])
        starts = [s for s in starts
                  if win_frac[0] <= s / n_frames < win_frac[1]]
    else:
        starts = starts[:n_windows]
    print(f"[{sid}] aligned windows: {len(starts)}")

    res = {"pa_init": [], "pa_virt_init": [], "pa_refined": [],
           "pa_virt_refined": [], "curve": {}, "grad_diag": {}, "n": 0}
    for st in starts:
        hs, poses0, trans0, anchors, det_n, det_mask = [], [], [], [], [], []
        for cam in cams:
            it = items[cam][st]
            batch = collate_windows([it])
            batch = move_batch(batch, device)
            config_id = torch.full((1,), config_value, device=device)
            v_feat, t_raw, t_phys, _ = condition_inputs(batch, config_id)
            with torch.no_grad():
                out = model(v_feat, t_raw, t_phys, config_id)
            hs.append(out[leaf][0].clone())                # (tw,n_tokens,256) 部位表示
            poses0.append(out["x0_hat"][0].clone())         # (tw,144)
            trans0.append(out["trans_hat"][0].clone())      # (tw,3)
            anchors.append(it["trans_anchor"].to(device))   # (3,) GT 平移（仅评估）
            kp, score = det[cam]
            seg = slice(st, st + 20)
            kp_n = torch.from_numpy(kp[seg].astype(np.float32)).to(device)
            sc = torch.from_numpy(score[seg].astype(np.float32)).to(device)
            det_n.append(kp_n[:, H26_ORDER])               # (tw,21,2) px
            det_mask.append((sc[:, H26_ORDER] >= CONF_THR).float())  # (tw,21)
        offsets = items[cams[0]][st]["offsets"].to(device)    # (24,3) 协议常量（评估）
        parents = items[cams[0]][st]["parents"].to(device)
        gt_kp = items[cams[0]][st]["kp_gt"].to(device)        # (tw,24,3) GT 仅评估

        def pa(pose, trans):
            kp = fk_pose6d(pose.reshape(1, -1, 144), trans.reshape(1, -1, 3),
                           offsets, parents)[0]
            return pa_mpjpe_np(kp.detach().cpu().numpy(), gt_kp.cpu().numpy())

        def arms_eval(leaves):
            """无梯度三臂评估：per-view 精修均值 + 精修后聚合。"""
            refined, refined_trans, refined_rots = [], [], []
            with torch.no_grad():
                for i, cam in enumerate(cams):
                    batch = collate_windows([items[cam][st]])
                    batch = move_batch(batch, device)
                    config_id = torch.full((1,), config_value, device=device)
                    v_feat, t_raw, t_phys, _ = condition_inputs(batch, config_id)
                    out = model(v_feat, t_raw, t_phys, config_id,
                                **{leaf + "_override": leaves[i].unsqueeze(0)})
                    refined.append(out["x0_hat"][0])
                    refined_trans.append(out["trans_hat"][0])
                    refined_rots.append(rot6d_to_rotmat(out["x0_hat"].reshape(-1, 24, 6)))
                virt_r = virtual_pose(torch.stack(refined_rots))
                vp_r = torch.cat([virt_r[..., 0], virt_r[..., 1]], -1).reshape(1, -1, 144)
                trans_mean_r = torch.stack(
                    [refined_trans[i] + anchors[i] for i in range(len(cams))]).mean(0)
                kp_vr = fk_pose6d(vp_r, trans_mean_r.reshape(1, -1, 3),
                                  offsets, parents)[0]
            refined_pa = np.mean([pa(refined[i], refined_trans[i] + anchors[i])
                                  for i in range(len(cams))])
            kp_vr_np = kp_vr.detach().cpu().numpy()
            gt_np = gt_kp.cpu().numpy()
            virt_refined_pa = pa_mpjpe_np(kp_vr_np, gt_np)
            virt_refined_mpjpe = mpjpe_np(kp_vr_np, gt_np)
            virt_refined_gmpjpe = gmpjpe_np(kp_vr_np, gt_np)
            return refined_pa, virt_refined_pa, virt_refined_mpjpe, \
                virt_refined_gmpjpe, kp_vr_np

        # 臂①：单视角 init（per-view 均值）
        init_pa = np.mean([pa(poses0[i], trans0[i] + anchors[i])
                           for i in range(len(cams))])
        # 臂②：多视角直接聚合（不迭代）
        rots0 = torch.stack([rot6d_to_rotmat(p.reshape(-1, 24, 6))
                             for p in poses0])              # (V,tw,24,3,3)
        virt0 = virtual_pose(rots0)                         # (tw,24,3,3)
        vp0 = torch.cat([virt0[..., 0], virt0[..., 1]], -1).reshape(1, -1, 144)
        trans_mean0 = torch.stack([trans0[i] + anchors[i]
                                   for i in range(len(cams))]).mean(0)
        kp_v0 = fk_pose6d(vp0, trans_mean0.reshape(1, -1, 3), offsets, parents)[0]
        virt_init_pa = pa_mpjpe_np(kp_v0.detach().cpu().numpy(), gt_kp.cpu().numpy())
        res["pa_init"].append(init_pa)
        res["pa_virt_init"].append(virt_init_pa)
        if len(cams) == 1:
            # 单视角：无 TTA 语义（多视角一致性最低两相机），只出 init 臂
            res["n"] += 1
            continue

        # 逐视角弱透视相机 init（无标定：仅模型自身 init 3D ↔ 检测 2D 的
        # PnP 式最小二乘，零外部标定）
        camR0, camS0, camT0 = [], [], []
        for i in range(len(cams)):
            rots_i = rot6d_to_rotmat(poses0[i].reshape(-1, 24, 6))
            so = smpl_layer(global_orient=rots_i[:, :1], body_pose=rots_i[:, 1:],
                            betas=torch.zeros(10, device=device).expand(
                                rots_i.shape[0], -1),
                            pose2rot=False)
            pts_i = torch.cat([so.joints[:, S24_BODY],
                               so.joints[:, J_EXTRA]], dim=1)  # (tw,21,3)
            X0 = (pts_i - pts_i[:, 13:14]).detach().cpu().numpy()  # 骨盆锚定（槽13=骨盆）
            r6, s_, t_ = init_weak_cam(X0, det_n[i].cpu().numpy(),
                                       det_mask[i].cpu().numpy())
            camR0.append(torch.from_numpy(r6).float().to(device))
            camS0.append(torch.from_numpy(s_).float().to(device))
            camT0.append(torch.from_numpy(t_).float().to(device))

        # 优化
        leaves = [h.reshape(-1, h.shape[-1]).clone().requires_grad_(True)
                  for h in hs]
        camR = [r.clone().requires_grad_(True) for r in camR0]
        camS = [s.clone().requires_grad_(True) for s in camS0]
        camT = [t.clone().requires_grad_(True) for t in camT0]
        beta = torch.zeros(10, device=device, requires_grad=True)   # 共享形状叶子
        opt = torch.optim.Adam(leaves + camR + camS + camT + [beta], lr=lr)

        def forward_step(i, cam, leaves_):
            batch = collate_windows([items[cam][st]])
            batch = move_batch(batch, device)
            config_id = torch.full((1,), config_value, device=device)
            v_feat, t_raw, t_phys, _ = condition_inputs(batch, config_id)
            out = model(v_feat, t_raw, t_phys, config_id,
                        **{leaf + "_override": leaves_[i].unsqueeze(0)})
            rots = rot6d_to_rotmat(out["x0_hat"].reshape(-1, 24, 6))  # (tw,24,3,3)
            smpl_out = smpl_layer(global_orient=rots[:, :1],
                                  body_pose=rots[:, 1:],
                                  betas=beta.expand(rots.shape[0], -1),
                                  pose2rot=False)  # 直通旋转矩阵（可微）
            pts = torch.cat([smpl_out.joints[:, S24_BODY],
                             smpl_out.joints[:, J_EXTRA]], dim=1)  # (tw,21,3)
            X = pts - pts[:, 13:14]                               # 骨盆锚定
            Rv = rot6d_to_rotmat(camR[i].reshape(-1, 6))          # (tw,3,3)
            Xr = torch.einsum("tij,tpj->tpi", Rv, X)
            proj = camS[i][:, None] * Xr[..., :2] + camT[i][:, None, :]  # 弱透视 px
            err = det_mask[i] * (proj - det_n[i]).pow(2).sum(-1)
            l2d_i = err.sum() / det_mask[i].sum().clamp(min=1)
            return out, rots, l2d_i

        for step in range(steps):
            lr_eff = lr * min(1.0, (step + 1) / max(1, warmup))
            opt.param_groups[0]["lr"] = lr_eff
            losses_2d, rots_all, poses_all = [], [], []
            for i, cam in enumerate(cams):
                out, rots, l2d_i = forward_step(i, cam, leaves)
                losses_2d.append(l2d_i)
                rots_all.append(rots)
                poses_all.append(out["x0_hat"][0])
            l2d = sum(losses_2d) / len(cams)
            rots = torch.stack(rots_all)                            # (V,tw,24,3,3)
            virt = virtual_pose(rots)
            lcon = (rots - virt).pow(2).mean()
            lreg = sum((leaves[i] - hs[i].reshape(-1, hs[i].shape[-1])).pow(2).mean()
                       for i in range(len(cams))) / len(cams) + \
                sum((poses_all[i] - poses0[i]).pow(2).mean() for i in range(len(cams))) / len(cams) + \
                sum((camR[i] - camR0[i]).pow(2).mean() for i in range(len(cams))) / len(cams) + \
                sum((camS[i] - camS0[i]).pow(2).mean() for i in range(len(cams))) / len(cams) + \
                sum((camT[i] - camT0[i]).pow(2).mean() for i in range(len(cams))) / len(cams) + \
                beta.pow(2).mean()
            loss = LAMBDA_2D * l2d + LAMBDA_CON * lcon + LAMBDA_REG * lreg
            if step == 0 and grad_diag and res["n"] == 0:
                # 损失尺度审计：各加权项的有效梯度范数（仅首窗第 0 步）
                diag = {"l2d_val": float(l2d.item()), "lcon_val": float(lcon.item()),
                        "lreg_val": float(lreg.item())}
                groups = {"h": leaves, "cam": camR + camS + camT, "beta": [beta]}
                for name, term in (("l2d", LAMBDA_2D * l2d),
                                   ("lcon", LAMBDA_CON * lcon),
                                   ("lreg", LAMBDA_REG * lreg)):
                    opt.zero_grad()
                    term.backward(retain_graph=True)
                    for gname, params in groups.items():
                        gn = float(np.sqrt(sum(
                            (p.grad.detach().pow(2).sum().item())
                            for p in params if p.grad is not None)))
                        diag[f"grad_{name}_{gname}"] = gn
                res["grad_diag"][str(st)] = diag
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], clip)
            opt.step()
            if (step + 1) in RECORD_STEPS:
                rpa, vpa, vmpjpe, vgmpjpe, _ = arms_eval(leaves)
                res["curve"].setdefault(str(step + 1), []).append(
                    {"pa_refined": rpa, "pa_virt_refined": vpa,
                     "mpjpe_virt_refined": vmpjpe,
                     "gmpjpe_virt_refined": vgmpjpe,
                     "l2d": float(l2d.item()), "lcon": float(lcon.item()),
                     "lreg": float(lreg.item())})
        # 终值（200 步记录点即终值；steps<200 时补录）
        if steps not in RECORD_STEPS:
            rpa, vpa, vmpjpe, vgmpjpe, _ = arms_eval(leaves)
            res["curve"].setdefault(str(steps), []).append(
                {"pa_refined": rpa, "pa_virt_refined": vpa,
                 "mpjpe_virt_refined": vmpjpe,
                 "gmpjpe_virt_refined": vgmpjpe})
        _, _, vmpjpe, vgmpjpe, kp_vr_np = arms_eval(leaves)
        final = res["curve"][str(steps)][-1]
        res["pa_refined"].append(final["pa_refined"])
        res["pa_virt_refined"].append(final["pa_virt_refined"])
        res.setdefault("mpjpe_virt_refined", []).append(final["mpjpe_virt_refined"])
        res.setdefault("gmpjpe_virt_refined", []).append(final["gmpjpe_virt_refined"])
        if dump_dir is not None:
            # E7 逐帧曲线用：每窗 virtual 精修聚合关节 + GT + 帧号
            frame_ids = np.arange(st, st + kp_vr_np.shape[0], dtype=np.int64)
            np.savez_compressed(Path(dump_dir) / f"{sid}_w{st}.npz",
                                joints=kp_vr_np.astype(np.float32),
                                gt=gt_kp.cpu().numpy().astype(np.float32),
                                frame_ids=frame_ids,
                                pa=float(final["pa_virt_refined"]),
                                mpjpe=float(final["mpjpe_virt_refined"]),
                                gmpjpe=float(final["gmpjpe_virt_refined"]))
        res["n"] += 1
        if res["n"] % 4 == 0:
            rpa, vpa = res["curve"][str(steps)][-1]["pa_refined"], \
                res["curve"][str(steps)][-1]["pa_virt_refined"]
            print(f"[{sid}] w{res['n']}: init {init_pa:.1f} virt0 {virt_init_pa:.1f} "
                  f"refined {rpa:.1f} virt_r {vpa:.1f} "
                  f"beta {beta.detach().cpu().numpy().round(2).tolist()}")
    return res


def summarize(res):
    out = {"n": res["n"]}
    for k in ("pa_init", "pa_virt_init", "pa_refined", "pa_virt_refined",
              "mpjpe_virt_refined", "gmpjpe_virt_refined"):
        out[k] = float(np.mean(res[k])) if res.get(k) else None
    out["curve"] = {}
    for step, rows in res["curve"].items():
        out["curve"][step] = {k: float(np.mean([r[k] for r in rows]))
                              for k in rows[0]}
    out["grad_diag"] = res["grad_diag"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="V3_Bv1", choices=tuple(TTA_MODELS))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--sessions", default=None, help="逗号分隔；缺省 = split 全集")
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--wins", default=None,
                    help="逗号分隔窗口序号（0 起，stride20 网格；E7 用，"
                         "缺省 = 前 n-windows 窗）")
    ap.add_argument("--win-frac", default=None,
                    help="\"f0,f1\"：取窗口序号占比在 [f0,f1) 的窗口"
                         "（E7 五阶段时间轴用；与 --wins 互斥）")
    ap.add_argument("--i", type=int, default=None, choices=(1, 2, 3, 4),
                    help="视角数：跑全部 C(4,i) 相机组合——std 列人口 = 组合间；"
                         "i=4 单一组合无组合间 std；缺省 = 固定组合 1,2,3,4")
    ap.add_argument("--cams", default=None,
                    help="逗号分隔相机子集（如 1,2）；与 --i 互斥")
    ap.add_argument("--seed", type=int, default=1,
                    help="种子（variant sdN 选多种子 ckpt；缺省 1）")
    ap.add_argument("--ckpt", default="last", choices=("best", "last"),
                    help="ckpt 选点口径（缺省 last）")
    ap.add_argument("--checkpoint", type=Path, help="Explicit frozen checkpoint; label version is read from its config")
    ap.add_argument("--dump-kp", default=None,
                    help="目录：每窗落 {sid}_w{st}.npz（virtual 精修聚合关节/"
                         "GT/帧号/mpjpe/gmpjpe，E7 逐帧曲线用）")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--grad-diag", action="store_true")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sids = (args.sessions.split(",") if args.sessions else
            (VAL_SESSIONS if args.split == "val" else TEST_SESSIONS))
    wins = ([int(x) for x in args.wins.split(",") if x.strip()]
            if args.wins else None)
    win_frac = None
    if args.win_frac:
        win_frac = tuple(float(x) for x in args.win_frac.split(","))
        if len(win_frac) != 2:
            raise SystemExit("--win-frac 需 \"f0,f1\" 两个数")
    if wins is not None and win_frac is not None:
        raise SystemExit("--wins 与 --win-frac 互斥")
    if args.dump_kp:
        Path(args.dump_kp).mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(RESULTS_ROOT / "metrics" /
                               f"anysole_tta_stage1_{args.model}_{args.split}"
                               + (f"_i{args.i}" if args.i is not None else
                                  (f"_cams{','.join(sorted(args.cams.split(',')))}"
                                   if args.cams else ""))
                               + f"_lr{args.lr}.json")
    model, saved = build_model(args.model, args.device, which=args.ckpt, checkpoint=args.checkpoint, seed=args.seed)
    contact_method = saved.get("contact_method", TTA_MODELS[args.model]["contact"])
    from anysole.data.contact_package import VERSIONED_CONTACT_METHODS, corrected_contact_provenance
    if contact_method in VERSIONED_CONTACT_METHODS:
        if saved.get("contact_label_package_sha256") != corrected_contact_provenance(contact_method)["contact_label_package_sha256"]:
            raise ValueError("TTA checkpoint contact package mismatch")
    smpl_layer = build_smpl_layer(args.device)
    from anysole.data.contact_package import sha256 as _sha256
    selected_ckpt = args.checkpoint or infer_checkpoint_path(args.model, args.ckpt)
    summary = {"sessions": {}, "checkpoint": str(selected_ckpt), "checkpoint_sha256": _sha256(selected_ckpt),
               "contact_method": contact_method, "contact_label_package_sha256": saved.get("contact_label_package_sha256"),
               "camera_calibration": "none", "extra_contact_loss": False}
    from itertools import combinations
    if args.i is not None and args.cams:
        raise SystemExit("--i 与 --cams 互斥")
    if args.cams:
        cams_list = [tuple(int(x) for x in args.cams.split(",") if x.strip())]
    elif args.i is not None:
        cams_list = list(combinations((1, 2, 3, 4), args.i))
    else:
        cams_list = [(1, 2, 3, 4)]
    metric_keys = ("pa_init", "pa_virt_init", "pa_refined", "pa_virt_refined",
                   "mpjpe_virt_refined", "gmpjpe_virt_refined")
    combo_mode = (args.i is not None) or (args.cams is not None)
    if combo_mode:
        summary["cams_list"] = [",".join(map(str, c)) for c in cams_list]
        summary["combos"] = {}
    for cams in cams_list:
        combo_key = ",".join(map(str, cams))
        combo_sessions = {}
        for sid in sids:
            res = run_session(model, smpl_layer, TTA_MODELS[args.model]["leaf"],
                              sid, args.n_windows, args.steps,
                              args.warmup, args.clip, args.lr, args.grad_diag,
                              args.device, wins=wins, dump_dir=args.dump_kp,
                              config_value=CONFIG_V,
                              win_frac=win_frac, contact_method=contact_method,
                              cams=cams)
            stats = summarize(res)
            if combo_mode:
                summary["sessions"].setdefault(sid, {})[combo_key] = stats
                combo_sessions[sid] = stats
            else:
                summary["sessions"][sid] = stats
            print(f"[{sid}] done: { {k: (round(v, 1) if v is not None else None) for k, v in stats.items() if k in ('pa_init','pa_virt_init','pa_refined','pa_virt_refined')} }")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        if combo_mode:
            combo_mean = {}
            for k in metric_keys:
                vals = [s[k] for s in combo_sessions.values() if s[k] is not None]
                combo_mean[k] = float(np.mean(vals)) if vals else None
            summary["combos"][combo_key] = combo_mean
            print(f"[combo {combo_key}] { {k: round(v, 1) for k, v in combo_mean.items()} }")
            Path(out_path).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if combo_mode:
        # std 列人口 = 组合间：均值/标准差均跨相机组合（i=4 单组合 → std=None）
        overall = {}
        for k in metric_keys:
            vals = [summary["combos"][ck][k] for ck in summary["combos"]
                    if summary["combos"][ck].get(k) is not None]
            overall[k] = {"mean": float(np.mean(vals)) if vals else None,
                          "std": float(np.std(vals)) if len(vals) > 1 else None}
        summary["overall"] = overall
    else:
        overall = {k: float(np.mean([s[k] for s in summary["sessions"].values()
                                     if s[k] is not None]))
                   for k in metric_keys}
    Path(out_path).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("overall:", overall)
    print("→", out_path)


if __name__ == "__main__":
    main()
