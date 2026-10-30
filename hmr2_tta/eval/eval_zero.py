"""阶段 1 · 零样本单视角评估（E1-1 证据源）。

双口径：
1. 官方口径：Evaluator（mode_mpjpe/mode_re，keypoint_list=[25..37,43]，pelvis=39）
2. 项目口径（主指标）：预测经真实焦距抬升到 MoSh 世界系 → 项目 SMPL-24 协议
   （J_regressor@vertices）→ anysole/utils/vision_smpl_metrics.evaluate_motion
   （与 WHAM/SAM3DB/GVHMR 同公式；两方都在 MoSh 世界系比较）

落盘（供 eval_crossview 复用，紧凑格式）：
  predictions/zero/cam{N}/{sid}.npz: pose_aa (N,72), betas (N,10), cam (N,3),
  joints44 (N,44,3) 模型空间, patch_w (N,), frame_id (N,)
汇总：metrics/zero/summary.json；逐会话明细 metrics/zero/detail/{sid}_cam{N}.json。

用法：python eval/eval_zero.py --split val --cams 1,2,3,4 --batch 32 [--sessions S13011]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmr2_tta.paths import RESULTS_ROOT, resolve_uri  # noqa: E402
from hmr2_tta.data.assemble_npz import build_smpl_layer  # noqa: E402
from hmr2_tta.eval.lift_utils import lift_to_world, lift_to_world_t, load_lift_chain, to_project_protocol, weak_persp_t  # noqa: E402
from hmr2_tta.models.loaders import load_hmr2_plain  # noqa: E402

KEYPOINT_LIST = list(range(25, 38)) + [43]
PELVIS_IND = 39


def collate(batch):
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], np.ndarray):
            out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
        else:
            out[k] = [b[k] for b in batch]
    return out


def smpl_forward_full(layer, poses_aa: np.ndarray, betas: np.ndarray):
    """(T,72) aa → joints44 (T,44,3) + vertices (T,6890,3) 模型空间。"""
    from scipy.spatial.transform import Rotation as R_
    rot = R_.from_rotvec(poses_aa.reshape(-1, 3)).as_matrix().reshape(len(poses_aa), 24, 3, 3)
    dev = next(layer.buffers()).device  # SMPL 层全 buffer 无 parameter
    rot = torch.from_numpy(rot.astype(np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(betas, dtype=np.float32)).to(dev)
    with torch.no_grad():
        out = layer(global_orient=rot[:, :1], body_pose=rot[:, 1:],
                    betas=b.expand(len(rot), -1))
    return out.joints.cpu().numpy(), out.vertices.cpu().numpy()


def project_metrics(pred: dict, layer, row: dict, cam: int, sid: str):
    """项目口径逐会话评估。pred = npz 载荷（pose_aa/betas/cam/joints44/patch_w/frame_id）。"""
    from anysole.data.smpl_io import load_smpl
    from anysole.utils import vision_smpl_metrics as vsm
    from hmr2_tta.paths import facts_session_dir

    # GT：MoSh 重采样（与装配同口径）；pred 帧 = npz 帧序（frame_id 升序）
    facts = np.load(facts_session_dir(sid, cam) / "frames.npz")
    smpl = load_smpl(resolve_uri(row["smpl_path"]), query_t=facts["mocap_time_s"])
    n = len(facts["frame_id"])
    gt_poses = smpl["poses_aa"].reshape(n, 72)
    gt_trans = smpl["trans_m"]
    gt_betas = smpl["betas"]
    gt_j44, gt_verts = smpl_forward_full(layer, gt_poses, gt_betas)
    gt_j24 = to_project_protocol(layer, torch.from_numpy(gt_verts)) + gt_trans[:, None, :]
    gt_v_world = gt_verts + gt_trans[:, None, :]
    gt_tj, gt_tv = smpl_forward_full(layer, np.zeros((1, 72), np.float32), gt_betas)
    g_idx = np.searchsorted(facts["frame_id"], pred["frame_id"])

    lift = load_lift_chain(date_of_manifest(row), cam)
    p_aa, p_cam, p_patch = pred["pose_aa"], pred["cam"], pred["patch_w"]
    m = len(p_aa)
    # 相机系平移：模型弱透视口径（s 深度 + 光轴校正）。
    # 注：GT-2D 最小二乘拟合曾实测引入 0.43m 深度偏差（预测姿态结构与 GT 2D 错位），
    # 已弃用——世界轨迹残差（ATE ~0.3m）登记为弱透视逐帧 s 噪声局限，TTA 阶段用 L_2D 精修。
    t_cam = weak_persp_t(p_cam, p_patch, lift, pred["center"])
    # 预测：44 关节抬升 + 原生 24（J_regressor@vertices）抬升；顶点同链抬升（A 旋转 + b 平移）
    p_j44_world = lift_to_world_t(pred["joints44"], t_cam, lift)
    _, p_verts = smpl_forward_full(layer, p_aa, pred["betas"][0])
    p_j24_world = lift_to_world_t(to_project_protocol(layer, torch.from_numpy(p_verts)),
                                  t_cam, lift)
    p_v_world = lift_to_world_t(p_verts, t_cam, lift)
    p_tj, p_tv = smpl_forward_full(layer, np.zeros((1, 72), np.float32), pred["betas"][0])

    times = facts["mocap_time_s"][g_idx]
    predicted = {"joints": p_j24_world, "vertices": p_v_world, "poses": p_aa,
                 "tpose_joints": np.repeat(p_tj, m, 0),
                 "tpose_vertices": np.repeat(p_tv, m, 0)}
    target = {"joints": gt_j24[g_idx], "vertices": gt_v_world[g_idx],
              "poses": gt_poses[g_idx], "tpose_joints": np.repeat(gt_tj, m, 0),
              "tpose_vertices": np.repeat(gt_tv, m, 0)}
    return vsm.evaluate_motion(predicted, target, times)


def date_of_manifest(row: dict) -> str:
    import re
    return re.search(r"(\d{8})", row["video_path"]).group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--cams", default="1,2,3,4")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sessions", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model, cfg = load_hmr2_plain()
    model.eval().to(args.device)
    layer = build_smpl_layer().to(args.device)
    index = json.loads((RESULTS_ROOT / "data/index" / f"{args.split}.json").read_text())
    from hmr2.datasets.image_dataset import ImageDataset
    from hmr2.utils.pose_utils import Evaluator

    summary = {}
    detail_dir = RESULTS_ROOT / "metrics/zero/detail"
    detail_dir.mkdir(parents=True, exist_ok=True)
    from hmr2_tta.data.assemble_npz import load_manifest_rows
    rows = load_manifest_rows(3)
    for cam in (int(c) for c in args.cams.split(",")):
        entries = [e for e in index if e["cam"] == cam
                   and (not args.sessions or e["sid"] in args.sessions)]
        if not entries:
            continue
        official_acc, project_acc = [], []
        for e in entries:
            ds = ImageDataset(cfg, e["npz"], img_dir="", train=False)
            evaluator = Evaluator(len(ds), KEYPOINT_LIST, PELVIS_IND)
            loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                                num_workers=0, collate_fn=collate)
            pose_aa, betas, cam_list, joints44, patch_w, frame_id, center = [], [], [], [], [], [], []
            for batch in loader:
                img = batch["img"].to(args.device)
                with torch.no_grad():
                    out = model.forward_step({"img": img})
                evaluator({"pred_keypoints_3d": out["pred_keypoints_3d"],
                           "pred_keypoints_2d": out["pred_keypoints_2d"]},
                          {"keypoints_3d": batch["keypoints_3d"].to(args.device),
                           "keypoints_2d": batch["keypoints_2d"].to(args.device)})
                params = out["pred_smpl_params"]
                rot = torch.cat([params["global_orient"], params["body_pose"]], 1).cpu().numpy()
                aa = R.from_matrix(rot.reshape(-1, 3, 3)).as_rotvec().reshape(len(img), 72)
                pose_aa.append(aa.astype(np.float32))
                betas.append(params["betas"].cpu().numpy().astype(np.float32))
                cam_list.append(out["pred_cam"].cpu().numpy().astype(np.float32))
                joints44.append(out["pred_keypoints_3d"].cpu().numpy().astype(np.float32))
                patch_w.append(np.asarray(batch["box_size"], dtype=np.float64))
                center.append(np.asarray(batch["box_center"], dtype=np.float64))
                frame_id.append(np.asarray(ds.data["frame_id"][batch["idx"]],
                                           dtype=np.int64))
            pred = {"pose_aa": np.concatenate(pose_aa), "betas": np.concatenate(betas),
                    "cam": np.concatenate(cam_list), "joints44": np.concatenate(joints44),
                    "patch_w": np.concatenate(patch_w), "frame_id": np.concatenate(frame_id),
                    "center": np.concatenate(center)}
            out_path = RESULTS_ROOT / "predictions/zero" / f"cam{cam}" / f"{e['sid']}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **pred)
            official = evaluator.get_metrics_dict()
            official_acc.append({k: float(v) for k, v in official.items()})
            proj = project_metrics(pred, layer, rows[e["sid"]], cam, e["sid"])
            project_acc.append(proj)
            (detail_dir / f"{e['sid']}_cam{cam}.json").write_text(
                json.dumps({"official": official_acc[-1], "project": proj}, indent=1),
                encoding="utf-8")
            print(f"[eval_zero] {e['sid']} cam{cam}: "
                  f"mode_mpjpe={official_acc[-1]['mode_mpjpe']:.0f} "
                  f"PA={proj['pa_mpjpe_mm']:.1f} PVE={proj['pve_mm']:.1f}")
        summary[f"cam{cam}"] = {
            "official": {k: float(np.mean([x[k] for x in official_acc]))
                         for k in official_acc[0]},
            "project": {k: float(np.mean([x[k] for x in project_acc]))
                        for k in project_acc[0]},
            "n_sessions": len(entries)}
        print(f"[eval_zero] cam{cam} 汇总: "
              f"mode_mpjpe={summary[f'cam{cam}']['official']['mode_mpjpe']:.1f} "
              f"PA={summary[f'cam{cam}']['project']['pa_mpjpe_mm']:.1f} "
              f"PVE={summary[f'cam{cam}']['project']['pve_mm']:.1f}")
    summary_path = RESULTS_ROOT / "metrics/zero/summary.json"
    if summary_path.exists():
        merged = json.loads(summary_path.read_text())
        merged.update(summary)
        summary = merged
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("→", summary_path)


if __name__ == "__main__":
    main()
