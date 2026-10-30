"""论文 TTA 复现评估（E3「TTA 原方法」证据源，2026-10-30 收口版，无标定）。

阶段 1（本脚本，hmr2 env）：同帧 4 视角 per-view token 优化 + virtual view；
每帧集合落盘 SMPL-24 原生关节（相机系：per-view 初值/精修 + virtual +
init 聚合），供阶段 2 用 AnySole 协议（窗口级 PA vs kp_gt，
anysole.utils.metrics，touch_gait env）评估——统一读出 SMPL-24、同一个
评估脚本。

设计（2026-10-30 收口 + 用户裁定）：
- 无标定：L_2D = 模型自带弱透视投影（deccam），零外部标定；
- L_2D 目标 = RTMPose-Halpe26 检测器（21 点监督子集，与 AnySole 线同一映射表）；
- 统一读出 SMPL-24：TTA 用原生 smplx 层（非 44 关节包装器）；
- 优化协议 = 论文：200 步、lr 0.1、20 步 warmup、梯度裁剪 0.1。

用法（hmr2 env）：
  python eval/eval_tta.py --sessions S13011 --n-frames 20
落盘：results/baselines/HMR2.0/predictions/tta_theta/{sid}.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # z_note/HMR2.0（hmr2_gait 包父目录）
from hmr2_tta.paths import RESULTS_ROOT, facts_session_dir  # noqa: E402
from hmr2_tta.data.assemble_npz import date_of, load_manifest_rows  # noqa: E402
from hmr2_tta.models.loaders import load_hmr2_plain_with_token  # noqa: E402
from hmr2_tta.models.tta import TTAFrameSet, init_virtual_view, rot_to_6d  # noqa: E402

CONF_THR = 0.5
# Halpe-26 → SMPL-24 监督子集（与 AnySole TTA 同一映射表，21 点）
H26_ORDER = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19,
             0, 20, 21, 22, 23, 24, 25]


def detector_2d(kp: np.ndarray, score: np.ndarray, center: np.ndarray,
                patch_w: float) -> np.ndarray:
    """RTMPose-Halpe26 帧级检测 (26,2)px → (21,3) 归一化 crop + conf（监督子集序）。"""
    norm = (kp - center) / patch_w
    out = np.zeros((len(H26_ORDER), 3), dtype=np.float32)
    for i, h in enumerate(H26_ORDER):
        if score[h] >= CONF_THR:
            out[i, :2] = norm[h]
            out[i, 2] = score[h]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="S13011")
    ap.add_argument("--n-frames", type=int, default=20, help="每会话采样帧集合数")
    ap.add_argument("--windows", type=int, default=0,
                    help="按 AnySole 窗口协议取前 K 个窗口的全部帧（0 = 用 --n-frames 采样）")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--lreg", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda:3")
    args = ap.parse_args()

    model, cfg = load_hmr2_plain_with_token()
    model.eval().to(args.device)
    rows = load_manifest_rows(3)

    for sid in args.sessions.split(","):
        facts = np.load(facts_session_dir(sid, 3) / "frames.npz")
        npz3 = np.load(RESULTS_ROOT / "data/npz/cam3" / f"{sid}.npz")
        cams = (1, 2, 3, 4)
        index = json.loads((RESULTS_ROOT / "data/index/test.json").read_text())
        from hmr2.datasets.image_dataset import ImageDataset
        dsets, preds, dets = {}, {}, {}
        for cam in cams:
            e = next(x for x in index if x["sid"] == sid and x["cam"] == cam)
            dsets[cam] = ImageDataset(cfg, e["npz"], img_dir="", train=False)
            preds[cam] = np.load(RESULTS_ROOT / "predictions/zero" / f"cam{cam}" /
                                 f"{sid}.npz")
            d = np.load(RESULTS_ROOT / "data/rtmpose26" / f"cam{cam}" / f"{sid}.npz")
            dets[cam] = d

        common = sorted(set(npz3["frame_id"].tolist()))
        if args.windows:
            frames = [f for f in common if f < args.windows * 20]
        else:
            frames = common[:: args.stride][: args.n_frames]
        out_all = {"frame_id": [], "joints_init_v": [], "joints_refined_v": [],
                   "joints_virtual": [], "joints_virt_init": []}
        for f in frames:
            tokens, inits, det2d = [], {"pose_6d": [], "betas": [], "cam": []}, []
            centers, patch_ws = [], []
            joints_init = []
            for cam in cams:
                loc = int(np.searchsorted(preds[cam]["frame_id"], f))
                item = dsets[cam][loc]
                img = torch.from_numpy(item["img"]).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    out = model.forward_step({"img": img})
                tokens.append(out["token_out"][0])
                rot = torch.cat([out["pred_smpl_params"]["global_orient"],
                                 out["pred_smpl_params"]["body_pose"]], 1)[0]
                inits["pose_6d"].append(rot_to_6d(rot).reshape(144).cpu())
                inits["betas"].append(out["pred_smpl_params"]["betas"][0].cpu())
                inits["cam"].append(out["pred_cam"][0].cpu())
                pw, ctr = item["box_size"], item["box_center"]
                centers.append(ctr)
                patch_ws.append(pw)
                gpos = int(np.searchsorted(dets[cam]["frame_id"], f))
                det2d.append(detector_2d(dets[cam]["kp"][gpos],
                                         dets[cam]["score"][gpos], ctr, pw))
            tta = TTAFrameSet(model, torch.stack(tokens).to(args.device),
                              {"pose_6d": torch.stack(inits["pose_6d"]).to(args.device),
                               "betas": torch.stack(inits["betas"]).to(args.device),
                               "cam": torch.stack(inits["cam"]).to(args.device)},
                              np.stack(det2d),
                              np.stack(centers), np.stack(patch_ws),
                              lambda_reg=args.lreg)
            # init SMPL-24（原生层正演 init 参数；不走 eval_zero 的 44 布局）
            from hmr2.utils.geometry import rot6d_to_rotmat
            with torch.no_grad():
                for vi in range(len(cams)):
                    rot_i0 = rot6d_to_rotmat(
                        inits["pose_6d"][vi].to(args.device).unsqueeze(0)).view(1, 24, 3, 3)
                    j_init, _ = tta._smpl_forward(
                        rot_i0, inits["betas"][vi].to(args.device).unsqueeze(0))
                    joints_init.append(j_init[0, :24].detach().cpu().numpy())
                virt0 = init_virtual_view(
                    torch.stack(inits["pose_6d"]).to(args.device),
                    torch.stack(inits["betas"]).to(args.device),
                    torch.stack(inits["cam"]).to(args.device))
                rot_v0 = rot6d_to_rotmat(
                    virt0["pose_6d"].to(args.device).unsqueeze(0)).view(24, 3, 3)
                j_v0, _ = tta._smpl_forward(rot_v0.unsqueeze(0),
                                            virt0["betas"].to(args.device).unsqueeze(0))
            tta.optimize(steps=args.steps, lr=args.lr, warmup=args.warmup,
                         clip=args.clip)
            out_t = tta.result()
            out_all["frame_id"].append(f)
            out_all["joints_init_v"].append(np.stack(joints_init))      # (V,24,3)
            out_all["joints_refined_v"].append(
                np.stack([out_t["joints_i"][v][:24] for v in range(4)]))
            out_all["joints_virtual"].append(out_t["joints0"][0][:24])
            out_all["joints_virt_init"].append(j_v0[0, :24].cpu().numpy())
            if (len(out_all["frame_id"]) % 5) == 0:
                print(f"[tta] {sid} f{f} done ({len(out_all['frame_id'])}/{len(frames)})")
        out_dir = RESULTS_ROOT / "predictions/tta_theta"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{sid}.npz",
                            frame_id=np.asarray(out_all["frame_id"]),
                            joints_init_v=np.stack(out_all["joints_init_v"]),
                            joints_refined_v=np.stack(out_all["joints_refined_v"]),
                            joints_virtual=np.stack(out_all["joints_virtual"]),
                            joints_virt_init=np.stack(out_all["joints_virt_init"]))
        print(f"[tta] {sid} → {out_dir / (sid + '.npz')}")


if __name__ == "__main__":
    main()
