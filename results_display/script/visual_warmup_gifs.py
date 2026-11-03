"""冷/热启动 T2M 生成的 motion GIF（视觉预热先导实验的可视化，2026-11-02）。

复用 visual_warmup_compare.rollout 的三臂（cold_T / V_to_T / VT_to_T，
前 40 帧前缀 + 全 T 后缀、预测状态跨窗接力、不注入 GT），对每个会话
渲染 4 栏 GIF：[cold_T | V_to_T | VT_to_T | GT]，后缀起始帧标记切换线。
产物：results_display/Etest/E7_DynamicModalityInputs/warmup_gifs/<session>.gif

用法：
    python results_display/script/visual_warmup_gifs.py \
        --checkpoint results/AnySole/V7_Cv5ARSS2_f6_soft_time_accel_state_smpl24_v1/\
tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_last.pt \
        --sessions S13023 S13061 S13071 S13092
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anysole.data.dataset import AnySoleDataset, load_split_ids
from anysole.eval import _load_model, _session_window_groups
from anysole.utils.geometry import fk_pose6d_np
from results_display.script.utils.render_common import (
    CANVAS_H, PANEL_W, SMPL_EDGES, INFO_FONT, TITLE_FONT, draw_text,
    render_skeleton_panel)
from results_display.script.visual_warmup_compare import ARMS, rollout

# 产物根：R1Test_visualize 下对应模型 variant 目录内 T2M_warm/（与 T2M=cold 并列）
OUT_ROOT = None  # main 内按 checkpoint 的模型名/变体名解析
PREFIX_FRAMES = 40  # 2 窗 × 20 帧


def yup_to_zup(j):
    """世界系 y-up（SMPL 原生，dataset kp_gt / 模型姿态空间）→ z-up
    （渲染引擎 project_joints 的口径，同导出管线 world_z_up）：绕 x 轴
    +90°，即 (x, y, z) → (x, -z, y)。"""
    return np.stack([j[..., 0], -j[..., 2], j[..., 1]], axis=-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--sessions", nargs="+", default=["S13023", "S13061", "S13071", "S13092"])
    ap.add_argument("--prefix-windows", type=int, default=2)
    ap.add_argument("--init-height", type=float, default=0.0,
                    help="先验初始高度（m，SMPL y-up；默认 0 = 旧口径）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args(argv)

    from PIL import Image
    from results_display.script.utils.render_common import project_joints

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = _load_model(checkpoint, cfg, torch.device(args.device)).eval()
    ds = AnySoleDataset(mode="eval", session_ids=args.sessions,
                        seq_root=Path(cfg["seq_root"]), split_csv=Path(cfg["split_csv"]),
                        cache_root=Path(cfg["cache_root"]), window_length=int(cfg["tw"]),
                        stride=int(cfg["tw"]), contact_method=cfg["contact_method"],
                        smpl_support_policy=cfg.get("smpl_support_policy", "exclude"),
                        smpl_roots=cfg.get("smpl_roots"),
                        v7_ar=bool(getattr(model, "ar", False)),
                        v7_ar2=bool(getattr(model, "ar2", False)),
                        obs_context=cfg.get("obs_context"))
    # 输出目录 = R1Test_visualize/<model>_<contact>/<variant>/T2M_warm
    from anysole.types import variant_from_config
    variant = variant_from_config(cfg)
    model_dir = f"{cfg['model_name']}_{cfg['contact_method']}"
    out_root = (ROOT / "results_display" / "ResultTest" / "R1Test_visualize"
                / model_dir / variant / "T2M_warm")
    out_root.mkdir(parents=True, exist_ok=True)
    for sid, indices in _session_window_groups(ds).items():
        session = next(s for s in ds.sessions if s["session_id"] == sid)
        cache = out_root / f"{sid}_arms.npz"
        if cache.is_file():
            c = np.load(cache, allow_pickle=True)
            generated = {"pose": c["pose"], "trans": c["trans"],
                         "frame_ids": c["frame_ids"],
                         "score_mask": c["score_mask"]}
            gt_kp = c["gt_kp"]
            print(f"{sid}: cached rollout reused", flush=True)
        else:
            print(f"rollout {sid} ...", flush=True)
            generated = rollout(ds, model, indices, torch.device(args.device),
                                init_height=args.init_height,
                                prefix_windows=args.prefix_windows,
                                progress=lambda n, t: print(f"  {n}/{t}", flush=True))
            frames = generated["frame_ids"]
            gt_kp = session["kp_gt"][frames]
            np.savez_compressed(cache, pose=generated["pose"],
                                trans=generated["trans"],
                                frame_ids=generated["frame_ids"],
                                score_mask=generated["score_mask"], gt_kp=gt_kp)
        frames = generated["frame_ids"]
        # 三臂 FK（y-up 世界关节）→ z-up 渲染口径
        pred_joints = []
        for arm in range(3):
            pred = fk_pose6d_np(generated["pose"][arm], generated["trans"][arm],
                                session["offsets"], session["parents"])
            pred_joints.append(yup_to_zup(pred))
        gt_kp = yup_to_zup(gt_kp)
        n = len(frames)
        prefix = args.prefix_windows * int(cfg["tw"])
        frames_out = []
        for t in range(n):
            panels = []
            for arm, label in enumerate(ARMS):
                # 直接喂世界关节（与 r_test1 引擎同路径：project_joints
                # 每栏自适应取景——不做外部归一化）
                panels.append(render_skeleton_panel(pred_joints[arm][t], label,
                                                    (80, 200, 255),
                                                    t, n, edges=SMPL_EDGES))
            panels.append(render_skeleton_panel(gt_kp[t], "GT", (255, 170, 80),
                                                t, n, edges=SMPL_EDGES))
            frame = np.concatenate(panels, axis=1)
            if t == prefix:
                frame = frame.copy()
                from PIL import ImageDraw
                draw = ImageDraw.Draw(Image.fromarray(frame))
                draw.line([(0, 30), (frame.shape[1], 30)], fill=(255, 80, 80), width=3)
                frame = np.asarray(frame)
            frames_out.append(Image.fromarray(frame))
        (out_root / "gif").mkdir(parents=True, exist_ok=True)
        out = out_root / "gif" / f"{sid}.gif"
        frames_out[0].save(out, save_all=True, append_images=frames_out[1:],
                           duration=1000 // args.fps, loop=0)
        print(f"wrote {out} ({n} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
