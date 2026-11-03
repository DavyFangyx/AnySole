"""Cell T2M 导出 GIF（2026-11-03）：四个现役模型的可视化对比。

直接消费 results/cells/<cell>/predictions/eval_motion/<sid>_T2M.npz 的
joint_xyz_world（已 z-up 显示口径）+ valid_mask + frame_indices，GT 关节
从该 cell 的 config.snapshot.yaml 重建 dataset 读取（kp_gt，y-up→z-up）。
每会话 2 栏 GIF：[Pred | GT]。零重训、零模型前向。

用法：
    python results_display/script/visual_cell_t2m_gifs.py \
        --cells V7_Cv5ARSS2_s1,V7_Cv7_gb_s1,V7_Cv5_gb_s1,V7_Cv5AR2M_s1 \
        --sessions S13023 S13061 S13071 S13092
产物：results_display/ResultTest/R1Test_visualize/<cell>/T2M/gif/<sid>.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anysole.data.dataset import AnySoleDataset
from anysole.train import load_config
from results_display.script.utils.render_common import (
    SMPL_EDGES, render_skeleton_panel)


def yup_to_zup(j):
    """世界系 y-up → z-up 显示口径（与 eval 导出同款）。"""
    j = np.asarray(j, dtype=np.float32)
    return np.stack((j[..., 0], -j[..., 2], j[..., 1]), axis=-1)


def dataset_for_cell(cell: str, session_ids: list[str]):
    cfg_path = ROOT / "results" / "cells" / cell / "config.snapshot.yaml"
    cfg = load_config(cfg_path)
    ds = AnySoleDataset(
        mode="eval", session_ids=session_ids,
        seq_root=Path(cfg["seq_root"]), split_csv=Path(cfg["split_csv"]),
        cache_root=Path(cfg["cache_root"]),
        window_length=int(cfg["tw"]), stride=int(cfg["tw"]),
        contact_method=cfg["contact_method"],
        smpl_support_policy=cfg.get("smpl_support_policy", "exclude"),
        smpl_roots=cfg.get("smpl_roots"),
        v7_ar=bool(cfg.get("v7_ar", False)),
        v7_ar2=bool(cfg.get("v7_ar2", False)),
        obs_context=cfg.get("obs_context"))
    return ds


def render_cell(cell: str, session_ids: list[str], mode: str, fps: int,
                out_base: Path) -> None:
    pred_root = ROOT / "results" / "cells" / cell / "predictions" / "eval_motion"
    ds = dataset_for_cell(cell, session_ids)
    sessions = {s["session_id"]: s for s in ds.sessions}
    out_root = out_base / cell / mode
    (out_root / "gif").mkdir(parents=True, exist_ok=True)
    for sid in session_ids:
        path = pred_root / f"{sid}_{mode}.npz"
        if not path.is_file():
            print(f"{cell} {sid}: no {mode} export, skipped")
            continue
        data = np.load(path)
        pred_j = np.asarray(data["joint_xyz_world"], dtype=np.float32)
        valid = np.asarray(data["valid_mask"], dtype=bool)
        frame_ids = np.asarray(data["frame_indices"], dtype=np.int64)
        gt_j = yup_to_zup(sessions[sid]["kp_gt"][frame_ids])
        n = int(valid.sum())
        if n == 0:
            print(f"{cell} {sid}: no valid frames, skipped")
            continue
        pred_j = pred_j[valid]
        gt_j = gt_j[valid]
        frames_out = []
        for t in range(n):
            panels = [
                render_skeleton_panel(pred_j[t], f"{cell} T2M", (80, 200, 255),
                                      t, n, edges=SMPL_EDGES),
                render_skeleton_panel(gt_j[t], "GT", (255, 170, 80),
                                      t, n, edges=SMPL_EDGES),
            ]
            frames_out.append(Image.fromarray(np.concatenate(panels, axis=1)))
        out = out_root / "gif" / f"{sid}.gif"
        frames_out[0].save(out, save_all=True, append_images=frames_out[1:],
                           duration=1000 // fps, loop=0)
        print(f"wrote {out} ({n} frames)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True,
                        help="逗号分隔 cell 名（results/cells 下）")
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--mode", default="T2M")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results_display" / "ResultTest"
                        / "R1Test_visualize")
    args = parser.parse_args(argv)
    for cell in args.cells.split(","):
        cell = cell.strip()
        render_cell(cell, args.sessions, args.mode, args.fps, args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
