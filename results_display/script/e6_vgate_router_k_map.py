#!/usr/bin/env python3
"""E6：competitive router 的 T 分支权重 k 的部位 × 遮挡图（2026-11-14）。

口径
----
- 模型：``V8_Cv7_PhysIMU_Router``（``route_router``），默认 checkpoint
  ``results/diagnostics/20261112_v8_gate_frozen/router_ft_enum_fS5S6S11.pt``。
- 划分：默认 fS5S6S11 test（S5/S6/S11，66 会话、1221 窗，stride 20），
  四档 clean / occ020 / occ050 / occ100 使用完全相同的运动窗，只替换视觉缓存。
- k = σ(s_v − s_t)，逐帧逐组；k 越大 T 分支权重越大，V 分支为 1−k。
  四组：根速度 / 根姿态＋躯干头臂 / 左腿左脚 / 右腿右脚。
  query 轴序 = [根速度] + PART_NAMES，故代表槽为 0/1/6/7。
- 热力图把每个 SMPL 关节按其所属部位赋 k，再经 SMPL skinning weights
  平滑铺到 6890 顶点，在零位姿 canonical mesh 上渲染。根速度不落在网格上，
  以脚注给出四档均值。
- 生成脚本只读 checkpoint / 缓存，不训练、不改模型。

产物（默认 ``results_display/Etest_crossfit_physimu_vgate/E6_ContactPhaseAnalysis/``）
  Main_Fig_E6_RouterK_SMPL_Front.png
  Main_Table1_E6_RouterK_byPart.csv
  detail/Main_Table1_E6_RouterK_byPart.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _entry in (REPO, SCRIPT_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from anysole.data.dataset import AnySoleDataset, load_split_ids  # noqa: E402
from anysole.types import (CONFIG_VT, JOINT_NAMES, PART_JOINTS,  # noqa: E402
                           PART_NAMES)

DIAG = REPO / "results" / "diagnostics" / "20261112_v8_gate_frozen"
sys.path.insert(0, str(DIAG))
from train_router_gate import build_router  # noqa: E402

# 与 E5 热力图共用 SMPL 渲染工具，保证两张图的网格/视角/留白一致。
import e5_v8_smpl_gain_heatmap as e5  # noqa: E402

DEFAULT_OUT = (REPO / "results_display" / "Etest_crossfit_physimu_vgate"
               / "E6_ContactPhaseAnalysis" / "detail" / "router_k")
DEFAULT_SPLIT = (REPO / "AnysoleWorkspace" / "protocol" / "splits" / "default"
                 / "splits_test_S5S6S11_val_S12.csv")
DEFAULT_VISUAL = (REPO / "AnysoleWorkspace" / "model_inputs" / "AnySole"
                  / "adapter_v1" / "visual" / "hrnet")

# query 轴序 = [根速度] + PART_NAMES；四组代表槽。
GROUPS = ("root_velocity", "upper_body", "left_leg_foot", "right_leg_foot")
GROUP_SLOTS = (0, 1, 6, 7)
GROUP_LABEL = {
    "root_velocity": "root velocity",
    "upper_body": "root pose + torso/head/arms",
    "left_leg_foot": "left leg + foot",
    "right_leg_foot": "right leg + foot",
}
PART_GROUP = {
    "root": "upper_body", "torso": "upper_body", "headneck": "upper_body",
    "l_arm": "upper_body", "r_arm": "upper_body",
    "l_leg": "left_leg_foot", "l_foot": "left_leg_foot",
    "r_leg": "right_leg_foot", "r_foot": "right_leg_foot",
}
CONDS = (
    ("clean", "cam3"),
    ("occ020", "cam3_occ_lower020"),
    ("occ050", "cam3_occ_lower050"),
    ("occ100", "cam3_occ_lower100"),
)
VIEWS = {
    "front": dict(azim=-64.0, elev=8.0),
}

# 与 E5 同一组蓝—灰—红停点；k=0.5 为中性灰。
CMAP_STOPS = [
    (0.00, "#0d366b"),
    (0.25, "#2a78d6"),
    (0.50, "#dedddb"),
    (0.75, "#e34948"),
    (1.00, "#7f1d1d"),
]


def _colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("router_k", CMAP_STOPS)


def _load_model(checkpoint: Path, device: torch.device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    model = build_router(cfg)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    return model, cfg, ck


def _replace_visual(ds: AnySoleDataset, visual_root: Path, cache: str) -> None:
    for session in ds.sessions:
        sid = session["session_id"]
        path = visual_root / cache / (sid + ".pt")
        if not path.is_file():
            raise FileNotFoundError(path)
        arr = torch.load(path, map_location="cpu", weights_only=False)
        arr = (arr.float().numpy() if torch.is_tensor(arr)
               else np.asarray(arr, dtype=np.float32))
        if arr.shape != session["V_feat"].shape:
            raise ValueError("%s: visual %s vs V_feat %s"
                             % (path, arr.shape, session["V_feat"].shape))
        session["V_feat"] = arr


def _collect_k(model, cfg, split_csv: Path, ids: list, visual_root: Path,
               device: torch.device, batch_size: int) -> dict:
    out = {}
    for cond, cache in CONDS:
        ds = AnySoleDataset(
            mode="eval", window_length=int(cfg["tw"]), session_ids=ids,
            cache_root=visual_root / cache,
            contact_method=str(cfg["contact_method"]),
            tactile_input=str(cfg["tactile_input"]), split_csv=split_csv)
        if cond != "clean":
            _replace_visual(ds, visual_root, cache)
        chunks = []
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            with torch.inference_mode():
                n = batch["V_feat"].shape[0]
                cid = torch.full((n,), CONFIG_VT, dtype=torch.long,
                                 device=device)
                v_tok, t_tok = model.encoders(
                    batch["V_feat"].to(device), batch["T_raw"].to(device), cid,
                    T_s2m=batch["T_s2m"].to(device),
                    T_phys=batch["T_phys"].to(device))
                present = torch.ones(n, dtype=v_tok.dtype, device=device)
                k = model._route_router_gate(
                    v_tok, t_tok, batch["T_phys"].to(device),
                    batch["T_s2m"].to(device), present, present)
                # (B, tw, 10, 1) -> 四个代表槽 -> (B, tw, 4)
                chunks.append(k[:, :, list(GROUP_SLOTS), 0].cpu().numpy())
        out[cond] = np.concatenate(chunks, axis=0)
        print("[k] %-6s windows=%d frames=%d"
              % (cond, out[cond].shape[0], out[cond].shape[0] * out[cond].shape[1]),
              flush=True)
    return out


def _stats(k: np.ndarray) -> dict:
    """k: (N, tw, 4) -> 组均值与窗间/窗内标准差。"""
    win_mean = k.mean(axis=1)                       # (N, 4)
    return {
        "mean": k.mean(axis=(0, 1)),
        "between_window_std": win_mean.std(axis=0),
        "within_window_std": k.std(axis=1).mean(axis=0),
        "frame_group_range": float((k.max(axis=2) - k.min(axis=2)).mean()),
        "n_windows": int(k.shape[0]),
        "n_frames": int(k.shape[0] * k.shape[1]),
    }


def _joint_values(means: np.ndarray) -> np.ndarray:
    """四组 k -> 24 关节 k（部位内常数）。"""
    group_mean = dict(zip(GROUPS, means))
    per_joint = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    for name, joints in zip(PART_NAMES, PART_JOINTS):
        value = group_mean[PART_GROUP[name]]
        for joint in joints:
            per_joint[int(joint)] = value
    return per_joint


def _render_one(verts_display: np.ndarray, faces: np.ndarray,
                vertex_values: np.ndarray, view: str, size: int = 480):
    from matplotlib.colors import Normalize

    colors = _colormap()(Normalize(0.0, 1.0, clip=True)(vertex_values))[:, :3]
    image = e5.render_mesh_frame(
        verts_display, faces, vertex_colors=colors, ambient=0.85,
        light_intensity=0.8, bg_color=e5.SURFACE, size=(size, size), **VIEWS[view])
    return e5._crop_body(image)


def _compose(images: list, means: dict, view: str, fold: str,
             stats: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image

    labels = [label for label, _ in CONDS]
    fig, axes = plt.subplots(1, len(labels),
                             figsize=(2.25 * len(labels) + 1.8, 3.4),
                             facecolor=e5.SURFACE)
    axes = np.atleast_1d(axes)
    for col, (label, ax) in enumerate(zip(labels, axes)):
        ax.imshow(images[col])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, fontsize=10.5, color=e5.INK, pad=4)
        m = means[label]
        ax.set_xlabel("upper %.2f  ·  L-leg %.2f  ·  R-leg %.2f"
                      % (m[1], m[2], m[3]), fontsize=8.2, color=e5.MUTED,
                      labelpad=2)
    root_vel = "  ".join("%s %.3f" % (label, means[label][0])
                         for label, _ in CONDS)
    fig.suptitle("Router T-branch weight by body region under 2D occlusion — %s view"
                 % view, fontsize=12.5, color=e5.INK, x=0.02, ha="left", y=0.985)
    fig.text(0.5, 0.055,
             "k = σ(s_V − s_T); k=0 → V only, k=1 → T only; per-frame per-group, "
             "test windows of %s (%d windows).\n"
             "Joint values smoothed by SMPL skinning weights. "
             "Root velocity (not on mesh): %s"
             % (fold, stats["n_windows"], root_vel),
             fontsize=8.3, color=e5.MUTED, ha="center", va="bottom")
    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap=_colormap())
    cax = fig.add_axes([0.905, 0.24, 0.012, 0.58])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("T-branch weight k", fontsize=9.0, color=e5.MUTED)
    cb.ax.tick_params(labelsize=8.5, colors=e5.MUTED)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.03, right=0.885, top=0.88, bottom=0.20,
                        wspace=0.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=e5.SURFACE)
    plt.close(fig)
    print("[png] %s" % path)


def _write_csv(path: Path, fold: str, checkpoint: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fold", "checkpoint", "condition", "group",
                         "k_mean", "k_between_window_std",
                         "k_within_window_std", "frame_group_range_mean",
                         "n_windows", "n_frames"])
        for cond, _ in CONDS:
            s = stats[cond]
            for idx, group in enumerate(GROUPS):
                writer.writerow([fold, checkpoint.name, cond, group,
                                 "%.6f" % s["mean"][idx],
                                 "%.6f" % s["between_window_std"][idx],
                                 "%.6f" % s["within_window_std"][idx],
                                 "%.6f" % s["frame_group_range"],
                                 s["n_windows"], s["n_frames"]])
    print("[csv] %s" % path)


def _write_md(path: Path, fold: str, checkpoint: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E6 主表：router 的 T 分支权重 k（部位 × 遮挡）",
        "",
        "口径：`%s`；test `%s`；窗口 stride 20；四档使用相同运动窗，"
        "只替换视觉缓存。" % (checkpoint.name, fold),
        "k = σ(s_v − s_t)，k 越大 T 分支权重越大；V 分支权重为 1−k。"
        "窗间标准差按窗均值计算，窗内标准差为各窗时间标准差的平均。",
        "",
        "| 组 | clean | occ020 | occ050 | occ100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for idx, group in enumerate(GROUPS):
        lines.append("| %s | %s |" % (
            GROUP_LABEL[group],
            " | ".join("%.3f" % stats[cond]["mean"][idx]
                       for cond, _ in CONDS)))
    lines += ["", "n_windows = %d；n_frames = %d（每档）。"
              % (stats["clean"]["n_windows"], stats["clean"]["n_frames"]), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print("[md] %s" % path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=DIAG / "router_ft_enum_fS5S6S11.pt")
    ap.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--views", default="front")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg, _ = _load_model(args.checkpoint, device)
    ids = load_split_ids(args.split_csv, "test")
    print("[cfg] checkpoint=%s split=%s test_sessions=%d"
          % (args.checkpoint.name, args.split_csv.name, len(ids)), flush=True)

    k = _collect_k(model, cfg, args.split_csv, ids, args.visual_root,
                   device, args.batch_size)
    stats = {cond: _stats(k[cond]) for cond, _ in CONDS}

    _write_csv(args.out_dir / "Main_Table1_E6_RouterK_byPart.csv",
               args.fold, args.checkpoint, stats)
    _write_md(args.out_dir / "detail" / "Main_Table1_E6_RouterK_byPart.md",
              args.fold, args.checkpoint, stats)

    verts, faces, weights = e5._canonical_mesh()
    verts_display = e5.smpl_yup_to_display(verts)
    means = {cond: stats[cond]["mean"] for cond, _ in CONDS}
    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        images = []
        for cond, _ in CONDS:
            per_joint = _joint_values(means[cond])
            vertex_values = e5._vertex_values(per_joint, weights)
            images.append(_render_one(verts_display, faces, vertex_values, view))
        _compose(images, means, view, args.fold, stats["clean"],
                 args.out_dir / ("Main_Fig_E6_RouterK_SMPL_%s.png"
                                 % view.capitalize()))

    print(json.dumps({cond: [round(float(x), 4) for x in stats[cond]["mean"]]
                      for cond, _ in CONDS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
