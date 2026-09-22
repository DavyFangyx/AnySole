#!/usr/bin/env python3
"""R_Test9 信任画像（missing-rate 任务书实验 5）。

E1 信任堆叠条（实验 1 跑完即免费）：读 R_Test5 的 grid_metrics.json 三角
（纯先验 / V-only / T-only）的每部位 PA-MPJPE → 每部位"信 V / 信 T / 信先验"画像
（分组柱 + 最低角标注）。纯数据读取，零推理。
E2 注意力（定性佐证，不当主证据）：VT 前向下给 decoder 各层 multihead_attn 挂
need_weights 钩子（不改模型、不落权重），把每部位查询对 F 的注意力按
V 段（F[:, :tw]）/ T 段（F[:, tw:]）聚合 → 每层 × 每部位 V 注意力占比热力图。
仅支持 gate="none" 的标准 decoder（V4B 口径）。

Usage (touch_gait env, 仓库根执行):
  python results_display/script/r_test9_trust.py --e1
  python results_display/script/r_test9_trust.py --e2 \
      --ckpt results/AnySole/V4B_joint_and/checkpoints/ckpt_last.pt --split val --sessions 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.eval import _load_model
from anysole.train import condition_inputs, load_config, move_batch, resolve_device
from anysole.types import CONFIG_VT

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_PRIOR = "#c3c2b7"   # 先验角（灰，非系列色）
C_V = "#2a78d6"       # V-only
C_T = "#eb6834"       # T-only

ANATOMICAL = ["root", "torso", "headneck",
              "l_arm", "r_arm", "l_leg", "r_leg", "l_foot", "r_foot"]
AGGREGATES = ["upper", "lower", "anklefoot", "hands"]
PARTS = ANATOMICAL + AGGREGATES


def pa_key(g: str) -> str:
    if g in ANATOMICAL:
        return "PA-MPJPE_part_%s" % g
    return "PA-MPJPE_%s" % g


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R_Test9 信任画像（任务书实验 5）")
    parser.add_argument("--e1", action="store_true", help="E1 信任堆叠条（读 grid_metrics.json）")
    parser.add_argument("--e2", action="store_true", help="E2 注意力聚合（需 GPU + ckpt）")
    parser.add_argument("--grid", type=Path,
                        default=REPO / "results_display" / "result" / "r_test5_rho_grid" / "grid_metrics.json")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=REPO / "anysole" / "configs" / "v1.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--sessions", type=int, default=None,
                        help="E2 用到的 session 数（默认全部）")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "result" / "r_test9_trust",
                        help="产物目录")
    return parser.parse_args(argv)


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_grid(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit("grid_metrics.json 缺失：%s（先跑 python -m anysole.rho_grid）" % path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def corner_values(data: dict, metric: str) -> dict:
    """三角（prior / V-only / T-only）的每部位指标（种子均值）。"""
    out = {}
    for name, (rV, rT) in (("prior", (0, 0)), ("V-only", (100, 0)), ("T-only", (0, 100))):
        vals = []
        for seed_key, cells in data["cells"].items():
            m = cells.get("rhoV%d_rhoT%d" % (rV, rT), {}).get(metric)
            if m is not None:
                vals.append(m)
        if not vals:
            raise SystemExit("网格缺 %s 角（%s）" % (name, metric))
        out[name] = float(np.mean(vals))
    return out


def run_e1(args: argparse.Namespace) -> int:
    data = load_grid(args.grid)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = PARTS + ["overall"]
    rows = []
    for g in groups:
        key = pa_key(g) if g != "overall" else "PA-MPJPE"
        row = {"group": g}
        for name, v in corner_values(data, key).items():
            row[name] = round(v, 2)
        best = min(("prior", "V-only", "T-only"), key=lambda n: row[n])
        row["trusts"] = {"prior": "先验", "V-only": "V", "T-only": "T"}[best]
        rows.append(row)

    with open(out_dir / "trust_profile.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "prior", "V-only", "T-only", "trusts"])
        w.writeheader()
        w.writerows(rows)

    labels = [r["group"] for r in rows]
    vals = {name: [r[name] for r in rows] for name in ("prior", "V-only", "T-only")}
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    x = np.arange(len(rows))
    width = 0.24
    for i, name in enumerate(("prior", "V-only", "T-only")):
        color = {"prior": C_PRIOR, "V-only": C_V, "T-only": C_T}[name]
        ax.bar(x + (i - 1) * width, vals[name], width, color=color, label=name, zorder=3)
    ymax = max(vals["prior"]) * 1.22
    for g, row in zip(x, rows):
        ax.text(g, ymax * 0.94, row["trusts"], ha="center", va="top",
                fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PA-MPJPE (mm)", color=INK2, fontsize=9)
    ax.set_ylim(0, ymax)
    ax.set_title("Trust profile by part: who does each part rely on (lower = better)",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "trust_profile.png", facecolor=SURFACE)
    plt.close(fig)
    for r in rows:
        print("  %-9s prior=%6.1f V=%6.1f T=%6.1f -> %s"
              % (r["group"], r["prior"], r["V-only"], r["T-only"], r["trusts"]))
    print("产物：%s（trust_profile.png / trust_profile.csv）" % out_dir)
    return 0


def run_e2(args: argparse.Namespace) -> int:
    if args.ckpt is None:
        raise SystemExit("E2 需要 --ckpt")
    device = resolve_device(args.device)
    config = load_config(args.config)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    saved = checkpoint.get("config", {})
    model = _load_model(checkpoint, config, device)
    model.eval()
    if model.pose_head.gate != "none":
        raise SystemExit("E2 仅支持 gate='none' 标准 decoder（got %r）" % model.pose_head.gate)
    tw = int(saved.get("tw", config["tw"]))
    session_ids = load_split_ids(Path(config["split_csv"]),
                                 {"val": "val", "test": "test"}[args.split])
    if args.sessions is not None:
        session_ids = session_ids[: args.sessions]
    dataset = AnySoleDataset(
        mode="eval",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=tw,
        session_ids=session_ids,
        contact_method=str(saved.get("contact_method", "joint_and")),
        tactile_input=str(saved.get("tactile_input", "raw108")),
        no_imu=bool(saved.get("no_imu", False)),
        v_input=str(saved.get("v_input", "hrnet")),
        f2_repr=bool(saved.get("f2_repr", False)),
        smpl_roots=saved.get("smpl_roots", config.get("smpl_roots")),
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset 无有效窗口")

    n_layers = model.pose_head.decoder.num_layers
    n_parts = model.pose_head.n_parts
    from anysole.utils.eval_protocol import PART_NAMES
    slot_names = list(PART_NAMES)[:n_parts]
    v_hmr_mode = str(saved.get("v_input", "hrnet")) == "hmr_gvhmr"
    collected = {layer_i: [] for layer_i in range(n_layers)}
    handles = []

    def make_hooks(layer_i, module):
        def pre_hook(mod, args, kwargs):
            kwargs["need_weights"] = True
            return None

        def forward_hook(mod, args, kwargs, output):
            weights = output[1] if isinstance(output, tuple) else None
            if weights is not None:
                collected[layer_i].append(weights.detach().cpu())
            return None

        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.register_forward_hook(forward_hook, with_kwargs=True))

    for layer_i, layer in enumerate(model.pose_head.decoder.layers):
        make_hooks(layer_i, layer.multihead_attn)

    try:
        groups: dict = {}
        for i in range(len(dataset)):
            groups.setdefault(dataset[i]["session_id"], []).append(i)
        with torch.inference_mode():
            for session_id, idxs in groups.items():
                raw = [dataset[i] for i in idxs]
                batch = move_batch(collate_windows(raw), device)
                bsz = batch["pose_gt"].shape[0]
                config_id = torch.full((bsz,), CONFIG_VT, device=device, dtype=torch.long)
                v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                model(v_feat, t_raw, t_phys, config_id, batch.get("session_id"),
                      T_s2m=t_s2m,
                      V_hmr=batch.get("V_hmr") if v_hmr_mode else None)
    finally:
        for h in handles:
            h.remove()

    # 每层：聚合 (B, L, S) → L=tw*n_parts → 每部位 (n_parts, S) → V 段占比
    v_share = np.zeros((n_layers, n_parts))
    for layer_i, ws in collected.items():
        w = torch.cat(ws, dim=0)                      # (N, L, S)
        N, L, S = w.shape
        if L != tw * n_parts or S != 2 * tw:
            raise RuntimeError("意外注意力形状 (%d,%d,%d)，期望 L=tw*%d, S=2*tw=%d"
                               % (N, L, S, n_parts, 2 * tw))
        w = w.view(N, tw, n_parts, S).mean(dim=(0, 1))  # (n_parts, S)
        v_share[layer_i] = (w[:, :tw].sum(dim=1) / (w.sum(dim=1) + 1e-12)).numpy()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "attention_trust.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + slot_names)
        for layer_i in range(n_layers):
            w.writerow([layer_i + 1] + [round(v, 4) for v in v_share[layer_i]])
    np.savez_compressed(out_dir / "attention_trust.npz", v_share=v_share)

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    from matplotlib.colors import LinearSegmentedColormap
    blue_steps = {250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7", 400: "#3987e5",
                  450: "#2a78d6", 500: "#256abf", 550: "#1c5cab", 600: "#184f95",
                  650: "#104281", 700: "#0d366b"}
    cmap = LinearSegmentedColormap.from_list(
        "blue_seq", [blue_steps[s] for s in sorted(blue_steps)], N=256)
    im = ax.imshow(v_share, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(n_layers):
        for j in range(n_parts):
            ax.text(j, i, "%.2f" % v_share[i, j], ha="center", va="center",
                    fontsize=8, color="#ffffff" if v_share[i, j] > 0.45 else INK)
    ax.set_xticks(range(n_parts))
    ax.set_xticklabels(slot_names, fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(["L%d" % (i + 1) for i in range(n_layers)], fontsize=8)
    ax.set_xlabel("part", color=INK2, fontsize=9)
    ax.set_ylabel("decoder layer", color=INK2, fontsize=9)
    ax.set_title("Attention V-share per layer × part (VT2M; 1.0 = 只看 V 段)",
                 color=INK, fontsize=11, loc="left", pad=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_trust.png", facecolor=SURFACE)
    plt.close(fig)
    print("每部位 V 注意力占比（全层均值）：")
    for j, p in enumerate(slot_names):
        print("  %-9s %.2f" % (p, v_share[:, j].mean()))
    print("产物：%s（attention_trust.png / .csv / .npz）" % out_dir)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.e1:
        return run_e1(args)
    if args.e2:
        return run_e2(args)
    parse_args(["--help"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
