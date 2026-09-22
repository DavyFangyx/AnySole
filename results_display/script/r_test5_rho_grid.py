#!/usr/bin/env python3
"""R_Test5 ρ 网格前端（missing-rate 任务书实验 1）：热力图 + 切片 + 角格自检表。

纯读 `anysole.rho_grid` 生成的 grid_metrics.json（零推理），出：
  - heatmap.png：6×6 PA-MPJPE 热力图（蓝序贯 ramp，越浅越好；格内 mean±std）
  - slices.png：两条 1D 切片（顶行 = T 全在扫 V；左列 = V 全在扫 T），
    与 fseries 角点参照一起画（取代 robust_vdrop/tdrop 口径）
  - corner_check.csv：三角 vs fseries 对应配置行的自检差（必须 ≈0，否则生成器有 bug）

Usage (touch_gait env, 仓库根执行):
  python results_display/script/r_test5_rho_grid.py
  python results_display/script/r_test5_rho_grid.py --metrics \
      results_display/result/r_test5_rho_grid/grid_metrics.json --metric MPJPE
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
from matplotlib.colors import LinearSegmentedColormap

# 蓝序贯 ramp（dataviz 参考调色板 sequential blue，250–700）
BLUE_STEPS = {
    250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7", 400: "#3987e5",
    450: "#2a78d6", 500: "#256abf", 550: "#1c5cab", 600: "#184f95",
    650: "#104281", 700: "#0d366b",
}
C_SWEEP_V = "#2a78d6"   # 顶行切片（扫 V）
C_SWEEP_T = "#eb6834"   # 左列切片（扫 T）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CORNER_CFG = {(100, 0): "V2M", (0, 100): "T2M", (100, 100): "VT2M"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R_Test5 ρ 网格前端")
    parser.add_argument("--metrics", type=Path,
                        default=REPO / "results_display" / "result" / "r_test5_rho_grid" / "grid_metrics.json")
    parser.add_argument("--metric", type=str, default="PA-MPJPE",
                        help="热力图主指标（任意 grid_metrics 单元格内的数值指标）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="产物目录（默认 = grid_metrics.json 所在目录）")
    return parser.parse_args(argv)


def load_grid(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit("grid_metrics.json 缺失：%s（先跑 python -m anysole.rho_grid）" % path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "cells" not in data:
        raise SystemExit("%s 不是 rho_grid 产物（缺 cells）" % path)
    return data


def aggregate(data: dict, metric: str) -> tuple:
    """rhos × rhos 的 (mean, std) 矩阵；多种子时按种子聚合。"""
    rhos = data["rhos"]
    n = len(rhos)
    mean = np.full((n, n), np.nan)
    std = np.full((n, n), np.nan)
    for i, rT in enumerate(rhos):
        for j, rV in enumerate(rhos):
            vals = []
            for seed_key, cells in data["cells"].items():
                m = cells.get("rhoV%d_rhoT%d" % (rV, rT), {}).get(metric)
                if m is not None:
                    vals.append(m)
            if vals:
                mean[i, j] = float(np.mean(vals))
                std[i, j] = float(np.std(vals)) if len(vals) > 1 else 0.0
    return mean, std


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def heatmap(data: dict, mean: np.ndarray, std: np.ndarray, metric: str,
            out_dir: Path) -> None:
    rhos = data["rhos"]
    cmap = LinearSegmentedColormap.from_list(
        "blue_seq", [BLUE_STEPS[s] for s in sorted(BLUE_STEPS)], N=256)
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    vmin, vmax = float(np.nanmin(mean)), float(np.nanmax(mean))
    im = ax.imshow(mean, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels(["%d%%" % r for r in rhos])
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels(["%d%%" % r for r in rhos])
    ax.set_xlabel("V retention", color=INK2, fontsize=9)
    ax.set_ylabel("T retention", color=INK2, fontsize=9)
    ax.set_title("ρ grid: %s (%s, %s)" % (metric, data["split"], data.get("modal", "")),
                 color=INK, fontsize=11, loc="left", pad=12)
    # 格内数值（mean±std）；文字墨色随底色明暗切换（对比度兜底）
    for i in range(len(rhos)):
        for j in range(len(rhos)):
            if not np.isnan(mean[i, j]):
                frac = (mean[i, j] - vmin) / max(vmax - vmin, 1e-9)
                color = "#ffffff" if frac > 0.45 else INK
                label = "%.1f" % mean[i, j] if std[i, j] < 0.05 \
                    else "%.1f±%.1f" % (mean[i, j], std[i, j])
                ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color=color)
    # 角标
    for (rV, rT), name in ([(100, 100), "VT2M"], [(100, 0), "V2M"],
                           [(0, 100), "T2M"], [(0, 0), "prior"]):
        ax.text(rhos.index(rV), rhos.index(rT) - 0.42, name,
                ha="center", va="top", fontsize=7, color=INK2)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    cbar.set_label(metric, color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", facecolor=SURFACE)
    plt.close(fig)


def slices(data: dict, mean: np.ndarray, metric: str, out_dir: Path) -> None:
    rhos = data["rhos"]
    # 顶行（T=100% 扫 V）与左列（V=100% 扫 T）
    top_row = mean[-1, :]          # T retention 100%
    left_col = mean[:, 0]          # V retention 100%
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style_ax(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    ax = axes[0]
    ax.plot(rhos, top_row, "-o", color=C_SWEEP_V, linewidth=2, markersize=5, zorder=3)
    ax.set_xlabel("V retention (T fixed 100%)", color=INK2, fontsize=9)
    ax.set_ylabel(metric, color=INK2, fontsize=9)
    ax.set_title("V sweep (was robust_vdrop)", color=INK, fontsize=10, loc="left", pad=10)
    if (100, 0) in CORNER_CFG:
        ax.text(100, top_row[-1], "V2M", ha="left", va="bottom", fontsize=8, color=C_SWEEP_V)
        ax.text(100, top_row[0], "VT2M", ha="left", va="bottom", fontsize=8, color=C_SWEEP_V)

    ax = axes[1]
    ax.plot(rhos, left_col, "-o", color=C_SWEEP_T, linewidth=2, markersize=5, zorder=3)
    ax.set_xlabel("T retention (V fixed 100%)", color=INK2, fontsize=9)
    ax.set_title("T sweep (was robust_tdrop)", color=INK, fontsize=10, loc="left", pad=10)
    if (0, 100) in CORNER_CFG:
        ax.text(100, left_col[-1], "T2M", ha="left", va="bottom", fontsize=8, color=C_SWEEP_T)
        ax.text(100, left_col[0], "VT2M", ha="left", va="bottom", fontsize=8, color=C_SWEEP_T)

    fig.suptitle("1D slices of the ρ grid (%s)" % metric, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "slices.png", facecolor=SURFACE)
    plt.close(fig)


def corner_table(data: dict, out_dir: Path) -> None:
    check = data.get("corner_check", {})
    if not check:
        print("corner_check 为空（fseries 缺失或未跑角格）")
        return
    rows = []
    for cfg, metrics in check.items():
        if cfg == "note":
            print("corner_check note: %s" % metrics)
            continue
        for metric, m in metrics.items():
            rows.append({"corner": cfg, "metric": metric, "grid": m["grid"],
                         "fseries": m["fseries"], "diff": m["diff"]})
    if not rows:
        return
    with open(out_dir / "corner_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    max_diff = max(abs(r["diff"]) for r in rows)
    print("corner_check: %d 行，最大 |diff| = %s %s"
          % (len(rows), max_diff, "（须 ≈0，否则生成器与 fseries 口径不一致）"))
    for r in rows:
        flag = "  <-- 不一致！" if abs(r["diff"]) > 1e-3 else ""
        print("  %-5s %-14s grid=%.4f fseries=%.4f diff=%+.4f%s"
              % (r["corner"], r["metric"], r["grid"], r["fseries"], r["diff"], flag))


def main(argv=None) -> int:
    args = parse_args(argv)
    data = load_grid(args.metrics)
    out_dir = args.out_dir or args.metrics.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = aggregate(data, args.metric)
    if np.isnan(mean).all():
        raise SystemExit("网格里没有指标 %r" % args.metric)
    heatmap(data, mean, std, args.metric, out_dir)
    slices(data, mean, args.metric, out_dir)
    corner_table(data, out_dir)
    print("产物：%s（heatmap.png / slices.png / corner_check.csv）" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
