#!/usr/bin/env python3
"""R_Test10 / B3: continuous rho-grid missingness display."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from utils.mpl_fonts import setup_cjk_fonts

setup_cjk_fonts()

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
AXIS = "#c3c2b7"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="R_Test10 / B3 rho-grid display")
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--metric", default="pa_mpjpe_mm")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit("rho 网格产物缺失：%s" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "cells" not in data:
        raise SystemExit("不是 rho-grid 产物：%s" % path)
    return data


def matrix(data, metric):
    rhos = data["rhos"]
    out = np.full((len(rhos), len(rhos)), np.nan)
    std = np.full_like(out, np.nan)
    for i, r_t in enumerate(rhos):
        for j, r_v in enumerate(rhos):
            values = []
            for cells in data["cells"].values():
                if metric in cells.get("rhoV%d_rhoT%d" % (r_v, r_t), {}):
                    values.append(cells["rhoV%d_rhoT%d" % (r_v, r_t)][metric])
            if values:
                out[i, j] = np.mean(values)
                std[i, j] = np.std(values) if len(values) > 1 else 0.0
    return out, std


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def draw(data, mean, std, metric, out):
    rhos = data["rhos"]
    cmap = LinearSegmentedColormap.from_list("b3_blue", ["#86b6ef", "#0d366b"])
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    vmin, vmax = float(np.nanmin(mean)), float(np.nanmax(mean))
    image = ax.imshow(mean, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(rhos)), ["%d%%" % x for x in rhos])
    ax.set_yticks(range(len(rhos)), ["%d%%" % x for x in rhos])
    ax.set_xlabel("V retention", color=MUTED)
    ax.set_ylabel("T retention", color=MUTED)
    ax.set_title("B3 rho grid: %s" % metric, loc="left", color=INK, pad=12)
    for i in range(len(rhos)):
        for j in range(len(rhos)):
            if np.isfinite(mean[i, j]):
                label = "%.1f" % mean[i, j]
                if np.isfinite(std[i, j]) and std[i, j] >= 0.05:
                    label += "±%.1f" % std[i, j]
                frac = (mean[i, j] - vmin) / max(vmax - vmin, 1e-9)
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="white" if frac > 0.45 else INK)
    for rv, rt, label in ((100, 100, "VT"), (100, 0, "V"), (0, 100, "T"), (0, 0, "Prior")):
        if rv in rhos and rt in rhos:
            ax.text(rhos.index(rv), rhos.index(rt) - 0.42, label,
                    ha="center", va="top", fontsize=7, color=MUTED)
    fig.colorbar(image, ax=ax, pad=0.02).set_label(metric)
    fig.tight_layout()
    fig.savefig(out / "heatmap.png", facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140)
    for ax in axes:
        style(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].plot(rhos, mean[-1, :], "-o", color=BLUE)
    axes[0].set(xlabel="V retention (T fixed 100%)", ylabel=metric, title="V sweep")
    axes[0].text(0, mean[-1, 0], "T", color=BLUE, fontsize=8)
    axes[0].text(100, mean[-1, -1], "VT", color=BLUE, fontsize=8)
    axes[1].plot(rhos, mean[:, -1], "-o", color=ORANGE)
    axes[1].set(xlabel="T retention (V fixed 100%)", title="T sweep")
    axes[1].text(0, mean[0, -1], "V", color=ORANGE, fontsize=8)
    axes[1].text(100, mean[-1, -1], "VT", color=ORANGE, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "slices.png", facecolor=SURFACE)
    plt.close(fig)


def criteria(data, mean, metric, out):
    rhos = data["rhos"]
    higher = metric in {"contact_mcc", "contact_f1", "contact_acc", "air_recall"}
    prior = mean[rhos.index(0), rhos.index(0)]
    rows = []
    for i, rt in enumerate(rhos):
        for j in range(len(rhos) - 1):
            a, b = mean[i, j], mean[i, j + 1]
            if np.isfinite(a) and np.isfinite(b) and (b < a if higher else b > a):
                rows.append({"axis": "V", "fixed": rt, "from": rhos[j], "to": rhos[j + 1]})
    for j, rv in enumerate(rhos):
        for i in range(len(rhos) - 1):
            a, b = mean[i, j], mean[i + 1, j]
            if np.isfinite(a) and np.isfinite(b) and (b < a if higher else b > a):
                rows.append({"axis": "T", "fixed": rv, "from": rhos[i], "to": rhos[i + 1]})
    (out / "b3_criteria.json").write_text(json.dumps({
        "metric": metric, "higher_is_better": higher,
        "valid_cells": int(np.isfinite(mean).sum()), "total_cells": int(mean.size),
        "cells_better_than_prior": int(sum(
            np.isfinite(mean[i, j]) and ((mean[i, j] > prior) if higher else (mean[i, j] < prior))
            for i in range(len(rhos)) for j in range(len(rhos)) if (i, j) != (rhos.index(0), rhos.index(0))
        )), "monotonicity_violations": rows,
        "note": "描述性检查；显著性需另按 session 统计。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def corners(data, out):
    rows = []
    for cfg, metrics in data.get("corner_check", {}).items():
        if cfg == "note":
            continue
        for metric, values in metrics.items():
            rows.append({"corner": cfg, "metric": metric, **values})
    if rows:
        with (out / "corner_check.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    data = load(args.metrics)
    out = args.out_dir or args.metrics.parent
    out.mkdir(parents=True, exist_ok=True)
    mean, std = matrix(data, args.metric)
    if np.isnan(mean).all():
        raise SystemExit("网格中没有指标：%s" % args.metric)
    draw(data, mean, std, args.metric, out)
    criteria(data, mean, args.metric, out)
    corners(data, out)
    print("B3 完成：%s（heatmap.png / slices.png / b3_criteria.json）" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
