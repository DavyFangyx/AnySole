#!/usr/bin/env python3
"""E4-A fusion-comparison figures (V2 formal eval, 2026-11-15)."""
from __future__ import annotations
import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "results/diagnostics/20261114_imu_eval_v2"
EV_ALL = REPO / "results/diagnostics/20261114_imu_eval_v2_all"
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate/E4_ComponentAblation"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODELS = (
    ("V8_IMU_Residual", "Residual Fusion (Ours)", "#2a78d6", "o"),
    ("V8_IMU_Router", "Competitive routing (router-only)", "#eb6834", "^"),
    ("V8_IMU_TGate", "Sensing-branch gate", "#1baf7a", "v"),
    ("V8_IMU_DualGate", "Independent dual-branch gate", "#8c564b", "X"),
)
METRICS = (("PA-MPJPE", "PA-MPJPE (mm)"),
           ("GlobalMPJPE", "GlobalMPJPE (mm)"),
           ("GroundPen（GT Anchored）", "GroundPen (mm)"))
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

def _path(model, cond, fold):
    for root in (EV, EV_ALL):
        p = root / model / cond / ("%s.json" % fold)
        if p.is_file():
            return p
    return None

def value(model, cond, fold, key):
    p = _path(model, cond, fold)
    if p is None:
        return None
    d = json.loads(p.read_text())
    return d.get("metrics", {}).get("VT2M", {}).get(key)

def ms(model, cond, key):
    vals = [value(model, cond, f, key) for f in FOLDS]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

def draw(ax, xs, conds, metric, title, xlabel):
    for model, label, color, marker in MODELS:
        means, stds, xx = [], [], []
        for x, cond in zip(xs, conds):
            m, s = ms(model, cond, metric)
            if m is None:
                continue
            xx.append(x); means.append(m); stds.append(s)
        if not xx:
            continue
        ax.plot(xx, means, color=color, lw=2.0, marker=marker, ms=4.5,
                label=label)
        ax.fill_between(xx, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.13, lw=0)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color="#e6e3de", lw=0.7)
    ax.set_facecolor(SURFACE)

def figure(name, xs, conds, xlabel, title):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), facecolor=SURFACE)
    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        draw(ax, xs, conds, metric, label, xlabel)
        ax.set_ylabel("mm", fontsize=8.5, color=MUTED)
        if xlabel == "Scene condition":
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["clean", "scene"])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4,
                   frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=11, color=INK, y=1.10)
    fig.tight_layout()
    p = OUT / "metrics" / (name + ".png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)

def main():
    figure("Main_Fig_E4_Fusion_Occlusion",
           [0, 20, 50, 90, 100],
           ["clean", "occ020", "occ050", "occ090", "occ100"],
           "Occlusion (%)", "Fusion comparison — lower-body occlusion")
    figure("Main_Fig_E4_Fusion_Scene",
           [0, 1], ["sceneocc_clean", "sceneocc"],
           "Scene condition", "Fusion comparison — scene occlusion")
    figure("Main_Fig_E4_Fusion_TimeGap",
           [0, 10, 20, 40, 60], ["clean", "gap010", "gap020", "gap040", "gap060"],
           "Dropout rho (%)", "Fusion comparison — time gap")

if __name__ == "__main__":
    main()
