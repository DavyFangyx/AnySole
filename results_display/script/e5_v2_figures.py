#!/usr/bin/env python3
"""E5 absolute-error figures for V8_IMU_Residual vs baselines (2026-11-15)."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate"
METRICS = (("PA-MPJPE", "PA-MPJPE (mm)"), ("PA-MPJPE_anklefoot", "Ankle/foot PA-MPJPE"),
           ("MPJPE", "MPJPE (mm)"), ("MPJRE", "MPJRE (deg)"),
           ("GroundPen", "Ground penetration"), ("FootSliding", "Foot sliding"))
SERIES = (
    ("V8_IMU_Residual @VT", "Residual @VT (Ours)", "#2a78d6", "o"),
    ("V8_IMU_Residual @V", "Residual @V", "#86b6ef", "o"),
    ("V8_IMU_Residual @T", "Residual @T (T+BiFoot)", "#4a3aa7", "D"),
    ("WHAM @V", "WHAM @V", "#eb6834", "^"),
    ("GVHMR @V", "GVHMR @V", "#1baf7a", "v"),
    ("MotionPRO @VT", "MotionPRO @VT", "#8c564b", "X"),
)
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

def parse(cell):
    if not cell:
        return None, None
    cell = cell.replace("±", "|").split("|")
    try:
        m = float(cell[0])
    except ValueError:
        return None, None
    s = float(cell[1]) if len(cell) > 1 else 0.0
    return m, s

def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))

def series(rows, model, metric, xmap):
    xs, ms, ss = [], [], []
    for r in rows:
        if r["模型"] != model:
            continue
        x = xmap(r["档位"]) if callable(xmap) else xmap.get(r["档位"])
        if x is None:
            continue
        m, s = parse(r.get(metric, ""))
        if m is None:
            continue
        xs.append(x); ms.append(m); ss.append(s)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ms[i] for i in order], [ss[i] for i in order]

def draw(ax, rows, metric, xmap, xlabel, title):
    for model, label, color, marker in SERIES:
        xs, ms, ss = series(rows, model, metric, xmap)
        if not xs:
            continue
        ax.plot(xs, ms, color=color, lw=2.0, marker=marker, ms=4.2, label=label)
        ax.fill_between(xs, [m - s for m, s in zip(ms, ss)],
                        [m + s for m, s in zip(ms, ss)],
                        color=color, alpha=0.13, lw=0)
    ax.set_title(title, fontsize=9.2, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.2, color=MUTED)
    ax.set_ylabel("deg" if "deg" in title else "mm", fontsize=8.2, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=7.8)
    ax.grid(color="#e6e3de", lw=0.7)
    ax.set_facecolor(SURFACE)

def make(scenario, path, xmap, xlabel, title, tag):
    rows = load(path)
    d = OUT / scenario
    d.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), facecolor=SURFACE)
    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        draw(ax, rows, metric, xmap, xlabel, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=6,
                   frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, fontsize=11, color=INK, y=1.04)
    fig.tight_layout()
    p = d / ("Main_Fig_E5_%s_Absolute.png" % tag)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)
    md = d / "metrics"
    md.mkdir(exist_ok=True)
    for metric, label in METRICS:
        fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
        draw(ax, rows, metric, xmap, xlabel, label)
        ax.legend(frameon=False, fontsize=7.6)
        fig.tight_layout()
        q = md / ("%s_absolute.png" % metric)
        fig.savefig(q, dpi=170, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)

def main():
    make("E5_OcclusionRobustness",
         OUT / "E5_OcclusionRobustness/Main_Table2_E5_Occlusion_MainVsBaselines.csv",
         {("occ%03d" % i): i for i in range(0, 91, 10)},
         "Occlusion (%)", "E5 occlusion robustness — absolute error", "Occlusion")
    make("E5_SceneOcclusion",
         OUT / "E5_SceneOcclusion/Main_Table2_E5_SceneOcclusion_MainVsBaselines.csv",
         {"scene_clean": 0, "scene": 1}, "Scene condition",
         "E5 scene occlusion — absolute error", "Scene")
    make("E5_TimeGap",
         OUT / "E5_TimeGap/Main_Table2_E5_TimeGap_MainVsBaselines.csv",
         {"gap000": 0, "gap010": 10, "gap020": 20, "gap040": 40, "gap060": 60},
         "Dropout rho (%)", "E5 time gap — absolute error", "TimeGap")

if __name__ == "__main__":
    main()
