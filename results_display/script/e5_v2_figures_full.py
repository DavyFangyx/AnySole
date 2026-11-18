#!/usr/bin/env python3
"""E5 all-metric V/T/VT + baseline figures (2026-11-17).

V8_IMU_Residual V/T/VT comes from Main_Table1_E5_*_Models.csv.
Baselines (WHAM, GVHMR, MotionPRO) are read directly from their raw
per-condition eval jsons (no cross-pipeline offset correction — this is a
quick-look sheet, not the paper's Main_Table2). A baseline line is only
drawn for metrics it actually reports.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results_display/Etest_crossfit_physimu_vgate"
OUT = SRC / "1_E5test"
MODEL = "V8_IMU_Residual"
BASE_ROOT = REPO / "results/baselines"

METRICS = (
    ("PA-MPJPE", "PA-MPJPE (mm)"),
    ("MPJPE", "MPJPE (mm)"),
    ("MPJRE", "MPJRE (deg)"),
    ("GlobalMPJPE", "GlobalMPJPE (mm)"),
    ("W-MPJPE100", "W-MPJPE@100 (mm)"),
    ("traj_ATE", "traj_ATE (mm)"),
    ("RootOrientationDrift", "Root orientation drift (deg)"),
    ("RootAbs", "RootAbs (mm)"),
    ("GroundPen", "Ground penetration, GT anchored (mm)"),
    ("FootSliding", "Foot sliding, GT anchored (mm)"),
    ("Jitter_pred", "Jitter_pred"),
)
ROWS = (("V2M", "@V", "#86b6ef", "o"),
        ("T2M", "@T (T+BiFoot)", "#4a3aa7", "D"),
        ("VT2M", "@VT (Ours)", "#2a78d6", "o"))
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

# metric -> (test_comparison.json summary key, fourfold_summary.json key)
BASELINE_KEYS = {
    "PA-MPJPE": ("pa_mpjpe_mm", "pa_mpjpe_mm"),
    "MPJPE": ("mpjpe_mm", "mpjpe_mm"),
    "MPJRE": ("mpjae_deg", None),
    "GlobalMPJPE": ("global_mpjpe_mm", "global_mpjpe_mm"),
    "W-MPJPE100": ("w_mpjpe100_mm", "w_mpjpe100_mm"),
    "traj_ATE": ("root_ate_mm", "root_ate_mm"),
    "RootOrientationDrift": ("root_orientation_drift_deg", None),
    "RootAbs": ("root_ate_mm", "root_ate_mm"),
    "GroundPen": ("ground_pen_mm", "ground_pen_mm"),
    "FootSliding": ("foot_sliding_mm", "foot_sliding_mm"),
    "Jitter_pred": ("jitter_pred_m_s3", "jitter_pred_m_s3"),
}
BASELINE_STYLE = {
    "WHAM": ("#eb6834", "^"),
    "GVHMR": ("#1baf7a", "v"),
    "MotionPRO": ("#8c564b", "X"),
}
LEVEL_SETS = {
    "occ": {("occ%03d" % i): "e5_occ_lower%03d" % i for i in range(10, 91, 10)}
            | {"occ000": "e5_clean"},
    "gap": {"gap000": "e5_clean", "gap010": "e5_gap_r010", "gap020": "e5_gap_r020",
            "gap040": "e5_gap_r040", "gap060": "e5_gap_r060"},
    "scene": {"scene_clean": "e5_scene_clean", "scene": "e5_scene"},
}


def parse(cell):
    if not cell or cell == "—":
        return None, None
    try:
        m, s = cell.split("±")
        return float(m), float(s)
    except ValueError:
        return None, None


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def series(rows, mode, metric, xmap):
    xs, ms, ss = [], [], []
    for r in rows:
        if r["模型"] != MODEL or r["行"] != mode:
            continue
        x = xmap.get(r["档位"])
        if x is None:
            continue
        m, s = parse(r.get(metric, ""))
        if m is None:
            continue
        xs.append(x); ms.append(m); ss.append(s)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ms[i] for i in order], [ss[i] for i in order]


def wham_gvhmr_value(model, level_dir, key):
    p = BASE_ROOT / model / "predictions" / level_dir / "eval_motion/metrics/test_comparison.json"
    if not p.is_file():
        return None
    v = json.loads(p.read_text(encoding="utf-8")).get("summary", {}).get(key)
    return float(v) if v is not None else None


def motionpro_value(level_dir, key):
    p = BASE_ROOT / "MotionPRO" / "predictions" / level_dir / "metrics/fourfold_summary.json"
    if not p.is_file():
        return None, None
    entry = json.loads(p.read_text(encoding="utf-8")).get(key)
    if not entry or entry.get("n_folds", 0) < 4:
        return None, None
    return float(entry["mean"]), float(entry["std"])


def baseline_series(level_kind, xmap, metric):
    sum_key, ff_key = BASELINE_KEYS[metric]
    level_dirs = LEVEL_SETS[level_kind]
    out = {}
    for model in ("WHAM", "GVHMR"):
        xs, ms = [], []
        for level, level_dir in level_dirs.items():
            x = xmap.get(level)
            if x is None:
                continue
            v = wham_gvhmr_value(model, level_dir, sum_key)
            if v is None:
                continue
            xs.append(x); ms.append(v)
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            out[model] = ([xs[i] for i in order], [ms[i] for i in order],
                          [0.0] * len(xs))
    if ff_key:
        xs, ms, ss = [], [], []
        for level, level_dir in level_dirs.items():
            x = xmap.get(level)
            if x is None:
                continue
            m, s = motionpro_value(level_dir, ff_key)
            if m is None:
                continue
            xs.append(x); ms.append(m); ss.append(s)
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            out["MotionPRO"] = ([xs[i] for i in order], [ms[i] for i in order],
                                [ss[i] for i in order])
    return out


def draw(ax, rows, metric, xmap, xlabel, title, level_kind):
    for mode, label, color, marker in ROWS:
        xs, ms, ss = series(rows, mode, metric, xmap)
        if not xs:
            continue
        ax.plot(xs, ms, color=color, lw=2.0, marker=marker, ms=4.2, label=label)
        ax.fill_between(xs, [m - s for m, s in zip(ms, ss)],
                        [m + s for m, s in zip(ms, ss)],
                        color=color, alpha=0.13, lw=0)
    for model, (xs, ms, ss) in baseline_series(level_kind, xmap, metric).items():
        color, marker = BASELINE_STYLE[model]
        ax.plot(xs, ms, color=color, lw=1.6, ls="--", marker=marker, ms=4.0,
                label=model + " @V" if model != "MotionPRO" else "MotionPRO @VT")
        if any(ss):
            ax.fill_between(xs, [m - s for m, s in zip(ms, ss)],
                            [m + s for m, s in zip(ms, ss)],
                            color=color, alpha=0.10, lw=0)
    ax.set_title(title, fontsize=9.2, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.2, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=7.8)
    ax.grid(color="#e6e3de", lw=0.7)
    ax.set_facecolor(SURFACE)


def make(scenario, table1_path, xmap, xlabel, title, tag, level_kind):
    rows = load(table1_path)
    d = OUT / scenario
    d.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(17.0, 10.2), facecolor=SURFACE)
    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        draw(ax, rows, metric, xmap, xlabel, label, level_kind)
    for ax in axes.ravel()[len(METRICS):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=6,
                   frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, fontsize=12, color=INK, y=1.06)
    fig.tight_layout()
    p = d / ("Main_Fig_E5_%s_AllMetrics.png" % tag)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)

    md = d / "metrics"
    md.mkdir(exist_ok=True)
    for metric, label in METRICS:
        fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
        draw(ax, rows, metric, xmap, xlabel, label, level_kind)
        ax.legend(frameon=False, fontsize=7.4)
        fig.tight_layout()
        q = md / ("%s.png" % metric)
        fig.savefig(q, dpi=170, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)


def main():
    make("E5_OcclusionRobustness",
         SRC / "E5_OcclusionRobustness/Main_Table1_E5_Occlusion_Models.csv",
         {("occ%03d" % i): i for i in range(0, 91, 10)},
         "Occlusion (%)", "E5 occlusion robustness — all metrics (occ0-90)",
         "Occlusion", "occ")
    make("E5_SceneOcclusion",
         SRC / "E5_SceneOcclusion/Main_Table1_E5_SceneOcclusion_Models.csv",
         {"scene_clean": 0, "scene": 1}, "Scene condition",
         "E5 scene occlusion — all metrics", "Scene", "scene")
    make("E5_TimeGap",
         SRC / "E5_TimeGap/Main_Table1_E5_TimeGap_Models.csv",
         {"gap000": 0, "gap010": 10, "gap020": 20, "gap040": 40, "gap060": 60},
         "Dropout rho (%)", "E5 time gap — all metrics", "TimeGap", "gap")


if __name__ == "__main__":
    main()
