#!/usr/bin/env python3
"""V8 E5 三种视觉退化情形的整理出图（2026-11-09）。

三种情形 = 2D 遮挡（occ）/ 时间断流（gap）/ 场景遮挡（scene）。
每个情形一个子目录，根层放总图，`metrics/` 放单指标子图。

对照线：
  V8_Cv5（主模型）
  V8_Cv7（无门控）
  WHAM  （已完成的纯视觉基线；仅 occ/gap，36-session canonical）
  T-only（纯触觉推理参考线）

数据源：
  V8 cells: results/cells/<fam>_<fold>_s1[_evalocc*/_evalgap*/_evalscene|evalsceneclean]
            metrics/test.json（四折 mean±std）
  WHAM:     results/baselines/WHAM/predictions/<variant>/eval_motion/metrics/
            test_comparison.json（summary）
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
BASE = REPO / "results_display" / "Etest_crossfit_v8"
WHAM = REPO / "results" / "baselines" / "WHAM" / "predictions"

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODELS = {"V8_Cv5": "V8_Cv5 (main)", "V8_Cv7": "V8_Cv7 (no gate)"}
T_LABEL = "T only (tactile, no V)"
METRICS = [
    ("PA-MPJPE", "PA-MPJPE (mm)", "PA-MPJPE"),
    ("ankle_foot_mpjpe", "ankle-foot MPJPE (mm)", "MPJPE_anklefoot"),
    ("GlobalMPJPE", "GlobalMPJPE (mm, unanchored)",
     "GlobalMPJPE"),
    ("FootSliding", "FootSliding, GT anchored (mm)",
     "FootSliding（GT Anchored）"),
    ("GroundPen", "GroundPen, GT anchored (mm)",
     "GroundPen（GT Anchored）"),
    ("traj_ATE", "traj_ATE, RootAbs (mm)", "RootAbs"),
]
BASELINE_KEYS = {
    "PA-MPJPE": "pa_mpjpe_mm", "ankle_foot_mpjpe": None,
    "GlobalMPJPE": "global_mpjpe_mm", "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm", "traj_ATE": None,
}
BASELINE_MODELS = ("WHAM", "GVHMR", "SAM3DB")
COLORS = {"V8_Cv5": "#2a78d6", "V8_Cv5_V": "#86b6ef",
          "V8_Cv7": "#52514e", "WHAM": "#eb6834",
          "GVHMR": "#1baf7a", "SAM3DB": "#e87ba4",
          "T-only": "#4a3aa7"}
MARKERS = {"V8_Cv5": "o", "V8_Cv5_V": "o", "V8_Cv7": "s",
           "WHAM": "^", "GVHMR": "v", "SAM3DB": "P", "T-only": "D"}
LABELS = {"V8_Cv5": "V8_Cv5 @VT (main)",
          "V8_Cv5_V": "V8_Cv5 @V (visual only)",
          "V8_Cv7": "V8_Cv7 @VT (no gate)",
          "WHAM": "WHAM @V",
          "GVHMR": "GVHMR @V", "SAM3DB": "SAM3DB @V",
          "T-only": T_LABEL}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _cell(fam: str, fold: str, suffix: str = "") -> Path:
    return CELLS / ("%s_%s_s1%s" % (fam, fold, suffix))


def _metrics(fam: str, fold: str, suffix: str) -> dict | None:
    p = _cell(fam, fold, suffix) / "metrics" / "test.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["metrics"]


def value(fam: str, fold: str, suffix: str, cond: str, key: str):
    m = _metrics(fam, fold, suffix)
    if not m:
        return None
    block = m.get(cond, {})
    return block.get(key)


def mean_std(values) -> tuple[float, float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def series(fam: str, cond: str, metric_key: str, suffixes: list[str]):
    means, stds = [], []
    for suffix in suffixes:
        vals = [value(fam, fold, suffix, cond, metric_key) for fold in FOLDS]
        m, s = mean_std(vals)
        means.append(m); stds.append(s)
    return np.asarray(means), np.asarray(stds)


def baseline_series(model: str, variants: list[str], metric: str):
    key = BASELINE_KEYS.get(metric)
    if key is None:
        return None
    root = REPO / "results" / "baselines" / model / "predictions"
    means = []
    for variant in variants:
        p = root / variant / "eval_motion" / "metrics" / "test_comparison.json"
        if not p.is_file():
            means.append(float("nan")); continue
        summary = json.loads(p.read_text(encoding="utf-8")).get("summary", {})
        means.append(float(summary.get(key, float("nan"))))
    return np.asarray(means)


CONDITIONS = {
    "occ": {
        "dir": "E5_OcclusionRobustness",
        "title": "2D occlusion",
        "xlabel": "Occlusion (%)",
        "x": [0, 20, 40, 60, 80, 100],
        "levels": ["", "_evalocc020", "_evalocc040", "_evalocc060",
                   "_evalocc080", "_evalocc100"],
        "baseline": ["e5_clean", "e5_occ_lower020", "e5_occ_lower040",
                 "e5_occ_lower060", "e5_occ_lower080", "e5_occ_lower100"],
    },
    "gap": {
        "dir": "E5_TimeGap",
        "title": "Temporal gap",
        "xlabel": "Dropout rho (%)",
        "x": [0, 10, 20, 40, 60, 80],
        "levels": ["", "_evalgap010", "_evalgap020", "_evalgap040",
                   "_evalgap060", "_evalgap080"],
        "baseline": ["e5_clean", "e5_gap_r010", "e5_gap_r020", "e5_gap_r040",
                 "e5_gap_r060", "e5_gap_r080"],
    },
    "scene": {
        "dir": "E5_SceneOcclusion",
        "title": "Scene occlusion (static column)",
        "xlabel": "Scene condition",
        "x": [0, 1],
        "levels": ["_evalsceneclean", "_evalscene"],
        "baseline": ["e5_scene_clean", "e5_scene"],
    },
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def _plot_series(ax, x, means, stds, name, relative=False):
    ax.plot(x, means, color=COLORS[name], lw=2.0, marker=MARKERS[name],
            ms=4.5, ls=":" if name == "V8_Cv5_V" else ("--" if name in ("V8_Cv7", "T-only") else "-"),
            label=LABELS[name])
    if np.isfinite(stds).any():
        ax.fill_between(x, means - stds, means + stds, color=COLORS[name],
                        alpha=0.12, lw=0)


KEY_BY_METRIC = {name: key for name, _, key in METRICS}


def _values_for(condition: str, metric: str) -> dict:
    cfg = CONDITIONS[condition]
    key = KEY_BY_METRIC[metric]
    out = {}
    for fam in MODELS:
        means, stds = series(fam, "VT2M", key, cfg["levels"])
        out[fam] = (means, stds)
        tmeans, tstds = series(fam, "T2M", key, cfg["levels"])
        if fam == "V8_Cv5":
            out["V8_Cv5_V"] = series("V8_Cv5", "V2M", key, cfg["levels"])
            out["T-only"] = (tmeans, tstds)
    for baseline in BASELINE_MODELS:
        vals = baseline_series(baseline, cfg["baseline"], metric)
        if vals is None:
            continue
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            continue
        if condition == "scene" and not finite.all():
            continue
        out[baseline] = (vals, np.zeros_like(vals))
    return out


def _relative(data: dict) -> dict:
    out = {}
    for name, (means, stds) in data.items():
        base = means[0]
        out[name] = (means - base, stds)
    return out


def figure_metric(condition: str, metric: str, label: str,
                  relative: bool, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = CONDITIONS[condition]
    data = _values_for(condition, metric)
    if relative:
        data = _relative(data)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    _style(ax)
    x = np.asarray(cfg["x"], dtype=float)
    for name in ("V8_Cv5", "V8_Cv5_V", "V8_Cv7", "WHAM", "GVHMR",
                 "SAM3DB", "T-only"):
        if name not in data:
            continue
        means, stds = data[name]
        _plot_series(ax, x, means, stds, name, relative)
    if relative:
        ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":")
    ax.set_xlabel(cfg["xlabel"], fontsize=9.5, color=MUTED)
    ax.set_ylabel(("Δ vs clean " if relative else "") + label,
                  fontsize=9.5, color=MUTED)
    ax.set_title(("Relative " if relative else "Absolute ") + label +
                 " — " + cfg["title"], fontsize=10.5, color=INK,
                 loc="left")
    if condition == "scene":
        ax.set_xticks(x)
        ax.set_xticklabels(["clean", "scene"])
    else:
        ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=8.2, loc="lower right" if condition == "scene" else "best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % path)


def figure_pa(condition: str, relative: bool, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = CONDITIONS[condition]
    data = _values_for(condition, "PA-MPJPE")
    if relative:
        data = _relative(data)
    fig, ax = plt.subplots(figsize=(9.2, 4.2), facecolor=SURFACE)
    _style(ax)
    x = np.asarray(cfg["x"], dtype=float)
    for name in ("V8_Cv5", "V8_Cv5_V", "V8_Cv7", "WHAM", "GVHMR",
                 "SAM3DB", "T-only"):
        if name not in data:
            continue
        means, stds = data[name]
        _plot_series(ax, x, means, stds, name, relative)
    if relative:
        ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":")
    ax.set_xlabel(cfg["xlabel"], fontsize=10, color=MUTED)
    ax.set_ylabel(("Δ vs clean " if relative else "") + "PA-MPJPE (mm)",
                  fontsize=10, color=MUTED)
    ax.set_title(("Relative" if relative else "Absolute") +
                 " PA-MPJPE — " + cfg["title"] +
                 " (4-fold mean; baselines only where completed)",
                 fontsize=11, color=INK, loc="left")
    if condition == "scene":
        ax.set_xticks(x); ax.set_xticklabels(["clean", "scene"])
    else:
        ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=8.6, loc="lower right" if condition == "scene" else "best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % path)


def figure_metrics(condition: str, relative: bool, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = CONDITIONS[condition]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), facecolor=SURFACE)
    x = np.asarray(cfg["x"], dtype=float)
    for ax, (metric, label, _) in zip(axes.flat, METRICS):
        _style(ax)
        data = _values_for(condition, metric)
        if relative:
            data = _relative(data)
        for name in ("V8_Cv5", "V8_Cv5_V", "V8_Cv7", "WHAM", "GVHMR",
                 "SAM3DB", "T-only"):
            if name not in data:
                continue
            means, stds = data[name]
            _plot_series(ax, x, means, stds, name, relative)
        if relative:
            ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":")
        ax.set_title(label, fontsize=9.8, color=INK, loc="left")
        ax.set_xlabel(cfg["xlabel"], fontsize=8.5, color=MUTED)
        ax.set_ylabel(("Δ vs clean" if relative else "value") +
                      " (mm)", fontsize=8.5, color=MUTED)
        if condition == "scene":
            ax.set_xticks(x); ax.set_xticklabels(["clean", "scene"])
        else:
            ax.set_xticks(x)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.6, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(("Relative" if relative else "Absolute") +
                 " metrics — " + cfg["title"] +
                 " (4-fold mean; baselines only where completed)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % path)


def write_detail(condition: str) -> None:
    cfg = CONDITIONS[condition]
    out = BASE / cfg["dir"] / "detail"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric, label, key in METRICS:
        for name, (means, stds) in _values_for(condition, metric).items():
            for x, m, s in zip(cfg["x"], means, stds):
                rows.append([metric, name, x, round(float(m), 3),
                             round(float(s), 3)])
    path = out / ("E5_%s_Values.csv" % condition)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "series", "level", "mean", "std"])
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def main() -> int:
    for condition, cfg in CONDITIONS.items():
        out = BASE / cfg["dir"]
        out.mkdir(parents=True, exist_ok=True)
        figure_pa(condition, False, out / "Main_Fig4_Gate_Absolute.png")
        figure_pa(condition, True, out / "Main_Fig5_Gate_Relative.png")
        figure_metrics(condition, False,
                       out / "Main_Fig6_Gate_Metrics_Absolute.png")
        figure_metrics(condition, True,
                       out / "Main_Fig7_Gate_Metrics_Relative.png")
        for metric, label, _ in METRICS:
            figure_metric(condition, metric, label, False,
                          out / "metrics" / ("%s_absolute.png" % metric))
            figure_metric(condition, metric, label, True,
                          out / "metrics" / ("%s_relative.png" % metric))
        write_detail(condition)
    print("[done] %s" % BASE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
