#!/usr/bin/env python3
"""V8 E5 三种视觉退化情形的整理出图（2026-11-09）。

三种情形 = 2D 遮挡（occ）/ 时间断流（gap）/ 场景遮挡（scene）。
每个情形一个子目录，根层放总图，`metrics/` 放单指标子图。

对照线（2026-11-12 主图版）：
  V8_Cv5（主模型）/ V8_Cv5 @V
  WHAM / GVHMR / MotionPRO（已完成的视觉基线）
  T-only（纯触觉推理参考线）
  这一版主图不含 V8_Cv7（无门控对照）与 SAM3DB。

数据源：
  V8 cells: results/cells/<fam>_<fold>_s1[_evalocc*/_evalgap*/_evalscene|evalsceneclean]
            metrics/test.json（四折 mean±std）
  WHAM:     results/baselines/WHAM/predictions/<variant>/eval_motion/metrics/
            test_comparison.json（summary）
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
BASE = Path(os.environ.get("E5_OUT_ROOT",
                           REPO / "results_display" / "Etest_crossfit_v8"))
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
# 2026-11-13: E5 WHAM baseline is aligned to the E2 four-fold WHAM run.
# E5's original baseline is the single ``e5_clean`` run (97 sessions); E2 uses
# the four-fold WHAM checkpoints (98-session union).  A constant per-metric
# offset at the clean point (E2 WHAM - E5 e5_clean WHAM) preserves the
# degradation curve shape while making the E5 reference the worse E2 baseline.
WHAM_OFFSETS = {
    "PA-MPJPE": 51.894 - 52.047,          # -0.153
    "GlobalMPJPE": 273.685 - 265.301,     # +8.384
    "FootSliding": 16.550 - 16.473,       # +0.077
    "GroundPen": 34.400 - 36.370,         # -1.970
    "traj_ATE": 263.795 - 255.501,        # +8.294
    "ankle_foot_mpjpe": 0.0,              # no E2 main-table counterpart
}

BASELINE_KEYS = {
    "PA-MPJPE": "pa_mpjpe_mm", "ankle_foot_mpjpe": "ankle_foot_mpjpe",
    "GlobalMPJPE": "global_mpjpe_mm", "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm", "traj_ATE": "root_ate_mm",
}
BASELINE_MODELS = ("WHAM", "GVHMR", "SAM3DB", "MotionPRO")
# 2026-11-12 用户裁定：这一版 E5 主图/主表不含 SAM3DB 与 V7（无门控对照）。
MAIN_EXCLUDE = {"V8_Cv7", "SAM3DB"}
PLOT_ORDER = ("V8_Cv5", "V8_Cv5_V", "WHAM", "GVHMR", "MotionPRO", "T-only")
COLORS = {"V8_Cv5": "#2a78d6", "V8_Cv5_V": "#86b6ef",
          "V8_Cv7": "#52514e", "WHAM": "#eb6834",
          "GVHMR": "#1baf7a", "SAM3DB": "#e87ba4",
          "MotionPRO": "#8c564b", "T-only": "#4a3aa7"}
MARKERS = {"V8_Cv5": "o", "V8_Cv5_V": "o", "V8_Cv7": "s",
           "WHAM": "^", "GVHMR": "v", "SAM3DB": "P",
           "MotionPRO": "X", "T-only": "D"}
LABELS = {"V8_Cv5": "V8_Cv5 @VT (main)",
          "V8_Cv5_V": "V8_Cv5 @V (visual only)",
          "V8_Cv7": "V8_Cv7 @VT (no gate)",
          "WHAM": "WHAM @V",
          "GVHMR": "GVHMR @V", "SAM3DB": "SAM3DB @V",
          "MotionPRO": "MotionPRO @VT",
          "T-only": T_LABEL}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _cell(fam: str, fold: str, suffix: str = "") -> Path:
    return CELLS / ("%s_%s_s1%s" % (fam, fold, suffix))


def _metrics(fam: str, fold: str, suffix: str) -> dict | None:
    root = _cell(fam, fold, suffix) / "metrics"
    sub = os.environ.get("E5_METRICS_SUBDIR", "").strip()
    if sub:
        root = root / sub
    p = root / "test.json"
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


def motionpro_series(variants: list[str], metric: str):
    """MotionPRO fourfold summaries: per-variant mean/std across folds."""
    key = BASELINE_KEYS.get(metric)
    if key is None:
        return None, None
    root = REPO / "results" / "baselines" / "MotionPRO" / "predictions"
    means, stds = [], []
    for variant in variants:
        p = root / variant / "metrics" / "fourfold_summary.json"
        if not p.is_file():
            means.append(float("nan")); stds.append(float("nan")); continue
        entry = json.loads(p.read_text(encoding="utf-8")).get(key)
        if not entry or entry.get("n_folds", 0) < 4:
            means.append(float("nan")); stds.append(float("nan")); continue
        means.append(float(entry["mean"])); stds.append(float(entry["std"]))
    return np.asarray(means), np.asarray(stds)


def baseline_series(model: str, variants: list[str], metric: str):
    key = BASELINE_KEYS.get(metric)
    if key is None:
        return None
    if model == "MotionPRO":
        return motionpro_series(variants, metric)[0]
    root = REPO / "results" / "baselines" / model / "predictions"
    means = []
    for variant in variants:
        p = root / variant / "eval_motion" / "metrics" / "test_comparison.json"
        if not p.is_file():
            means.append(float("nan")); continue
        summary = json.loads(p.read_text(encoding="utf-8")).get("summary", {})
        value = summary.get(key)
        if value is None:
            extra = p.parent / "extra_metrics.json"
            if extra.is_file():
                value = json.loads(extra.read_text(encoding="utf-8")).get(key)
        means.append(float(value) if value is not None else float("nan"))
    arr = np.asarray(means)
    if model == "WHAM":
        arr = arr + WHAM_OFFSETS.get(metric, 0.0)
    return arr


CONDITIONS = {
    "occ": {
        "dir": "E5_OcclusionRobustness",
        "title": "2D occlusion",
        "xlabel": "Occlusion (%)",
        "x": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "levels": ["", "_evalocc010", "_evalocc020", "_evalocc030",
                   "_evalocc040", "_evalocc050", "_evalocc060",
                   "_evalocc070", "_evalocc080", "_evalocc090"],
        "baseline": ["e5_clean", "e5_occ_lower010", "e5_occ_lower020",
                 "e5_occ_lower030", "e5_occ_lower040", "e5_occ_lower050",
                 "e5_occ_lower060", "e5_occ_lower070", "e5_occ_lower080",
                 "e5_occ_lower090"],
    },
    "gap": {
        "dir": "E5_TimeGap",
        "title": "Temporal gap",
        "xlabel": "Dropout rho (%)",
        "x": [0, 10, 20, 40, 60],
        "levels": ["", "_evalgap010", "_evalgap020", "_evalgap040",
                   "_evalgap060"],
        "baseline": ["e5_clean", "e5_gap_r010", "e5_gap_r020", "e5_gap_r040",
                 "e5_gap_r060"],
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
BIFOOT_DIR = (REPO / "results" / "diagnostics" / "20261111_variant_sweep"
              / "clean_e2_t_bifoot" / "V8_Cv5")
E5_SCENE_DIAG = REPO / "results" / "diagnostics" / "20261107_e5_sceneocc_full"
BIFOOT_KEYS = {"GlobalMPJPE": "GlobalMPJPE", "FootSliding": "FootSliding",
               "GroundPen": "GroundPen", "traj_ATE": "traj_ATE_3d_rms"}


def _bifoot_t_only(condition: str, metric: str):
    """T-only reference now uses the shared BiFoot branch, not the legacy root.

    Pose metrics (PA / ankle-foot) stay on the legacy T2M read; the trajectory/
    physical metrics come from the T+BiFoot evaluation (GT-start direct cumsum;
    traj_ATE keeps RootAbs 3D-RMS semantics).
    """
    key = BIFOOT_KEYS.get(metric)
    if key is None:
        return None
    clean, scene = [], []
    for fold in FOLDS:
        rows = json.loads((BIFOOT_DIR / ("clean_e2_%s.json" % fold))
                          .read_text(encoding="utf-8"))
        w = [r["n_valid"] for r in rows]
        clean.append(float(np.average([r[key] for r in rows], weights=w)))
        sd = json.loads((E5_SCENE_DIAG / ("%s_scene" % fold) / "metrics"
                         / "session_detail_test.json").read_text(encoding="utf-8"))
        sw = {r["session_id"]: r["n_frames"]
              for r in sd["conditions"]["T2M"]}
        srows = [r for r in rows if r["session"] in sw]
        sws = [sw[r["session"]] for r in srows]
        scene.append(float(np.average([r[key] for r in srows], weights=sws)))
    cmean, cstd = mean_std(clean)
    smean, sstd = mean_std(scene)
    if condition == "scene":
        # T-only ignores the visual cache: the scene "clean" point must use
        # the same scene session subset as the scene point, so the reference
        # line is flat (only pose metrics are level-independent; root metrics
        # are measured on the same frames).
        return np.asarray([smean, smean]), np.asarray([sstd, sstd])
    n = len(CONDITIONS[condition]["levels"])
    return np.full(n, cmean), np.full(n, cstd)


def _values_for(condition: str, metric: str) -> dict:
    cfg = CONDITIONS[condition]
    key = KEY_BY_METRIC[metric]
    out = {}
    for fam in MODELS:
        if fam in MAIN_EXCLUDE:
            continue
        means, stds = series(fam, "VT2M", key, cfg["levels"])
        out[fam] = (means, stds)
        tmeans, tstds = series(fam, "T2M", key, cfg["levels"])
        if fam == "V8_Cv5":
            out["V8_Cv5_V"] = series("V8_Cv5", "V2M", key, cfg["levels"])
            override = _bifoot_t_only(condition, metric)
            if override is not None:
                tmeans, tstds = override
            out["T-only"] = (tmeans, tstds)
    for baseline in BASELINE_MODELS:
        if baseline in MAIN_EXCLUDE:
            continue
        if baseline == "MotionPRO":
            vals, errs = motionpro_series(cfg["baseline"], metric)
        else:
            vals = baseline_series(baseline, cfg["baseline"], metric)
            errs = np.zeros_like(vals) if vals is not None else None
        if vals is None:
            continue
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            continue
        if condition == "scene" and not finite.all():
            continue
        out[baseline] = (vals, errs)
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
    for name in PLOT_ORDER:
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
    for name in PLOT_ORDER:
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
        for name in PLOT_ORDER:
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
        figure_metrics(condition, False,
                       out / "Main_Fig6_Gate_Metrics_Absolute.png")
        for metric, label, _ in METRICS:
            figure_metric(condition, metric, label, False,
                          out / "metrics" / ("%s_absolute.png" % metric))
        if os.environ.get("E5_RELATIVE_ARCHIVE", "") == "1":
            rel = out / "detail" / "relative_archive"
            figure_pa(condition, True, rel / "Main_Fig5_Gate_Relative.png")
            figure_metrics(condition, True,
                           rel / "Main_Fig7_Gate_Metrics_Relative.png")
            for metric, label, _ in METRICS:
                figure_metric(condition, metric, label, True,
                              rel / "metrics" / ("%s_relative.png" % metric))
        write_detail(condition)
    print("[done] %s" % BASE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
