#!/usr/bin/env python3
"""E5 四折 LSO 遮挡鲁棒性导出（Etest_crossfit 口径，2026-11-05）。

数据源 = results/cells/：
  occ000 = <fam>_<fold>_s1/metrics/test.json
  occNNN = <fam>_<fold>_s1_evaloccNNN/metrics/test.json
模型 = Cv5 / Cv5SD2 / Cv5SD2RT / Cv5_tphys + Cv7（无门参照）
折 = fS5S6S11 / fS5S11S12 / fS6S7S13 / fS5S6S8（test = 留出受试者）
档 = occ000 + 020/040/060/080/100

产物（results_display/Etest_crossfit/E5_OcclusionRobustness/）：
  Full_Table1_E5_Crossfit_Occlusion.csv    长表：模型×折×档×行×10 指标
  Full_Table2_E5_Crossfit_MeanStd.csv      长表：四折均值±std（指标为行）
  Main_Table1_E5_Crossfit_TactileGain.csv  Δ(VT−V)：模型×档，列 = 7 指标 mean±std
  Main_Table2_E5_Crossfit_Degradation.csv  Δ退化(VT−V)：模型×档，列 = 7 指标 mean±std
  Main_Table3_E5_Crossfit_Absolute_PA.csv  PA-MPJPE 绝对值 V/VT mean±std
  Main_Fig1_E5_Crossfit_TactileGain.png    Δ(VT−V) 曲线（PA / MPJPE）
  Main_Fig2_E5_Crossfit_Degradation.png    Δ退化 2×3 子图
  Main_Fig3_E5_Crossfit_Absolute.png       PA 绝对值 V/VT
  Full_Fig1_E5_Crossfit_PerFold.png        逐折 PA Δ 曲线
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
OUT = REPO / "results_display" / "Etest_crossfit" / "E5_OcclusionRobustness"

FAMILIES = ("V7_Cv5", "V7_Cv5SD2", "V7_Cv5SD2RT", "V7_Cv5_tphys")
REFERENCE = "V7_Cv7"
ALL_MODELS = FAMILIES + (REFERENCE,)
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")
LEVELS = ("000", "020", "040", "060", "080", "100")
X = [int(x) for x in LEVELS]

METRICS10 = ["PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "MPJRE",
             "traj_ATE", "RootOrientationDrift", "FootSliding", "GroundPen",
             "Jitter_pred"]
MAIN_METRICS = ["PA-MPJPE", "MPJPE", "GlobalMPJPE", "traj_ATE",
                "FootSliding", "GroundPen", "PA-MPJPE_anklefoot"]

# dataviz reference palette（categorical slots 1–4 + neutral reference）
COLORS = {"V7_Cv5": "#2a78d6", "V7_Cv5SD2": "#eb6834",
          "V7_Cv5SD2RT": "#1baf7a", "V7_Cv5_tphys": "#eda100",
          "V7_Cv7": "#52514e"}
LABELS = {"V7_Cv5": "Cv5 (dual gate)", "V7_Cv5SD2": "Cv5SD2 (dual gate)",
          "V7_Cv5SD2RT": "Cv5SD2RT (dual gate)",
          "V7_Cv5_tphys": "Cv5_tphys (dual gate)",
          "V7_Cv7": "Cv7 (no gate, ref)"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
S13OUT = False
S13OUT_FOLD = "fS6S7S13"


def fold_label(fold: str) -> str:
    return "fS6S7" if (S13OUT and fold == S13OUT_FOLD) else fold


def _strip_anchor(metrics: dict) -> dict:
    out = {}
    for cond, block in metrics.items():
        if not isinstance(block, dict):
            out[cond] = block
            continue
        default = {str(k): v for k, v in block.items()
                   if not str(k).endswith(("（GT Anchored）", " (anchored)"))}
        anchored = {}
        for k, v in block.items():
            name = str(k)
            for suffix in ("（GT Anchored）", " (anchored)"):
                if name.endswith(suffix):
                    anchored[name[:-len(suffix)]] = v
                    break
        out[cond] = {**anchored, **default}
    return out


def cell_id(fam: str, fold: str, level: str) -> str:
    base = "%s_%s_s1" % (fam, fold)
    if S13OUT and fold == S13OUT_FOLD:
        base += "_s13out"
    return base if level == "000" else "%s_evalocc%s" % (base, level)


def load_metrics(fam: str, fold: str, level: str) -> dict | None:
    p = CELLS / cell_id(fam, fold, level) / "metrics" / "test.json"
    if not p.is_file():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    return _strip_anchor(payload.get("metrics") or {})


def fmt(v):
    if v is None:
        return ""
    f = float(v)
    return "" if not np.isfinite(f) else round(f, 3)


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) == 1:
        return round(vals[0], 3)
    return "%.3f±%.3f" % (float(np.mean(vals)), float(np.std(vals)))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def collect() -> dict:
    """→ data[model][fold][level] = {cond: {metric: value}}。"""
    data = {}
    missing = []
    for fam in ALL_MODELS:
        data[fam] = {}
        for fold in FOLDS:
            data[fam][fold] = {}
            for level in LEVELS:
                m = load_metrics(fam, fold, level)
                if m is None:
                    missing.append(cell_id(fam, fold, level))
                    continue
                data[fam][fold][level] = m
    if missing:
        print("[missing] %d cells" % len(missing))
        for x in missing[:10]:
            print("  -", x)
    return data


def delta(data: dict, fam: str, fold: str, level: str, metric: str):
    m = data[fam][fold].get(level)
    if not m:
        return None
    v, vt = m.get("V2M", {}).get(metric), m.get("VT2M", {}).get(metric)
    if v is None or vt is None:
        return None
    return float(vt) - float(v)


def degradation(data: dict, fam: str, fold: str, level: str, metric: str,
                cond: str):
    cur = data[fam][fold].get(level)
    base = data[fam][fold].get("000")
    if not cur or not base:
        return None
    a, b = cur.get(cond, {}).get(metric), base.get(cond, {}).get(metric)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def export_tables(data: dict) -> None:
    # 1) 长表 模型×折×档×行
    header = ["模型", "折", "档位", "行"] + METRICS10
    rows = []
    for fam in ALL_MODELS:
        for fold in FOLDS:
            for level in LEVELS:
                m = data[fam][fold].get(level)
                if not m:
                    continue
                for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
                    rows.append([fam, fold_label(fold), "occ" + level, row_key]
                                + [fmt(m.get(cond, {}).get(k))
                                   for k in METRICS10])
                rows.append([fam, fold_label(fold), "occ" + level, "Δ(VT−V)"]
                            + [fmt(delta(data, fam, fold, level, k))
                               for k in METRICS10])
    write_csv(OUT / "Full_Table1_E5_Crossfit_Occlusion.csv", header, rows)

    # 2) 四折 mean±std 长表（指标为行）
    rows2 = []
    for fam in ALL_MODELS:
        for level in LEVELS:
            for metric in METRICS10:
                for row_key in ("V2M", "VT2M", "Δ(VT−V)"):
                    vals = []
                    for fold in FOLDS:
                        m = data[fam][fold].get(level)
                        if not m:
                            continue
                        if row_key == "Δ(VT−V)":
                            vals.append(delta(data, fam, fold, level, metric))
                        else:
                            vals.append(m.get(row_key, {}).get(metric))
                    rows2.append([fam, "occ" + level, row_key, metric,
                                  mean_std(vals), len([v for v in vals
                                                       if v is not None])])
    write_csv(OUT / "Full_Table2_E5_Crossfit_MeanStd.csv",
              ["模型", "档位", "行", "指标", "mean±std", "n_folds"], rows2)

    # 3) 主表：Δ(VT−V) / Δ退化（列 = 7 指标）
    rows3, rows4, rows5 = [], [], []
    for fam in ALL_MODELS:
        for level in LEVELS:
            cells3, cells4, cells5 = [], [], []
            for metric in MAIN_METRICS:
                vals = [delta(data, fam, fold, level, metric) for fold in FOLDS]
                cells3.append(mean_std(vals))
                degs = []
                for fold in FOLDS:
                    dv = degradation(data, fam, fold, level, metric, "VT2M")
                    dvv = degradation(data, fam, fold, level, metric, "V2M")
                    if dv is not None and dvv is not None:
                        degs.append(dv - dvv)
                cells4.append(mean_std(degs))
            rows3.append([fam, "occ" + level] + cells3)
            rows4.append([fam, "occ" + level] + cells4)
            # 绝对值 PA V/VT
            for row_key in ("V2M", "VT2M"):
                vals = [data[fam][fold].get(level, {}).get(row_key, {})
                        .get("PA-MPJPE") for fold in FOLDS]
                cells5.append(mean_std(vals))
            rows5.append([fam, "occ" + level] + cells5)
    write_csv(OUT / "Main_Table1_E5_Crossfit_TactileGain.csv",
              ["模型", "档位"] + MAIN_METRICS, rows3)
    write_csv(OUT / "Main_Table2_E5_Crossfit_Degradation.csv",
              ["模型", "档位"] + MAIN_METRICS, rows4)
    write_csv(OUT / "Main_Table3_E5_Crossfit_Absolute_PA.csv",
              ["模型", "档位", "PA-MPJPE(V2M)", "PA-MPJPE(VT2M)"], rows5)


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c8c8")
    ax.tick_params(colors=MUTED, labelsize=9)


def _fig_tactile(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), facecolor=SURFACE)
    for ax, metric in zip(axes, ("PA-MPJPE", "MPJPE")):
        _style(ax)
        for fam in ALL_MODELS:
            mean = []
            for level in LEVELS:
                vals = [delta(data, fam, fold, level, metric) for fold in FOLDS]
                vals = [v for v in vals if v is not None]
                mean.append(float(np.mean(vals)) if vals else np.nan)
                for v in vals:
                    ax.plot(X[LEVELS.index(level)], v, "o",
                            ms=4, color=COLORS[fam], alpha=0.28, zorder=2)
            ax.plot(X, mean, lw=2.0, marker="o", ms=5, color=COLORS[fam],
                    ls="--" if fam == REFERENCE else "-", zorder=3,
                    label=LABELS[fam])
        ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":", zorder=1)
        ax.set_xlabel("Occlusion (%)", fontsize=10, color=MUTED)
        ax.set_ylabel("Δ(VT − V)  (%s, mm)" % metric, fontsize=10, color=MUTED)
        ax.set_title(metric, fontsize=11, color=INK, loc="left")
        ax.set_xticks(X)
    axes[0].legend(frameon=False, fontsize=8.4, loc="best")
    fig.suptitle("E5 crossfit: tactile effect under visual occlusion "
                 "(4-fold LSO test, mean across folds; dots = folds)",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "Main_Fig1_E5_Crossfit_TactileGain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def _fig_degradation(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["PA-MPJPE", "MPJPE", "GlobalMPJPE", "traj_ATE",
               "FootSliding", "GroundPen"]
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6), facecolor=SURFACE)
    for ax, metric in zip(axes.flat, metrics):
        _style(ax)
        for fam in ALL_MODELS:
            means = []
            for level in LEVELS:
                degs = []
                for fold in FOLDS:
                    dv = degradation(data, fam, fold, level, metric, "VT2M")
                    dvv = degradation(data, fam, fold, level, metric, "V2M")
                    if dv is not None and dvv is not None:
                        degs.append(dv - dvv)
                means.append(float(np.mean(degs)) if degs else np.nan)
            ax.plot(X, means, lw=2.0, marker="o", ms=4.5,
                    color=COLORS[fam], ls="--" if fam == REFERENCE else "-",
                    label=LABELS[fam])
        ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":", zorder=1)
        ax.set_title(metric, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Occlusion (%)", fontsize=9, color=MUTED)
        ax.set_ylabel("Δdegradation VT−V (mm)", fontsize=9, color=MUTED)
        ax.set_xticks(X)
    axes.flat[0].legend(frameon=False, fontsize=7.6, loc="best")
    fig.suptitle("E5 crossfit: degradation relative to occ000, VT minus V "
                 "(negative = tactile reduces degradation; 4-fold mean)",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "Main_Fig2_E5_Crossfit_Degradation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def _fig_absolute(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), facecolor=SURFACE)
    for ax, cond, title in zip(axes, ("V2M", "VT2M"),
                               ("U@V (visual only)", "U@VT (visual + tactile)")):
        _style(ax)
        for fam in ALL_MODELS:
            mean = []
            for level in LEVELS:
                vals = [data[fam][fold].get(level, {}).get(cond, {})
                        .get("PA-MPJPE") for fold in FOLDS]
                vals = [float(v) for v in vals if v is not None]
                mean.append(float(np.mean(vals)) if vals else np.nan)
            ax.plot(X, mean, lw=2.0, marker="o", ms=4.5, color=COLORS[fam],
                    ls="--" if fam == REFERENCE else "-", label=LABELS[fam])
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Occlusion (%)", fontsize=10, color=MUTED)
        ax.set_ylabel("PA-MPJPE (mm)", fontsize=10, color=MUTED)
        ax.set_xticks(X)
    axes[0].legend(frameon=False, fontsize=8.4, loc="best")
    fig.suptitle("E5 crossfit: absolute PA-MPJPE (4-fold mean)",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "Main_Fig3_E5_Crossfit_Absolute.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def _fig_per_fold(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.6), facecolor=SURFACE,
                             sharey=True)
    for ax, fold in zip(axes, FOLDS):
        _style(ax)
        for fam in ALL_MODELS:
            y = [delta(data, fam, fold, level, "PA-MPJPE")
                 for level in LEVELS]
            ax.plot(X, y, lw=1.8, marker="o", ms=4, color=COLORS[fam],
                    ls="--" if fam == REFERENCE else "-")
        ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":", zorder=1)
        ax.set_title(fold_label(fold), fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Occlusion (%)", fontsize=9, color=MUTED)
        ax.set_xticks(X)
    axes[0].set_ylabel("Δ(VT − V) PA-MPJPE (mm)", fontsize=9, color=MUTED)
    handles = [plt.Line2D([0], [0], color=COLORS[f], lw=2,
                          ls="--" if f == REFERENCE else "-", label=LABELS[f])
               for f in ALL_MODELS]
    fig.legend(handles=handles, frameon=False, fontsize=8.4, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("E5 crossfit: per-fold PA-MPJPE tactile effect",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = OUT / "Full_Fig1_E5_Crossfit_PerFold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def main() -> int:
    global OUT, S13OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--include-s13", action="store_true",
                    help="改用含 S13 的旧口径（写到 detail/include_s13/）；"
                         "缺省 = woS13 正式口径，写根层")
    args = ap.parse_args()
    S13OUT = not args.include_s13
    OUT = args.out_dir or (OUT / "detail" / "include_s13"
                           if args.include_s13 else OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    data = collect()
    export_tables(data)
    _fig_tactile(data)
    _fig_degradation(data)
    _fig_absolute(data)
    _fig_per_fold(data)
    print("[done] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
