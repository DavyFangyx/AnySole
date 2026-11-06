#!/usr/bin/env python3
"""E6 四折 LSO 门控 × 遮挡：汇总表 + 主图（Etest_crossfit 口径，2026-11-05）。

输入 = results_display/Etest_crossfit/E6_gate_occlusion/
        fold_gate_occlusion_long.csv（e6_fold_gate_occlusion.py 产物）
输出 = results_display/Etest_crossfit/E6_ContactPhaseAnalysis/gate_occlusion/
        Full_Table1_E6_Crossfit_GateOcclusion_MeanStd.csv
        Main_Fig1_E6_Crossfit_GateOcclusion.png
        Full_Fig1_E6_Crossfit_GateOcclusion_PerFold.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results_display" / "Etest_crossfit" / "E6_gate_occlusion"
OUT = (REPO / "results_display" / "Etest_crossfit" /
       "E6_ContactPhaseAnalysis" / "gate_occlusion")
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")
FAMILIES = ("V7_Cv5", "V7_Cv5SD2", "V7_Cv5SD2RT", "V7_Cv5_tphys")
LEVELS = ("020", "040", "060", "080", "100")
X = [int(x) for x in LEVELS]
COLORS = {"V7_Cv5": "#2a78d6", "V7_Cv5SD2": "#eb6834",
          "V7_Cv5SD2RT": "#1baf7a", "V7_Cv5_tphys": "#eda100",
          "V7_Cv7": "#52514e"}
LABELS = {"V7_Cv5": "Cv5 (dual gate)", "V7_Cv5SD2": "Cv5SD2 (dual gate)",
          "V7_Cv5SD2RT": "Cv5SD2RT (dual gate)",
          "V7_Cv5_tphys": "Cv5_tphys (dual gate)",
          "V7_Cv7": "Cv7 (no gate, ref)"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
METRIC = "PA-MPJPE"


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(v):
    return None if v in (None, "") else float(v)


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
    rows = _read(SRC / "fold_gate_occlusion_long.csv")
    # by[(family, level)][fold] = row（只取主指标，避免多指标互相覆盖）
    out: dict = {}
    for r in rows:
        if r["metric"] != METRIC:
            continue
        out.setdefault((r["family"], r["level"]), {})[r["fold"]] = r
    return out


def export_table(data: dict) -> None:
    rows = []
    for fam in FAMILIES:
        for level in LEVELS:
            by_fold = data.get((fam, level), {})
            if not by_fold:
                continue
            ref_net = [ _f(r.get("ref_net")) for r in by_fold.values()]
            cand_net = [_f(r.get("cand_net")) for r in by_fold.values()]
            net_delta = [_f(r.get("net_delta")) for r in by_fold.values()]
            supp = [_f(r.get("suppression_pct")) for r in by_fold.values()]
            pres = [_f(r.get("preservation_pct")) for r in by_fold.values()]
            rows.append([
                fam, "occ" + level, METRIC,
                mean_std(ref_net), mean_std(cand_net), mean_std(net_delta),
                mean_std(supp), mean_std(pres),
                sum(1 for v in net_delta if v is not None),
                sum(1 for v in supp if v is not None),
                sum(1 for v in pres if v is not None),
            ])
    write_csv(OUT / "Full_Table1_E6_Crossfit_GateOcclusion_MeanStd.csv",
              ["模型", "档位", "指标", "ref_net(ΔE_Cv7)", "cand_net(ΔE)",
               "net_delta", "suppression(%)", "preservation(%)",
               "n_folds", "n_supp", "n_pres"], rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def figure_main(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), facecolor=SURFACE)
    # (a) ΔE 净值
    ax = axes[0]
    _style(ax)
    ref_means = []
    for level in LEVELS:
        vals = [_f(r.get("ref_net")) for r in data.get(
            (FAMILIES[0], level), {}).values()]
        ref_means.append(float(np.mean([v for v in vals if v is not None]))
                          if any(v is not None for v in vals) else np.nan)
    ax.plot(X, ref_means, lw=2.2, marker="s", ms=5.5, color=COLORS["V7_Cv7"],
            ls="--", label=LABELS["V7_Cv7"])
    for fam in FAMILIES:
        means = []
        for level in LEVELS:
            vals = [_f(r.get("cand_net")) for r in data.get((fam, level), {}).values()]
            means.append(float(np.mean([v for v in vals if v is not None]))
                         if any(v is not None for v in vals) else np.nan)
        ax.plot(X, means, lw=2.0, marker="o", ms=4.5, color=COLORS[fam],
                label=LABELS[fam])
    ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":", zorder=1)
    ax.set_title("Tactile effect ΔE = VT − V (lower = helps)", fontsize=10.5,
                 color=INK, loc="left")
    ax.set_xlabel("Occlusion (%)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("PA-MPJPE ΔE (mm)", fontsize=9.5, color=MUTED)
    ax.set_xticks(X)
    ax.legend(frameon=False, fontsize=7.6, loc="best")

    # (b) 有害分量绝对值 ΔE⁺（比值在高遮挡分母趋零，只入表）
    ax = axes[1]
    _style(ax)
    ref_plus = []
    for level in LEVELS:
        vals = [_f(r.get("ref_plus")) for r in data.get(
            (FAMILIES[0], level), {}).values()]
        vals = [v for v in vals if v is not None]
        ref_plus.append(float(np.mean(vals)) if vals else np.nan)
    ax.plot(X, ref_plus, lw=2.2, marker="s", ms=5.5, color=COLORS["V7_Cv7"],
            ls="--", label=LABELS["V7_Cv7"])
    for fam in FAMILIES:
        ys = []
        for level in LEVELS:
            vals = [_f(r.get("cand_plus"))
                    for r in data.get((fam, level), {}).values()]
            vals = [v for v in vals if v is not None]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(X, ys, lw=2.0, marker="o", ms=4.5, color=COLORS[fam],
                label=LABELS[fam])
    ax.set_title("Harmful component ΔE⁺ (lower = better)", fontsize=10.5,
                 color=INK, loc="left")
    ax.set_xlabel("Occlusion (%)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("ΔE⁺ (mm)", fontsize=9.5, color=MUTED)
    ax.set_xticks(X)
    ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":")

    # (c) 有益分量绝对值 ΔE⁻
    ax = axes[2]
    _style(ax)
    ref_minus = []
    for level in LEVELS:
        vals = [_f(r.get("ref_minus")) for r in data.get(
            (FAMILIES[0], level), {}).values()]
        vals = [v for v in vals if v is not None]
        ref_minus.append(float(np.mean(vals)) if vals else np.nan)
    ax.plot(X, ref_minus, lw=2.2, marker="s", ms=5.5, color=COLORS["V7_Cv7"],
            ls="--", label=LABELS["V7_Cv7"])
    for fam in FAMILIES:
        ys = []
        for level in LEVELS:
            vals = [_f(r.get("cand_minus"))
                    for r in data.get((fam, level), {}).values()]
            vals = [v for v in vals if v is not None]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(X, ys, lw=2.0, marker="o", ms=4.5, color=COLORS[fam],
                label=LABELS[fam])
    ax.set_title("Helpful component ΔE⁻ (more negative = better)",
                 fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("Occlusion (%)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("ΔE⁻ (mm)", fontsize=9.5, color=MUTED)
    ax.set_xticks(X)
    ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":")

    fig.suptitle("E6 crossfit: gate × occlusion, PA-MPJPE, 4-fold mean "
                 "(absolute ΔE components; ratios in table)", fontsize=11.5,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "Main_Fig1_E6_Crossfit_GateOcclusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def figure_per_fold() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read(SRC / "fold_gate_occlusion_long.csv")
    by = {}
    for r in rows:
        if r["metric"] != METRIC:
            continue
        by[(r["fold"], r["family"], r["level"])] = r
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.6), facecolor=SURFACE,
                             sharey=True)
    for ax, fold in zip(axes, FOLDS):
        _style(ax)
        ref = [_f(by.get((fold, FAMILIES[0], lv), {}).get("ref_net"))
               for lv in LEVELS]
        ax.plot(X, ref, lw=2.2, marker="s", ms=5, color=COLORS["V7_Cv7"],
                ls="--", label=LABELS["V7_Cv7"])
        for fam in FAMILIES:
            ys = [_f(by.get((fold, fam, lv), {}).get("cand_net"))
                  for lv in LEVELS]
            ax.plot(X, ys, lw=1.8, marker="o", ms=4, color=COLORS[fam])
        ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":", zorder=1)
        ax.set_title(fold, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Occlusion (%)", fontsize=9, color=MUTED)
        ax.set_xticks(X)
    axes[0].set_ylabel("PA-MPJPE ΔE (mm)", fontsize=9, color=MUTED)
    handles = [plt.Line2D([0], [0], color=COLORS[f], lw=2,
                          ls="--" if f == "V7_Cv7" else "-", label=LABELS[f])
               for f in ("V7_Cv7",) + FAMILIES]
    fig.legend(handles=handles, frameon=False, fontsize=8.4, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("E6 crossfit: gate × occlusion per fold (PA-MPJPE)",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = OUT / "Full_Fig1_E6_Crossfit_GateOcclusion_PerFold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def main() -> int:
    global SRC, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    SRC, OUT = args.src, args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    data = collect()
    export_table(data)
    figure_main(data)
    figure_per_fold()
    print("[done] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
