#!/usr/bin/env python3
"""E6 四折 LSO 门控两张主表（occ000，Etest_crossfit 口径，2026-11-05）。

与 E6 门控主表同口径 ΔE = ΔE⁺ + ΔE⁻（逐会话 n_frames 加权）：
  Δ_s = VT_s − V_s（正 = 触觉引入退化）
  ΔE⁺ = Σ w max(Δ,0)/Σw（有害分量）；ΔE⁻ = Σ w min(Δ,0)/Σw（有益分量）
  抑制率 = 1 − ΔE⁺_cand/ΔE⁺_ref；保留率 = ΔE⁻_cand/ΔE⁻_ref
参照 = Cv7（无门）；候选 = Cv5 / Cv5SD2 / Cv5SD2RT / Cv5_tphys。
数据源 = results/cells/<fam>_<fold>_s1/metrics/session_detail_test.json。

注：四折 test 会话不在 canonical 的 session_foot_contact_counts.json 中，
FootSliding 权重退化为 n_frames（与 E6-C 折脚本同一处理）。

产物：results_display/Etest_crossfit/E6_ContactPhaseAnalysis/gate_tables/
  Full_Table1_E6_Crossfit_GateHarmSuppression.csv
  Full_Table2_E6_Crossfit_GateGainPreservation.csv
  detail/PerFold_E6_Crossfit_GateDecomposition.csv
  Main_Fig1_E6_Crossfit_GateEffect.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
OUT = (REPO / "results_display" / "Etest_crossfit" /
       "E6_ContactPhaseAnalysis" / "gate_tables")
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")
REFERENCE = "V7_Cv7"
CANDIDATES = ("V7_Cv5", "V7_Cv5SD2", "V7_Cv5SD2RT", "V7_Cv5_tphys")
ALL_MODELS = CANDIDATES + (REFERENCE,)
METRICS = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "W-MPJPE100": "w_mpjpe100_mm",
    "MPJRE": "mpjae_deg",
    "RootOrientationDrift": "root_orientation_drift_deg",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}
KEY_METRICS = tuple(METRICS)
MIN_EFFECT = 0.05
COLORS = {"V7_Cv5": "#2a78d6", "V7_Cv5SD2": "#eb6834",
          "V7_Cv5SD2RT": "#1baf7a", "V7_Cv5_tphys": "#eda100",
          "V7_Cv7": "#52514e"}
LABELS = {"V7_Cv5": "Cv5", "V7_Cv5SD2": "Cv5SD2",
          "V7_Cv5SD2RT": "Cv5SD2RT", "V7_Cv5_tphys": "Cv5_tphys",
          "V7_Cv7": "Cv7 (no gate)"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
S13OUT = False
S13OUT_FOLD = "fS6S7S13"


def cell_id(fam: str, fold: str) -> str:
    base = "%s_%s_s1" % (fam, fold)
    if S13OUT and fold == S13OUT_FOLD:
        return base + "_s13out"
    return base


def decompose(cell: str) -> dict | None:
    p = CELLS / cell / "metrics" / "session_detail_test.json"
    if not p.is_file():
        return None
    detail = json.loads(p.read_text(encoding="utf-8"))["conditions"]
    vt = {e["session_id"]: e["metrics"] for e in detail["VT2M"]}
    v = {e["session_id"]: e["metrics"] for e in detail["V2M"]}
    n = {e["session_id"]: e["n_frames"] for e in detail["V2M"]}
    sids = sorted(set(v) & set(vt))
    if not sids:
        return None
    out = {}
    for name, key in METRICS.items():
        pairs = [(vt[s].get(key), v[s].get(key)) for s in sids]
        if any(a is None or b is None for a, b in pairs):
            continue
        delta = np.array([a - b for a, b in pairs], dtype=float)
        w = np.array([float(n[s]) for s in sids])
        total = w.sum()
        out[name] = {
            "net": float((w * delta).sum() / total),
            "plus": float((w * np.maximum(delta, 0.0)).sum() / total),
            "minus": float((w * np.minimum(delta, 0.0)).sum() / total),
        }
    return out


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
    """→ per_fold[(fam, fold)][metric] = components。"""
    out = {}
    missing = []
    for fold in FOLDS:
        for fam in ALL_MODELS:
            d = decompose(cell_id(fam, fold))
            if d is None:
                missing.append((fam, fold))
                continue
            out[(fam, fold)] = d
    if missing:
        print("[missing]", missing)
    return out


def export_tables(data: dict) -> dict:
    per_fold_rows = []
    for (fam, fold), d in data.items():
        for metric in KEY_METRICS:
            if metric not in d:
                continue
            x = d[metric]
            per_fold_rows.append([fam, fold, metric, round(x["net"], 3),
                                  round(x["plus"], 3), round(x["minus"], 3)])
    write_csv(OUT / "detail" / "PerFold_E6_Crossfit_GateDecomposition.csv",
              ["模型", "折", "指标", "ΔE_net", "ΔE_plus", "ΔE_minus"],
              per_fold_rows)

    # 抑制 / 保留（四折 mean±std；百分比仅在分母 >= MIN_EFFECT 时定义）
    h1 = ["模型", "指标", "ΔE⁺_mean±std(mm)", "抑制率_mean±std(%)",
          "n_folds"]
    h2 = ["模型", "指标", "ΔE⁻_mean±std(mm)", "保留率_mean±std(%)",
          "n_folds"]
    rows1, rows2 = [], []
    for fam in ALL_MODELS:
        for metric in KEY_METRICS:
            plus_vals, supp_vals, minus_vals, pres_vals = [], [], [], []
            for fold in FOLDS:
                ref = data.get((REFERENCE, fold), {}).get(metric)
                cand = data.get((fam, fold), {}).get(metric)
                if not ref or not cand:
                    continue
                plus_vals.append(cand["plus"])
                minus_vals.append(cand["minus"])
                if fam != REFERENCE:
                    if abs(ref["plus"]) >= MIN_EFFECT:
                        supp_vals.append(100.0 * (1.0 - cand["plus"] / ref["plus"]))
                    if abs(ref["minus"]) >= MIN_EFFECT:
                        pres_vals.append(100.0 * cand["minus"] / ref["minus"])
            rows1.append([fam, metric, mean_std(plus_vals),
                          "" if fam == REFERENCE else mean_std(supp_vals),
                          len(plus_vals)])
            rows2.append([fam, metric, mean_std(minus_vals),
                          "" if fam == REFERENCE else mean_std(pres_vals),
                          len(minus_vals)])
    write_csv(OUT / "Full_Table1_E6_Crossfit_GateHarmSuppression.csv", h1, rows1)
    write_csv(OUT / "Full_Table2_E6_Crossfit_GateGainPreservation.csv", h2, rows2)
    return {"plus": rows1, "minus": rows2}


def figure(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = "PA-MPJPE"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), facecolor=SURFACE)
    for ax, key, title, ylab in (
            (axes[0], "plus", "Harmful component ΔE⁺ (lower = better)",
             "ΔE⁺ (mm)"),
            (axes[1], "minus", "Helpful component ΔE⁻ (more negative = better)",
             "ΔE⁻ (mm)")):
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.25, lw=0.6, axis="y")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        labels, means, stds = [], [], []
        for fam in ALL_MODELS:
            vals = []
            for fold in FOLDS:
                d = data.get((fam, fold), {}).get(metric)
                if d:
                    vals.append(d[key])
            labels.append(LABELS[fam])
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals)) if vals else 0.0)
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, width=0.62, capsize=3,
               color=[COLORS[f] for f in ALL_MODELS],
               edgecolor=SURFACE, linewidth=1.5, error_kw={"ecolor": MUTED,
                                                           "lw": 0.9})
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.6)
        ax.axhline(0.0, color="#9a9890", lw=0.9)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_ylabel(ylab, fontsize=9.5, color=MUTED)
    fig.suptitle("E6 crossfit gate effect at occ000 (PA-MPJPE; 4-fold "
                 "mean ± std across folds)", fontsize=11.5, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "Main_Fig1_E6_Crossfit_GateEffect.png"
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
    figure(data)
    print("[done] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
