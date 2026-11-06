#!/usr/bin/env python3
"""E6 四折 LSO 接触相位增益汇总（Etest_crossfit 口径，2026-11-05）。

读取 anysole_tta/e6_phase_analysis.py 逐折产物：
  results_display/Etest_crossfit/E6_ContactPhaseAnalysis/<metric>/fold_<fold>/
      detail/e6_table.json
四折按 动作 × 模型 × 脚 × 相位 聚合为 mean±std。

产物：
  Full_Table1_E6_Crossfit_PhaseGain_<metric>.csv   长表（逐折均值 + 四折 mean±std）
  Main_Table1_E6_Crossfit_PhaseGain_<metric>.csv   主表：G 的 mean±std（左右脚合并）
  Main_Fig1_E6_Crossfit_PhaseGain_<metric>.png     3×3 动作面板，x=相位，y=G
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results_display" / "Etest_crossfit" / "E6_ContactPhaseAnalysis"
OUT = BASE
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")
FAMILIES = ("V7_Cv5", "V7_Cv5SD2", "V7_Cv5SD2RT", "V7_Cv5_tphys", "V7_Cv7")
PHASES = ("0", "1", "2", "3")
PHASE_EN = ("Stance", "Swing", "Strike", "Off")
ACTION_ORDER = ("walk", "run", "squat", "stand_shift", "single_stand",
                "jump_jack", "lateral", "sit_stand", "stairs")
COLORS = {"V7_Cv5": "#2a78d6", "V7_Cv5SD2": "#eb6834",
          "V7_Cv5SD2RT": "#1baf7a", "V7_Cv5_tphys": "#eda100",
          "V7_Cv7": "#52514e"}
LABELS = {"V7_Cv5": "Cv5 (dual gate)", "V7_Cv5SD2": "Cv5SD2 (dual gate)",
          "V7_Cv5SD2RT": "Cv5SD2RT (dual gate)",
          "V7_Cv5_tphys": "Cv5_tphys (dual gate)",
          "V7_Cv7": "Cv7 (no gate, ref)"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
S13OUT = False
S13OUT_FOLD = "fS6S7S13"
S13OUT_DIR = "fold_fS6S7_s13out"


def family_of(model: str, fold: str) -> str:
    model = model.replace("_s13out", "")
    return model.replace("_%s_s1" % fold, "")


def load_fold(metric: str, fold: str) -> dict | None:
    if S13OUT and fold == S13OUT_FOLD:
        p = BASE / metric / S13OUT_DIR / "detail" / "e6_table.json"
    else:
        p = BASE / metric / ("fold_" + fold) / "detail" / "e6_table.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(v):
    return "" if v is None or not np.isfinite(float(v)) else round(float(v), 3)


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


def collect(metric: str) -> dict:
    """→ by_action[action][family][foot][phase] = list[(fold, V, VT, G)]"""
    out: dict = {}
    for fold in FOLDS:
        table = load_fold(metric, fold)
        if table is None:
            print("[missing] %s fold %s" % (metric, fold))
            continue
        for action, payload in table.items():
            models = payload["table"]
            for model, feet in models.items():
                fam = family_of(model, fold)
                if fam not in FAMILIES:
                    continue
                for foot, phases in feet.items():
                    for ph in PHASES:
                        entry = phases.get(ph)
                        if entry is None:
                            continue
                        out.setdefault(action, {}).setdefault(fam, {}) \
                           .setdefault(foot, {}).setdefault(ph, []) \
                           .append((fold, entry["V"], entry["VT"], entry["G"]))
    return out


def export_tables(metric: str, data: dict) -> None:
    long_rows, main_rows = [], []
    for action in ACTION_ORDER:
        if action not in data:
            continue
        for fam in FAMILIES:
            if fam not in data[action]:
                continue
            for foot in ("left", "right"):
                for ph in PHASES:
                    vals = data[action].get(fam, {}).get(foot, {}).get(ph, [])
                    if not vals:
                        continue
                    V = mean_std([v[1] for v in vals])
                    VT = mean_std([v[2] for v in vals])
                    G = mean_std([v[3] for v in vals])
                    long_rows.append([action, fam, foot, PHASE_EN[int(ph)],
                                      V, VT, G, len(vals)])
            # 左右脚合并（同相位等权）
            for ph in PHASES:
                gvals = []
                for foot in ("left", "right"):
                    gvals.extend([v[3] for v in data[action].get(fam, {})
                                  .get(foot, {}).get(ph, [])])
                main_rows.append([action, fam, PHASE_EN[int(ph)],
                                  mean_std(gvals),
                                  len([x for x in gvals if x is not None])])
    write_csv(OUT / ("Full_Table1_E6_Crossfit_PhaseGain_%s.csv" % metric),
              ["动作", "模型", "脚", "相位", "V_mean±std(mm)", "VT_mean±std(mm)",
               "G_mean±std(mm)", "n_folds"], long_rows)
    write_csv(OUT / ("Main_Table1_E6_Crossfit_PhaseGain_%s.csv" % metric),
              ["动作", "模型", "相位", "G_mean±std(mm)", "n"], main_rows)


def figure(metric: str, data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actions = [a for a in ACTION_ORDER if a in data]
    n = len(actions)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow),
                             facecolor=SURFACE, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, action in zip(axes, actions):
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for fam in FAMILIES:
            ys = []
            for ph in PHASES:
                g = []
                for foot in ("left", "right"):
                    g.extend([v[3] for v in data[action].get(fam, {})
                              .get(foot, {}).get(ph, [])])
                ys.append(float(np.mean(g)) if g else np.nan)
            ax.plot(range(4), ys, lw=1.9, marker="o", ms=4.5,
                    color=COLORS[fam], ls="--" if fam == "V7_Cv7" else "-",
                    label=LABELS[fam])
        ax.axhline(0.0, color="#9a9890", lw=0.8, ls=":", zorder=1)
        ax.set_title(action, fontsize=10.5, color=INK, loc="left")
        ax.set_xticks(range(4))
        ax.set_xticklabels(PHASE_EN, fontsize=8.5)
        ax.set_ylabel("G = VT − V (mm)", fontsize=9, color=MUTED)
    for ax in axes[n:]:
        ax.set_axis_off()
    handles = [plt.Line2D([0], [0], color=COLORS[f], lw=2,
                          ls="--" if f == "V7_Cv7" else "-", label=LABELS[f])
               for f in FAMILIES]
    fig.legend(handles=handles, frameon=False, fontsize=8.6, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    suffix = " (S13-out fold fS6S7)" if S13OUT else ""
    fig.suptitle("E6 crossfit: phase-wise tactile effect, G = VT − V "
                 "(%s; negative = tactile helps; 4-fold mean)%s"
                 % (metric, suffix),
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = OUT / ("Main_Fig1_E6_Crossfit_PhaseGain_%s.png" % metric)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def main() -> int:
    global OUT, S13OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default="ankle_foot_mpjpe,pa_mpjpe")
    ap.add_argument("--include-s13", action="store_true",
                    help="改用含 S13 的旧口径（写到 detail/include_s13/）；"
                         "缺省 = woS13 正式口径，写根层")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    S13OUT = not args.include_s13
    if args.out_dir is not None:
        OUT = args.out_dir
    elif args.include_s13:
        OUT = BASE / "detail" / "include_s13"
    for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        data = collect(metric)
        if not data:
            print("[skip] %s: no fold tables" % metric)
            continue
        export_tables(metric, data)
        figure(metric, data)
    print("[done] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
