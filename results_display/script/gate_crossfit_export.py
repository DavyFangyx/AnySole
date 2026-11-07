#!/usr/bin/env python3
"""E5 crossfit 口径：V 门 vs 无 V 门 vs 无 T 门（Absolute + 相对增益）。

数据源（V8 主基座 + V8 消融）：
- 无 V 门 = V8_Cv5（结构 = Cv5SD2；T 门开、V 门关）；
- 无 T 门 = V8_Cv7（V8 消融：无门）；
- V 门 = results/diagnostics/20261108_gate_level/report_<fold>.json
  的 per_level.e_pa（全身体 PA-MPJPE，输出空间 y=(1-g)T+gVT）。
  V2M 行与 V8_Cv5 相同（V 门只作用在 VT 行）。
- 对照消融 = V8_Cv4（单门）、V8_Cv2（dropout）、V8_Cv6（始终 VT）。
折：fS5S6S11 / fS5S11S12 / fS6S7 / fS5S6S8，V8 cells 同名折叠。
四折均值；图内点 = 逐折。

产物（results_display/Etest_crossfit/E5_OcclusionRobustness/）：
  Main_Fig4_Gate_Absolute.png    U@V / U@VT 绝对 PA-MPJPE（三组）
  Main_Fig5_Gate_Relative.png    Δ vs 无 V 门 / Δ vs 无门 / Δ(VT−V)
  Main_Table4_Gate_Absolute.csv  模型×档位 V2M/VT2M mean±std
  Main_Table5_Gate_Relative.csv  档位×相对量 mean±std
  Full_Table3_Gate_PerFold.csv   逐折长表

用法：python results_display/script/gate_crossfit_export.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "results_display/script"))

GATE_OUT = REPO / "results/diagnostics/20261108_gate_level"
OUT = REPO / "results_display/Etest_crossfit/E5_OcclusionRobustness"
LEVELS = ("000", "020", "040", "060", "080", "100")
X = [int(x) for x in LEVELS]
GATE_FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
# 主基座 = V8（结构 = Cv5SD2）；消融 = V8_Cv4/Cv2/Cv6/Cv7
GROUPS = ("Gate", "V8_Cv5", "V8_Cv7")
ABLATIONS = ("V8_Cv4", "V8_Cv2", "V8_Cv6")
ALL_GROUPS = GROUPS + ABLATIONS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
COLORS = {"Gate": "#2a78d6", "V8_Cv5": "#eb6834", "V8_Cv7": "#52514e",
          "V8_Cv4": "#1baf7a", "V8_Cv2": "#eda100", "V8_Cv6": "#7a5af8"}
LABELS = {"Gate": "V gate (T gate + V gate)",
          "V8_Cv5": "V8 / Cv5SD2 (T gate, no V gate)",
          "V8_Cv7": "V8 ablation: no gate",
          "V8_Cv4": "V8 ablation: single gate",
          "V8_Cv2": "V8 ablation: dropout",
          "V8_Cv6": "V8 ablation: always VT"}


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) == 1:
        return round(vals[0], 3)
    return "%.3f±%.3f" % (float(np.mean(vals)), float(np.std(vals)))


def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def load_gate(fold: str) -> dict:
    return json.loads((GATE_OUT / ("report_%s.json" % fold)).read_text(encoding="utf-8"))


def load_cell(fam: str, fold: str, level: str) -> dict:
    base = "results/cells/%s_%s_s1" % (fam, fold)
    if level != "000":
        base += "_evalocc%s" % level
    p = REPO / base / "metrics/test.json"
    if not p.is_file():
        raise RuntimeError("missing cell %s" % p)
    return json.loads(p.read_text(encoding="utf-8"))["metrics"]


def collect() -> dict:
    """data[group][fold][level] = {'v': ..., 'vt': ...} (PA-MPJPE, mm)."""
    data = {g: {f: {} for f in GATE_FOLDS} for g in ALL_GROUPS}
    for fold in GATE_FOLDS:
        rep = load_gate(fold)
        for level in LEVELS:
            cells = {g: load_cell(g, fold, level) for g in ALL_GROUPS if g != "Gate"}
            gate = rep["per_level"][level]
            # gate 的 no-V-gate 基座必须与 V8_Cv5 cell 逐位对齐（同一 ckpt/test）
            base_vt = float(cells["V8_Cv5"]["VT2M"]["PA-MPJPE"])
            if abs(float(gate["e_pa_vt"]) - base_vt) > 0.01:
                raise RuntimeError("gate base mismatch %s/%s: %.3f vs %.3f"
                                   % (fold, level, gate["e_pa_vt"], base_vt))
            data["Gate"][fold][level] = {
                "v": float(cells["V8_Cv5"]["V2M"]["PA-MPJPE"]),  # V-only 未受 V 门影响
                "vt": float(gate["e_pa"]),                        # 输出融合后的全身体 PA
            }
            for g in ALL_GROUPS:
                if g == "Gate":
                    continue
                data[g][fold][level] = {
                    "v": float(cells[g]["V2M"]["PA-MPJPE"]),
                    "vt": float(cells[g]["VT2M"]["PA-MPJPE"]),
                }
    return data


def mean_at(data: dict, group: str, level: str, key: str) -> float:
    vals = [data[group][f][level][key] for f in GATE_FOLDS]
    return float(np.mean(vals))


def fold_vals(data: dict, group: str, level: str, key: str):
    return [data[group][f][level][key] for f in GATE_FOLDS]


def relative(data: dict, level: str):
    """Δ vs no-V (Gate−V8_Cv5), Δ vs no-gate (V8_Cv7), tactile Δ(VT−V)."""
    out = {}
    for f in GATE_FOLDS:
        g = data["Gate"][f][level]
        c5 = data["V8_Cv5"][f][level]
        c7 = data["V8_Cv7"][f][level]
        rec = {
            "gate_minus_v8": g["vt"] - c5["vt"],
            "gate_minus_nogate": g["vt"] - c7["vt"],
            "v8_minus_nogate": c5["vt"] - c7["vt"],
            "tactile_gate": g["vt"] - g["v"],
            "tactile_v8": c5["vt"] - c5["v"],
            "tactile_nogate": c7["vt"] - c7["v"],
        }
        for a in ABLATIONS:
            rec["%s_minus_nogate" % a] = data[a][f][level]["vt"] - c7["vt"]
        out[f] = rec
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c8c8")
    ax.tick_params(colors=MUTED, labelsize=9)


def _mean_line(ax, data, group, key, ls="-", color=None, label=None):
    color = color or COLORS[group]
    mean = [mean_at(data, group, l, key) for l in LEVELS]
    for l in LEVELS:
        for v in fold_vals(data, group, l, key):
            ax.plot(int(l), v, "o", ms=3.5, color=color, alpha=0.22, zorder=2)
    ax.plot(X, mean, lw=2.0, marker="o", ms=4.5, color=color, ls=ls,
            label=label, zorder=3)


def _fig_absolute(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 单张 absolute：VT 三组 + V8 消融 + 单 V 参考 + 单 T 参考
    fig, ax = plt.subplots(figsize=(10.8, 5.0), facecolor=SURFACE)
    _style(ax)
    for group in ("V8_Cv5", "V8_Cv7"):
        _mean_line(ax, data, group, "vt",
                   ls="--" if group == "V8_Cv7" else "-", label=LABELS[group])
    _mean_line(ax, data, "Gate", "vt", label=LABELS["Gate"])
    for group in ABLATIONS:
        _mean_line(ax, data, group, "vt", ls=":",
                   color=COLORS[group], label=LABELS[group])

    # 单 V 参考：V8_Cv5 的 V2M 行（V 门不作用；随遮挡退化）
    _mean_line(ax, data, "V8_Cv5", "v", ls="-.",
               color="#8c6d31", label="V only (visual, no tactile)")

    # 单 T 参考：V8_Cv5 的 T2M 行（V 遮挡不影响 T 行，逐档常数）
    t_vals = np.asarray([[load_gate(f)["per_level"][l]["e_pa_t"]
                          for l in LEVELS] for f in GATE_FOLDS])
    t_mean, t_std = t_vals.mean(axis=0), t_vals.std(axis=0)
    ax.fill_between(X, t_mean - t_std, t_mean + t_std,
                    color="#6b6b6b", alpha=0.10, lw=0, zorder=0)
    ax.plot(X, t_mean, color="#6b6b6b", lw=1.4, ls=(0, (2, 2)),
            label="T only (tactile, no V)", zorder=1.5)

    ax.set_xlabel("Occlusion (%)", fontsize=10, color=MUTED)
    ax.set_ylabel("PA-MPJPE (mm)", fontsize=10, color=MUTED)
    ax.set_xticks(X)
    ax.legend(frameon=False, fontsize=8.0, loc="lower right", ncol=2)
    fig.suptitle("E5 crossfit: absolute PA-MPJPE (4-fold mean; "
                 "fS6S7 = woS13)", fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "Main_Fig4_Gate_Absolute.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def _fig_relative(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.0), facecolor=SURFACE)
    titles = ("Δ vs no V gate (Gate − V8)",
              "Δ vs no gate (V8_Cv7)",
              "Δ(VT − V) tactile gain")
    ylabels = ("Δ PA-MPJPE (mm; negative = V gate helps)",
               "Δ PA-MPJPE (mm; negative = better)",
               "Δ(VT − V) PA-MPJPE (mm; negative = tactile helps)")
    for ax, title, ylab in zip(axes, titles, ylabels):
        _style(ax)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Occlusion (%)", fontsize=9.5, color=MUTED)
        ax.set_ylabel(ylab, fontsize=9.0, color=MUTED)
        ax.set_xticks(X)
        ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":", zorder=1)

    rel = {l: relative(data, l) for l in LEVELS}
    # panel 1: Gate - Cv5SD2
    mean = [float(np.mean([rel[l][f]["gate_minus_v8"] for f in GATE_FOLDS]))
            for l in LEVELS]
    for l in LEVELS:
        for f in GATE_FOLDS:
            axes[0].plot(int(l), rel[l][f]["gate_minus_v8"], "o", ms=3.5,
                         color=COLORS["Gate"], alpha=0.22, zorder=2)
    axes[0].plot(X, mean, lw=2.0, marker="o", ms=4.5, color=COLORS["Gate"],
                 label="Gate − V8 (Cv5SD2)")
    axes[0].legend(frameon=False, fontsize=8.6, loc="best")

    # panel 2: vs V8_Cv7 (no gate), core + ablations
    for key, group, ls, lab in (
            ("gate_minus_nogate", "Gate", "-", "Gate − Cv7 (T+V gate vs no gate)"),
            ("v8_minus_nogate", "V8_Cv5", "-", "V8 − Cv7 (T gate vs no gate)")):
        mean = [float(np.mean([rel[l][f][key] for f in GATE_FOLDS]))
                for l in LEVELS]
        for l in LEVELS:
            for f in GATE_FOLDS:
                axes[1].plot(int(l), rel[l][f][key], "o", ms=3.5,
                             color=COLORS[group], alpha=0.20, zorder=2)
        axes[1].plot(X, mean, lw=2.0, marker="o", ms=4.5, color=COLORS[group],
                     ls=ls, label=lab)
    for group in ABLATIONS:
        key = "%s_minus_nogate" % group
        mean = [float(np.mean([rel[l][f][key] for f in GATE_FOLDS]))
                for l in LEVELS]
        axes[1].plot(X, mean, lw=1.4, color=COLORS[group], ls=":",
                     label="%s − Cv7" % group, alpha=0.9)
    axes[1].legend(frameon=False, fontsize=7.6, loc="best")

    # panel 3: tactile gain per group
    for group, key in (("Gate", "tactile_gate"), ("V8_Cv5", "tactile_v8"),
                       ("V8_Cv7", "tactile_nogate")):
        mean = [float(np.mean([rel[l][f][key] for f in GATE_FOLDS]))
                for l in LEVELS]
        for l in LEVELS:
            for f in GATE_FOLDS:
                axes[2].plot(int(l), rel[l][f][key], "o", ms=3.5,
                             color=COLORS[group], alpha=0.20, zorder=2)
        axes[2].plot(X, mean, lw=2.0, marker="o", ms=4.5, color=COLORS[group],
                     ls="--" if group == "V8_Cv7" else "-",
                     label="VT−V " + group)
    axes[2].legend(frameon=False, fontsize=8.2, loc="best")

    fig.suptitle("E5 crossfit: gate effects (4-fold LSO mean; "
                 "fS6S7 = woS13; negative = improvement)",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "Main_Fig5_Gate_Relative.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def export_tables(data: dict) -> None:
    # absolute: group × level × {V2M,VT2M} mean±std
    rows = []
    for group in ALL_GROUPS:
        for level in LEVELS:
            rows.append([group, "occ" + level,
                         mean_std([data[group][f][level]["v"] for f in GATE_FOLDS]),
                         mean_std([data[group][f][level]["vt"] for f in GATE_FOLDS])])
    write_csv(OUT / "Main_Table4_Gate_Absolute.csv",
              ["模型", "档位", "PA-MPJPE(V2M)", "PA-MPJPE(VT2M)"], rows)

    # relative: level × mean±std
    rel = {l: relative(data, l) for l in LEVELS}
    rel_keys = ["gate_minus_v8", "gate_minus_nogate", "v8_minus_nogate",
                "tactile_gate", "tactile_v8", "tactile_nogate"]
    rel_keys += ["%s_minus_nogate" % a for a in ABLATIONS]
    rows = []
    for level in LEVELS:
        cells = [mean_std([rel[level][f][k] for f in GATE_FOLDS]) for k in rel_keys]
        rows.append(["occ" + level] + cells)
    write_csv(OUT / "Main_Table5_Gate_Relative.csv",
              ["档位", "Δ(Gate−V8)", "Δ(Gate−Cv7)", "Δ(V8−Cv7)",
               "Δ(VT−V) Gate", "Δ(VT−V) V8", "Δ(VT−V) Cv7"]
              + ["Δ(%s−Cv7)" % a for a in ABLATIONS], rows)

    # per-fold long
    rows = []
    for fold in GATE_FOLDS:
        for level in LEVELS:
            g = data["Gate"][fold][level]
            c5 = data["V8_Cv5"][fold][level]
            c7 = data["V8_Cv7"][fold][level]
            relf = rel[level][fold]
            row = [fold, "occ" + level, g["v"], g["vt"], c5["vt"], c7["vt"],
                   relf["gate_minus_v8"], relf["gate_minus_nogate"],
                   relf["v8_minus_nogate"], relf["tactile_gate"],
                   relf["tactile_v8"], relf["tactile_nogate"]]
            row += [relf["%s_minus_nogate" % a] for a in ABLATIONS]
            rows.append(row)
    write_csv(OUT / "Full_Table3_Gate_PerFold.csv",
              ["折", "档位", "V2M", "VT2M Gate", "VT2M V8", "VT2M Cv7",
               "Gate−V8", "Gate−Cv7", "V8−Cv7",
               "VT−V Gate", "VT−V V8", "VT−V Cv7"]
              + ["%s−Cv7" % a for a in ABLATIONS], rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = collect()
    export_tables(data)
    _fig_absolute(data)
    _fig_relative(data)
    print("[done] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
