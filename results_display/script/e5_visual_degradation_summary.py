#!/usr/bin/env python3
"""E5 视觉退化论文汇总表（2026-11-12）。

T-only rows: after this exporter, run
``results_display/script/e5_t_bifoot_reference_replace.py`` to replace the
legacy T2M reference with the shared BiFoot branch.

输出到 `results_display/Etest_crossfit_v8/E5_VisualDegradation_Summary/`：
- `Main_Table_E5_VisualDegradation_Full.csv`：全部子情况 × 六指标（长表）。
- `Main_Table_E5_VisualDegradation_Main.tex`：论文主表（三个指标面板）。
- `Main_Table_E5_VisualDegradation_Main.md`：同内容 Markdown 预览。

主表行 = WHAM / GVHMR / MotionPRO / V8_Cv5 @V / V8_Cv5 @VT / T-only；
这一版不含 SAM3DB、V7。子情况取代表性档位：
2D 遮挡 0/50/90%，时间断流 0/40/60%，场景 clean/scene。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT))

from v8_e5_degradation_figures import (  # noqa: E402
    BASELINE_KEYS, BIFOOT_KEYS, CONDITIONS, METRICS, _bifoot_t_only,
    baseline_series, motionpro_series, series,
)

OUT = Path(os.environ.get(
    "E5_SUMMARY_OUT",
    REPO / "results_display" / "Etest_crossfit_v8" / "E5_VisualDegradation_Summary"))

SERIES = ("WHAM @V", "GVHMR @V", "MotionPRO @VT",
          "V8_Cv5 @V", "V8_Cv5 @VT", "T-only")
MODEL_SERIES = {"V8_Cv5 @V": "V2M", "V8_Cv5 @VT": "VT2M", "T-only": "T2M"}
MODE_ZH = {"occ": "2D遮挡", "gap": "时间断流", "scene": "场景遮挡"}
# representative levels: (condition, index in CONDITIONS[*]["x"])
MAIN_LEVELS = [("occ", 0), ("occ", 5), ("occ", 9),
               ("gap", 0), ("gap", 3), ("gap", 4),
               ("scene", 0), ("scene", 1)]
MAIN_METRICS = ("PA-MPJPE", "ankle_foot_mpjpe", "traj_ATE")
METRIC_LABEL = {"PA-MPJPE": "PA-MPJPE", "ankle_foot_mpjpe": "Ankle-foot MPJPE",
                "traj_ATE": "traj_ATE"}
FULL_METRICS = tuple(name for name, _, _ in METRICS)


def value_at(condition: str, metric: str, series_name: str, idx: int):
    """Return (mean, std_or_None) for one series at one level."""
    cfg = CONDITIONS[condition]
    key = next(k for name, _, k in METRICS if name == metric)
    if series_name == "T-only":
        # Pose metrics stay on the T2M read; trajectory/physical metrics use
        # the shared T+BiFoot branch (same as the E5 figures).
        if metric in BIFOOT_KEYS:
            override = _bifoot_t_only(condition, metric)
            if override is not None:
                return override[0][idx], override[1][idx]
        suffix = cfg["levels"][idx] if condition == "scene" else ""
        means, stds = series("V8_Cv5", "T2M", key, [suffix])
        return means[0], stds[0]
    if series_name == "MotionPRO @VT":
        means, stds = motionpro_series(cfg["baseline"], metric)
        return means[idx], stds[idx]
    if series_name in MODEL_SERIES:
        means, stds = series("V8_Cv5", MODEL_SERIES[series_name], key, cfg["levels"])
        return means[idx], stds[idx]
    base = series_name.split()[0]
    vals = baseline_series(base, cfg["baseline"], metric)
    if vals is None:
        return float("nan"), None
    return vals[idx], None


def fmt(mean: float, std: float | None, digits=1) -> str:
    if mean is None or not np.isfinite(float(mean)):
        return "—"
    if std is None or not np.isfinite(float(std)):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f}±{float(std):.{digits}f}"


def write_full() -> None:
    path = OUT / "Main_Table_E5_VisualDegradation_Full.csv"
    header = ["退化模式", "子情况", "模型"] + [
        label for label in ("PA-MPJPE (mm)", "Ankle-foot MPJPE (mm)",
                            "GlobalMPJPE (mm)", "FootSliding (mm)",
                            "GroundPen (mm)", "traj_ATE (mm)")
    ]
    rows = []
    for condition, cfg in CONDITIONS.items():
        for idx, level in enumerate(cfg["x"]):
            sub = f"{condition}{int(level):03d}" if condition != "scene" else (
                "scene_clean" if idx == 0 else "scene")
            for name in SERIES:
                cells = []
                for metric in FULL_METRICS:
                    m, s = value_at(condition, metric, name, idx)
                    cells.append(fmt(m, s))
                rows.append([MODE_ZH[condition], sub, name] + cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("[csv]", path, len(rows))


def _main_matrix() -> dict:
    data = {}
    for condition, idx in MAIN_LEVELS:
        sub = (f"{condition}{int(CONDITIONS[condition]['x'][idx]):03d}"
               if condition != "scene" else ("clean" if idx == 0 else "scene"))
        for name in SERIES:
            data[(condition, sub, name)] = {
                metric: value_at(condition, metric, name, idx)[0]
                for metric in FULL_METRICS
            }
    return data


def write_wide_csv(data: dict) -> None:
    heads = []
    for condition, idx in MAIN_LEVELS:
        if condition == "scene":
            heads.append("Scene clean" if idx == 0 else "Scene occl.")
        else:
            tag = "2D" if condition == "occ" else "Gap"
            heads.append(f"{tag} {int(CONDITIONS[condition]['x'][idx])}%")
    path = OUT / "Main_Table_E5_VisualDegradation_Main.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["指标", "模型"] + heads)
        for metric in MAIN_METRICS:
            for name in SERIES:
                vals = []
                for condition, idx in MAIN_LEVELS:
                    sub = (f"{condition}{int(CONDITIONS[condition]['x'][idx]):03d}"
                           if condition != "scene" else ("clean" if idx == 0 else "scene"))
                    m = data[(condition, sub, name)][metric]
                    vals.append("—" if not np.isfinite(float(m)) else f"{float(m):.1f}")
                w.writerow([METRIC_LABEL[metric], name] + vals)
    print("[csv]", path)


def write_markdown(data: dict) -> None:
    heads = []
    for condition, idx in MAIN_LEVELS:
        if condition == "scene":
            heads.append("Scene clean" if idx == 0 else "Scene occl.")
        else:
            tag = "2D" if condition == "occ" else "Gap"
            heads.append(f"{tag} {int(CONDITIONS[condition]['x'][idx])}%")
    lines = ["2D occlusion: 0/50/90%; temporal gap: 0/40/60%; scene: clean/occluded.", ""]
    for metric in MAIN_METRICS:
        lines += [f"**{METRIC_LABEL[metric]} (mm)**", "",
                  "| 模型 | " + " | ".join(heads) + " |",
                  "|---|" + "---|" * len(heads)]
        for name in SERIES:
            vals = []
            for condition, idx in MAIN_LEVELS:
                sub = (f"{condition}{int(CONDITIONS[condition]['x'][idx]):03d}"
                       if condition != "scene" else ("clean" if idx == 0 else "scene"))
                m = data[(condition, sub, name)][metric]
                vals.append("—" if not np.isfinite(float(m)) else f"{float(m):.1f}")
            lines.append(f"| {name} | " + " | ".join(vals) + " |")
        lines.append("")
    path = OUT / "Main_Table_E5_VisualDegradation_Main.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[md]", path)


def write_tex(data: dict) -> None:
    heads = []
    for condition, idx in MAIN_LEVELS:
        if condition == "scene":
            heads.append("clean" if idx == 0 else "scene")
        else:
            heads.append(f"{int(CONDITIONS[condition]['x'][idx])}\\%")
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Performance under three visual degradation modes (4-fold mean, mm).}",
        r"\label{tab:e5-visual-degradation}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\scriptsize",
        r"\begin{tabular}{l ccc ccc cc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{2D occlusion} & \multicolumn{3}{c}{Temporal gap} & \multicolumn{2}{c}{Scene occl.} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"Model & " + " & ".join(heads) + r" \\",
        r"\midrule",
    ]
    for mi, metric in enumerate(MAIN_METRICS):
        if mi:
            tex.append(r"\midrule")
        tex.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(heads) + 1,
                                                           METRIC_LABEL[metric]))
        for name in SERIES:
            vals = []
            for condition, idx in MAIN_LEVELS:
                sub = (f"{condition}{int(CONDITIONS[condition]['x'][idx]):03d}"
                       if condition != "scene" else ("clean" if idx == 0 else "scene"))
                m = data[(condition, sub, name)][metric]
                vals.append("--" if not np.isfinite(float(m)) else f"{float(m):.1f}")
            tex.append(name.replace("_", r"\_") + " & " + " & ".join(vals) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path = OUT / "Main_Table_E5_VisualDegradation_Main.tex"
    path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("[tex]", path)


def write_long_tex(data: dict) -> None:
    """Tidy appendix table: one row per mode/sub-case/model, all six metrics."""
    metric_labels = ["PA-MPJPE", "Ank.-foot", "GlobalMPJPE",
                     "FootSliding", "GroundPen", "traj\\_ATE"]
    tex = [
        r"% requires booktabs, multirow",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Visual degradation summary (4-fold mean, mm).}",
        r"\label{tab:e5-visual-degradation-long}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\small",
        r"\begin{tabular}{lll cccccc}",
        r"\toprule",
        r"Mode & Sub-case & Model & " + " & ".join(metric_labels) + r" \\",
        r"\midrule",
    ]
    labels = {"occ": "2D occlusion", "gap": "Temporal gap", "scene": "Scene occl."}
    for condition, idx in MAIN_LEVELS:
        key_sub = (f"{condition}{int(CONDITIONS[condition]['x'][idx]):03d}"
                   if condition != "scene" else ("clean" if idx == 0 else "scene"))
        display_sub = (f"{int(CONDITIONS[condition]['x'][idx])}\\%"
                       if condition != "scene" else key_sub)
        for j, name in enumerate(SERIES):
            vals = []
            for metric in FULL_METRICS:
                m = data[(condition, key_sub, name)][metric]
                vals.append("--" if not np.isfinite(float(m)) else f"{float(m):.1f}")
            mode_cell = (r"\multirow{5}{*}{%s}" % labels[condition]) if j == 0 else ""
            sub_cell = (r"\multirow{5}{*}{%s}" % display_sub) if j == 0 else ""
            tex.append(mode_cell + " & " + sub_cell + " & " +
                       name.replace("_", r"\_") + " & " + " & ".join(vals) + r" \\")
        tex.append(r"\cmidrule(lr){1-9}")
    tex[-1] = r"\bottomrule"
    tex += [r"\end{tabular}", r"\end{table*}"]
    path = OUT / "Main_Table_E5_VisualDegradation_Long.tex"
    path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("[tex]", path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_full()
    data = _main_matrix()
    write_wide_csv(data)
    write_markdown(data)
    write_tex(data)
    write_long_tex(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
