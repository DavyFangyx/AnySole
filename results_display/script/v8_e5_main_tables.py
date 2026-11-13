#!/usr/bin/env python3
"""E5 主表（2026-11-12 用户裁定版）。

这一版 E5 主图/主表不含 SAM3DB 与 V8_Cv7（无门控对照）。
- Main_Table1_*_Models.csv：四模型（Cv5/Cv4/Cv2/Cv6）× 档位 × 行 × 10 指标。
- Main_Table2_*_MainVsBaselines.csv：V8_Cv5 @VT / @V、WHAM @V、GVHMR @V、
  MotionPRO @VT × 档位 × 6 指标（与主图同口径）。
数据源与主图脚本一致；基线只记录已完成的 WHAM/GVHMR/MotionPRO。
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT))

from v8_e5_degradation_figures import (  # noqa: E402
    BASELINE_KEYS, CONDITIONS, FOLDS, METRICS, baseline_series, mean_std, series,
)

CELLS = REPO / "results" / "cells"
OUT = Path(os.environ.get("E5_OUT_ROOT",
                          REPO / "results_display" / "Etest_crossfit_v8"))

FAMS_MAIN = tuple(x.strip() for x in os.environ.get(
    "E5_MODELS", "V8_Cv5,V8_Cv4,V8_Cv2,V8_Cv6").split(",") if x.strip())
METRICS10 = ["PA-MPJPE", "MPJPE", "MPJRE",
             "GlobalMPJPE", "W-MPJPE100", "traj_ATE", "RootOrientationDrift",
             "GroundPen", "FootSliding", "Jitter_pred"]
MAIN6 = ("PA-MPJPE", "ankle_foot_mpjpe", "GlobalMPJPE",
         "FootSliding", "GroundPen", "traj_ATE")
READ_KEY = {"traj_ATE": "RootAbs"}
DIRNAME = {"occ": "E5_OcclusionRobustness", "gap": "E5_TimeGap",
           "scene": "E5_SceneOcclusion"}
FILEPFX = {"occ": "Occlusion", "gap": "TimeGap", "scene": "SceneOcclusion"}
BASELINES = ("WHAM", "GVHMR", "MotionPRO")

# (level_tag, display, cell_suffix)
LEVELS = {
    "occ": [("000", "occ000", "")] + [
        (l, "occ" + l, "_evalocc" + l)
        for l in ("010", "020", "030", "040", "050", "060", "070", "080", "090")
    ],
    "gap": [("000", "gap000", "")] + [
        (l, "gap" + l, "_evalgap" + l) for l in ("010", "020", "040", "060")
    ],
    "scene": [("clean", "scene_clean", "_evalsceneclean"),
              ("scene", "scene", "_evalscene")],
}


def strip_anchor(metrics: dict) -> dict:
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


def load_cell(fam: str, fold: str, suffix: str) -> dict | None:
    root = CELLS / ("%s_%s_s1%s" % (fam, fold, suffix)) / "metrics"
    sub = os.environ.get("E5_METRICS_SUBDIR", "").strip()
    if sub:
        root = root / sub
    p = root / "test.json"
    if not p.is_file():
        return None
    return strip_anchor(json.loads(p.read_text(encoding="utf-8"))["metrics"])


def motionpro_series(variants, metric: str):
    """MotionPRO per-variant four-fold mean±std (per-fold predictions dirs)."""
    key = BASELINE_KEYS.get(metric)
    if key is None:
        return None, None
    if isinstance(variants, str):
        variants = [variants]
    means, stds = [], []
    for variant in variants:
        p = REPO / "results/baselines/MotionPRO/predictions" / variant \
            / "metrics" / "fourfold_summary.json"
        if not p.is_file():
            means.append(np.nan); stds.append(np.nan); continue
        entry = json.loads(p.read_text(encoding="utf-8")).get(key)
        if not entry or entry.get("n_folds", 0) < 4:
            means.append(np.nan); stds.append(np.nan); continue
        means.append(float(entry["mean"])); stds.append(float(entry["std"]))
    return np.asarray(means), np.asarray(stds)


def fmt_cell(values) -> str:
    vals = [float(v) for v in values
            if v is not None and np.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) == 1:
        return "%.3f" % vals[0]
    return "%.3f±%.3f" % (float(np.mean(vals)), float(np.std(vals)))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def models_table(condition: str) -> None:
    rows = []
    for fam in FAMS_MAIN:
        for _tag, display, suffix in LEVELS[condition]:
            for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
                cells = []
                for metric in METRICS10:
                    key = READ_KEY.get(metric, metric)
                    vals = []
                    for fold in FOLDS:
                        m = load_cell(fam, fold, suffix)
                        if m:
                            vals.append(m.get(cond, {}).get(key))
                    cells.append(fmt_cell(vals))
                rows.append([fam, display, row_key] + cells)
            delta = []
            for metric in METRICS10:
                key = READ_KEY.get(metric, metric)
                vals = []
                for fold in FOLDS:
                    m = load_cell(fam, fold, suffix)
                    if not m:
                        continue
                    vt = m.get("VT2M", {}).get(key)
                    v = m.get("V2M", {}).get(key)
                    if vt is not None and v is not None:
                        vals.append(float(vt) - float(v))
                delta.append(fmt_cell(vals))
            rows.append([fam, display, "Δ(VT−V)"] + delta)
    out = OUT / DIRNAME[condition] / (
        "Main_Table1_E5_%s_Models.csv" % FILEPFX[condition])
    write_csv(out, ["模型", "档位", "行"] + METRICS10, rows)


def main_vs_baselines(condition: str) -> None:
    cfg = CONDITIONS[condition]
    rows = []
    for idx, (_tag, display, _suffix) in enumerate(LEVELS[condition]):
        block = {"V8_Cv5 @VT": [], "V8_Cv5 @V": [],
                 "WHAM @V": [], "GVHMR @V": [], "MotionPRO @VT": []}
        for metric in MAIN6:
            key = next(k for name, _, k in METRICS if name == metric)
            means, stds = series("V8_Cv5", "VT2M", key, cfg["levels"])
            block["V8_Cv5 @VT"].append((means[idx], stds[idx]))
            means, stds = series("V8_Cv5", "V2M", key, cfg["levels"])
            block["V8_Cv5 @V"].append((means[idx], stds[idx]))
            for base in BASELINES:
                if base == "MotionPRO":
                    means, stds = motionpro_series(cfg["baseline"], metric)
                    block["MotionPRO @VT"].append((means[idx], stds[idx]))
                else:
                    vals = baseline_series(base, cfg["baseline"], metric)
                    v = vals[idx] if vals is not None else np.nan
                    block["%s @V" % base].append((v, None))
        for name in ("V8_Cv5 @VT", "V8_Cv5 @V", "WHAM @V", "GVHMR @V",
                     "MotionPRO @VT"):
            cells = []
            for v, s in block[name]:
                if v is None or not np.isfinite(float(v)):
                    cells.append("")
                elif s is None or not np.isfinite(float(s)):
                    cells.append("%.3f" % float(v))
                else:
                    cells.append("%.3f±%.3f" % (float(v), float(s)))
            row_key = "@VT" if name.endswith("@VT") else "@V"
            rows.append([name, display, row_key] + cells)
    out = OUT / DIRNAME[condition] / (
        "Main_Table2_E5_%s_MainVsBaselines.csv" % FILEPFX[condition])
    write_csv(out, ["模型", "档位", "行"] + list(MAIN6), rows)


def main() -> int:
    for condition in ("occ", "gap", "scene"):
        models_table(condition)
        main_vs_baselines(condition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
