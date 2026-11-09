#!/usr/bin/env python3
"""四折 LSO woS13 纯视觉基线表（2026-11-08）。

数据源：results/baselines/<Model>/metrics/<fold>_comparison.json（schema v2，
由 Baselines.utils.evaluate 按 crossfit_woS13 折表计算；帧加权聚合）。
产物：results_display/Etest_crossfit/Full_Table5_Crossfit_Baselines_test.csv

列序与 Full_Table1_Crossfit_Main_test.csv 一致（E2_METRICS10，2026-11-05
用户裁定）；行 = 基线 × 折 + mean±std。视觉基线全部是 V2M（真 V、零 T），
条件标签 U@V。n 列与主表一致，表示该折结果是否存在（1/0），不是会话数；
会话数见 detail 文件。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results_display" / "Etest_crossfit"
BASELINES = REPO / "results" / "baselines"

METRICS = ["GroundPen", "FootSliding", "RootOrientationDrift",
           "PA-MPJPE", "MPJPE", "MPJRE", "W-MPJPE100",
           "GlobalMPJPE", "traj_ATE", "Jitter_pred"]
NATIVE_KEY = {
    "GroundPen": "ground_pen_mm",
    "FootSliding": "foot_sliding_mm",
    "RootOrientationDrift": "root_orientation_drift_deg",
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "MPJRE": "mpjae_deg",
    "W-MPJPE100": "w_mpjpe100_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "traj_ATE": "root_ate_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}
MODELS = (("WHAM", "WHAM"), ("SAM3DB", "SAM3DB"), ("GVHMR", "GVHMR 2024"))
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")


def fmt(value):
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(number) else round(number, 3)


def mean_std(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not values:
        return ""
    if len(values) == 1:
        return round(values[0], 3)
    return "%.3f±%.3f" % (float(np.mean(values)), float(np.std(values)))


def load_summary(model: str, fold: str) -> dict | None:
    path = BASELINES / model / "metrics" / ("%s_comparison.json" % fold)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (payload.get("summary") or {}, int(payload.get("n_sessions") or 0))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def main() -> int:
    header = ["模型", "条件", "折", "n"] + METRICS
    rows = []
    detail = []
    reference = {}
    main_path = OUT / "Full_Table1_Crossfit_Main_test.csv"
    if main_path.is_file():
        with main_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("模型") == "V7_Cv5" and row.get("条件") == "U@V":
                    reference[row.get("折")] = row
    for model, display in MODELS:
        summaries = {}
        for fold in FOLDS:
            loaded = load_summary(model, fold)
            if loaded is None:
                rows.append([display, "U@V", fold, 0] + [""] * len(METRICS))
                continue
            summary, n_sessions = loaded
            summaries[fold] = summary
            rows.append([display, "U@V", fold, 1]
                        + [fmt(summary.get(NATIVE_KEY[key])) for key in METRICS])
            detail.append([display, fold, n_sessions, int(summary.get("n_valid_frames") or 0)])
        if summaries:
            rows.append([display, "U@V", "mean±std", len(summaries)]
                        + [mean_std([summaries[fold].get(NATIVE_KEY[key])
                                     for fold in summaries]) for key in METRICS])
    write_csv(OUT / "Full_Table5_Crossfit_Baselines_test.csv", header, rows)
    write_csv(OUT / "Full_Table5_Crossfit_Baselines_detail.csv",
              ["模型", "折", "n_sessions", "n_valid_frames"], detail)

    if reference:
        delta_rows = []
        for model, display in MODELS:
            per_fold = {}
            for fold in FOLDS:
                loaded = load_summary(model, fold)
                ref = reference.get(fold)
                if loaded is None or ref is None:
                    delta_rows.append([display, fold, 0] + [""] * len(METRICS))
                    continue
                summary, _n = loaded
                values = {}
                for key in METRICS:
                    base = summary.get(NATIVE_KEY[key])
                    try:
                        anchor = float(ref[key])
                    except (TypeError, ValueError):
                        anchor = None
                    values[key] = (None if base is None or anchor is None
                                   or not np.isfinite(float(base))
                                   else float(base) - anchor)
                per_fold[fold] = values
                delta_rows.append([display, fold, 1] + [fmt(values[key]) for key in METRICS])
            if per_fold:
                delta_rows.append([display, "mean±std", len(per_fold)]
                                  + [mean_std([per_fold[fold][key] for fold in per_fold])
                                     for key in METRICS])
        write_csv(OUT / "Full_Table6_Crossfit_BaselineDelta_test.csv",
                  ["模型", "折", "n"] + METRICS, delta_rows)

    # 合并视图：不修改主表，把基线行插在 U@V 块末尾（主表 V 行的直接对照）。
    if main_path.is_file():
        with main_path.open(encoding="utf-8-sig", newline="") as handle:
            main_rows = list(csv.reader(handle))
        if main_rows:
            main_header, main_body = main_rows[0], main_rows[1:]
            baseline_header = header
            baseline_body = [row for row in rows]
            if main_header == baseline_header:
                insert_at = max((i for i, row in enumerate(main_body)
                                 if len(row) > 1 and row[1] == "U@V"), default=-1) + 1
                merged = main_body[:insert_at] + baseline_body + main_body[insert_at:]
                write_csv(OUT / "Full_Table1_Crossfit_Main_test_with_baselines.csv",
                          main_header, merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
