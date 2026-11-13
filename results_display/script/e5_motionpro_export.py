#!/usr/bin/env python3
"""MotionPRO E5 逐档主表（与 Main_Table2_E5_*_MainVsBaselines.csv 同口径）。

数据源：results/baselines/MotionPRO/predictions/<variant>/metrics/fourfold_summary.json
（逐折评估后的四折 mean±std）。产物（每情形一张，列 = 主图同口径 6 指标）：

    E5_OcclusionRobustness/Main_Table_MotionPRO_Occlusion.csv
    E5_TimeGap/Main_Table_MotionPRO_TimeGap.csv
    E5_SceneOcclusion/Main_Table_MotionPRO_SceneOcclusion.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT))

from v8_e5_degradation_figures import BASELINE_KEYS, CONDITIONS  # noqa: E402

OUT = REPO / "results_display" / "Etest_crossfit_v8"
PRED = REPO / "results/baselines/MotionPRO/predictions"
MAIN6 = ("PA-MPJPE", "ankle_foot_mpjpe", "GlobalMPJPE",
         "FootSliding", "GroundPen", "traj_ATE")


def fmt(v):
    if v is None or not np.isfinite(float(v)):
        return ""
    return "%.3f" % float(v)


def fmt_ms(entry):
    if entry is None or entry.get("n_folds", 0) < 4:
        return ""
    return "%.3f±%.3f" % (float(entry["mean"]), float(entry["std"]))


def main() -> int:
    for condition, cfg in CONDITIONS.items():
        rows = []
        for level, display in zip(cfg["baseline"], cfg["x"]):
            p = PRED / level / "metrics" / "fourfold_summary.json"
            payload = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
            cells = []
            for metric in MAIN6:
                key = BASELINE_KEYS.get(metric)
                entry = payload.get(key)
                cells.append(fmt_ms(entry) if entry is not None else "")
            rows.append(["MotionPRO", f"occ {display}" if condition == "occ"
                         else f"gap {display}" if condition == "gap"
                         else ("scene clean" if display == 0 else "scene"),
                         "@VT"] + cells)
        out = OUT / cfg["dir"] / "detail" / ("Main_Table_MotionPRO_%s.csv"
                                  % {"occ": "Occlusion", "gap": "TimeGap",
                                     "scene": "SceneOcclusion"}[condition])
        with out.open("w", newline="", encoding="utf-8-sig") as fh:
            import csv
            writer = csv.writer(fh)
            writer.writerow(["模型", "档位", "行"] + list(MAIN6))
            writer.writerows(rows)
        print(f"[csv] {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
