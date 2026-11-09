#!/usr/bin/env python3
"""指标信息台账（2026-11-08）：V8 四折 woS13 × 全部可用指标。

目的：给论文指标选择提供**信息底账**，不做取舍。列出
V8_Cv5 / V8_Cv7 的 V、VT、Δ(VT−V)，以及 WHAM / GVHMR / SAM3DB 的 V 块四折值
（仅标准十指标可比），并标注 E5 遮挡轴 / E6 相位轴是否可用、与基线是否同定义。

产物：results_display/Etest_crossfit_v8/detail/Metric_Inventory_20261108.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
BASE = REPO / "results" / "baselines"
OUT = REPO / "results_display" / "Etest_crossfit_v8" / "detail" / "Metric_Inventory_20261108.csv"

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODELS = ("V8_Cv5", "V8_Cv7")
BASELINES = ("WHAM", "GVHMR", "SAM3DB")

# 基线 summary key ↔ 本表指标（仅标准十指标）
BASELINE_KEYS = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "W-MPJPE100": "w_mpjpe100_mm",
    "RootAbs": "root_ate_mm",          # 近似对照：自锚绝对根误差 vs 基线 raw root ATE
    "MPJRE": "mpjae_deg",
    "traj_ATE": "root_ate_mm",
    "RootOrientationDrift": "root_orientation_drift_deg",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "Jitter_pred": "jitter_pred_m_s3",
    # 展示层吃锚列与基线同源
    "FootSliding（GT Anchored）": "foot_sliding_mm",
    "GroundPen（GT Anchored）": "ground_pen_mm",
}
# 与基线同定义的指标；traj_ATE 为定义不同（主模型链式无锚 vs 基线 raw root ATE）
COMPARABLE = {k: "yes" for k in BASELINE_KEYS}
COMPARABLE["traj_ATE"] = "旧链式；论文不用，改用 RootAbs"
COMPARABLE["RootAbs"] = "论文 traj_ATE（近似基线 raw）"
COMPARABLE["GlobalMPJPE"] = "yes（无锚）"
COMPARABLE["W-MPJPE100"] = "yes（无锚）"
COMPARABLE["FootSliding"] = "yes（吃锚）"
COMPARABLE["GroundPen"] = "yes（GT floor）"
COMPARABLE["FootSliding（GT Anchored）"] = "yes（吃锚）"
COMPARABLE["GroundPen（GT Anchored）"] = "yes（GT floor）"

# E5 遮挡轴可用（v8_crossfit_export.py 的 10 指标 + Full_Table2 的触觉专项）
E5_METRICS = {
    "PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "MPJRE", "traj_ATE",
    "RootOrientationDrift", "FootSliding", "GroundPen", "Jitter_pred",
    "GroundPenFrac（GT Anchored）", "ground_pen_foot_mm",
    "PA-MPJPE_anklefoot", "PA-MPJPE_l_foot", "PA-MPJPE_r_foot",
    "MPJPE_anklefoot", "PA-MPJPE_upper",
}
E6_METRICS = {"PA-MPJPE", "PA-MPJPE_anklefoot", "MPJPE_anklefoot"}


def layer_of(key: str) -> str:
    if key.startswith(("PA-MPJPE", "MPJPE", "MPJRE", "pve_mm", "shape_vertex_std")):
        return "姿态"
    if key.startswith(("GlobalMPJPE", "W-MPJPE100", "WA-MPJPE100", "traj",
                       "Root", "root_", "WinDisp", "Anchor", "seam_jump",
                       "accel_dist_err_upper")):
        return "轨迹/全局"
    if key.startswith(("GroundPen", "ground_pen", "FootSliding", "foot_sliding",
                       "joint_limit")):
        return "物理"
    if key.startswith(("Jitter", "AccelError", "accel_")):
        return "时序"
    if key.startswith(("contact", "air_recall")):
        return "接触"
    return "其他"


def direction_of(key: str) -> str:
    if key.startswith(("contact_f1", "contact_acc", "contact_recall",
                       "contact_balanced_acc", "air_recall")):
        return "越高越好"
    if key.startswith(("shape_vertex_std", "seam_jump_gt", "accel_mag_gt",
                       "jitter_gt")):
        return "诊断"
    return "越低越好"


def metric_keys() -> list[str]:
    p = CELLS / "V8_Cv5_fS5S6S11_s1" / "metrics" / "test.json"
    return sorted(json.loads(p.read_text(encoding="utf-8"))["metrics"]["VT2M"].keys())


def cell_mean_std(model: str, cond: str, key: str) -> tuple[float, float, int]:
    vals = []
    for fold in FOLDS:
        p = CELLS / ("%s_%s_s1" % (model, fold)) / "metrics" / "test.json"
        if not p.is_file():
            continue
        block = json.loads(p.read_text(encoding="utf-8"))["metrics"][cond]
        v = block.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(mean: float, std: float, n: int) -> str:
    if n == 0:
        return ""
    return "%.3f±%.3f" % (mean, std) if n > 1 else "%.3f" % mean


def baseline_mean(key: str) -> dict[str, str]:
    out = {}
    bkey = BASELINE_KEYS.get(key)
    if not bkey:
        return {b: "" for b in BASELINES}
    for b in BASELINES:
        vals = []
        for fold in FOLDS:
            p = BASE / b / "metrics" / ("%s_comparison.json" % fold)
            if not p.is_file():
                continue
            s = json.loads(p.read_text(encoding="utf-8")).get("summary") or {}
            v = s.get(bkey)
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        out[b] = fmt(float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else ""
    return out


def main() -> int:
    keys = metric_keys()
    rows = []
    for key in keys:
        vals = {}
        for model in MODELS:
            for cond in ("V2M", "VT2M"):
                m, s, n = cell_mean_std(model, cond, key)
                vals[(model, cond)] = m
                vals[(model, cond, "txt")] = fmt(m, s, n)
        d5 = vals[("V8_Cv5", "VT2M")] - vals[("V8_Cv5", "V2M")]
        d7 = vals[("V8_Cv7", "VT2M")] - vals[("V8_Cv7", "V2M")]
        bl = baseline_mean(key)
        rows.append([
            layer_of(key), key, direction_of(key),
            vals[("V8_Cv5", "V2M", "txt")], vals[("V8_Cv5", "VT2M", "txt")],
            "%+.3f" % d5 if np.isfinite(d5) else "",
            vals[("V8_Cv7", "V2M", "txt")], vals[("V8_Cv7", "VT2M", "txt")],
            "%+.3f" % d7 if np.isfinite(d7) else "",
            bl["WHAM"], bl["GVHMR"], bl["SAM3DB"],
            COMPARABLE.get(key, ""),
            "yes" if key in E5_METRICS else "",
            "yes" if key in E6_METRICS else "",
        ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["层面", "指标", "方向",
                    "Cv5_V", "Cv5_VT", "Cv5_Δ(VT−V)",
                    "Cv7_V", "Cv7_VT", "Cv7_Δ(VT−V)",
                    "WHAM_V", "GVHMR_V", "SAM3DB_V",
                    "与基线同定义", "E5遮挡轴", "E6相位轴"])
        w.writerows(rows)
    print("[csv] %s (%d metrics)" % (OUT, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
