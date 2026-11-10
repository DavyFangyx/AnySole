#!/usr/bin/env python3
"""E6-A 接触相位：V8 四折 LSO woS13 聚合表（2026-11-08）。

数据源：``results_display/Etest_crossfit_v8/E6_ContactPhaseAnalysis/
{pa_mpjpe,ankle_foot_mpjpe}/fold_*/detail/e6_table.json``。

口径：
- G = VT 误差 − V 误差（mm；负 = 触觉有帮助）。
- 相位 = f6_soft 的 0/1/2/3 = 稳定支撑 / 摆动 / 落脚附近 / 离地附近。
- 聚合 = 4 折 × 9 动作 × 左右脚的非空单元；``G`` 的 std 按折内动作均值
  的 4 折标准差（不把帧当独立样本）。
- 只保留论文消融五模型（Cv5/Cv7/Cv4/Cv2/Cv6）；RM/PO/VG/T1 已判失败，
  不进主表。

产物：
  Main_Table1_E6_Crossfit_PhaseGainV8_pa_mpjpe.csv
  Main_Table1_E6_Crossfit_PhaseGainV8_ankle_foot_mpjpe.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results_display" / "Etest_crossfit_v8" / "E6_ContactPhaseAnalysis"
OUT = ROOT
# E6 只做 Cv5（门控）vs Cv7（无门控）；其余模型留给 E4 消融。
MODELS = ("V8_Cv5", "V8_Cv7")
PHASES = (("0", "Stance"), ("1", "Swing"), ("2", "Strike"), ("3", "Off"))


def family(cell_model: str) -> str:
    return cell_model.split("_fS")[0]


def collect(metric_dir: str) -> dict:
    """{(model, phase): {"G": [fold_mean...], "V": [...], "VT": [...], "n": int}}"""
    folds = sorted((ROOT / metric_dir).glob("fold_*"))
    acc: dict[tuple[str, str], dict] = {}
    for fold_dir in folds:
        path = fold_dir / "detail" / "e6_table.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_fold: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
        for action, block in payload.items():
            for cell_model, feet in (block.get("table") or {}).items():
                model = family(cell_model)
                if model not in MODELS:
                    continue
                for phase_key, phase_name in PHASES:
                    vals = []
                    for foot in ("left", "right"):
                        entry = (feet.get(foot) or {}).get(phase_key)
                        if entry and entry.get("G") is not None:
                            vals.append((float(entry["G"]), float(entry["V"]), float(entry["VT"])))
                    if not vals:
                        continue
                    g = float(np.mean([v[0] for v in vals]))
                    v = float(np.mean([v[1] for v in vals]))
                    vt = float(np.mean([v[2] for v in vals]))
                    per_fold.setdefault((model, phase_name), []).append((g, v, vt))
        for key, vals in per_fold.items():
            bucket = acc.setdefault(key, {"G": [], "V": [], "VT": [], "n": 0})
            bucket["G"].append(float(np.mean([x[0] for x in vals])))
            bucket["V"].append(float(np.mean([x[1] for x in vals])))
            bucket["VT"].append(float(np.mean([x[2] for x in vals])))
            bucket["n"] += len(vals)
    return acc


def write(metric_dir: str, out_name: str, out_dir: Path | None = None) -> None:
    acc = collect(metric_dir)
    rows = []
    for model in MODELS:
        for _, phase_name in PHASES:
            bucket = acc.get((model, phase_name))
            if not bucket:
                continue
            g = np.asarray(bucket["G"])
            rows.append([model, phase_name,
                         "%.3f±%.3f" % (g.mean(), g.std(ddof=0)) if len(g) > 1 else "%.3f" % g.mean(),
                         "%.3f" % float(np.mean(bucket["V"])),
                         "%.3f" % float(np.mean(bucket["VT"])),
                         len(g), bucket["n"]])
    path = (out_dir or OUT) / out_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["模型", "相位", "G_mean±std(mm)", "V_mean(mm)", "VT_mean(mm)",
                    "n_folds", "n_action_fold_cells"])
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def collect_extra(metric: str) -> dict:
    """相位物理/接触指标（vert/pen mm，contact %）。

    数据源 = ``pa_mpjpe/fold_*/detail/e6_table_metrics.json`` 的
    ``table[cell_model][foot][phase][metric] = {V, VT, G}``。
    """
    folds = sorted((ROOT / "pa_mpjpe").glob("fold_*"))
    acc: dict[tuple[str, str], dict] = {}
    for fold_dir in folds:
        path = fold_dir / "detail" / "e6_table_metrics.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_fold: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
        for action, block in payload.items():
            for cell_model, feet in (block.get("table") or {}).items():
                model = family(cell_model)
                if model not in MODELS:
                    continue
                for phase_key, phase_name in PHASES:
                    vals = []
                    for foot in ("left", "right"):
                        entry = (feet.get(foot) or {}).get(phase_key)
                        m = (entry or {}).get(metric)
                        if m and m.get("G") is not None:
                            vals.append((float(m["G"]), float(m["V"]), float(m["VT"])))
                    if not vals:
                        continue
                    per_fold.setdefault((model, phase_name), []).append(
                        (float(np.mean([v[0] for v in vals])),
                         float(np.mean([v[1] for v in vals])),
                         float(np.mean([v[2] for v in vals]))))
        for key, vals in per_fold.items():
            bucket = acc.setdefault(key, {"G": [], "V": [], "VT": [], "n": 0})
            bucket["G"].append(float(np.mean([x[0] for x in vals])))
            bucket["V"].append(float(np.mean([x[1] for x in vals])))
            bucket["VT"].append(float(np.mean([x[2] for x in vals])))
            bucket["n"] += len(vals)
    return acc


def write_extra(metric: str, direction: str, out_name: str,
                out_dir: Path | None = None) -> None:
    acc = collect_extra(metric)
    rows = []
    for model in MODELS:
        for _, phase_name in PHASES:
            bucket = acc.get((model, phase_name))
            if not bucket:
                continue
            g = np.asarray(bucket["G"])
            rows.append([model, phase_name,
                         "%.3f±%.3f" % (g.mean(), g.std(ddof=0)) if len(g) > 1 else "%.3f" % g.mean(),
                         "%.3f" % float(np.mean(bucket["V"])),
                         "%.3f" % float(np.mean(bucket["VT"])),
                         direction, len(g), bucket["n"]])
    path = (out_dir or OUT) / out_name
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["模型", "相位", "G_mean±std", "V_mean", "VT_mean", "方向",
                    "n_folds", "n_action_fold_cells"])
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def g_fmt(bucket: dict | None) -> str:
    if not bucket:
        return ""
    g = np.asarray(bucket["G"])
    return "%.3f±%.3f" % (g.mean(), g.std(ddof=0)) if len(g) > 1 else "%.3f" % g.mean()


def write_three_layers() -> None:
    """相位增益总表：PA / GlobalMPJPE / pen 三层各一个代表指标。"""
    acc = {
        "PA-MPJPE": collect("pa_mpjpe"),
        "GlobalMPJPE": collect("global_mpjpe"),
        "pen（GroundPen）": collect_extra("pen"),
    }
    rows = []
    for model in MODELS:
        for _, phase_name in PHASES:
            row = [model, phase_name]
            n = 0
            for name in ("PA-MPJPE", "GlobalMPJPE", "pen（GroundPen）"):
                bucket = acc[name].get((model, phase_name))
                row.append(g_fmt(bucket))
                if bucket:
                    n = max(n, len(bucket["G"]))
            row.append(n)
            rows.append(row)
    path = OUT / "Main_Table1_E6_PhaseGain_ThreeLayers.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["模型", "相位", "G_PA-MPJPE", "G_GlobalMPJPE",
                    "G_pen（GroundPen）", "n_folds"])
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def main() -> int:
    supp = OUT / "supplementary"
    # 主表：姿态 PA + 物理 pen（GroundPen 的相位版）；其余下沉 supplementary。
    write("pa_mpjpe", "Main_Table1_E6_Crossfit_PhaseGainV8_pa_mpjpe.csv")
    write("global_mpjpe", "Main_Table1_E6_Crossfit_PhaseGainV8_global_mpjpe.csv")
    write_extra("pen", "负=有帮助（mm，GroundPen 相位版）",
                "Main_Table1_E6_Crossfit_PhaseGainV8_pen.csv")
    write_three_layers()
    write("ankle_foot_mpjpe", "E6_PhaseGainV8_ankle_foot.csv", supp)
    write_extra("vert", "负=有帮助（mm）", "E6_PhaseGainV8_vert.csv", supp)
    write_extra("contact", "正=有帮助（%）", "E6_PhaseGainV8_contact.csv", supp)
    for stale in (
        "Main_Table1_E6_Crossfit_PhaseGainV8_ankle_foot_mpjpe.csv",
        "Main_Table1_E6_Crossfit_PhaseGainV8_vert.csv",
        "Main_Table1_E6_Crossfit_PhaseGainV8_contact.csv",
    ):
        p = OUT / stale
        if p.exists():
            p.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
