#!/usr/bin/env python3
"""E2 main-performance table for V8_IMU_Residual + baselines (V2 formal eval)."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results/diagnostics/20261114_imu_eval_v2_all/V8_IMU_Residual/clean"
OLD = REPO / "results_display/Etest_crossfit_v8/E2_MainPerformance/Main_Table1_E2_MainPerformance.csv"
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate/E2_MainPerformance"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODES = (("V2M", "U@V"), ("T2M", "U@T"), ("VT2M", "U@VT"))
MAIN = (("PA-MPJPE", "PA-MPJPE"), ("MPJPE", "MPJPE"),
        ("GlobalMPJPE", "GlobalMPJPE"), ("RootAbs", "traj_ATE"),
        ("RootOrientationDrift", "RootOrientationDrift"),
        ("GroundPen（GT Anchored）", "GroundPen"),
        ("FootSliding（GT Anchored）", "FootSliding"))
DETAIL = MAIN + (("MPJPE_anklefoot", "MPJPE_anklefoot"), ("MPJRE", "MPJRE"),
                 ("W-MPJPE100", "W-MPJPE100"), ("RootAbs", "RootAbs"),
                 ("Jitter_pred", "Jitter_pred"))

def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

def fmt(vals):
    m, s = ms(vals)
    return "-" if m is None else "%.3f±%.3f" % (m, s)

def load_baselines():
    rows = []
    if not OLD.is_file():
        return rows
    for r in csv.DictReader(open(OLD, encoding="utf-8-sig")):
        if r["模型"].startswith("V8_") or r["模型"].startswith("SAM3DB"):
            continue
        if r.get("条件") not in ("U@V", "U@T", "U@VT"):
            continue
        if r["模型"] == "Step2Motion（position 单锚）":
            r["模型"] = "Step2Motion"
        rows.append(r)
    return rows

def main():
    tb_path = (REPO / "results/diagnostics/20261114_imu_eval_v2_motion"
               / "V8_IMU_Residual_TBiFoot_yaw_all.json")
    tb_all = json.loads(tb_path.read_text()) if tb_path.is_file() else None
    tb = tb_all.get("clean") if tb_all else None
    res_rows = {}
    for mode, cond_label in MODES:
        rec = {"模型": "V8_IMU_Residual", "条件": cond_label, "n": 4}
        for key, label in MAIN:
            if mode == "T2M" and tb is not None and label in tb["mean"]:
                # T channel is one whole: T2M pose + E9 BiFoot trajectory.
                rec[label] = "%.3f±%.3f" % (tb["mean"][label], tb["std"][label])
                continue
            rec[label] = fmt([json.loads((SRC / ("%s.json" % f)).read_text())
                              ["metrics"][mode].get(key) for f in FOLDS])
        res_rows[cond_label] = rec
    base_rows = load_baselines()
    rows = []
    for cond_label in ("U@V", "U@T", "U@VT"):
        rows.extend([r for r in base_rows if r.get("条件") == cond_label])
        rows.append(res_rows[cond_label])
    fields = ["模型", "条件", "n"] + [x for _, x in MAIN]
    p = OUT / "Main_Table1_E2_MainPerformance.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", p)

    # detail: per-mode full metrics + per-fold
    detail = []
    for mode, cond_label in MODES:
        rec = {"模型": "V8_IMU_Residual", "条件": cond_label}
        for key, label in DETAIL:
            rec[label] = fmt([json.loads((SRC / ("%s.json" % f)).read_text())
                              ["metrics"][mode].get(key) for f in FOLDS])
        detail.append(rec)
    df = ["模型", "条件"] + [x for _, x in DETAIL]
    dp = OUT / "detail" / "Main_Table1_E2_MainPerformance.md"
    lines = ["# E2 原生通道诊断明细（V8_IMU_Residual，Clean，四折 mean±std）", "",
             "| " + " | ".join(df) + " |", "|" + "---|" * len(df)]
    for r in detail:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in df) + " |")
    dp.write_text("\n".join(lines) + "\n")
    print("wrote", dp)

    per = []
    for mode, cond_label in MODES:
        for f in FOLDS:
            d = json.loads((SRC / ("%s.json" % f)).read_text())["metrics"][mode]
            rec = {"模型": "V8_IMU_Residual", "条件": cond_label, "fold": f}
            for key, label in DETAIL:
                rec[label] = d.get(key)
            per.append(rec)
    pf = ["模型", "条件", "fold"] + [x for _, x in DETAIL]
    pp = OUT / "detail" / "E2_MainPerformance_perFold.csv"
    with pp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pf)
        w.writeheader()
        for r in per:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in pf})
    print("wrote", pp)

if __name__ == "__main__":
    main()
