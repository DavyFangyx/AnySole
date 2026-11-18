#!/usr/bin/env python3
"""E5 visual-degradation summary for the new Residual model (mean values)."""
from __future__ import annotations
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate/E5_VisualDegradation_Summary"
OCC = REPO / "results_display/Etest_crossfit_physimu_vgate/E5_OcclusionRobustness/Main_Table2_E5_Occlusion_MainVsBaselines.csv"
SCN = REPO / "results_display/Etest_crossfit_physimu_vgate/E5_SceneOcclusion/Main_Table2_E5_SceneOcclusion_MainVsBaselines.csv"
GAP = REPO / "results_display/Etest_crossfit_physimu_vgate/E5_TimeGap/Main_Table2_E5_TimeGap_MainVsBaselines.csv"
METRICS = (("PA-MPJPE", "PA-MPJPE"), ("ankle_foot_mpjpe", "Ankle-foot MPJPE"),
           ("GlobalMPJPE", "GlobalMPJPE"), ("FootSliding", "FootSliding"),
           ("GroundPen", "GroundPen"), ("traj_ATE", "traj_ATE"))
MODELS = ("V8_IMU_Residual @VT", "V8_IMU_Residual @V", "V8_IMU_Residual @T",
          "WHAM @V", "GVHMR @V", "MotionPRO @VT")
COLS = (("2D 0%", OCC, "occ000"), ("2D 50%", OCC, "occ050"),
        ("2D 90%", OCC, "occ090"), ("Gap 0%", GAP, "gap000"),
        ("Gap 40%", GAP, "gap040"), ("Gap 60%", GAP, "gap060"),
        ("Scene clean", SCN, "scene_clean"), ("Scene occl.", SCN, "scene"))

def mean(cell):
    if not cell:
        return None
    return cell.split("±")[0]

def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))

def main():
    tables = {str(p): load(p) for _, p, _ in COLS}
    rows = []
    for key, label in METRICS:
        for model in MODELS:
            rec = {"指标": label, "模型": model}
            for col, path, level in COLS:
                vals = [r for r in tables[str(path)]
                        if r["模型"] == model and r["档位"] == level]
                rec[col] = mean(vals[0].get(key, "")) if vals else None
            rows.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["指标", "模型"] + [c for c, _, _ in COLS]
    p = OUT / "Main_Table_E5_VisualDegradation_Main.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in fields})
    print("wrote", p)
    lines = ["# E5 视觉退化汇总（V8_IMU_Residual）", "",
             "| 指标 | 模型 | " + " | ".join(c for c, _, _ in COLS) + " |",
             "|---|---|" + "---:|" * len(COLS)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |")
    (OUT / "Main_Table_E5_VisualDegradation_Main.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "Main_Table_E5_VisualDegradation_Main.md")

if __name__ == "__main__":
    main()
