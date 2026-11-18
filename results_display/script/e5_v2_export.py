#!/usr/bin/env python3
"""E5 tables for V8_IMU_Residual from the formal V2 eval (2026-11-15)."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "results/diagnostics/20261114_imu_eval_v2_all/V8_IMU_Residual"
OLD = REPO / "results_display/Etest_crossfit_v8"
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
TBI_PATH = (REPO / "results/diagnostics/20261114_imu_eval_v2_motion"
            / "V8_IMU_Residual_TBiFoot_yaw_all.json")
TBI_ALL = json.loads(TBI_PATH.read_text()) if TBI_PATH.is_file() else {}
TBI = TBI_ALL.get("clean")
TBI_SCEN = TBI_ALL
MODE_KEY = {"V2M": "V2M", "T2M": "T2M", "VT2M": "VT2M"}
METRICS = ("PA-MPJPE", "PA-MPJPE_anklefoot", "MPJPE", "MPJPE_anklefoot", "MPJRE",
           "GlobalMPJPE", "W-MPJPE100", "RootOrientationDrift", "RootAbs",
           "FootSliding（GT Anchored）", "GroundPen（GT Anchored）",
           "traj_ATE", "Jitter_pred")

BASE_ROOT = REPO / "results/baselines"
# 档位 -> each baseline model's predictions subdir name
LEVEL_TO_BASELINE_DIR = {
    **{("occ%03d" % i): ("e5_clean" if i == 0 else "e5_occ_lower%03d" % i)
       for i in range(0, 91, 10)},
    **{("gap%03d" % i): ("e5_clean" if i == 0 else "e5_gap_r%03d" % i)
       for i in (0, 10, 20, 40, 60)},
    "scene_clean": "e5_scene_clean", "scene": "e5_scene",
}


def baseline_mpjpe_mpjre(model, level):
    """MPJPE/MPJRE read directly from each baseline's raw eval json.

    Not present in the legacy Etest_crossfit_v8 MainVsBaselines CSV, so this
    pulls straight from predictions/ instead of going through baseline_rows().
    MotionPRO's fourfold_summary has no MPJRE; use the side file instead
    (2026-11-17，results/baselines/<model>/_e5_anklefoot_mpjre.json — the
    fourfold eval script never passed rotations into the canonical solver).
    """
    level_dir = LEVEL_TO_BASELINE_DIR.get(level)
    if level_dir is None:
        return "—", "—"
    if model == "MotionPRO":
        p = BASE_ROOT / model / "predictions" / level_dir / "metrics/fourfold_summary.json"
        if not p.is_file():
            return "—", "—"
        entry = json.loads(p.read_text(encoding="utf-8")).get("mpjpe_mm")
        mpjpe = "%.3f±%.3f" % (entry["mean"], entry["std"]) if entry and entry.get("n_folds", 0) >= 4 else "—"
        side = BASE_ROOT / model / "_e5_anklefoot_mpjre.json"
        mpjre = "—"
        if side.is_file():
            rec = json.loads(side.read_text(encoding="utf-8")).get(level_dir, {}).get("mpjae_deg")
            if rec and rec.get("n_folds", 0) >= 4:
                mpjre = "%.3f±%.3f" % (rec["mean"], rec["std"])
        return mpjpe, mpjre
    p = (BASE_ROOT / model / "predictions" / level_dir
         / "eval_motion/metrics/test_comparison.json")
    if not p.is_file():
        return "—", "—"
    summary = json.loads(p.read_text(encoding="utf-8")).get("summary", {})
    mpjpe = summary.get("mpjpe_mm")
    mpjre = summary.get("mpjae_deg")
    return ("%.3f" % mpjpe if mpjpe is not None else "—",
            "%.3f" % mpjre if mpjre is not None else "—")


def baseline_anklefoot_pa(model, level):
    """PA-MPJPE_anklefoot from the 2026-11-17 side-computation (none of the
    baselines' canonical solvers output a per-part PA-MPJPE)."""
    level_dir = LEVEL_TO_BASELINE_DIR.get(level)
    if level_dir is None:
        return "—"
    side = BASE_ROOT / model / "_e5_anklefoot_mpjre.json"
    if not side.is_file():
        return "—"
    rec = json.loads(side.read_text(encoding="utf-8")).get(level_dir, {})
    if model == "MotionPRO":
        v = rec.get("pa_mpjpe_anklefoot_mm")
        return "%.3f±%.3f" % (v["mean"], v["std"]) if v and v.get("n_folds", 0) >= 4 else "—"
    v = rec.get("pa_mpjpe_anklefoot_mm")
    return "%.3f" % v if v is not None else "—"

OCC = [("occ000", "clean", 0), ("occ010", "occ010", 10),
       ("occ020", "occ020", 20), ("occ030", "occ030", 30),
       ("occ040", "occ040", 40), ("occ050", "occ050", 50),
       ("occ060", "occ060", 60), ("occ070", "occ070", 70),
       ("occ080", "occ080", 80), ("occ090", "occ090", 90)]
SCENE = [("scene_clean", "sceneocc_clean", 0), ("scene", "sceneocc", 1)]
GAP = [("gap000", "clean", 0), ("gap010", "gap010", 10),
       ("gap020", "gap020", 20), ("gap040", "gap040", 40),
       ("gap060", "gap060", 60)]

def metric(cond, fold, mode, key):
    p = EVAL / cond / ("%s.json" % fold)
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    return d.get("metrics", {}).get(mode, {}).get(key)

def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

def fmt(vals):
    m, s = ms(vals)
    return "-" if m is None else "%.3f±%.3f" % (m, s)

def residual_row(cond, mode):
    return {k: fmt([metric(cond, f, mode, k) for f in FOLDS]) for k in METRICS}

def tb_row(cond, tbdata):
    """T channel as one whole: T2M pose + E9 BiFoot trajectory.

    Pose-only metrics (ankle MPJPE, MPJRE, RootOrientationDrift) come from the
    T2M row; trajectory/Global/GroundPen/FootSliding/W/Jitter come from the
    T+BiFoot combination. RootAbs is not defined for this mixed-anchor row.
    """
    out = residual_row(cond, "T2M")
    if tbdata is None:
        out["traj_ATE"] = "—"
        return out
    mapping = {
        "PA-MPJPE": "PA-MPJPE", "MPJPE": "MPJPE",
        "GlobalMPJPE": "GlobalMPJPE", "W-MPJPE100": "W-MPJPE100",
        "traj_ATE": "traj_ATE",
        "FootSliding（GT Anchored）": "FootSliding",
        "GroundPen（GT Anchored）": "GroundPen",
        "Jitter_pred": "Jitter_pred",
        "RootOrientationDrift": "RootOrientationDrift",
        "MPJPE_anklefoot": "MPJPE_anklefoot",
    }
    for k, src in mapping.items():
        out[k] = ("%.3f±%.3f" % (tbdata["mean"][src], tbdata["std"][src])
                  if src in tbdata["mean"] else "—")
    out["RootAbs"] = "—"
    return out

def residual_delta(cond):
    return {k: fmt([(metric(cond, f, "VT2M", k) or 0) - (metric(cond, f, "V2M", k) or 0)
                    for f in FOLDS]) for k in METRICS}

def baseline_rows(csv_path, level):
    out = []
    if not csv_path.is_file():
        return out
    for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        if r["档位"] != level:
            continue
        if r["模型"].startswith("V8_"):
            continue
        model = r["模型"].split(" ")[0]
        mpjpe, mpjre = baseline_mpjpe_mpjre(model, level)
        row = {k: r.get(k, "") for k in
               ("模型", "档位", "行", "PA-MPJPE", "ankle_foot_mpjpe",
                "GlobalMPJPE", "FootSliding", "GroundPen", "traj_ATE")}
        row["MPJPE"] = mpjpe
        row["MPJRE"] = mpjre
        row["PA-MPJPE_anklefoot"] = baseline_anklefoot_pa(model, level)
        out.append(row)
    return out

def write_table1(path, levels, tbdata=None):
    fields = ["模型", "档位", "行", "PA-MPJPE", "MPJPE", "MPJRE",
              "GlobalMPJPE", "W-MPJPE100", "traj_ATE", "RootOrientationDrift", "RootAbs",
              "GroundPen", "FootSliding", "Jitter_pred"]
    rows = []
    for level, cond, _ in levels:
        for mode in ("V2M", "T2M", "VT2M"):
            rec = tb_row(cond, tbdata) if (tbdata is not None and mode == "T2M") else residual_row(cond, mode)
            rows.append({"模型": "V8_IMU_Residual", "档位": level, "行": mode,
                         "PA-MPJPE": rec["PA-MPJPE"], "MPJPE": rec["MPJPE"],
                         "MPJRE": rec["MPJRE"],
                         "GlobalMPJPE": rec["GlobalMPJPE"],
                         "W-MPJPE100": rec["W-MPJPE100"],
                         "traj_ATE": (rec["traj_ATE"] if (tbdata is not None and mode == "T2M")
                                      else "—" if mode == "T2M" else rec["RootAbs"]),
                         "RootOrientationDrift": rec["RootOrientationDrift"],
                         "RootAbs": ("—" if mode == "T2M" else rec["RootAbs"]),
                         "GroundPen": rec["GroundPen（GT Anchored）"],
                         "FootSliding": rec["FootSliding（GT Anchored）"],
                         "Jitter_pred": rec["Jitter_pred"]})
        d = residual_delta(cond)
        rows.append({"模型": "V8_IMU_Residual", "档位": level, "行": "Δ(VT−V)",
                     "PA-MPJPE": d["PA-MPJPE"], "MPJPE": d["MPJPE"],
                     "MPJRE": d["MPJRE"], "GlobalMPJPE": d["GlobalMPJPE"],
                     "W-MPJPE100": d["W-MPJPE100"],
                     "traj_ATE": d["RootAbs"],
                     "RootOrientationDrift": d["RootOrientationDrift"],
                     "RootAbs": d["RootAbs"],
                     "GroundPen": d["GroundPen（GT Anchored）"],
                     "FootSliding": d["FootSliding（GT Anchored）"],
                     "Jitter_pred": d["Jitter_pred"]})
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", path)

def write_table2(path, levels, old_csv, tbdata=None):
    fields = ["模型", "档位", "行", "PA-MPJPE", "PA-MPJPE_anklefoot", "MPJPE",
              "MPJRE", "ankle_foot_mpjpe", "GlobalMPJPE", "FootSliding",
              "GroundPen", "traj_ATE"]
    rows = []
    for level, cond, _ in levels:
        for label, mode in (("@VT", "VT2M"), ("@V", "V2M"), ("@T", "T2M")):
            rec = tb_row(cond, tbdata) if (tbdata is not None and label == "@T") else residual_row(cond, mode)
            rows.append({"模型": "V8_IMU_Residual " + label, "档位": level,
                         "行": label,
                         "PA-MPJPE": rec["PA-MPJPE"],
                         "PA-MPJPE_anklefoot": rec["PA-MPJPE_anklefoot"],
                         "MPJPE": rec["MPJPE"],
                         "MPJRE": rec["MPJRE"],
                         "ankle_foot_mpjpe": rec["MPJPE_anklefoot"],
                         "GlobalMPJPE": rec["GlobalMPJPE"],
                         "FootSliding": rec["FootSliding（GT Anchored）"],
                         "GroundPen": rec["GroundPen（GT Anchored）"],
                         "traj_ATE": (rec["traj_ATE"] if (tbdata is not None and label == "@T")
                                      else "—" if label == "@T" else rec["RootAbs"])})
        rows.extend(baseline_rows(old_csv, level))
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", path)


def write_per_fold(scenario_dir, levels, tag):
    out = OUT / scenario_dir / "detail"
    out.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "condition", "fold"] + list(METRICS)
    rows = []
    for _, cond, _ in levels:
        for mode in ("V2M", "T2M", "VT2M"):
            for fold in FOLDS:
                rec = {"mode": mode, "condition": cond, "fold": fold}
                for key in METRICS:
                    rec[key] = metric(cond, fold, mode, key)
                rows.append(rec)
    p = out / ("E5_%s_perFold.csv" % tag)
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print("wrote", p)

def main():
    specs = [
        ("E5_OcclusionRobustness", OCC, OLD / "E5_OcclusionRobustness/Main_Table2_E5_Occlusion_MainVsBaselines.csv", "Occlusion", TBI),
        ("E5_SceneOcclusion", SCENE, OLD / "E5_SceneOcclusion/Main_Table2_E5_SceneOcclusion_MainVsBaselines.csv", "SceneOcclusion", TBI_SCEN.get("sceneocc")),
        ("E5_TimeGap", GAP, OLD / "E5_TimeGap/Main_Table2_E5_TimeGap_MainVsBaselines.csv", "TimeGap", TBI_SCEN.get("gap010")),
    ]
    for d, levels, old_csv, tag, tbdata in specs:
        out = OUT / d
        out.mkdir(parents=True, exist_ok=True)
        write_table1(out / ("Main_Table1_E5_%s_Models.csv" % tag), levels, tbdata)
        write_table2(out / ("Main_Table2_E5_%s_MainVsBaselines.csv" % tag), levels, old_csv, tbdata)
        write_per_fold(d, levels, tag)

if __name__ == "__main__":
    main()
