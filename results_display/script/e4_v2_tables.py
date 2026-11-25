#!/usr/bin/env python3
"""E4-A / E4-B tables from the formal V2 eval briefs (2026-11-15).

Main metrics: PA-MPJPE, GlobalMPJPE and GroundPen. MPJPE, RootAbs,
FootSliding and ankle MPJPE stay in detail.
"""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "results/diagnostics/20261125_e4_selfphys"
EVAL_OLD = REPO / "results/diagnostics/20261114_imu_eval_v2"
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate/E4_ComponentAblation"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
CONDS = ("clean", "occ020", "occ050", "occ090", "sceneocc")
METRICS = (
    ("PA-MPJPE", "pa", "PA-MPJPE"),
    ("GlobalMPJPE", "global", "GlobalMPJPE"),
    ("GroundPen（GT Anchored）", "groundpen", "GroundPen"),
)
DETAIL_EXTRA = (("MPJPE", "mpjpe"), ("RootAbs", "traj"),
                ("FootSliding（GT Anchored）", "footslide"),
                ("MPJPE_anklefoot", "ankle"))
FUSION = (
    ("V8_IMU_Residual", "Residual Fusion (Ours)", "enum base + occlusion-aware FT"),
    ("V8_IMU_Router", "Competitive routing", "router-only, frozen base"),
    ("V8_IMU_TGate", "Sensing-branch gating", "Cv5 dual T gate, 740 ep"),
    ("V8_IMU_DualGate", "Independent dual-branch gating", "frozen base + post-stage gate"),
)
TRAINING = (
    ("V8_IMU_Residual", "Residual Fusion (Ours)", "full training strategy"),
    ("V8_IMU_Residual_woAdapt", "w/o adaptation", "no FT, enum base"),
    ("V8_IMU_Residual_woEnum", "no-enum FT", "no enum pretrain, same FT"),
    ("V8_IMU_Residual_DecoderOnly", "decoder-only FT", "only pose_head"),
    ("V8_IMU_Residual_CleanOnly", "clean-only FT", "clean cache only"),
)

def read_metric(model, cond, fold, key):
    # 2026-11-25 重评树优先（含自锚物理列），旧 EV 树回落。
    p = EVAL / model / cond / ("%s.json" % fold)
    if not p.is_file():
        p = EVAL_OLD / model / cond / ("%s.json" % fold)
        if not p.is_file():
            p = EVAL_OLD / model / cond / ("%s_brief.json" % fold)
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    m = d.get("metrics", {}).get("VT2M", {})
    # 物理列优先自锚版（08 §3.3）；旧树无该字段时回落 GT Anchored。
    if key == "GroundPen（GT Anchored）" and "GroundPen" in m:
        return m.get("GroundPen")
    if key == "FootSliding（GT Anchored）" and "FootSliding" in m:
        return m.get("FootSliding")
    return m.get(key)

def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0), len(vals)

def rows_for(models):
    rows = []
    for model, label, adaptation in models:
        for cond in CONDS:
            rec = {"model": model, "label": label, "adaptation": adaptation,
                   "condition": cond, "n_folds": 0}
            for key, short, _ in METRICS:
                m, s, n = agg([read_metric(model, cond, f, key) for f in FOLDS])
                rec[short + "_mean"] = m
                rec[short + "_std"] = s
                rec["n_folds"] = max(rec["n_folds"], n)
            rows.append(rec)
    return rows

def write(rows, name):
    p = OUT / name
    fields = ["模型", "条件"] + [lab for _, _, lab in METRICS]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({"模型": r["label"], "条件": r["condition"],
                        **{lab: fmt(r[s + "_mean"], r[s + "_std"])
                           for _, s, lab in METRICS}})
    print("wrote", p)

def fmt(v, s):
    return "-" if v is None else "%.2f±%.2f" % (v, s)

def write_md(rows, name, title):
    lines = ["# " + title, "",
             "四折 mean±std，VT 行，PA-MPJPE / GlobalMPJPE / GroundPen（mm）。", ""]
    models = []
    for r in rows:
        if r["model"] not in [m[0] for m in models]:
            models.append((r["model"], r["label"], r["adaptation"]))
    for model, label, adaptation in models:
        lines += ["## %s" % label, ""]
        lines.append("| 条件 | " + " | ".join(lab for _, _, lab in METRICS) + " |")
        lines.append("|---|" + "---:|" * len(METRICS))
        for r in rows:
            if r["model"] != model:
                continue
            lines.append("| %s | %s |" % (r["condition"], " | ".join(
                fmt(r[s + "_mean"], r[s + "_std"]) for _, s, _ in METRICS)))
        lines.append("")
    (OUT / "detail" / name).write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "detail" / name)

def per_fold_rows(models):
    rows = []
    for model, label, adaptation in models:
        for cond in CONDS:
            for fold in FOLDS:
                rec = {"model": model, "label": label, "condition": cond, "fold": fold}
                for item in METRICS + DETAIL_EXTRA:
                    key, short = item[0], item[1]
                    rec[short] = read_metric(model, cond, fold, key)
                rows.append(rec)
    return rows

def write_per_fold(rows, name):
    p = OUT / "detail" / name
    fields = (["model", "label", "condition", "fold"]
              + [s for _, s, _ in METRICS] + [s for _, s in DETAIL_EXTRA])
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print("wrote", p)

if __name__ == "__main__":
    a = rows_for(FUSION)
    b = rows_for(TRAINING)
    write(a, "Main_Table1_E4_FusionComparison.csv")
    write(b, "Main_Table2_E4_TrainingAblation.csv")
    write_md(a, "Main_Table1_E4_FusionComparison.md", "E4-A 融合方案比较")
    write_md(b, "Main_Table2_E4_TrainingAblation.md", "E4-B 训练策略消融")
    write_per_fold(per_fold_rows(FUSION), "E4_FusionComparison_perFold.csv")
    write_per_fold(per_fold_rows(TRAINING), "E4_TrainingAblation_perFold.csv")
