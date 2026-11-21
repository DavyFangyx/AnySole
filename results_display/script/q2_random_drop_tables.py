#!/usr/bin/env python3
"""Q2 随机 modality dropout 对照表（V/T/VT 三行 × 条件，四折 mean±std）。

主模型（enum FT）= results/diagnostics/20261114_imu_eval_v2_all/V8_IMU_Residual
随机臂（random FT）= results/diagnostics/20261119_random_drop_eval/V8_IMU_RandomDrop
口径与 E4-B 一致：PA-MPJPE / GlobalMPJPE / GroundPen（GT Anchored）；detail 含 MPJPE。
"""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO / "results/diagnostics/20261114_imu_eval_v2_all/V8_IMU_Residual"
RAND = REPO / "results/diagnostics/20261119_random_drop_eval/V8_IMU_RandomDrop"
OUT = REPO / ("results_display/Etest_crossfit_physimu_vgate/Supplement/"
              "02_姿态恢复与训练策略/Q2：数据增强的方式使用的是三种配置枚举/实验结果")
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
CONDS = ("clean", "occ020", "occ050", "occ090", "sceneocc")
ROWS = ("VT2M", "V2M", "T2M")
ROW_LABELS = {"VT2M": "VT", "V2M": "V", "T2M": "T"}
METRICS = (
    ("PA-MPJPE", "pa"),
    ("GlobalMPJPE", "global"),
    ("GroundPen（GT Anchored）", "groundpen"),
)
DETAIL_EXTRA = (("MPJPE", "mpjpe"),)

ARMS = (
    ("main", MAIN, "三档枚举 FT（主模型 V8_IMU_Residual）"),
    ("random", RAND, "随机 modality dropout FT（V8_IMU_RandomDrop）"),
)


def read_metric(root: Path, cond: str, fold: str, row: str, key: str):
    for name in ("%s.json" % fold, "%s_brief.json" % fold):
        p = root / cond / name
        if p.is_file():
            d = json.loads(p.read_text())
            return d.get("metrics", {}).get(row, {}).get(key)
    return None


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt(v, s):
    return "-" if v is None else "%.2f±%.2f" % (v, s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Q2 对照：三档枚举 FT vs 随机 modality dropout FT", "",
          "四折 mean±std；V/T/VT 三行；PA-MPJPE / GlobalMPJPE / GroundPen（mm）。"
          "两臂唯一差异为训练监督方式（每窗枚举三档 vs 每窗随机采样一档），"
          "基座（enum 740ep）、遮挡条件、epoch、val 选点准则完全一致。", ""]
    per_fold = []
    for tag, root, label in ARMS:
        md += ["## %s" % label, ""]
        for row in ROWS:
            md += ["### %s 行" % ROW_LABELS[row], ""]
            md.append("| 条件 | " + " | ".join(lab for _, lab in
                       (("PA-MPJPE", "PA-MPJPE"), ("GlobalMPJPE", "GlobalMPJPE"),
                        ("GroundPen（GT Anchored）", "GroundPen"))) + " |")
            md.append("|---|" + "---:|" * 3)
            for cond in CONDS:
                cells = []
                for key, short in METRICS:
                    m, s, _ = agg([read_metric(root, cond, f, row, key)
                                   for f in FOLDS])
                    cells.append(fmt(m, s))
                md.append("| %s | %s |" % (cond, " | ".join(cells)))
            md.append("")
        for cond in CONDS:
            for fold in FOLDS:
                for row in ROWS:
                    rec = {"arm": tag, "row": row, "condition": cond,
                           "fold": fold}
                    for key, short in METRICS + DETAIL_EXTRA:
                        rec[short] = read_metric(root, cond, fold, row, key)
                    per_fold.append(rec)
        md.append("")
    (OUT / "Q2_enum_vs_random_drop.md").write_text("\n".join(md) + "\n")

    fields = (["arm", "row", "condition", "fold"]
              + [s for _, s in METRICS] + [s for _, s in DETAIL_EXTRA])
    with (OUT / "Q2_enum_vs_random_drop_perFold.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in per_fold:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
