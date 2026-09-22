#!/usr/bin/env python3
"""R_Test10 单模态训练对比（missing-rate 任务书实验 6）：数据整理脚本。

把主线 / V-only（--config-probs 0,1,0 训出）/ T-only（0,0,1 训出）三份
fseries 整理成逐配置 × 逐指标对比表（csv）。预期：T-only 赢接触/下肢，
V-only 赢全局/yaw——与 R_Test6 的互补 bar 互相印证。
（nodrop 模型可选 --fs-nodrop 一并入表，对应 R_Test7 C1 的训练侧。）

Usage (touch_gait env, 仓库根执行):
  python results_display/script/r_test10_singlemodal_compare.py \
      --fs-vonly results/AnySole/<vonly>/metrics/val_fseries.json \
      --fs-tonly results/AnySole/<tonly>/metrics/val_fseries.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

KEY_METRICS = ["PA-MPJPE", "MPJPE", "PA-MPJPE_lower", "PA-MPJPE_anklefoot",
               "PA-MPJPE_upper", "PA-MPJPE_hands", "RTE_norm", "yaw_abs_deg",
               "yaw_drift_deg", "contact_f1", "contact_acc", "air_recall",
               "foot_slide_mm", "seam_jump_mm", "jitter_mm",
               "joint_limit_viol_elbow", "joint_limit_viol_knee"]
CONFIGS = ("VT2M", "V2M", "T2M")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R_Test10 单模态训练对比表")
    parser.add_argument("--fs-main", type=Path,
                        default=REPO / "results" / "AnySole" / "V4B_joint_and" / "metrics" / "val_fseries.json")
    parser.add_argument("--fs-vonly", type=Path, default=None)
    parser.add_argument("--fs-tonly", type=Path, default=None)
    parser.add_argument("--fs-nodrop", type=Path, default=None,
                        help="可选：nodrop 模型（R_Test7 C1）一并入表")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "result" / "r_test10_singlemodal_compare",
                        help="产物目录")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    models = {}
    for name, path in (("main", args.fs_main), ("vonly", args.fs_vonly),
                       ("tonly", args.fs_tonly), ("nodrop", args.fs_nodrop)):
        if path is None:
            continue
        if not path.is_file():
            raise SystemExit("fseries 缺失：%s（%s）" % (path, name))
        models[name] = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    if len(models) < 2:
        raise SystemExit("至少需要 main + 一个对比模型（--fs-vonly/--fs-tonly）")

    rows = []
    for cfg in CONFIGS:
        for metric in KEY_METRICS:
            row = {"config": cfg, "metric": metric}
            for name, m in models.items():
                if metric in m.get(cfg, {}):
                    row[name] = round(m[cfg][metric], 4)
            if len(row) > 2:
                rows.append(row)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "singlemodal_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    names = list(rows[0].keys())[2:]
    print("R_Test10 对比表（%s）：" % ", ".join(names))
    header = "%-6s %-24s" % ("cfg", "metric") + "".join("%12s" % n for n in names)
    print(header)
    for r in rows:
        line = "%-6s %-24s" % (r["config"], r["metric"])
        line += "".join("%12s" % ("%.4f" % r[n] if n in r else "-") for n in names)
        print(line)
    print("产物：%s/singlemodal_compare.csv" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
