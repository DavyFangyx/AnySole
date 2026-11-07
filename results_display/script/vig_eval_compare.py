#!/usr/bin/env python3
"""V8_Cv5VIG 评估对比（修改 15）：三块口径。

1. base clean（v1_fold 折配置 test）：VIG vs V8_Cv5（回归，PA 税应 ≤ +0.5）
   与 vs V8_Cv7（「视觉正常还打不打得过 CV7」）。
2. E5 场景遮挡（v1_sceneocc 折配置 test）：VIG vs V7_Cv5 全量参照
   （20261107_e5_sceneocc_full），逐折 VT2M/V2M/T2M + scene−clean 退化。
3. E6 occ 档位（v1_occXXX 折配置 test）：VIG vs V8_Cv5 evalocc cells。

只读，不写评估产物；输出 results/diagnostics/20261107_stage2_vig/compare.json。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
VIG_ROOT = REPO / "results/diagnostics/20261107_stage2_vig"
C5_SCENE = REPO / "results/diagnostics/20261107_e5_sceneocc_full"
CELLS = REPO / "results/cells"
METRICS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE（GT Anchored）")
CONDS = ("V2M", "VT2M", "T2M")
LEVELS = ("020", "040", "060", "080", "100")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def conds(path: Path) -> dict:
    return load(path)["metrics"]


def cell_metrics(cell: str) -> dict:
    return conds(CELLS / cell / "metrics/test.json")


def vig_eval(fold: str, name: str) -> dict:
    return conds(VIG_ROOT / ("eval_%s" % fold) / name / "metrics/test.json")


def mean_folds(vals):
    return float(np.mean([v for v in vals if v is not None]))


def main():
    report = {"folds": list(FOLDS),
              "clean": {"per_fold": {}, "VIG": {}, "Cv5": {}, "Cv7": {}},
              "scene": {"per_fold": {}}, "occ": {"per_fold": {}}
              }

    # 1. base clean
    print("== 1. base clean (VT2M / PA-MPJPE) ==")
    print("%-10s %8s %8s %8s | %6s %6s" % ("fold", "VIG", "Cv5", "Cv7", "dCv5", "dCv7"))
    for fold in FOLDS:
        vig = vig_eval(fold, "clean")
        c5 = cell_metrics("V8_Cv5_%s_s1" % fold)
        c7 = cell_metrics("V8_Cv7_%s_s1" % fold)
        rec = {}
        for m in METRICS:
            v, a, b = (vig["VT2M"][m], c5["VT2M"][m], c7["VT2M"][m])
            rec[m] = {"VIG": v, "Cv5": a, "Cv7": b,
                      "d_vs_Cv5": v - a, "d_vs_Cv7": v - b}
        report["clean"]["per_fold"][fold] = rec
        pa = rec["PA-MPJPE"]
        print("%-10s %8.2f %8.2f %8.2f | %+6.2f %+6.2f"
              % (fold, pa["VIG"], pa["Cv5"], pa["Cv7"], pa["d_vs_Cv5"], pa["d_vs_Cv7"]))
    for key in ("VIG", "Cv5", "Cv7"):
        report["clean"][key] = {
            m: mean_folds([report["clean"]["per_fold"][f][m][key] for f in FOLDS])
            for m in METRICS}
    d5 = mean_folds([report["clean"]["per_fold"][f]["PA-MPJPE"]["d_vs_Cv5"] for f in FOLDS])
    d7 = mean_folds([report["clean"]["per_fold"][f]["PA-MPJPE"]["d_vs_Cv7"] for f in FOLDS])
    print("mean     %8.2f %8.2f %8.2f | %+6.2f %+6.2f\n"
          % (report["clean"]["VIG"]["PA-MPJPE"], report["clean"]["Cv5"]["PA-MPJPE"],
             report["clean"]["Cv7"]["PA-MPJPE"], d5, d7))

    # 2. E5 scene occlusion
    print("== 2. E5 scene occ (VT2M / PA-MPJPE; scene − clean) ==")
    print("%-10s %12s %12s | %8s | %8s %8s" % ("fold", "VIG_scene", "Cv5_scene",
                                               "dVIG", "VIG_deg", "Cv5_deg"))
    for fold in FOLDS:
        vig_s = vig_eval(fold, "sceneocc_scene")
        vig_c = vig_eval(fold, "sceneocc_clean")
        c5_s = conds(C5_SCENE / ("%s_scene" % fold) / "metrics/test.json")
        c5_c = conds(C5_SCENE / ("%s_clean" % fold) / "metrics/test.json")
        rec = {}
        for cond in CONDS:
            rec[cond] = {m: {"VIG_scene": vig_s[cond][m],
                             "Cv5_scene": c5_s[cond][m],
                             "VIG_clean": vig_c[cond][m],
                             "Cv5_clean": c5_c[cond][m],
                             "d_scene": vig_s[cond][m] - c5_s[cond][m],
                             "VIG_deg": vig_s[cond][m] - vig_c[cond][m],
                             "Cv5_deg": c5_s[cond][m] - c5_c[cond][m]}
                         for m in METRICS}
        report["scene"]["per_fold"][fold] = rec
        pa = rec["VT2M"]["PA-MPJPE"]
        print("%-10s %12.2f %12.2f | %+8.2f | %8.2f %8.2f"
              % (fold, pa["VIG_scene"], pa["Cv5_scene"], pa["d_scene"],
                 pa["VIG_deg"], pa["Cv5_deg"]))
    for cond in CONDS:
        for m in METRICS:
            report["scene"]["mean_%s_%s" % (cond, m)] = {
                "VIG_scene": mean_folds([report["scene"]["per_fold"][f][cond][m]["VIG_scene"] for f in FOLDS]),
                "Cv5_scene": mean_folds([report["scene"]["per_fold"][f][cond][m]["Cv5_scene"] for f in FOLDS]),
                "d_scene": mean_folds([report["scene"]["per_fold"][f][cond][m]["d_scene"] for f in FOLDS]),
                "VIG_deg": mean_folds([report["scene"]["per_fold"][f][cond][m]["VIG_deg"] for f in FOLDS]),
                "Cv5_deg": mean_folds([report["scene"]["per_fold"][f][cond][m]["Cv5_deg"] for f in FOLDS])}
    m = report["scene"]["mean_VT2M_PA-MPJPE"]
    print("mean     %12.2f %12.2f | %+8.2f | %8.2f %8.2f\n"
          % (m["VIG_scene"], m["Cv5_scene"], m["d_scene"], m["VIG_deg"], m["Cv5_deg"]))

    # 3. E6 occ levels
    print("== 3. E6 occ levels (VT2M / PA-MPJPE; VIG − Cv5) ==")
    hdr = "%-10s " % "fold" + "".join("%9s" % l for l in LEVELS)
    print(hdr)
    for fold in FOLDS:
        row = []
        for level in LEVELS:
            try:
                vig = vig_eval(fold, "occ%s" % level)["VT2M"]["PA-MPJPE"]
                c5 = cell_metrics("V8_Cv5_%s_s1_evalocc%s" % (fold, level))["VT2M"]["PA-MPJPE"]
                row.append(vig - c5)
            except FileNotFoundError:
                row.append(None)
        report["occ"]["per_fold"][fold] = {l: (None if row[i] is None else float(row[i]))
                                           for i, l in enumerate(LEVELS)}
        print("%-10s " % fold + "".join("%+9.2f" % (v if v is not None else float("nan")) for v in row))
    report["occ"]["mean"] = {l: mean_folds([report["occ"]["per_fold"][f][l] for f in FOLDS])
                             for l in LEVELS}
    print("%-10s " % "mean" + "".join("%+9.2f" % report["occ"]["mean"][l] for l in LEVELS))

    out = VIG_ROOT / "compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n[out] %s" % out)


if __name__ == "__main__":
    main()
