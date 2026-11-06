#!/usr/bin/env python3
"""E5 四折遮挡的 S13 敏感性（2026-11-05）。

背景：11 号文档裁定 S13 只作 train/val、不进留出 test；fS6S7S13 折的
E5 occ 格仍池化了 S13（25 格待裁定）。本脚本用 session_detail_test.json
的逐会话 PA-MPJPE 按 n_frames 重池化（主会话已验证与 test.json 逐位一致），
给出「含 S13 / 剔 S13」的四折均值对照，只作敏感性，不改主表。

产物：results_display/Etest_crossfit/E5_OcclusionRobustness/Supplementary/
      Full_Table1_E5_Crossfit_S13Sensitivity_PA.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
OUT = (REPO / "results_display" / "Etest_crossfit" / "E5_OcclusionRobustness"
       / "Supplementary")
FAMS = ("V7_Cv5", "V7_Cv5SD2", "V7_Cv5SD2RT", "V7_Cv5_tphys", "V7_Cv7")
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")
LEVELS = ("000", "020", "040", "060", "080", "100")
KEY = "pa_mpjpe_mm"


def cell(fam: str, fold: str, level: str) -> str:
    base = "%s_%s_s1" % (fam, fold)
    return base if level == "000" else "%s_evalocc%s" % (base, level)


def fold_delta(fam: str, fold: str, level: str, exclude_s13: bool):
    p = CELLS / cell(fam, fold, level) / "metrics" / "session_detail_test.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))["conditions"]
    vt = {e["session_id"]: e for e in d["VT2M"]}
    v = {e["session_id"]: e for e in d["V2M"]}
    sids = [s for s in sorted(set(v) & set(vt))
            if not (exclude_s13 and s.startswith("S13"))]
    if not sids:
        return None
    w = np.array([float(v[s]["n_frames"]) for s in sids])
    delta = np.array([vt[s]["metrics"][KEY] - v[s]["metrics"][KEY] for s in sids])
    return float((w * delta).sum() / w.sum())


def main() -> int:
    rows = []
    for fam in FAMS:
        for level in LEVELS:
            cur, exc = [], []
            for fold in FOLDS:
                a = fold_delta(fam, fold, level, False)
                if a is not None:
                    cur.append(a)
                if fold == "fS6S7S13":
                    b = fold_delta(fam, fold, level, True)
                    exc.append(b if b is not None else a)
                else:
                    exc.append(a)
            rows.append([fam, "occ" + level,
                         round(float(np.mean(cur)), 3),
                         round(float(np.mean(exc)), 3),
                         round(float(np.mean(exc)) - float(np.mean(cur)), 3)])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "Full_Table1_E5_Crossfit_S13Sensitivity_PA.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["模型", "档位", "4折均值_含S13", "4折均值_剔S13",
                    "Δ(剔−含)"])
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (p, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
