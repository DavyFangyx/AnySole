#!/usr/bin/env python3
"""E5 接触口径统一：同机制对照数字（2026-11-02 用户裁定「接触指标口径统一」）。

问题：cell 级 contact_acc 混两套机制——V2M 用足高判据（soft_contact 几何反推，
recall≈0.53），VT2M/T2M 用模型触觉接触头（V41ContactHead，recall≈0.91）。
Δ(VT−V) 因此混合「机制」与「模态」两个变量。

公平设计（三行两机制）：
- 足高判据行（V / VT 同机制）：对两条件的预测关节做同一 soft 接触反推。
  若 VT≈V，说明骨骼层面加触觉不改变接触反推——排除「关节修正副产物」质疑。
- 接触头行（仅 VT/T 推理有触觉 token 驱动）：V-only 无触觉通道、无接触头，
  故「V=足高判据 vs VT=接触头」即各条件最强接触感知方式的对比，
  差值 = 触觉通道本身的贡献。

本脚本从各 cell 的逐帧预测（predictions/eval_motion）离线池化足高判据接触
准确率（canonical test 36 会话，两脚），写入 e5_contact_kin.json 供
e_results_export.py 的表 3 消费。V2M 池化值与 cell 级 0.608 一致性已核验
（occ000 0.617，帧覆盖口径差异 0.9pp）。
用法（touch_gait env，CPU）：
  python results_display/script/e5_contact_kin.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from anysole_tta.e6_phase_analysis import build_session_sets, load_session  # noqa: E402

OUT = REPO / "results_display/Etest/E5_OcclusionRobustness/detail/e5_contact_kin.json"
OCC_CELLS = {"occ000": "V7_Cv5_s1"}
for _o in ("010", "020", "030", "040", "050", "060", "070", "080", "090", "100"):
    OCC_CELLS["occ" + _o] = "V7_Cv5_s1_evalocc" + _o


def pooled(sessions: list[str], cell: str) -> dict[str, dict]:
    acc = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in ("V2M", "VT2M")}
    for sid in sessions:
        d = load_session(cell, sid)
        n = len(d["phase"])
        valid = d["valid"][:n]
        gt = d["contact_gt"][:n]
        for cond in ("V2M", "VT2M"):
            cp = d[cond + "_cpred"][:n]
            a = acc[cond]
            a["tp"] += int((cp & gt & valid[:, None]).sum())
            a["fp"] += int((cp & ~gt & valid[:, None]).sum())
            a["fn"] += int((~cp & gt & valid[:, None]).sum())
            a["tn"] += int((~cp & ~gt & valid[:, None]).sum())
    out = {}
    for cond, a in acc.items():
        p = a["tp"] / max(a["tp"] + a["fp"], 1)
        r = a["tp"] / max(a["tp"] + a["fn"], 1)
        out[cond] = {
            "acc": (a["tp"] + a["tn"]) / max(sum(a.values()), 1),
            "f1": 2 * p * r / max(p + r, 1e-8),
            "recall": r,
            "air_recall": a["tn"] / max(a["tn"] + a["fp"], 1)}
    return out


def main() -> int:
    sessions = sorted({s for v in build_session_sets().values() for s in v})
    payload = {}
    for occ, cell in OCC_CELLS.items():
        payload[occ] = pooled(sessions, cell)
        print("[E5] %s %s: V_kin acc=%.3f / VT_kin acc=%.3f" %
              (occ, cell, payload[occ]["V2M"]["acc"], payload[occ]["VT2M"]["acc"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("→", OUT.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
