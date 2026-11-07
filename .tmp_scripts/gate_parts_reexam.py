#!/usr/bin/env python3
"""V7 门控组（B/LR/TLR/Cv7）部位限定效果复查（2026-11-07）。

问题：LR/TLR 的左右脚分组门被判「无效果」——是否被 PA 指标与全周期
平均掩盖？检验：window_detail_test.json 逐部位 PA × 动作 × 相位 分解，
对比 B=V7_Cv5 / LR / TLR / Cv7（无门）。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole_tta.phase_common import load_contact, phase_labels  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CELLS = REPO / "results" / "cells"
OUT = REPO / ".tmp_scripts" / "gate_parts_reexam_out"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ("V7_Cv5_s1", "V7_Cv5LR_s1", "V7_Cv5TLR_s1", "V7_Cv7_s1")
ACTION_GROUPS = {"03": "walk", "04": "walk", "05": "run", "08": "run",
                 "06": "squat", "01": "stand_shift", "02": "single_stand",
                 "07": "jump_jack", "09": "lateral", "10": "sit_stand",
                 "11": "stairs"}
PARTS = ("full", "upper", "lower", "anklefoot", "l_foot", "r_foot")
METRIC = {"full": "pa_mpjpe_mm", "upper": "PA-MPJPE_upper",
          "lower": "PA-MPJPE_lower", "anklefoot": "PA-MPJPE_anklefoot",
          "l_foot": "PA-MPJPE_l_foot", "r_foot": "PA-MPJPE_r_foot"}

_session_action: dict[str, str] = {}
with open(REPO / "AnysoleWorkspace/protocol/manifests/"
          "session_manifest.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        _session_action[row["session_id"]] = ACTION_GROUPS.get(
            row["action"], row["action"])


def load_cell(cell: str) -> dict:
    d = json.loads(
        (CELLS / cell / "metrics" / "window_detail_test.json").read_text())
    vt = d["conditions"]["VT2M"]
    v = d["conditions"]["V2M"]
    return {
        "sids": vt["session_ids"],
        "fs": np.asarray(vt["frame_starts"], dtype=int),
        "harm": {p: np.asarray(vt["metrics"][METRIC[p]], dtype=np.float64)
                 - np.asarray(v["metrics"][METRIC[p]], dtype=np.float64)
                 for p in PARTS},
        "vt": {p: np.asarray(vt["metrics"][METRIC[p]], dtype=np.float64)
               for p in PARTS},
        "v": {p: np.asarray(v["metrics"][METRIC[p]], dtype=np.float64)
              for p in PARTS},
    }


def main():
    cells = {m: load_cell(m) for m in MODELS}
    sids = cells[MODELS[0]]["sids"]
    contact = {}
    for sid in sorted(set(sids)):
        c, fid, valid = load_contact(sid)
        assert fid[0] == 0 and np.all(np.diff(fid) == 1)
        contact[sid] = (c, valid)

    # 每窗：动作、相位比例（逐部位窗口相同）
    fs = cells[MODELS[0]]["fs"]
    win_action = np.asarray([_session_action.get(s, "?") for s in sids])
    win_swing = np.zeros(len(sids), dtype=np.float32)
    win_phase = np.zeros((len(sids), 4), dtype=np.float32)  # 窗内四相位帧占比
    for i, (sid, s0) in enumerate(zip(sids, fs)):
        c, valid = contact[sid]
        ph = phase_labels(c)[s0:s0 + 20]
        vseg = valid[s0:s0 + 20].astype(bool)
        if vseg.sum() == 0:
            win_swing[i] = np.nan
            continue
        ph = ph[vseg]
        win_swing[i] = (ph == 1).mean()
        for k in range(4):
            win_phase[i, k] = (ph == k).mean()

    # 1) 池化 & 按动作：逐部位 harm 均值
    with open(OUT / "harm_by_action_part.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "action", "part", "n_win", "mean_harm",
                    "mean_V", "mean_VT"])
        for m in MODELS:
            for a in ("walk", "run", "lateral", "stairs", "squat",
                      "stand_shift", "jump_jack", "sit_stand",
                      "single_stand", "ALL"):
                sel = (win_action == a) if a != "ALL" else np.ones(
                    len(sids), bool)
                for p in PARTS:
                    h = cells[m]["harm"][p][sel]
                    w.writerow([m, a, p, int(sel.sum()),
                                round(float(h.mean()), 2),
                                round(float(cells[m]["v"][p][sel].mean()), 2),
                                round(float(cells[m]["vt"][p][sel].mean()), 2)])

    # 2) 相位加权：窗内相位占比加权分解（全池 + 按动作）
    with open(OUT / "harm_by_phase_part.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "action", "phase", "part", "mean_harm",
                    "mean_V", "mean_VT", "n_win"])
        for m in MODELS:
            for a in ("walk", "run", "lateral", "ALL"):
                sel = (win_action == a) if a != "ALL" else np.ones(
                    len(sids), bool)
                for k, pname in enumerate(
                        ("stance", "swing", "strike", "off")):
                    wts = win_phase[sel, k]
                    ok = wts > 0.5  # 该窗主要由该相位构成
                    if ok.sum() < 5:
                        continue
                    for p in PARTS:
                        h = cells[m]["harm"][p][sel][ok]
                        w.writerow([m, a, pname, p,
                                    round(float(h.mean()), 2),
                                    round(float(cells[m]["v"][p][sel][ok].mean()), 2),
                                    round(float(cells[m]["vt"][p][sel][ok].mean()), 2),
                                    int(ok.sum())])

    # 3) 门值（TLR npz）按动作读出
    print("=== TLR gate values by action (S_GateDiag npz) ===")
    npz = np.load(REPO / "results_display" / "Etest" / "S_GateDiag"
                  / "tlr_gate_diag.npz", allow_pickle=True)
    print("files:", npz.files)


if __name__ == "__main__":
    main()
