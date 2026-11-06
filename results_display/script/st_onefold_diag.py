#!/usr/bin/env python3
"""ST 一折风险诊断（2026-11-06）：下游指标 + g_t 的 held-out/train 分层。

用户判据（2026-11-06）：
  1. 下游：这一折的 U@V / U@VT / U@T、Δ(VT−V)，与同折现有 Cv5（无 ST）比。
  2. 机制：直接读 g_t（route_gate），按「是不是这一折的 held-out 受试者」
     分层——若推导成立，held-out 侧的 g_t 中位数显著高于训练内受试者。
     Cv5 对照同做一遍，ST 的特异位移 =
     (ST_test − ST_train) − (Cv5_test − Cv5_train)。

下游读 test.json（训练格自带 test eval）；门值用
`e6_gate_values_occlusion.py` 的 build_model/build_loader/run_level 复用，
在 fold yaml 的 train / test 两个 split 上各前向一次（VT 行，clean 档）。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/st_onefold_diag.py \
      --st-cell V8_Cv5ST_fS5S6S11_s1 --control-cell V8_Cv5_fS5S6S11_s1 \
      --fold fS5S6S11 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results_display" / "script"))

from e6_gate_values_occlusion import (  # noqa: E402
    GROUPS, GROUP_QUERIES, build_loader, build_model, run_level)

METRICS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE（GT Anchored）")
MODES = ("V2M", "VT2M", "T2M")
OUT = REPO / "results_display" / "Etest_crossfit" / "detail"


def load_test_metrics(cell: str) -> dict:
    path = REPO / "results" / "cells" / cell / "metrics" / "test.json"
    if not path.is_file():
        raise FileNotFoundError("missing test metrics (cell not finished?): %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def downstream_table(st: dict, control: dict) -> list[dict]:
    rows = []
    for metric in METRICS:
        vals = {}
        for tag, data in (("ST", st), ("Cv5", control)):
            v = float(data["V2M"][metric])
            vt = float(data["VT2M"][metric])
            vals[tag] = {"V": v, "VT": vt, "T": float(data["T2M"][metric]),
                         "dVT-V": vt - v}
        for key in ("V", "VT", "T", "dVT-V"):
            rows.append({
                "metric": metric, "column": key,
                "ST": vals["ST"][key], "Cv5": vals["Cv5"][key],
                "ST-Cv5": vals["ST"][key] - vals["Cv5"][key]})
    return rows


def _group_vals(gates: np.ndarray, queries) -> np.ndarray:
    return gates[..., list(queries)].reshape(-1)


def gate_rows(model: str, split: str, gates: np.ndarray, sids: list[str]) -> list[dict]:
    rows = []
    for k, name in enumerate(GROUPS):
        vals = _group_vals(gates, GROUP_QUERIES[k])
        rows.append({
            "model": model, "split": split, "group": name, "n": int(vals.size),
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
        })
    all_vals = gates.reshape(-1)
    rows.append({
        "model": model, "split": split, "group": "all", "n": int(all_vals.size),
        "mean": float(all_vals.mean()), "median": float(np.median(all_vals)),
        "p05": float(np.percentile(all_vals, 5)),
        "p95": float(np.percentile(all_vals, 95)),
    })
    return rows


def per_session_medians(model: str, split: str, gates: np.ndarray,
                        sids: list[str]) -> list[dict]:
    out = []
    for sid in sorted(set(sids)):
        idx = np.array([i for i, s in enumerate(sids) if s == sid])
        row = {"model": model, "split": split, "session": sid, "n_windows": int(idx.size)}
        for k, name in enumerate(GROUPS):
            row[name] = float(np.median(_group_vals(gates[idx], GROUP_QUERIES[k])))
        row["all"] = float(np.median(gates[idx].reshape(-1)))
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--st-cell", default="V8_Cv5ST_fS5S6S11_s1")
    ap.add_argument("--control-cell", default="V8_Cv5_fS5S6S11_s1")
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)

    # 1) 下游：同一 test split（held-out 受试者），ST vs 同折 Cv5。
    rows = downstream_table(load_test_metrics(args.st_cell),
                            load_test_metrics(args.control_cell))
    print("\n== downstream (test split = held-out subjects) ==")
    print("%-28s %-6s %10s %10s %10s" % ("metric", "col", "ST", "Cv5", "ST-Cv5"))
    for r in rows:
        print("%-28s %-6s %10.3f %10.3f %+10.3f"
              % (r["metric"], r["column"], r["ST"], r["Cv5"], r["ST-Cv5"]))

    # 2) 门值：同一 ckpt 在 fold 的 train / test 两个 split 上各前向一次。
    fold_yaml = args.fold[1:] if args.fold.startswith("f") else args.fold
    config_path = "anysole/configs/v1_fold_%s.yaml" % fold_yaml
    gate_rows_all, session_rows, medians = [], [], {}
    for model, cell in (("ST", args.st_cell), ("Cv5", args.control_cell)):
        net, ckpt_path, t_phys_v2 = build_model(cell, device)
        print("\n[ckpt:%s] %s" % (model, ckpt_path))
        medians[model] = {}
        for split in ("train", "test"):
            loader = build_loader(config_path, split, args.batch_size,
                                  t_phys_v2=t_phys_v2)
            gates, sids = run_level(net, loader, device, "000")
            gate_rows_all.extend(gate_rows(model, split, gates, sids))
            session_rows.extend(per_session_medians(model, split, gates, sids))
            medians[model][split] = {
                name: float(np.median(_group_vals(gates, GROUP_QUERIES[k])))
                for k, name in enumerate(GROUPS)}
            medians[model][split]["all"] = float(np.median(gates.reshape(-1)))
            print("[%s/%s] windows=%d  %s" % (
                model, split, gates.shape[0],
                " ".join("%s=%.3f" % (name, medians[model][split][name])
                         for name in (*GROUPS, "all"))))

    print("\n== g_t median: held-out vs train (same ckpt) ==")
    print("%-4s %-12s %10s %10s %10s" % ("model", "group", "train", "test", "test-train"))
    for model in ("ST", "Cv5"):
        for name in (*GROUPS, "all"):
            tr, te = medians[model]["train"][name], medians[model]["test"][name]
            print("%-4s %-12s %10.4f %10.4f %+10.4f"
                  % (model, name, tr, te, te - tr))
    for name in (*GROUPS, "all"):
        st_shift = (medians["ST"]["test"][name] - medians["ST"]["train"][name])
        cv_shift = (medians["Cv5"]["test"][name] - medians["Cv5"]["train"][name])
        print("ST-specific shift [%s]: %+.4f  (ST %+.4f - Cv5 %+.4f)"
              % (name, st_shift - cv_shift, st_shift, cv_shift))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "st_onefold_%s" % args.fold
    (out_dir / (stem + ".json")).write_text(json.dumps({
        "st_cell": args.st_cell, "control_cell": args.control_cell,
        "fold": args.fold, "downstream": rows,
        "gate_rows": gate_rows_all, "session_medians": session_rows,
        "medians": medians,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n[json] %s" % (out_dir / (stem + ".json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
