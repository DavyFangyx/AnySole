#!/usr/bin/env python3
"""修改 15 门控 oracle 前沿诊断（只诊断、不作结果汇报）：

1. 拟合集（A 臂 train+val × {000,040,100}）落盘 + 关键分位数：
   clean(000) ig_s 的 p05/p10（左尾）与 occ100 ig_s 的 p50/p75/p90/p95
   （右尾）——margin 抬 b = 阈值 −b/a 左移，撞进右尾多少，跑前即可预判。
2. test 逐窗数据落盘（test × 六档，A 头）。
3. **oracle 扫描（作弊、仅诊断）**：a 固定 32，在 test 上扫 b ∈ [−4,8]，
   逐 b 算 000 税与 100 档兑现率 → 税–兑现前沿；检查预注册目标
   （000 税 ≤ +0.5 且 100 兑现 ≥ 50%）是否在前沿可达域内：
   - 可达 → 校准泛化问题（拟合的 b 落点不对），改留出/交叉拟合；
   - 不可达 → 方法天花板（单阈值标量 IG 的 clean 左尾与 occ100 右尾
     在受试者偏移下不可分），换门信号（按受试者归一化 / 时间平滑 /
     多特征），不再调 margin。

用法：python results_display/script/ig_oracle_frontier.py --fold fS5S6S11 --device cuda:2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "results_display/script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402
from ig_contact_pilot import ContactHead  # noqa: E402
from ig9_gate_smoke import (collect_gate_rows, ema_by_session,  # noqa: E402
                            gate_mix, single_session_loader)
from pa_grid_oracle import const_w  # noqa: E402
from stage2_gate_pilot import win_lower_pa  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
FIT_LEVELS = ("000", "040", "100")
OUT = REPO / "results/diagnostics/20261107_stage2_vig"
FOLD_CSV = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12.csv",
    "fS5S11S12": "splits_test_S5S11S12_val_S13.csv",
    "fS6S7": "splits_test_S6S7_val_S14.csv",
    "fS5S6S8": "splits_test_S5S6S8_val_S7.csv",
}


def fold_sessions(csv_name, split):
    rows = list(csv.DictReader(open(
        REPO / "AnysoleWorkspace/protocol/splits/default" / csv_name)))
    return [r[split] for r in rows if r[split]]


def collect_levels(net, head, cfg_path, splits, levels, device, bbox_cache,
                   batch_size, tp, tag, save_name=None):
    data = {}
    for level in levels:
        cfg_l = ("anysole/configs/v1_fold_%s.yaml" % cfg_path if level == "000"
                 else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, cfg_path))
        parts = []
        for split in splits:
            loader = build_loader(cfg_l, split, batch_size, t_phys_v2=tp)
            parts.append(collect_gate_rows(net, head, loader, device, bbox_cache,
                                           split, level))
        r = {k: np.concatenate([p[k] for p in parts])
             for k in ("session", "p_v", "p_t", "gt", "e_vt", "ig",
                       "rtm", "v_tok", "contact")}
        r["ig_s"] = ema_by_session(r["ig"], r["session"])
        data[level] = r
        print("[%s/%s] n=%d" % (tag, level, len(r["ig"])))
    if save_name:
        np.savez_compressed(
            OUT / save_name,
            level=np.concatenate([np.full(len(data[l]["ig"]), int(l)) for l in levels]),
            session=np.concatenate([data[l]["session"] for l in levels]),
            ig=np.concatenate([data[l]["ig"] for l in levels]),
            ig_s=np.concatenate([data[l]["ig_s"] for l in levels]),
            rtm=np.concatenate([data[l]["rtm"] for l in levels]),
            e_vt=np.concatenate([data[l]["e_vt"] for l in levels]),
            p_v=np.concatenate([data[l]["p_v"] for l in levels]),
            p_t=np.concatenate([data[l]["p_t"] for l in levels]),
            gt=np.concatenate([data[l]["gt"] for l in levels]),
            v_tok=np.concatenate([data[l]["v_tok"] for l in levels]),
            contact=np.concatenate([data[l]["contact"] for l in levels]))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--a", type=float, default=32.0)
    args = ap.parse_args()
    device = torch.device(args.device)
    fold_yaml = args.fold[1:]
    csv_name = FOLD_CSV[args.fold]
    net, ckpt, tp = build_model("V8_Cv5_%s_s1" % args.fold, device)
    print("[ckpt] %s" % ckpt)
    bbox_cache = {}
    head = ContactHead(256).to(device)
    head.load_state_dict(torch.load(OUT / ("contact_head_%s_A.pt" % args.fold),
                                    map_location="cpu"))
    head.eval(); head.requires_grad_(False)

    # 1. 拟合集落盘 + 分位数
    fit = collect_levels(net, head, fold_yaml, ("train", "val"), FIT_LEVELS,
                         device, bbox_cache, args.batch_size, tp, "fit",
                         save_name="ig_frontier_fit_%s.npz" % args.fold)
    quant = {}
    for lvl in FIT_LEVELS:
        ig = fit[lvl]["ig_s"]
        quant[lvl] = {"p05": float(np.percentile(ig, 5)),
                      "p10": float(np.percentile(ig, 10)),
                      "p50": float(np.median(ig)),
                      "p75": float(np.percentile(ig, 75)),
                      "p90": float(np.percentile(ig, 90)),
                      "p95": float(np.percentile(ig, 95))}
    print("[fit-quant] clean p05/p10=%.2f/%.2f | occ100 p50/p75/p90/p95=%.2f/%.2f/%.2f/%.2f"
          % (quant["000"]["p05"], quant["000"]["p10"], quant["100"]["p50"],
             quant["100"]["p75"], quant["100"]["p90"], quant["100"]["p95"]))

    # 2. test 落盘（六档）
    test_sessions = fold_sessions(csv_name, "test")
    test = {}
    for level in LEVELS:
        cfg_l = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                 else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
        loader = single_session_loader(cfg_l, test_sessions, args.batch_size, tp)
        r = collect_gate_rows(net, head, loader, device, bbox_cache, "test", level)
        r["ig_s"] = ema_by_session(r["ig"], r["session"])
        test[level] = r
        print("[test/%s] n=%d" % (level, len(r["ig"])))
    np.savez_compressed(
        OUT / ("ig_frontier_test_%s.npz" % args.fold),
        level=np.concatenate([np.full(len(test[l]["ig"]), int(l)) for l in LEVELS]),
        session=np.concatenate([test[l]["session"] for l in LEVELS]),
        ig=np.concatenate([test[l]["ig"] for l in LEVELS]),
        ig_s=np.concatenate([test[l]["ig_s"] for l in LEVELS]),
        rtm=np.concatenate([test[l]["rtm"] for l in LEVELS]),
        e_vt=np.concatenate([test[l]["e_vt"] for l in LEVELS]),
        p_v=np.concatenate([test[l]["p_v"] for l in LEVELS]),
        p_t=np.concatenate([test[l]["p_t"] for l in LEVELS]),
        gt=np.concatenate([test[l]["gt"] for l in LEVELS]),
        v_tok=np.concatenate([test[l]["v_tok"] for l in LEVELS]),
        contact=np.concatenate([test[l]["contact"] for l in LEVELS]))

    # 3. oracle 扫描：a 固定，b 在 test 上扫
    w_const = {}
    for lvl in LEVELS:
        if lvl in fit:
            w_const[lvl], _ = const_w(fit[lvl]["p_v"], fit[lvl]["p_t"], fit[lvl]["gt"])
        else:
            w_const[lvl] = float("nan")
    a = args.a
    frontier = []
    for b in np.arange(-4.0, 8.01, 0.25):
        row = {"b": float(b)}
        for lvl in LEVELS:
            r = test[lvl]
            z = a * r["ig_s"] + b
            g = 1.0 / (1.0 + np.exp(-z))
            e_mix = win_lower_pa(gate_mix(g, r["p_t"], r["p_v"]), r["gt"])
            row["tax_%s" % lvl] = float(np.mean(e_mix - r["e_vt"]))
            row["gain_%s" % lvl] = float(np.mean(r["e_vt"] - e_mix))
        # 100 档档位常数收益（拟合集 w）
        lc_mix = gate_mix(np.full(len(test["100"]["ig"]), w_const["100"]),
                          test["100"]["p_t"], test["100"]["p_v"])
        e_lc = win_lower_pa(lc_mix, test["100"]["gt"])
        gain_lc = float(np.mean(test["100"]["e_vt"] - e_lc))
        row["realization_100"] = row["gain_100"] / gain_lc if gain_lc > 0.5 else None
        frontier.append(row)
    best_tax = min(frontier, key=lambda r: abs(r["tax_000"]))
    feasible = [r for r in frontier
                if r["tax_000"] <= 0.5 and r["realization_100"] is not None
                and r["realization_100"] >= 0.5]
    best_feasible = None
    if feasible:
        best_feasible = max(feasible, key=lambda r: r["realization_100"])
    print("[frontier] b sweep −4..8 (a=%.0f): %d points" % (a, len(frontier)))
    print("[frontier] target reachable: %s" % ("YES" if feasible else "NO"))
    if best_feasible:
        r = best_feasible
        print("[frontier] best feasible b=%.2f tax000=%+.1f real100=%.0f%% gain100=%+.1f"
              % (r["b"], r["tax_000"], 100 * r["realization_100"], r["gain_100"]))
    else:
        # 报告两个端点：最优税（无兑现约束）与最优兑现（税≤X 的最优点）
        r1 = min(frontier, key=lambda r: abs(r["tax_000"]))
        print("[frontier] min|tax| b=%.2f tax000=%+.1f real100=%s" % (
            r1["b"], r1["tax_000"],
            "%.0f%%" % (100 * r1["realization_100"]) if r1["realization_100"] else "n/a"))
        r2 = max(frontier, key=lambda r: r["realization_100"] or -99)
        print("[frontier] max real b=%.2f tax000=%+.1f real100=%s" % (
            r2["b"], r2["tax_000"],
            "%.0f%%" % (100 * r2["realization_100"]) if r2["realization_100"] else "n/a"))
    (OUT / ("ig_frontier_%s.json" % args.fold)).write_text(
        json.dumps({"fold": args.fold, "a": a, "fit_quantiles": quant,
                    "w_const_100": w_const["100"],
                    "frontier": frontier, "target_reachable": bool(feasible),
                    "best_feasible": best_feasible},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("[out] %s" % (OUT / ("ig_frontier_%s.json" % args.fold)))


if __name__ == "__main__":
    main()
