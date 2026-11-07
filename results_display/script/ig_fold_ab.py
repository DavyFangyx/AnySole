#!/usr/bin/env python3
"""修改 15 门控专家训练范围两臂对照（四折 × test 验收）。

A 臂：IG 接触头 + 门参数 (a,b) 在 **train+val** 上训练/拟合 → test。
B 臂：只在 **train** 上训练/拟合 → test。
两臂共用同一次 test 前向（tokens 落内存，两个头各算一次 IG）。

【口径（2026-11-07，预注册）】
- 头：ContactHead 逐帧 v_tok 训练（A=fold train+val，B=fold train）。
- 门：g = σ(a·IG_s + b)（阈值 logistic；前两版形式——无截距、中心化——
  均已证明数学上不可满足 p05 约束，见 ig_lso_gate.py 头注）；
  IG_s = EMA(α=0.7) 会话内；rtm 实测随档恒定（579）→ 剔除。
  拟合集 = 臂训练集 × {000,040,100}；网格 a ∈ {0.5..32}、b ∈ {−4..8}；
  硬约束 000 档 g 的 p05 ≥ 0.99（在拟合集上；margin 版——0.9 会把 b 选在
  约束边界，test clean OOD 左尾掉到门下，税 −1.2~−5.1；0.99 上移 b ≈2.4
  制造余量）；目标 = min Σ 三档 (1−g)T+gV 的 PA 均值。
- 测试：fold test 全部受试者 × {000,020,040,060,080,100}（fold occ 配置）。
  逐折逐臂报：000 税（e_mix−e_vt）、020-100 收益、100 档兑现率
  （vs 拟合集 const_w 的档位常数）。
- 通过线（预注册）：臂内 ≥3/4 折满足「000 税 ≤ +0.5 且 100 档兑现率 ≥ 50%」。
  060/080 窗级不可达 = 既定 limitation，不作失败条件。

用法：python results_display/script/ig_fold_ab.py --fold fS5S6S11 --device cuda:2
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
from ig_contact_pilot import ContactHead, train_head  # noqa: E402
from ig9_gate_smoke import (collect_gate_rows, collect_head_rows,  # noqa: E402
                            ema_by_session, gate_mix, ig_of,
                            single_session_loader)
from pa_grid_oracle import const_w  # noqa: E402
from stage2_gate_pilot import win_lower_pa  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
FIT_LEVELS = ("000", "040", "100")
EMA_ALPHA = 0.7
A_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
B_GRID = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
CLEAN_P05 = 0.99
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


def fit_gate(data, device, tag):
    best = None
    best_p05 = (0.0, None)
    for a in A_GRID:
        for b in B_GRID:
            ok = True; obj = 0.0
            for level in FIT_LEVELS:
                r = data[level]
                z = a * r["ig_s"] + b
                g = 1.0 / (1.0 + np.exp(-z))
                e = win_lower_pa(gate_mix(g, r["p_t"], r["p_v"]), r["gt"])
                obj += float(e.mean())
                if level == "000":
                    p05 = float(np.percentile(g, 5))
                    if p05 > best_p05[0]:
                        best_p05 = (p05, (a, b, obj))
                    if p05 < CLEAN_P05:
                        ok = False
            if ok and (best is None or obj < best[0]):
                best = (obj, a, b)
    if best is None:
        print("[fit/%s] INFEASIBLE; max clean p05=%.2f at (a,b)=%s"
              % (tag, best_p05[0], best_p05[1][:2]))
        return None
    _, a, b = best
    print("[fit/%s] a=%.2f b=%.2f obj=%.1f" % (tag, a, b, best[0]))
    return a, b


def collect_split_rows(net, head, cfg_path, splits, levels, device, bbox_cache,
                       batch_size, tp, tag):
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
        print("[%s/%s] n=%d IG med=%.2f p05=%.2f" % (
            tag, level, len(r["ig"]), float(np.median(r["ig"])),
            float(np.percentile(r["ig_s"], 5))))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--arms", choices=("A", "B", "both"), default="A")
    args = ap.parse_args()
    device = torch.device(args.device)
    fold_yaml = args.fold[1:]
    csv_name = FOLD_CSV[args.fold]
    net, ckpt, tp = build_model("V8_Cv5_%s_s1" % args.fold, device)
    print("[ckpt] %s" % ckpt)
    bbox_cache = {}

    # 头：A = train+val，B = train
    arms = [t for t in (("A", ("train", "val")), ("B", ("train",)))
            if t[0] in args.arms or args.arms == "both"]
    print("== heads (%s)" % ", ".join(t[0] for t in arms))
    heads = {}
    for tag, splits in arms:
        tr_tok, tr_con = [], []
        for split in splits:
            loader = build_loader("anysole/configs/v1_fold_%s.yaml" % fold_yaml,
                                  split, args.batch_size, t_phys_v2=tp)
            r = collect_head_rows(net, loader, device)
            tr_tok.append(r["v_tok"]); tr_con.append(r["contact"])
        tr = {"v_tok": np.concatenate(tr_tok), "contact": np.concatenate(tr_con)}
        print("[head/%s] windows=%d" % (tag, len(tr["v_tok"])))
        heads[tag] = train_head(tr, tr, device)
        torch.save(heads[tag].state_dict(),
                   OUT / ("contact_head_%s_%s.pt" % (args.fold, tag)))

    # 门拟合集：A = train+val，B = train
    print("== gate fit")
    fits = {}
    for tag, splits in arms:
        data = collect_split_rows(net, heads[tag], fold_yaml, splits, FIT_LEVELS,
                                  device, bbox_cache, args.batch_size, tp, "fit/" + tag)
        ab = fit_gate(data, device, tag)
        fits[tag] = (data, ab)

    # 测试：fold test 全部会话 × 六档（tokens 共用，两臂各算 IG）
    print("== test")
    test_sessions = fold_sessions(csv_name, "test")
    report = {"fold": args.fold, "test_sessions": test_sessions,
              "variants": {}}
    for tag, splits in arms:
        data, ab = fits[tag]
        if ab is None:
            report["variants"][tag] = {"feasible": False}
            continue
        a, b = ab
        head = heads[tag]
        w_const = {l: (const_w(data[l]["p_v"], data[l]["p_t"], data[l]["gt"])[0]
                       if l in data else float("nan")) for l in LEVELS}
        per_level = {}
        for level in LEVELS:
            cfg_l = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                     else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            loader = single_session_loader(cfg_l, test_sessions, args.batch_size, tp)
            r = collect_gate_rows(net, head, loader, device, bbox_cache, "test", level)
            ig_s = ema_by_session(r["ig"], r["session"])
            z = a * ig_s + b
            g = 1.0 / (1.0 + np.exp(-z))
            e_mix = win_lower_pa(gate_mix(g, r["p_t"], r["p_v"]), r["gt"])
            e_v = win_lower_pa(r["p_v"], r["gt"])
            e_t = win_lower_pa(r["p_t"], r["gt"])
            rec = {"level": level, "n": len(g),
                   "IG_med": float(np.median(r["ig"])),
                   "g_median": float(np.median(g)), "g_p05": float(np.percentile(g, 5)),
                   "e_v": float(e_v.mean()), "e_t": float(e_t.mean()),
                   "e_vt": float(r["e_vt"].mean()), "e_mix": float(e_mix.mean()),
                   "gain_vig": float(np.mean(r["e_vt"] - e_mix))}
            if level in data and np.isfinite(w_const[level]):
                lc = gate_mix(np.full(len(g), w_const[level]), r["p_t"], r["p_v"])
                e_lc = win_lower_pa(lc, r["gt"])
                rec["e_levelconst"] = float(e_lc.mean())
                rec["gain_levelconst"] = float(np.mean(r["e_vt"] - e_lc))
                rec["realization"] = (rec["gain_vig"] / rec["gain_levelconst"]
                                      if rec["gain_levelconst"] > 0.5 else None)
            per_level[level] = rec
            print("[%s/test/%s] n=%d IG med=%.2f | g med=%.2f p05=%.2f | e_v=%.1f e_t=%.1f e_vt=%.1f e_gate=%.1f%s | gain=%+.1f%s"
                  % (tag, level, rec["n"], rec["IG_med"], rec["g_median"],
                     rec["g_p05"], rec["e_v"], rec["e_t"], rec["e_vt"], rec["e_mix"],
                     "" if "gain_levelconst" not in rec else " e_lc=%.1f" % rec["e_levelconst"],
                     rec["gain_vig"],
                     "" if "realization" not in rec else " lc=%+.1f real=%s" % (
                         rec["gain_levelconst"],
                         "%.0f%%" % (100 * rec["realization"]) if rec["realization"] is not None else "n/a")))
        report["variants"][tag] = {"a": a, "b": b,
                                   "w_const": {l: w_const[l] for l in LEVELS},
                                   "per_level": per_level}
    (OUT / ("ig_fold_ab_%s.json" % args.fold)).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[out] %s" % (OUT / ("ig_fold_ab_%s.json" % args.fold)))


if __name__ == "__main__":
    main()
