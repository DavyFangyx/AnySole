#!/usr/bin/env python3
"""修改 15 数据归因验证·留受试者版：一折（fS5S6S11），test = 2 受试者（S6+S11），
IG 接触头在其余 7 受试者上训练（严格 leave-subject-out），门在 val（S12）拟合。

【口径（2026-11-07）】
- 头训练：fS5S6S11 train（S7/S8/S10/S13/S14，15 会话）+ fS6S7 train 的 S5 会话
  + val S12（2 会话）= 7 受试者、26 会话，逐帧 v_tok 训 ContactHead。
- 门拟合：val（S12）× {000,040,100}；g = σ(a·IG_s + b)（阈值 logistic，
  斜率 a ≥ 0、截距 b ∈ ℝ），IG_s = EMA(α=0.7) 会话内；网格 a ∈ {0.5..32}、
  b ∈ {−4..8}；硬约束 000 档 g 的 p05 ≥ 0.9；目标 = min Σ 三档 (1−g)T+gV
  的 PA 均值。rtm（bbox 高度）实测随档不变（579 恒定）→ 剔除（被截距吸收）。
  【形式两误更正】①无截距 σ(a·IG+b·rtm)：特征非负 → g≥0.5 恒成立；
  ②中心化 σ(a·(IG−μ)+…)：p05 窗偏离恒为负 → 约束数学上不可满足。
  正确两参数形式 = 显式阈值 logistic。
- 测试：S6+S11 全部 test 会话 × {000,020,040,060,080,100}（fold occ 配置）。
  逐受试者逐档报：IG 中位、g 中位/p05、e_v/e_t/e_vt/e_mix、相对 VT 与
  档位常数（w 由 val 拟合集 const_w 估计）的收益；旧头（fS5S6S11 训）IG
  分布作对照。
- 判读：可行性冒烟；看「头不再受试者 OOD 后 clean 能否守住 g≈1、
  高遮挡能否压低 g」。约束不可行时报告最大可达 clean p05 作诊断。

用法：python results_display/script/ig_lso_gate.py --device cuda:6
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
                            ema_by_session, gate_mix, single_session_loader)
from pa_grid_oracle import const_w  # noqa: E402
from stage2_gate_pilot import win_lower_pa  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
FIT_LEVELS = ("000", "040", "100")
EMA_ALPHA = 0.7
A_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
B_GRID = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
CLEAN_P05 = 0.9
OUT = REPO / "results/diagnostics/20261107_stage2_vig"
OLD_HEAD = REPO / "results/diagnostics/20261107_ig_contact_pilot/contact_head.pt"
NEW_HEAD = OUT / "contact_head_lso7.pt"
TEST_SUBJECT_PREFIXES = ("S60", "S61", "S11")


def fold_sessions(csv_name, split):
    rows = list(csv.DictReader(open(
        REPO / "AnysoleWorkspace/protocol/splits/default" / csv_name)))
    return [r[split] for r in rows if r[split]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-head", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    net, ckpt, tp = build_model("V8_Cv5_fS5S6S11_s1", device)
    print("[ckpt] %s" % ckpt)
    bbox_cache = {}

    if not args.skip_head:
        print("== head training on 7 subjects (26 sessions)")
        tr_tok, tr_con = [], []
        for r in (collect_head_rows(
                net, build_loader("anysole/configs/v1_fold_S5S6S11.yaml", "train",
                                  args.batch_size, t_phys_v2=tp), device),
                  collect_head_rows(
                net, build_loader("anysole/configs/v1_fold_S5S6S11.yaml", "val",
                                  args.batch_size, t_phys_v2=tp), device),
                  collect_head_rows(
                net, single_session_loader(
                    "anysole/configs/v1_fold_S6S7.yaml",
                    [s for s in fold_sessions("splits_test_S6S7_val_S14.csv", "train")
                     if s[:3] in ("S50", "S51")],
                    args.batch_size, tp), device)):
            tr_tok.append(r["v_tok"]); tr_con.append(r["contact"])
        tr = {"v_tok": np.concatenate(tr_tok), "contact": np.concatenate(tr_con)}
        print("[head] train windows=%d" % len(tr["v_tok"]))
        va = collect_head_rows(
            net, build_loader("anysole/configs/v1_fold_S5S6S11.yaml", "val",
                              args.batch_size, t_phys_v2=tp), device)
        head = train_head(tr, va, device)
        torch.save(head.state_dict(), NEW_HEAD)
        print("[head] saved %s" % NEW_HEAD)
    else:
        head = ContactHead(256).to(device)
        head.load_state_dict(torch.load(NEW_HEAD, map_location="cpu"))
    head.eval(); head.requires_grad_(False)
    old_head = ContactHead(256).to(device)
    old_head.load_state_dict(torch.load(OLD_HEAD, map_location="cpu"))
    old_head.eval(); old_head.requires_grad_(False)

    # 门拟合：val（S12）× {000,040,100}
    print("== gate fit on val (S12)")
    data = {}
    for level in FIT_LEVELS:
        cfg_path = ("anysole/configs/v1_fold_S5S6S11.yaml" if level == "000"
                    else "anysole/configs/v1_occ%s_fold_S5S6S11.yaml" % level)
        loader = build_loader(cfg_path, "val", args.batch_size, t_phys_v2=tp)
        data[level] = collect_gate_rows(net, head, loader, device, bbox_cache, "val", level)
        r = data[level]
        print("[fit/%s] n=%d IG med=%.2f mean=%.2f p05=%.2f p95=%.2f" % (
            level, len(r["ig"]), float(np.median(r["ig"])), float(r["ig_s"].mean()),
            float(np.percentile(r["ig_s"], 5)), float(np.percentile(r["ig_s"], 95))))
    np.savez_compressed(OUT / "ig_lso_fit_rows.npz",
                        level=np.concatenate([np.full(len(data[l]["ig"]), int(l)) for l in FIT_LEVELS]),
                        ig_s=np.concatenate([data[l]["ig_s"] for l in FIT_LEVELS]),
                        rtm=np.concatenate([data[l]["rtm"] for l in FIT_LEVELS]),
                        p_v=np.concatenate([data[l]["p_v"] for l in FIT_LEVELS]),
                        p_t=np.concatenate([data[l]["p_t"] for l in FIT_LEVELS]),
                        gt=np.concatenate([data[l]["gt"] for l in FIT_LEVELS]),
                        e_vt=np.concatenate([data[l]["e_vt"] for l in FIT_LEVELS]))
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
        print("[fit] INFEASIBLE; max clean p05=%.2f at (a,b)=%s" % (best_p05[0], best_p05[1][:2]))
        return
    _, a, b = best
    print("[fit] a=%.2f b=%.2f obj=%.1f" % (a, b, best[0]))
    w_const = {}
    for level in LEVELS:
        if level in data:
            w_const[level], _ = const_w(data[level]["p_v"], data[level]["p_t"], data[level]["gt"])
        else:
            w_const[level] = float("nan")

    # 测试：S6+S11 全部 test 会话 × 六档
    print("== test: S6 + S11")
    test_sessions = [s for s in fold_sessions("splits_test_S5S6S11_val_S12.csv", "test")
                     if s[:3] in TEST_SUBJECT_PREFIXES]
    print("[test] sessions=%s" % test_sessions)
    report = {"a": a, "b": b,
              "test_sessions": test_sessions,
              "w_const": {l: w_const[l] for l in LEVELS}, "per_subject": {}}
    for subj_prefix, subj in (("S6", ("S60", "S61")), ("S11", ("S11",))):
        for level in LEVELS:
            cfg_path = ("anysole/configs/v1_fold_S5S6S11.yaml" if level == "000"
                        else "anysole/configs/v1_occ%s_fold_S5S6S11.yaml" % level)
            sids = [s for s in test_sessions if s[:3] in subj]
            if not sids:
                continue
            loader = single_session_loader(cfg_path, sids, args.batch_size, tp)
            r = collect_gate_rows(net, head, loader, device, bbox_cache, "test", level)
            z = a * r["ig_s"] + b
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
            if np.isfinite(w_const[level]):
                lc = gate_mix(np.full(len(g), w_const[level]), r["p_t"], r["p_v"])
                e_lc = win_lower_pa(lc, r["gt"])
                rec["e_levelconst"] = float(e_lc.mean())
                rec["gain_levelconst"] = float(np.mean(r["e_vt"] - e_lc))
                rec["realization"] = (rec["gain_vig"] / rec["gain_levelconst"]
                                      if rec["gain_levelconst"] > 0.5 else None)
            report["per_subject"].setdefault(subj_prefix, {})[level] = rec
            print("[%s/%s] n=%d IG med=%.2f | g med=%.2f p05=%.2f | e_v=%.1f e_t=%.1f e_vt=%.1f e_gate=%.1f%s | gain=%+.1f%s"
                  % (subj_prefix, level, rec["n"], rec["IG_med"], rec["g_median"],
                     rec["g_p05"], rec["e_v"], rec["e_t"], rec["e_vt"], rec["e_mix"],
                     "" if "gain_levelconst" not in rec else " e_lc=%.1f" % rec["e_levelconst"],
                     rec["gain_vig"],
                     "" if "realization" not in rec else " lc=%+.1f real=%s" % (
                         rec["gain_levelconst"],
                         "%.0f%%" % (100 * rec["realization"]) if rec["realization"] is not None else "n/a")))
    (OUT / "ig_lso.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print("[out] %s" % (OUT / "ig_lso.json"))


if __name__ == "__main__":
    main()
