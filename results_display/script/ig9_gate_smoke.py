#!/usr/bin/env python3
"""修改 15 数据归因验证：IG 接触头在 9 受试者（act8020 train）上训练，
门（中心化两参数 logistic）在 act8020 val（同受试者、会话折外）拟合，
最终只在 test 的单个样本上实验。

【口径（2026-11-07）】
- 头：ContactHead（ig_contact_pilot 同款）在 act8020 train（109 会话、9 受试者）
  逐帧 v_tok 上训练，act8020 val 早停。旧头（fS5S6S11 训）作对照保留。
- 门：g = σ(a·(IG_s−μ_ig) + b·(rtm−μ_rtm))，μ 由拟合集估计；IG_s = EMA(α=0.7)
  会话内平滑；rtm = bbox_cam3 逐窗均值高度（px）。网格 a,b ∈ {0.25..32}²。
  拟合集 = act8020 val × {000,040,100}；硬约束 000 档 g 的 p05 ≥ 0.9；
  目标 = min Σ 三档 (1−g)T+gV 的 PA 均值。
- 测试：单会话（--session，默认 S11013 = S11、fS5S6S11 test ∩ act8020 val）
  × {000,020,040,060,080,100}（fold occ 配置）。逐档报：IG 中位、g 中位/p05、
  e_v/e_t/e_vt/e_mix、相对 VT 与相对档位常数（w 由拟合集 const_w 估计）的收益。
- 判读：这是可行性冒烟（单样本），不是正式评估；重点看「头不再受试者 OOD 后，
  门能否在 clean 保持 g≈1 且高遮挡压低 g」。

用法：python results_display/script/ig9_gate_smoke.py --device cuda:6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "results_display/script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402
from ig_contact_pilot import ContactHead, head_bce, train_head  # noqa: E402
from pa_grid_oracle import const_w  # noqa: E402
from stage2_gate_pilot import pred_kp, win_lower_pa  # noqa: E402
from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.train import condition_inputs, load_config, move_batch  # noqa: E402
from anysole.types import CONFIG_T, CONFIG_V, CONFIG_VT  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
FIT_LEVELS = ("000", "040", "100")
EMA_ALPHA = 0.7
AB_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
CLEAN_P05 = 0.9
OUT = REPO / "results/diagnostics/20261107_stage2_vig"
BBOX = REPO / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/bbox_cam3"
OLD_HEAD = REPO / "results/diagnostics/20261107_ig_contact_pilot/contact_head.pt"
NEW_HEAD = OUT / "contact_head_act8020.pt"


def act8020_sessions(split):
    rows = list(csv.DictReader(open(
        REPO / "AnysoleWorkspace/protocol/splits/default/splits_action_8020.csv")))
    return [r[split] for r in rows if r[split]]


def single_session_loader(config_path, session_id, batch_size, t_phys_v2):
    config = load_config(REPO / config_path)
    tw = int(config.get("tw", 20))
    sids = session_id if isinstance(session_id, (list, tuple)) else [session_id]
    dataset = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]), cache_root=Path(config["cache_root"]),
        window_length=tw, session_ids=sids,
        contact_method=str(config["contact_method"]), tactile_input="raw108",
        no_imu=False, v_input="hrnet", f2_repr=False,
        smpl_roots=config.get("smpl_roots"), cam_id=int(config.get("cam_id", 3)),
        t_phys_v2=t_phys_v2, stride=None, obs_context=tw,
        smpl_support_policy="exclude")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=2, collate_fn=collate_windows, pin_memory=True)


def ema_by_session(sig, sessions):
    out = np.empty_like(sig)
    prev_s, prev_v = None, None
    for i, (s, v) in enumerate(zip(sessions, sig)):
        if s != prev_s:
            prev_v = None
        prev_v = v if prev_v is None else EMA_ALPHA * v + (1 - EMA_ALPHA) * prev_v
        out[i] = prev_v
        prev_s = s
    return out


def collect_head_rows(net, loader, device):
    v_tok, contact, sessions = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        n = batch["V_feat"].shape[0]
        cid_vt = torch.full((n,), CONFIG_VT, dtype=torch.long, device=device)
        with torch.no_grad():
            vf, tr, tp, s2 = condition_inputs(batch, cid_vt)
            o = net(vf, tr, tp, cid_vt, batch.get("subject_ids"), T_s2m=s2)
        v_tok.append(o["v_tok"].detach().cpu().numpy())
        contact.append(batch["contact_gt"].detach().cpu().numpy())
        sessions.extend(batch["session_id"])
    return {"v_tok": np.concatenate(v_tok), "contact": np.concatenate(contact),
            "session": np.asarray(sessions)}


def ig_of(head, tok, contact, device):
    with torch.no_grad():
        x = torch.from_numpy(tok).float().to(device)
        c = torch.from_numpy(contact).float().to(device)
        e_v = F.binary_cross_entropy_with_logits(head(x), c, reduction="none").mean(dim=(1, 2))
        e_0 = F.binary_cross_entropy_with_logits(head(torch.zeros_like(x)), c,
                                                 reduction="none").mean(dim=(1, 2))
    return (1.0 - e_v / e_0.clamp(min=1e-6)).cpu().numpy()


def collect_gate_rows(net, head, loader, device, bbox_cache, split, level):
    rows = {k: [] for k in ("session", "p_v", "p_t", "gt", "e_vt", "ig", "rtm",
                            "v_tok", "contact")}
    for batch in loader:
        batch = move_batch(batch, device)
        n = batch["V_feat"].shape[0]
        cid_v = torch.full((n,), CONFIG_V, dtype=torch.long, device=device)
        cid_t = torch.full((n,), CONFIG_T, dtype=torch.long, device=device)
        cid_vt = torch.full((n,), CONFIG_VT, dtype=torch.long, device=device)
        with torch.no_grad():
            vf, tr, tp, s2 = condition_inputs(batch, cid_v)
            out_v = net(vf, tr, tp, cid_v, batch.get("subject_ids"), T_s2m=s2)
            vf, tr, tp, s2 = condition_inputs(batch, cid_t)
            out_t = net(vf, tr, tp, cid_t, batch.get("subject_ids"), T_s2m=s2)
            vf, tr, tp, s2 = condition_inputs(batch, cid_vt)
            out_vt = net(vf, tr, tp, cid_vt, batch.get("subject_ids"), T_s2m=s2)
        tok = out_vt["v_tok"].detach().cpu().numpy()
        contact = batch["contact_gt"].detach().cpu().numpy()
        ig = ig_of(head, tok, contact, device)
        p_v = pred_kp(out_v, batch); p_t = pred_kp(out_t, batch)
        p_vt = pred_kp(out_vt, batch)
        gt = batch["kp_gt"].detach().cpu().numpy()
        sessions = batch["session_id"]
        starts = batch["frame_start"].detach().cpu().tolist()
        rtm = []
        for sid, st in zip(sessions, starts):
            bb = bbox_cache.get(sid)
            if bb is None:
                bb = np.load(BBOX / ("%s.npy" % sid))
                bbox_cache[sid] = bb
            seg = bb[int(st):int(st) + gt.shape[1]]
            rtm.append(float((seg[:, 4] - seg[:, 2]).mean()) if len(seg) else float("nan"))
        rows["session"].extend(sessions)
        rows["p_v"].append(p_v); rows["p_t"].append(p_t); rows["gt"].append(gt)
        rows["e_vt"].append(win_lower_pa(p_vt, gt))
        rows["ig"].append(ig); rows["rtm"].extend(rtm)
        rows["v_tok"].append(tok); rows["contact"].append(contact)
    out = {"session": np.asarray(rows["session"]),
           "p_v": np.concatenate(rows["p_v"]), "p_t": np.concatenate(rows["p_t"]),
           "gt": np.concatenate(rows["gt"]), "e_vt": np.concatenate(rows["e_vt"]),
           "ig": np.concatenate(rows["ig"]), "rtm": np.asarray(rows["rtm"]),
           "v_tok": np.concatenate(rows["v_tok"]),
           "contact": np.concatenate(rows["contact"]),
           "split": split, "level": level}
    out["ig_s"] = ema_by_session(out["ig"], out["session"])
    return out


def gate_mix(g, p_t, p_v):
    return (1.0 - g)[:, None, None, None] * p_t + g[:, None, None, None] * p_v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--session", default="S11013")
    ap.add_argument("--skip-head", action="store_true",
                    help="跳过阶段 A，直接载入已有 contact_head_act8020.pt")
    args = ap.parse_args()
    device = torch.device(args.device)
    cell = "V8_Cv5_fS5S6S11_s1"
    net, ckpt, tp = build_model(cell, device)
    print("[ckpt] %s" % ckpt)
    bbox_cache = {}

    # 阶段 A：9 受试者头训练
    if not args.skip_head:
        print("== stage A: head on act8020 train (9 subjects)")
        tr_loader = build_loader("anysole/configs/v1_act8020.yaml", "train",
                                 args.batch_size, t_phys_v2=tp)
        tr = collect_head_rows(net, tr_loader, device)
        va_loader = build_loader("anysole/configs/v1_act8020.yaml", "val",
                                 args.batch_size, t_phys_v2=tp)
        va = collect_head_rows(net, va_loader, device)
        print("[head] train windows=%d val windows=%d" % (len(tr["v_tok"]), len(va["v_tok"])))
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

    # 阶段 B：门拟合（act8020 val × {000,040,100}）
    print("== stage B: gate fit on act8020 val")
    data = {}
    for level in FIT_LEVELS:
        cfg_path = ("anysole/configs/v1_act8020.yaml" if level == "000"
                    else "anysole/configs/v1_act8020_occ%s.yaml" % level)
        loader = build_loader(cfg_path, "val", args.batch_size, t_phys_v2=tp)
        data[level] = collect_gate_rows(net, head, loader, device, bbox_cache, "val", level)
        r = data[level]
        print("[fit/%s] n=%d IG med=%.2f rtm med=%.0f" % (level, len(r["ig"]),
              float(np.median(r["ig"])), float(np.nanmedian(r["rtm"]))))
    for level in FIT_LEVELS:
        r = data[level]
        print("[fit/%s] IG mean=%.2f p05=%.2f p95=%.2f | rtm mean=%.0f p05=%.0f p95=%.0f"
              % (level, float(r["ig_s"].mean()), float(np.percentile(r["ig_s"], 5)),
                 float(np.percentile(r["ig_s"], 95)), float(r["rtm"].mean()),
                 float(np.percentile(r["rtm"], 5)), float(np.percentile(r["rtm"], 95))))
    np.savez_compressed(OUT / ("ig9_fit_rows.npz"),
                        level=np.concatenate([np.full(len(data[l]["ig"]), int(l))
                                              for l in FIT_LEVELS]),
                        ig_s=np.concatenate([data[l]["ig_s"] for l in FIT_LEVELS]),
                        rtm=np.concatenate([data[l]["rtm"] for l in FIT_LEVELS]),
                        p_v=np.concatenate([data[l]["p_v"] for l in FIT_LEVELS]),
                        p_t=np.concatenate([data[l]["p_t"] for l in FIT_LEVELS]),
                        gt=np.concatenate([data[l]["gt"] for l in FIT_LEVELS]),
                        e_vt=np.concatenate([data[l]["e_vt"] for l in FIT_LEVELS]))
    all_ig = np.concatenate([data[l]["ig_s"] for l in FIT_LEVELS])
    all_rtm = np.concatenate([data[l]["rtm"] for l in FIT_LEVELS])
    mu_ig = float(np.nanmean(all_ig)); mu_rtm = float(np.nanmean(all_rtm))
    best = None
    best_p05 = (0.0, None)   # 不可行时的最大 clean p05（诊断）
    for a in AB_GRID:
        for b in AB_GRID:
            ok = True; obj = 0.0
            for level in FIT_LEVELS:
                r = data[level]
                z = a * (r["ig_s"] - mu_ig) + b * (r["rtm"] - mu_rtm)
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
        print("[fit] INFEASIBLE; max clean p05=%.2f at (a,b)=%s obj=%.1f"
              % (best_p05[0], best_p05[1][:2], best_p05[1][2]))
        return
    _, a, b = best
    print("[fit] mu_ig=%.3f mu_rtm=%.0f a=%.2f b=%.2f obj=%.1f" % (mu_ig, mu_rtm, a, b, best[0]))
    w_const = {}
    for level in LEVELS:
        if level in data:
            w_const[level], _ = const_w(data[level]["p_v"], data[level]["p_t"],
                                        data[level]["gt"])
        else:
            w_const[level] = float("nan")

    # 阶段 C：单样本测试（fold occ 配置）
    print("== stage C: single test session %s" % args.session)
    fold_yaml = "S5S6S11"
    report = {"session": args.session, "a": a, "b": b,
              "mu_ig": mu_ig, "mu_rtm": mu_rtm,
              "w_const": {l: w_const[l] for l in LEVELS}, "per_level": {}}
    for level in LEVELS:
        cfg_path = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                    else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
        loader = single_session_loader(cfg_path, args.session, args.batch_size, tp)
        r = collect_gate_rows(net, head, loader, device, bbox_cache, "test", level)
        z = a * (r["ig_s"] - mu_ig) + b * (r["rtm"] - mu_rtm)
        g = 1.0 / (1.0 + np.exp(-z))
        e_mix = win_lower_pa(gate_mix(g, r["p_t"], r["p_v"]), r["gt"])
        e_v = win_lower_pa(r["p_v"], r["gt"])
        e_t = win_lower_pa(r["p_t"], r["gt"])
        rec = {"level": level, "n": len(g),
               "IG_med": float(np.median(r["ig"])),
               "g_median": float(np.median(g)), "g_p05": float(np.percentile(g, 5)),
               "g_p95": float(np.percentile(g, 95)),
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
        report["per_level"][level] = rec
        print("[test/%s] n=%d IG med=%.2f | g med=%.2f p05=%.2f | e_v=%.1f e_t=%.1f e_vt=%.1f e_gate=%.1f%s | gain=%+.1f%s"
              % (level, rec["n"], rec["IG_med"], rec["g_median"], rec["g_p05"],
                 rec["e_v"], rec["e_t"], rec["e_vt"], rec["e_mix"],
                 "" if "gain_levelconst" not in rec else " e_lc=%.1f" % rec["e_levelconst"],
                 rec["gain_vig"],
                 "" if "realization" not in rec else " lc=%+.1f real=%s" % (
                     rec["gain_levelconst"],
                     "%.0f%%" % (100 * rec["realization"]) if rec["realization"] is not None else "n/a")))
    (OUT / ("ig9_%s.json" % args.session)).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[out] %s" % (OUT / ("ig9_%s.json" % args.session)))


if __name__ == "__main__":
    main()
