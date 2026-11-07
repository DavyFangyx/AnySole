#!/usr/bin/env python3
"""修改 15 最后一轮低成本验证：门 = σ(a·EMA(IG) + b·rtm)，折外拟合，分位数约束。

【预先注册（2026-11-07，执行前写死）】
- 信号 IG：1 − BCE(h(V),c)/BCE(h(∅),c)，h = 冻结接触头
  （results/diagnostics/20261107_ig_contact_pilot/contact_head.pt，fS5S6S11 训），
  c = f6_soft 接触 GT（信号上界；运行时以 T 行接触头伪标签替代 = 落地说）。
- 平滑：窗口序列内 EMA，α=0.7，会话边界重置。
- RTMPose 摘要 rtm：bbox_cam3 检测框逐窗均值高度（px 原始值；score 列恒为 1
  无信息）。已知限制：bbox 来自 clean 视频（CLIFF），合成遮挡评估中 rtm 不随
  档位退化，档位分辨完全依赖 IG。
- 映射：g = σ(a·(IG_s−μ_ig) + b·(rtm−μ_rtm))，a,b ≥ 0，μ 由 val 估计
  （两参数 logistic、特征中心化 = 隐式截距；对两信号保序）。
  网格 a,b ∈ {0.25,0.5,1,2,4,8,16,32}²。
  【形式修正（首轮后）】原形式 σ(a·IG_s + b·rtm) 无截距且特征非负 →
  g ≥ 0.5 恒成立，数学上无法压制 V——首轮四折因此全部塌缩为 g≡1。
- 拟合：val（折外）。硬约束：val 000 与 020 各自 g 的 p05 ≥ 0.9；
  目标 = min Σ_档 val 上 (1−g)T+gV 的 PA 均值（可行集内）。
- 报告（test 逐折）：000/020 税（e_mix−e_vt）、060/080/100 兑现率
  （gain_vig/gain_levelconst，档位常数 w 由 val const_w 拟合）、可行 (a,b)、
  clean g 的 p05。

【通过线】≥3/4 折满足「000 税 ≤ +0.5 且 100 档兑现率 ≥ 50%」→ 模型侧落地；
否则本线关闭，写否定结论。fS6S7（val=S14）档位常数本身为负，其 100 档记「无空间」。

用法：python results_display/script/ig_rtm_gate_verify.py --fold fS5S6S11 --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "results_display/script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402
from ig_contact_pilot import ContactHead  # noqa: E402
from pa_grid_oracle import const_w  # noqa: E402
from stage2_gate_pilot import pred_kp, win_lower_pa  # noqa: E402
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import CONFIG_T, CONFIG_V, CONFIG_VT  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
SPLITS = ("val", "test")
EMA_ALPHA = 0.7
AB_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
CLEAN_P05 = 0.9
OUT = REPO / "results/diagnostics/20261107_stage2_vig"
BBOX = REPO / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/bbox_cam3"
HEAD = REPO / "results/diagnostics/20261107_ig_contact_pilot/contact_head.pt"


def ema_by_session(sig, sessions):
    """窗口序列内 EMA（α=0.7），会话边界重置。输入按 loader 顺序（会话内连续）。"""
    out = np.empty_like(sig)
    prev_s, prev_v = None, None
    for i, (s, v) in enumerate(zip(sessions, sig)):
        if s != prev_s:
            prev_v = None
        prev_v = v if prev_v is None else EMA_ALPHA * v + (1 - EMA_ALPHA) * prev_v
        out[i] = prev_v
        prev_s = s
    return out


def collect(net, head, loader, device, bbox_cache, split, level):
    rows = {k: [] for k in ("session", "start", "p_v", "p_t", "gt", "e_vt",
                            "ig", "rtm")}
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
            v_tok = out_vt["v_tok"]
            c = batch["contact_gt"]
            e_v = F.binary_cross_entropy_with_logits(head(v_tok), c, reduction="none").mean(dim=(1, 2))
            e_0 = F.binary_cross_entropy_with_logits(head(torch.zeros_like(v_tok)), c,
                                                     reduction="none").mean(dim=(1, 2))
            ig = (1.0 - e_v / e_0.clamp(min=1e-6)).cpu().numpy()
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
        rows["start"].extend(starts)
        rows["p_v"].append(p_v); rows["p_t"].append(p_t); rows["gt"].append(gt)
        rows["e_vt"].append(win_lower_pa(p_vt, gt))
        rows["ig"].append(ig)
        rows["rtm"].extend(rtm)
    out = {"session": np.asarray(rows["session"]),
           "p_v": np.concatenate(rows["p_v"]), "p_t": np.concatenate(rows["p_t"]),
           "gt": np.concatenate(rows["gt"]), "e_vt": np.concatenate(rows["e_vt"]),
           "ig": np.concatenate(rows["ig"]), "rtm": np.asarray(rows["rtm"]),
           "split": split, "level": level}
    out["ig_s"] = ema_by_session(out["ig"], out["session"])
    return out


def gate_mix(g, p_t, p_v):
    return (1.0 - g)[:, None, None, None] * p_t + g[:, None, None, None] * p_v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    fold_yaml = args.fold[1:] if args.fold.startswith("f") else args.fold
    cell = "V8_Cv5_%s_s1" % args.fold
    net, ckpt, tp = build_model(cell, device)
    print("[ckpt] %s" % ckpt)
    head = ContactHead(256).to(device)
    head.load_state_dict(torch.load(HEAD, map_location="cpu"))
    head.eval(); head.requires_grad_(False)
    bbox_cache = {}

    data = {}
    for split in splits:
        for level in levels:
            cfg_path = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                        else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            loader = build_loader(cfg_path, split, args.batch_size, t_phys_v2=tp)
            data[(split, level)] = collect(net, head, loader, device, bbox_cache,
                                           split, level)
            r = data[(split, level)]
            print("[%s/%s] n=%d IG med=%.2f rtm med=%.0f" % (
                split, level, len(r["ig"]), float(np.median(r["ig"])),
                float(np.nanmedian(r["rtm"]))))

    # 拟合：val 折外；特征中心化（μ 由 val 估计，隐式截距）；
    # 硬约束 clean p05 ≥ 0.9；目标 = min Σ_档 e_mix
    all_ig = np.concatenate([data[("val", l)]["ig_s"] for l in levels])
    all_rtm = np.concatenate([data[("val", l)]["rtm"] for l in levels])
    mu_ig = float(np.nanmean(all_ig)); mu_rtm = float(np.nanmean(all_rtm))
    best = None
    for a in AB_GRID:
        for b in AB_GRID:
            ok = True
            obj = 0.0
            for level in levels:
                r = data[("val", level)]
                z = a * (r["ig_s"] - mu_ig) + b * (r["rtm"] - mu_rtm)
                g = 1.0 / (1.0 + np.exp(-z))
                e = win_lower_pa(gate_mix(g, r["p_t"], r["p_v"]), r["gt"])
                obj += float(e.mean())
                if level in ("000", "020") and float(np.percentile(g, 5)) < CLEAN_P05:
                    ok = False
            if ok and (best is None or obj < best[0]):
                best = (obj, a, b)
    report_meta = {"mu_ig": mu_ig, "mu_rtm": mu_rtm}
    # 落盘逐窗数据（供拟合变体离线复算，免重复前向）
    for split in splits:
        rows_save = {k: [] for k in ("session", "level", "ig", "ig_s", "rtm",
                                     "p_v", "p_t", "gt", "e_vt")}
        for level in levels:
            r = data[(split, level)]
            rows_save["session"].append(r["session"])
            rows_save["level"].append(np.full(len(r["ig"]), int(level)))
            for k in ("ig", "ig_s", "rtm", "e_vt"):
                rows_save[k].append(r[k])
            rows_save["p_v"].append(r["p_v"]); rows_save["p_t"].append(r["p_t"])
            rows_save["gt"].append(r["gt"])
        np.savez_compressed(OUT / ("igrtm_rows_%s_%s.npz" % (args.fold, split)),
                            session=np.concatenate(rows_save["session"]),
                            level=np.concatenate(rows_save["level"]),
                            ig=np.concatenate(rows_save["ig"]),
                            ig_s=np.concatenate(rows_save["ig_s"]),
                            rtm=np.concatenate(rows_save["rtm"]),
                            e_vt=np.concatenate(rows_save["e_vt"]),
                            p_v=np.concatenate(rows_save["p_v"]),
                            p_t=np.concatenate(rows_save["p_t"]),
                            gt=np.concatenate(rows_save["gt"]))
    if best is None:
        print("[fit] INFEASIBLE: no (a,b) satisfies clean p05 >= %.1f on val" % CLEAN_P05)
        (OUT / ("igrtm_%s.json" % args.fold)).write_text(
            json.dumps({"fold": args.fold, "feasible": False, **report_meta},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    _, a, b = best
    print("[fit] mu_ig=%.3f mu_rtm=%.0f a=%.1f b=%.1f (val obj=%.1f)"
          % (mu_ig, mu_rtm, a, b, best[0]))

    report = {"fold": args.fold, "ckpt": str(ckpt), "a": a, "b": b,
              "ema_alpha": EMA_ALPHA, "clean_p05_constraint": CLEAN_P05,
              "per_level": {}, **report_meta}
    for split in ("val", "test"):
        for level in levels:
            r = data[(split, level)]
            z = a * (r["ig_s"] - mu_ig) + b * (r["rtm"] - mu_rtm)
            g = 1.0 / (1.0 + np.exp(-z))
            e_mix = win_lower_pa(gate_mix(g, r["p_t"], r["p_v"]), r["gt"])
            w_l = None
            if split == "test":
                w_l, _ = const_w(data[("val", level)]["p_v"],
                                 data[("val", level)]["p_t"],
                                 data[("val", level)]["gt"])
                lc = gate_mix(np.full(len(g), w_l), r["p_t"], r["p_v"])
                e_lc = win_lower_pa(lc, r["gt"])
            rec = {"split": split, "level": level, "n": len(g),
                   "g_median": float(np.median(g)), "g_p05": float(np.percentile(g, 5)),
                   "g_p95": float(np.percentile(g, 95)),
                   "e_vt": float(r["e_vt"].mean()), "e_mix": float(e_mix.mean()),
                   "tax": float(np.mean(e_mix - r["e_vt"]))}
            if split == "test":
                rec["gain_vig"] = float(np.mean(r["e_vt"] - e_mix))
                rec["w_const"] = float(w_l)
                rec["e_levelconst"] = float(e_lc.mean())
                rec["gain_levelconst"] = float(np.mean(r["e_vt"] - e_lc))
                rec["realization"] = (rec["gain_vig"] / rec["gain_levelconst"]
                                      if rec["gain_levelconst"] > 0.5 else None)
            report["per_level"]["%s/%s" % (split, level)] = rec
            if split == "test":
                print("[test/%s] g med=%.2f p05=%.2f | e_vt=%.1f e_gate=%.1f e_lc=%.1f | tax=%+.1f gain=%+.1f lc=%+.1f real=%s"
                      % (level, rec["g_median"], rec["g_p05"], rec["e_vt"],
                         rec["e_mix"], rec["e_levelconst"], rec["tax"],
                         rec["gain_vig"], rec["gain_levelconst"],
                         "%.0f%%" % (100 * rec["realization"]) if rec["realization"] is not None else "n/a"))
    (OUT / ("igrtm_%s.json" % args.fold)).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[out] %s" % (OUT / ("igrtm_%s.json" % args.fold)))


if __name__ == "__main__":
    main()
