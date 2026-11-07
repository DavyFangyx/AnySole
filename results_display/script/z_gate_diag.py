#!/usr/bin/env python3
"""V 门 z 分布诊断（修改 15）：test 上逐档 z（route_v_mlp 输出，校准后仿射口径）。

回答校准可行性的核心问题：clean（000/020）左尾与高遮挡（060-100）窗在
z 上是否可分（AUROC）。仿射 (α,β) 不改变 AUROC（单调），故一次收集即可；
校准前后 g 中位数由 z₀=(z−β)/α 反推（前=2σ(z₀) 无 cap，后=min(2σ(z),1)）。

用法：python results_display/script/z_gate_diag.py --fold fS5S6S11 --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "results_display/script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e6_gate_values_occlusion import build_loader  # noqa: E402
from anysole.models.v8_cv5vig import build as build_vig  # noqa: E402
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import CONFIG_VT  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
OUT = REPO / "results/diagnostics/20261107_stage2_vig"


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    v = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(v, kind="mergesort")
    lab = lab[order]
    ranks = np.arange(1, len(v) + 1)
    r_pos = ranks[lab == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    fold_yaml = args.fold[1:] if args.fold.startswith("f") else args.fold
    ckpt = Path(args.ckpt) if args.ckpt else OUT / ("V8_Cv5VIG_%s.pt" % args.fold)

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    src_cfg = payload.get("config", {}) or {}
    calib = {}
    mf = OUT / ("metrics_%s.json" % args.fold)
    if mf.is_file():
        calib = json.loads(mf.read_text(encoding="utf-8")).get("calib", {})
    alpha = float(calib.get("alpha", 1.0)); beta = float(calib.get("beta", 0.0))
    cfg = {"d_model": 256, "tw": 20, "dropout": 0.1, "pose_layers": 6,
           "v7_task_routes": True, "v7_root_sd": True,
           "v7_v_gate": True, "v7_gate_2sigma": False,
           "v7_trans_anchor_head": True}
    net = build_vig(cfg).to(device)
    missing, unexpected = net.load_state_dict(state, strict=False)
    net.eval()
    print("[ckpt] %s missing=%s unexpected=%s" % (ckpt, missing, unexpected))
    print("[calib] alpha=%.2f beta=%.2f" % (alpha, beta))

    tp = bool(src_cfg.get("t_phys_v2", False))
    out = {"fold": args.fold, "ckpt": str(ckpt), "calib": calib,
           "per_level": {}, "auroc": {}}
    zs_by_level = {}
    for level in levels:
        cfg_path = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                    else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
        loader = build_loader(cfg_path, "test", args.batch_size, t_phys_v2=tp)
        zs = []
        for batch in loader:
            batch = move_batch(batch, device)
            n = batch["V_feat"].shape[0]
            cid_vt = torch.full((n,), CONFIG_VT, dtype=torch.long, device=device)
            with torch.no_grad():
                vf, tr, tp2, s2 = condition_inputs(batch, cid_vt)
                o = net(vf, tr, tp2, cid_vt, batch.get("subject_ids"), T_s2m=s2)
                z = net.route_v_mlp(net._gate_input(o["v_tok"].mean(1)[:, None, :],
                                                    o["t_tok"].mean(1)[:, None, :])).squeeze(-1)
            zs.append(z.detach().cpu().numpy())
        z = np.concatenate(zs)
        zs_by_level[level] = z
        z0 = (z - beta) / alpha
        g_pre = 2.0 / (1.0 + np.exp(-z0))
        g_post = np.minimum(2.0 / (1.0 + np.exp(-z)), 1.0)
        out["per_level"][level] = {
            "n": int(len(z)),
            "z_mean": float(z.mean()), "z_p05": float(np.percentile(z, 5)),
            "z_median": float(np.median(z)), "z_p95": float(np.percentile(z, 95)),
            "g_pre_median": float(np.median(g_pre)), "g_pre_p05": float(np.percentile(g_pre, 5)),
            "g_post_median": float(np.median(g_post)), "g_post_p05": float(np.percentile(g_post, 5))}
        print("[%s] n=%d z med=%+.2f [p05 %+.2f p95 %+.2f] | g_pre med=%.2f (p05 %.2f) -> g_post med=%.2f (p05 %.2f)"
              % (level, len(z), out["per_level"][level]["z_median"],
                 out["per_level"][level]["z_p05"], out["per_level"][level]["z_p95"],
                 out["per_level"][level]["g_pre_median"], out["per_level"][level]["g_pre_p05"],
                 out["per_level"][level]["g_post_median"], out["per_level"][level]["g_post_p05"]))

    out["auroc"] = {
        "clean_vs_high": auc(np.concatenate([zs_by_level[l] for l in ("000", "020")]),
                             np.concatenate([zs_by_level[l] for l in ("060", "080", "100")])),
        "000_vs_100": auc(zs_by_level["000"], zs_by_level["100"]),
        "020_vs_100": auc(zs_by_level["020"], zs_by_level["100"]),
        "000_vs_060": auc(zs_by_level["000"], zs_by_level["060"]),
    }
    print("[auroc] clean(000+020) vs high(060-100)=%.3f | 000vs100=%.3f 020vs100=%.3f 000vs060=%.3f"
          % (out["auroc"]["clean_vs_high"], out["auroc"]["000_vs_100"],
             out["auroc"]["020_vs_100"], out["auroc"]["000_vs_060"]))
    (OUT / ("zdiag_%s.json" % args.fold)).write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[out] %s" % (OUT / ("zdiag_%s.json" % args.fold)))


if __name__ == "__main__":
    main()
