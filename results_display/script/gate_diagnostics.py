#!/usr/bin/env python3
"""门头诊断 D1/D2/D3（2026-11-07，不重训）。

D1 标签信度：同一窗拆两半（帧 0:10 / 10:20）各算一个池化 w*，跨窗相关 r，
    Spearman–Brown 整窗信度 = 2r/(1+r)；门头与 g* 的相关上限 ≈ √信度。
D2 ∅ 对照：V 行换成 V_feat 全零（模型先验输出），重算 w* 与 oracle 增益；
    若低遮挡段与原始 g*/增益差别不大，g* 主要在测收缩效应。
D3 IG 混淆：控制 e_0 后看 IG~g* 的偏相关；负号消失/翻正即证实
    「IG 代理 T 强弱」的混淆。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/gate_diagnostics.py \
      --cell V8_Cv5_fS5S6S11_s1 --fold fS5S6S11 --device cuda
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
from stage2_gate_pilot import (GateHead, feats, oracle_g, pred_kp,  # noqa: E402
                               spearman, win_lower_pa)
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import CONFIG_T, CONFIG_V, CONFIG_VT  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
IG_DIR = REPO / "results/diagnostics/20261107_ig_contact_pilot"
S2_DIR = REPO / "results/diagnostics/20261107_stage2_gate_pilot"
OUT = REPO / "results/diagnostics/20261107_gate_diagnostics"


def w_pooled(pred_v, pred_t, gt, sl):
    """池化 half 帧的 w*：<V−T, GT−T> / ||V−T||² on frames sl。"""
    d = pred_v[:, sl] - pred_t[:, sl]
    r = gt[:, sl] - pred_t[:, sl]
    num = (d * r).sum(axis=(1, 2, 3))
    den = (d * d).sum(axis=(1, 2, 3))
    return num / np.maximum(den, 1e-8)


def run_rows(net, loader, device, split, level, max_batches=0):
    rows = {k: [] for k in ("session", "start", "w_half1", "w_half2",
                            "g_star", "g_star_black", "gain", "gain_black")}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = move_batch(batch, device)
        n = batch["V_feat"].shape[0]
        cid_vt = torch.full((n,), CONFIG_VT, dtype=torch.long, device=device)
        cid_v = torch.full((n,), CONFIG_V, dtype=torch.long, device=device)
        cid_t = torch.full((n,), CONFIG_T, dtype=torch.long, device=device)
        with torch.no_grad():
            vf, tr, tp, s2 = condition_inputs(batch, cid_v)
            out_v = net(vf, tr, tp, cid_v, batch.get("subject_ids"), T_s2m=s2)
            vf, tr, tp, s2 = condition_inputs(batch, cid_t)
            out_t = net(vf, tr, tp, cid_t, batch.get("subject_ids"), T_s2m=s2)
            # V=black：V_feat 全零的 V 行（模型先验输出）
            vf, tr, tp, s2 = condition_inputs(batch, cid_v)
            out_b = net(torch.zeros_like(vf), tr, tp, cid_v,
                        batch.get("subject_ids"), T_s2m=s2)
        p_v = pred_kp(out_v, batch); p_t = pred_kp(out_t, batch); p_b = pred_kp(out_b, batch)
        gt = batch["kp_gt"].detach().cpu().numpy()
        tw = p_v.shape[1]; mid = tw // 2
        sl1 = slice(0, mid); sl2 = slice(mid, tw)
        w1 = np.clip(w_pooled(p_v, p_t, gt, sl1), 0, 1)
        w2 = np.clip(w_pooled(p_v, p_t, gt, sl2), 0, 1)
        g = oracle_g(p_v, p_t, gt).mean(axis=1)
        gb = oracle_g(p_b, p_t, gt).mean(axis=1)
        e_v = win_lower_pa(p_v, gt); e_t = win_lower_pa(p_t, gt)
        e_b = win_lower_pa(p_b, gt)
        gain = e_t - (g * e_v + (1 - g) * e_t)
        gain_b = e_t - (gb * e_b + (1 - gb) * e_t)
        rows["session"].extend(batch["session_id"])
        rows["start"].extend(batch["frame_start"].detach().cpu().tolist())
        rows["w_half1"].append(w1); rows["w_half2"].append(w2)
        rows["g_star"].append(g); rows["g_star_black"].append(gb)
        rows["gain"].append(gain); rows["gain_black"].append(gain_b)
    out = {k: np.asarray(v) for k, v in rows.items() if k in ("session", "start")}
    for k in ("w_half1", "w_half2", "g_star", "g_star_black", "gain", "gain_black"):
        out[k] = np.concatenate(rows[k])
    out["split"] = split; out["level"] = level
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="V8_Cv5_fS5S6S11_s1")
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    net, ckpt, tp = build_model(args.cell, device)
    print("[ckpt] %s" % ckpt)
    head = ContactHead(256).to(device)
    head.load_state_dict(torch.load(IG_DIR / "contact_head.pt", map_location="cpu"))
    head.eval(); head.requires_grad_(False)
    gate = GateHead(256).to(device)
    gate.load_state_dict(torch.load(S2_DIR / "gate_head.pt", map_location="cpu"))
    gate.eval()
    fold_yaml = args.fold[1:] if args.fold.startswith("f") else args.fold
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    report = {"cell": args.cell, "fold": args.fold, "per_level": []}

    for split in splits:
        ig_rows = np.load(IG_DIR / ("rows_%s.npz" % split))
        s2_rows = np.load(S2_DIR / ("rows_%s.npz" % split))
        for level in levels:
            cfg = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                   else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            loader = build_loader(cfg, split, args.batch_size, t_phys_v2=tp)
            r = run_rows(net, loader, device, split, level)
            m = (s2_rows["level"] == int(level))
            ig = s2_rows["ig"][m]; g = s2_rows["g_star"][m]
            # D3: e_0 from contact head on zero tokens + contact_gt (IG rows, same order)
            cm = (ig_rows["level"] == int(level))
            contact = torch.from_numpy(ig_rows["contact"][cm]).float().to(device)
            v_zero = torch.zeros_like(torch.from_numpy(ig_rows["v_tok"][cm]).float().to(device))
            with torch.no_grad():
                e0 = F.binary_cross_entropy_with_logits(head(v_zero), contact,
                                                        reduction="none").mean(dim=(1, 2)).cpu().numpy()
            # D1: reliability and upper bound
            r_sb_raw = spearman(r["w_half1"], r["w_half2"])
            r_sb = 2 * r_sb_raw / (1 + r_sb_raw) if np.isfinite(r_sb_raw) and r_sb_raw > -1 else float("nan")
            upper = np.sqrt(max(r_sb, 0.0)) if np.isfinite(r_sb) else float("nan")
            # D3: partial IG~g controlling e_0 (rank partial)
            def rank(v):
                o = np.argsort(v); rr = np.empty(len(v)); rr[o] = np.arange(len(v))
                for val in np.unique(v):
                    idx = np.where(v == val)[0]; rr[idx] = rr[idx].mean()
                return rr
            X = np.stack([np.ones_like(e0), rank(e0)], axis=1)
            bx, *_ = np.linalg.lstsq(X, rank(ig), rcond=None)
            by, *_ = np.linalg.lstsq(X, rank(g), rcond=None)
            ex = rank(ig) - X @ bx; ey = rank(g) - X @ by
            ex -= ex.mean(); ey -= ey.mean()
            den = np.sqrt((ex * ex).sum() * (ey * ey).sum())
            partial = float((ex * ey).sum() / den) if den > 0 else float("nan")
            # gate head ghat~g* on the same rows (reference for D1)
            xg = torch.from_numpy(np.concatenate([s2_rows["v_tok"][m], s2_rows["t_tok"][m],
                                                  ig[:, None]], axis=1)).float().to(device)
            with torch.no_grad():
                p = gate(xg).cpu().numpy()
            row = {"split": split, "level": level, "n": len(r["g_star"]),
                   "D1_r_half": float(r_sb_raw), "D1_reliability": float(r_sb),
                   "D1_upper_sqrt": float(upper),
                   "D1_ghat_vs_g": spearman(p, g),
                   "D2_g_star": float(np.mean(r["g_star"])),
                   "D2_g_star_black": float(np.mean(r["g_star_black"])),
                   "D2_gain": float(np.mean(r["gain"])),
                   "D2_gain_black": float(np.mean(r["gain_black"])),
                   "D3_ig_vs_g": spearman(ig, g),
                   "D3_ig_vs_g_control_e0": partial}
            report["per_level"].append(row)
            print("[%s/%s] D1 r=%.2f rel=%.2f upper=%.2f ghat~g*=%+.3f | "
                  "D2 g*=%.2f black=%.2f gain=%.1f black_gain=%.1f | "
                  "D3 IG~g*=%+.2f partial(e0)=%+.2f"
                  % (split, level, r_sb_raw, r_sb, upper, row["D1_ghat_vs_g"],
                     row["D2_g_star"], row["D2_g_star_black"],
                     row["D2_gain"], row["D2_gain_black"],
                     row["D3_ig_vs_g"], row["D3_ig_vs_g_control_e0"]))
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print("[out] %s" % out_dir)


if __name__ == "__main__":
    main()
