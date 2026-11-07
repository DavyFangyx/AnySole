#!/usr/bin/env python3
"""收缩 / V 信任拆分（2026-11-07）：常数 w 基线 + (a*,b*) + 加权 IG~b*。

1. 常数 w 基线：train 上拟全局 w（最小二乘），test 上比较 (1−w)T+wV 与 VT 行。
2. (a*,b*)：ŷ = T + a(∅−T) + b(V−∅)，逐窗解 2×2 正规方程；
   a* = 向先验收缩，b* = V 超出先验的信任。
3. IG~b*：按 ||V−∅||² 加权（b* 在 V≈∅ 时病态但权重≈0），test 上检验
   「接触信息」与「姿态信任」是否同一类量。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/ab_decomp.py \
      --cell V8_Cv5_fS5S6S11_s1 --fold fS5S6S11 --device cuda
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

from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402
from stage2_gate_pilot import pred_kp, win_lower_pa  # noqa: E402
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import CONFIG_T, CONFIG_V, CONFIG_VT  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
S2_DIR = REPO / "results/diagnostics/20261107_stage2_gate_pilot"
OUT = REPO / "results/diagnostics/20261107_ab_decomp"


def solve_ab(p_v, p_t, p_0, gt):
    d1 = p_0 - p_t
    d2 = p_v - p_0
    r = gt - p_t
    a11 = (d1 * d1).sum(axis=(2, 3))
    a22 = (d2 * d2).sum(axis=(2, 3))
    a12 = (d1 * d2).sum(axis=(2, 3))
    b1 = (d1 * r).sum(axis=(2, 3))
    b2 = (d2 * r).sum(axis=(2, 3))
    det = a11 * a22 - a12 * a12
    ok = np.abs(det) > 1e-8
    a = np.where(ok, (a22 * b1 - a12 * b2) / np.where(ok, det, 1), np.nan)
    b = np.where(ok, (a11 * b2 - a12 * b1) / np.where(ok, det, 1), np.nan)
    return a, b, a22


def run_rows(net, loader, device, split, level):
    keys = ("session", "start", "p_v", "p_t", "p_0", "gt",
            "a", "b", "w2", "e_v", "e_t", "e_p", "e_vt", "v_tok", "t_tok")
    rows = {k: [] for k in keys}
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
            vf, tr, tp, s2 = condition_inputs(batch, cid_v)
            out_0 = net(torch.zeros_like(vf), tr, tp, cid_v,
                        batch.get("subject_ids"), T_s2m=s2)
            vf, tr, tp, s2 = condition_inputs(batch, cid_vt)
            out_vt = net(vf, tr, tp, cid_vt, batch.get("subject_ids"), T_s2m=s2)
        p_v = pred_kp(out_v, batch); p_t = pred_kp(out_t, batch)
        p_0 = pred_kp(out_0, batch); p_vt = pred_kp(out_vt, batch)
        gt = batch["kp_gt"].detach().cpu().numpy()
        a, b, w2 = solve_ab(p_v, p_t, p_0, gt)
        rows["session"].extend(batch["session_id"])
        rows["start"].extend(batch["frame_start"].detach().cpu().tolist())
        rows["p_v"].append(p_v); rows["p_t"].append(p_t)
        rows["p_0"].append(p_0); rows["gt"].append(gt)
        rows["a"].append(np.nanmean(a, axis=1)); rows["b"].append(np.nanmean(b, axis=1))
        rows["w2"].append(np.nanmean(w2, axis=1))
        rows["e_v"].append(win_lower_pa(p_v, gt)); rows["e_t"].append(win_lower_pa(p_t, gt))
        rows["e_p"].append(win_lower_pa(p_0, gt)); rows["e_vt"].append(win_lower_pa(p_vt, gt))
        rows["v_tok"].append(out_vt["v_tok"].mean(dim=1).detach().cpu().numpy())
        rows["t_tok"].append(out_vt["t_tok"].mean(dim=1).detach().cpu().numpy())
    out = {k: np.asarray(v) for k, v in rows.items() if k in ("session", "start")}
    for k in ("p_v", "p_t", "p_0", "gt", "a", "b", "w2", "e_v", "e_t", "e_p", "e_vt",
              "v_tok", "t_tok"):
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
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    net, ckpt, tp = build_model(args.cell, device)
    print("[ckpt] %s" % ckpt)
    fold_yaml = args.fold[1:] if args.fold.startswith("f") else args.fold
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for split in splits:
        for level in levels:
            cfg = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                   else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            loader = build_loader(cfg, split, args.batch_size, t_phys_v2=tp)
            data[(split, level)] = run_rows(net, loader, device, split, level)
            print("[%s/%s] n=%d" % (split, level, len(data[(split, level)]["a"])))

    # 常数 w：train 全档最小二乘 (1−w)T+wV vs GT
    tr = {k: np.concatenate([data[("train", l)][k] for l in levels])
          for k in ("p_v", "p_t", "gt")}
    num = float(np.sum((tr["p_v"] - tr["p_t"]) * (tr["gt"] - tr["p_t"])))
    den = float(np.sum((tr["p_v"] - tr["p_t"]) ** 2))
    w_const = float(np.clip(num / max(den, 1e-8), 0.0, 1.0))
    print("[const] w=%.4f" % w_const)

    report = {"cell": args.cell, "fold": args.fold, "w_const": w_const, "per_level": []}
    s2 = np.load(S2_DIR / "rows_test.npz") if (S2_DIR / "rows_test.npz").is_file() else None
    for split in [s for s in ("val", "test") if s in splits]:
        for level in levels:
            r = data[(split, level)]
            mix = (1.0 - w_const) * r["p_t"] + w_const * r["p_v"]
            e_mix = win_lower_pa(mix, r["gt"])
            row = {"split": split, "level": level, "n": len(r["a"]),
                   "a_mean": float(np.nanmean(r["a"])),
                   "b_mean": float(np.nanmean(r["b"])),
                   "b_p05": float(np.nanpercentile(r["b"], 5)),
                   "b_p95": float(np.nanpercentile(r["b"], 95)),
                   "e_v": float(np.mean(r["e_v"])), "e_t": float(np.mean(r["e_t"])),
                   "e_p": float(np.mean(r["e_p"])), "e_vt": float(np.mean(r["e_vt"])),
                   "e_mix_const": float(np.mean(e_mix)),
                   "vt_minus_mix_const": float(np.mean(r["e_vt"] - e_mix))}
            if s2 is not None and split == "test":
                m = s2["level"] == int(level)
                ig = s2["ig"][m]
                wgt = r["w2"]
                okm = np.isfinite(r["b"]) & np.isfinite(ig) & (wgt > 1e-6)
                if okm.sum() >= 5:
                    def rank(v):
                        o = np.argsort(v); rr = np.empty(len(v)); rr[o] = np.arange(len(v))
                        for val in np.unique(v):
                            idx = np.where(v == val)[0]; rr[idx] = rr[idx].mean()
                        return rr
                    rx, ry = rank(ig[okm]), rank(r["b"][okm])
                    ww = wgt[okm]
                    mx = np.sum(rx * ww) / ww.sum(); my = np.sum(ry * ww) / ww.sum()
                    cov = np.sum(ww * (rx - mx) * (ry - my))
                    den2 = np.sqrt(np.sum(ww * (rx - mx) ** 2) * np.sum(ww * (ry - my) ** 2))
                    row["ig_b_weighted_spearman"] = float(cov / den2) if den2 > 0 else float("nan")
                    row["ig_b_n"] = int(okm.sum())
            report["per_level"].append(row)
            print("[%s/%s] a*=%.2f b*=%.2f [p05 %.2f p95 %.2f] | e_v=%.1f e_t=%.1f e_p=%.1f e_vt=%.1f e_mix=%.1f | vt−mix=%+.1f%s"
                  % (split, level, row["a_mean"], row["b_mean"], row["b_p05"], row["b_p95"],
                     row["e_v"], row["e_t"], row["e_p"], row["e_vt"], row["e_mix_const"],
                     row["vt_minus_mix_const"],
                     "" if "ig_b_weighted_spearman" not in row else
                     " | IG~b*(w)=%+.2f" % row["ig_b_weighted_spearman"]))
    for split in splits:
        np.savez_compressed(
            out_dir / ("rows_%s.npz" % split),
            session=np.concatenate([data[(split, l)]["session"] for l in levels]),
            start=np.concatenate([data[(split, l)]["start"] for l in levels]),
            level=np.concatenate([np.full(len(data[(split, l)]["a"]), int(l)) for l in levels]),
            a=np.concatenate([data[(split, l)]["a"] for l in levels]),
            b=np.concatenate([data[(split, l)]["b"] for l in levels]),
            w2=np.concatenate([data[(split, l)]["w2"] for l in levels]),
            e_v=np.concatenate([data[(split, l)]["e_v"] for l in levels]),
            e_t=np.concatenate([data[(split, l)]["e_t"] for l in levels]),
            e_p=np.concatenate([data[(split, l)]["e_p"] for l in levels]),
            e_vt=np.concatenate([data[(split, l)]["e_vt"] for l in levels]))
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print("[out] %s" % out_dir)


if __name__ == "__main__":
    main()
