#!/usr/bin/env python3
"""PA 网格 oracle + 自检（2026-11-07）。

修正上一轮的两处定义问题：
1. 上一轮 D2 的 oracle 增益是 V/T 两条误差曲线的线性插值，不是真实混合
   预测的 PA 误差；常数混合增益却是真实 PA。两者不可比。
2. w*/a*/b* 在未对齐原始坐标里做最小二乘，PA 会消掉平移旋转；且 (∅−T)
   与 (V−∅) 近共线，b* 被放大到 ±3。

本脚本直接在评估指标（PA-MPJPE lower）上做逐窗网格搜索：
  w ∈ 21 格点； (a,b) ∈ 11×11 格点；每个格点用真实混合预测算 PA。
自检：每一档 oracle 增益（VT − 最优逐窗混合）必须 ≥ 最优常数混合增益；
不通过则 b*/IG 结论维持暂定。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/pa_grid_oracle.py \
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

from ab_decomp import run_rows  # noqa: E402
from stage2_gate_pilot import win_lower_pa  # noqa: E402
from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
S2_DIR = REPO / "results/diagnostics/20261107_stage2_gate_pilot"
OUT = REPO / "results/diagnostics/20261107_pa_grid_oracle"

W_GRID = np.linspace(0.0, 1.0, 21)
A_GRID = np.linspace(-0.5, 2.0, 11)
B_GRID = np.linspace(-1.0, 1.5, 11)


def w_oracle(p_v, p_t, gt):
    """逐窗 w 网格：返回 (w*, 最优 PA, VT 混合误差?)."""
    errs = np.stack([win_lower_pa((1 - w) * p_t + w * p_v, gt) for w in W_GRID], axis=1)
    idx = np.argmin(errs, axis=1)
    return W_GRID[idx], errs[np.arange(len(idx)), idx]


def ab_oracle(p_v, p_t, p_0, gt):
    best_e = np.full(len(gt), np.inf)
    best_a = np.zeros(len(gt)); best_b = np.zeros(len(gt))
    for a in A_GRID:
        for b in B_GRID:
            mix = p_t + a * (p_0 - p_t) + b * (p_v - p_0)
            e = win_lower_pa(mix, gt)
            upd = e < best_e
            best_e[upd] = e[upd]; best_a[upd] = a; best_b[upd] = b
    return best_a, best_b, best_e


def const_w(p_v, p_t, gt):
    errs = [np.mean(win_lower_pa((1 - w) * p_t + w * p_v, gt)) for w in W_GRID]
    i = int(np.argmin(errs))
    return float(W_GRID[i]), float(errs[i])


def const_ab(p_v, p_t, p_0, gt):
    best = (np.inf, 0.0, 0.0)
    for a in A_GRID:
        for b in B_GRID:
            e = float(np.mean(win_lower_pa(p_t + a * (p_0 - p_t) + b * (p_v - p_0), gt)))
            if e < best[0]:
                best = (e, float(a), float(b))
    return best


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
            print("[%s/%s] n=%d" % (split, level, len(data[(split, level)]["e_v"])))

    tr = {k: np.concatenate([data[("train", l)][k] for l in levels])
          for k in ("p_v", "p_t", "p_0", "gt")}
    w_const, _ = const_w(tr["p_v"], tr["p_t"], tr["gt"])
    e_ab, a_const, b_const = const_ab(tr["p_v"], tr["p_t"], tr["p_0"], tr["gt"])
    print("[const] w=%.3f  (a,b)=(%.2f,%.2f) PA=%.2f" % (w_const, a_const, b_const, e_ab))

    report = {"cell": args.cell, "fold": args.fold,
              "w_const": w_const, "a_const": a_const, "b_const": b_const,
              "per_level": []}
    s2 = np.load(S2_DIR / "rows_test.npz") if (S2_DIR / "rows_test.npz").is_file() else None
    for split in [s for s in ("val", "test") if s in splits]:
        for level in levels:
            r = data[(split, level)]
            e_vt = r["e_vt"]
            # w oracle
            w_star, e_w = w_oracle(r["p_v"], r["p_t"], r["gt"])
            mix_c = (1 - w_const) * r["p_t"] + w_const * r["p_v"]
            e_c = win_lower_pa(mix_c, r["gt"])
            gain_w = float(np.mean(e_vt - e_w))
            gain_c = float(np.mean(e_vt - e_c))
            # ab oracle
            a_star, b_star, e_ab_or = ab_oracle(r["p_v"], r["p_t"], r["p_0"], r["gt"])
            mix_ab = r["p_t"] + a_const * (r["p_0"] - r["p_t"]) + b_const * (r["p_v"] - r["p_0"])
            e_ab_c = win_lower_pa(mix_ab, r["gt"])
            gain_ab = float(np.mean(e_vt - e_ab_or))
            gain_ab_c = float(np.mean(e_vt - e_ab_c))
            row = {"split": split, "level": level, "n": len(w_star),
                   "w_star_mean": float(np.mean(w_star)),
                   "gain_w_oracle": gain_w, "gain_w_const": gain_c,
                   "check_w": bool(gain_w >= gain_c - 1e-6),
                   "gain_ab_oracle": gain_ab, "gain_ab_const": gain_ab_c,
                   "check_ab": bool(gain_ab >= gain_ab_c - 1e-6),
                   "a_star_mean": float(np.mean(a_star)),
                   "b_star_mean": float(np.mean(b_star)),
                   "b_star_p05": float(np.percentile(b_star, 5)),
                   "b_star_p95": float(np.percentile(b_star, 95))}
            if s2 is not None and split == "test":
                m = s2["level"] == int(level)
                ig = s2["ig"][m]
                wgt = r["w2"]
                okm = np.isfinite(b_star) & np.isfinite(ig) & (wgt > 1e-6)
                if okm.sum() >= 5:
                    def rank(v):
                        o = np.argsort(v); rr = np.empty(len(v)); rr[o] = np.arange(len(v))
                        for val in np.unique(v):
                            idx = np.where(v == val)[0]; rr[idx] = rr[idx].mean()
                        return rr
                    rx, ry = rank(ig[okm]), rank(b_star[okm])
                    ww = wgt[okm]
                    mx = np.sum(rx * ww) / ww.sum(); my = np.sum(ry * ww) / ww.sum()
                    cov = np.sum(ww * (rx - mx) * (ry - my))
                    den = np.sqrt(np.sum(ww * (rx - mx) ** 2) * np.sum(ww * (ry - my) ** 2))
                    row["ig_b_weighted_spearman"] = float(cov / den) if den > 0 else float("nan")
                    row["ig_b_n"] = int(okm.sum())
            report["per_level"].append(row)
            print("[%s/%s] w: or=%.1f const=%.1f %s | ab: or=%.1f const=%.1f %s | "
                  "w*=%.2f a*=%.2f b*=%.2f [%.2f,%.2f]%s"
                  % (split, level, gain_w, gain_c, "OK" if row["check_w"] else "FAIL",
                     gain_ab, gain_ab_c, "OK" if row["check_ab"] else "FAIL",
                     row["w_star_mean"], row["a_star_mean"], row["b_star_mean"],
                     row["b_star_p05"], row["b_star_p95"],
                     "" if "ig_b_weighted_spearman" not in row else
                     " | IG~b*(w)=%+.2f" % row["ig_b_weighted_spearman"]))
    for split in splits:
        np.savez_compressed(
            out_dir / ("rows_%s.npz" % split),
            session=np.concatenate([data[(split, l)]["session"] for l in levels]),
            start=np.concatenate([data[(split, l)]["start"] for l in levels]),
            level=np.concatenate([np.full(len(data[(split, l)]["e_v"]), int(l)) for l in levels]),
            e_v=np.concatenate([data[(split, l)]["e_v"] for l in levels]),
            e_t=np.concatenate([data[(split, l)]["e_t"] for l in levels]),
            e_p=np.concatenate([data[(split, l)]["e_p"] for l in levels]),
            e_vt=np.concatenate([data[(split, l)]["e_vt"] for l in levels]),
            w2=np.concatenate([data[(split, l)]["w2"] for l in levels]))
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print("[out] %s" % out_dir)


if __name__ == "__main__":
    main()
