#!/usr/bin/env python3
"""四折泄漏核对 + 交叉拟合 oracle（2026-11-07）。

对四个 Cv5 折（fS5S6S11 / fS5S11S12 / fS6S7 / fS5S6S8）：

1. train/val/test × 6 档的 V/T/VT 行 PA —— 看 T 行过拟合与排序翻转是否普遍。
2. val-fit：在 val（对主干真 held-out）上拟合常数 w/(a,b)，test 上评估，
   作为无泄漏基线；报 b*（val-fit）与 IG~b*。
3. 交叉拟合窗级 oracle：同窗前 half 帧选 w/(a,b)，后 half 评估，再反向；
   平均后剩下的是门头真正可能拿到的上界。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/leak_check_crossfit.py --device cuda
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
from pa_grid_oracle import (A_GRID, B_GRID, W_GRID, const_ab, const_w)  # noqa: E402
from stage2_gate_pilot import win_lower_pa  # noqa: E402
from ig_contact_pilot import ContactHead, train_head, run_split as ig_run_split  # noqa: E402
from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
LEVELS = ("000", "020", "040", "060", "080", "100")
OUT = REPO / "results/diagnostics/20261107_leak_check_crossfit"


def crossfit_w(p_v, p_t, gt):
    """逐窗：half 帧选 w，另一半评估，双向取平均。"""
    H = p_v.shape[1] // 2
    evals = []
    for sel, ev in ((slice(0, H), slice(H, None)), (slice(H, None), slice(0, H))):
        errs = np.stack([win_lower_pa((1 - w) * p_t[:, sel] + w * p_v[:, sel],
                                      gt[:, sel]) for w in W_GRID], axis=1)  # (B,21)
        w = W_GRID[np.argmin(errs, axis=1)]
        sh = (-1, 1, 1, 1)
        mix = (1 - w.reshape(sh)) * p_t[:, ev] + w.reshape(sh) * p_v[:, ev]
        evals.append(win_lower_pa(mix, gt[:, ev]))
    return np.mean(np.stack(evals, 0), 0)


def crossfit_ab(p_v, p_t, p_0, gt):
    """逐窗：half 帧选 (a,b)，另一半评估，双向取平均；返回 per-window b*。"""
    H = p_v.shape[1] // 2
    evals, bs = [], []
    for sel, ev in ((slice(0, H), slice(H, None)), (slice(H, None), slice(0, H))):
        B = len(gt)
        best_e = np.full(B, np.inf); best_a = np.zeros(B); best_b = np.zeros(B)
        for a in A_GRID:
            for b in B_GRID:
                mix = p_t[:, sel] + a * (p_0[:, sel] - p_t[:, sel]) + b * (p_v[:, sel] - p_0[:, sel])
                e = win_lower_pa(mix, gt[:, sel])
                upd = e < best_e
                best_e[upd] = e[upd]; best_a[upd] = a; best_b[upd] = b
        sh = (-1, 1, 1, 1)
        mix_ev = (p_t[:, ev] + best_a.reshape(sh) * (p_0[:, ev] - p_t[:, ev])
                  + best_b.reshape(sh) * (p_v[:, ev] - p_0[:, ev]))
        evals.append(win_lower_pa(mix_ev, gt[:, ev]))
        bs.append(best_b)
    b_pw = np.mean(np.stack(bs, 0), 0)
    return np.mean(np.stack(evals, 0), 0), float(np.mean(b_pw)), b_pw


def ig_b(head, ig_rows, b_star, level, device):
    # ig_rows 已经是单档（ig_run_split 的返回值），level 仅保留给调用签名。
    x = torch.from_numpy(ig_rows["v_tok"]).float().to(device)
    c = torch.from_numpy(ig_rows["contact"]).float().to(device)
    with torch.no_grad():
        ev = torch.nn.functional.binary_cross_entropy_with_logits(
            head(x), c, reduction="none").mean(dim=(1, 2)).cpu().numpy()
        e0 = torch.nn.functional.binary_cross_entropy_with_logits(
            head(torch.zeros_like(x)), c, reduction="none").mean(dim=(1, 2)).cpu().numpy()
    ig = 1.0 - ev / np.maximum(e0, 1e-6)
    ok = np.isfinite(b_star) & np.isfinite(ig) & (e0 > 1e-4)
    if ok.sum() < 5:
        return float("nan")
    def rank(v):
        o = np.argsort(v); rr = np.empty(len(v)); rr[o] = np.arange(len(v))
        for val in np.unique(v):
            idx = np.where(v == val)[0]; rr[idx] = rr[idx].mean()
        return rr
    rx, ry = rank(ig[ok]), rank(b_star[ok])
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def run_fold(fold, device, batch, levels, splits, do_ig=True, out_dir=None):
    cell = "V8_Cv5_%s_s1" % fold
    net, ckpt, tp = build_model(cell, device)
    fold_yaml = fold[1:] if fold.startswith("f") else fold
    data, igdata = {}, {}
    for split in splits:
        for level in levels:
            cfg = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                   else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            loader = build_loader(cfg, split, batch, t_phys_v2=tp)
            data[(split, level)] = run_rows(net, loader, device, split, level)
            if do_ig:
                loader = build_loader(cfg, split, batch, t_phys_v2=tp)
                igdata[(split, level)] = ig_run_split(net, loader, device, split, level)
    out = {"fold": fold, "ckpt": str(ckpt), "rows": data}

    # val-fit constants
    va = {k: np.concatenate([data[("val", l)][k] for l in levels])
          for k in ("p_v", "p_t", "p_0", "gt")}
    w_v, _ = const_w(va["p_v"], va["p_t"], va["gt"])
    e_v, a_v, b_v = const_ab(va["p_v"], va["p_t"], va["p_0"], va["gt"])
    out["val_const"] = {"w": w_v, "a": a_v, "b": b_v}
    # 按档常数：每档在 val 上拟合 (a,b)
    plc = {}
    for level in levels:
        va_l = data[("val", level)]
        e_l, a_l, b_l = const_ab(va_l["p_v"], va_l["p_t"], va_l["p_0"], va_l["gt"])
        plc[level] = {"a": float(a_l), "b": float(b_l), "val_pa": float(e_l)}
    out["per_level_const"] = plc
    # contact head if IG
    head = None
    if do_ig:
        tr_ig = {k: np.concatenate([igdata[("train", l)][k] for l in levels])
                 for k in ("v_tok", "contact", "e_pose", "jerk_gt", "speed_gt")}
        va_ig = {k: np.concatenate([igdata[("val", l)][k] for l in levels])
                 for k in ("v_tok", "contact", "e_pose", "jerk_gt", "speed_gt")}
        head = train_head(tr_ig, va_ig, device, v_drop=0.25)
    out["per_level"] = []
    for level in levels:
        r = data[("test", level)]
        e_vt = r["e_vt"]
        # val-fit constant on test
        mix = (1 - w_v) * r["p_t"] + w_v * r["p_v"]
        e_c = win_lower_pa(mix, r["gt"])
        mix_ab = r["p_t"] + a_v * (r["p_0"] - r["p_t"]) + b_v * (r["p_v"] - r["p_0"])
        e_ab_c = win_lower_pa(mix_ab, r["gt"])
        # cross-fit oracle
        e_w = crossfit_w(r["p_v"], r["p_t"], r["gt"])
        e_ab, b_cf, b_cf_pw = crossfit_ab(r["p_v"], r["p_t"], r["p_0"], r["gt"])
        pl = plc[level]
        mix_l = (r["p_t"] + pl["a"] * (r["p_0"] - r["p_t"])
                 + pl["b"] * (r["p_v"] - r["p_0"]))
        e_l = win_lower_pa(mix_l, r["gt"])
        row = {"level": level, "n": len(e_vt),
               "gain_w_crossfit": float(np.mean(e_vt - e_w)),
               "gain_w_valconst": float(np.mean(e_vt - e_c)),
               "gain_ab_crossfit": float(np.mean(e_vt - e_ab)),
               "gain_ab_valconst": float(np.mean(e_vt - e_ab_c)),
               "gain_ab_levelconst": float(np.mean(e_vt - e_l)),
               "b_crossfit": b_cf, "b_valconst": b_v,
               "a_levelconst": pl["a"], "b_levelconst": pl["b"]}
        if head is not None:
            row["ig_b_crossfit"] = ig_b(head, igdata[("test", level)], b_cf_pw, level, device)
            row["ig_b_valfit"] = ig_b(head, igdata[("test", level)],
                                      np.full(len(e_vt), b_v), level, device)
        out["per_level"].append(row)
    if do_ig:
        torch.save(head.state_dict(), Path(out_dir or OUT) / ("head_%s.pt" % fold))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--folds", default=",".join(FOLDS))
    ap.add_argument("--no-ig", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    folds = tuple(s.strip() for s in args.folds.split(",") if s.strip())
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    report = {"folds": list(folds), "levels": list(levels), "per_fold": []}
    for fold in folds:
        print("===== %s" % fold)
        out = run_fold(fold, device, args.batch_size, levels, ("train", "val", "test"),
                       do_ig=not args.no_ig, out_dir=out_dir)
        # train/val/test table
        for split in ("train", "val", "test"):
            for level in levels:
                r = out["rows"][(split, level)]
                print("[%s/%s] V=%.1f T=%.1f VT=%.1f VT-T=%+.1f"
                      % (split, level, np.mean(r["e_v"]), np.mean(r["e_t"]),
                         np.mean(r["e_vt"]), np.mean(r["e_vt"] - r["e_t"])))
        print("[%s] val const w=%.3f (a,b)=(%.2f,%.2f)" % (fold, out["val_const"]["w"],
                                                           out["val_const"]["a"],
                                                           out["val_const"]["b"]))
        for row in out["per_level"]:
            print("[%s/%s] w: cf=%+.1f valconst=%+.1f | ab: cf=%+.1f valconst=%+.1f levelconst=%+.1f | b_cf=%.2f%s"
                  % (fold, row["level"], row["gain_w_crossfit"], row["gain_w_valconst"],
                     row["gain_ab_crossfit"], row["gain_ab_valconst"], row["gain_ab_levelconst"], row["b_crossfit"],
                     "" if "ig_b_crossfit" not in row else
                     " | IG~b: cf=%+.2f valfit=%+.2f" % (row["ig_b_crossfit"], row["ig_b_valfit"])))
        # strip the big arrays before json
        report["per_fold"].append({k: v for k, v in out.items() if k != "rows"})
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print("[out] %s" % out_dir)


if __name__ == "__main__":
    main()
