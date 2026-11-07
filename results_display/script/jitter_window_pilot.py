#!/usr/bin/env python3
"""Jitter_pred 逐窗验证 + ST 场景验证（2026-11-06）。

背景：E5 的 Jitter_pred 在 occ 轴上干净 604 -> occ100 3219（×5.3，11 档
单调），但记忆里 T2M 三失效解耦记过「抖动 91% 集中在窗接缝」——窗口边界
是独立抖动源，可能把聚合曲线拉成假单调。本脚本做两件事：

1. 把 Jitter_pred 从会话聚合降到**逐窗**（每窗 20 帧，pelvis-align 后按
   eval 同一定义 `temporal_metrics`），在 occ000/010/020/040/060/080/100
   上重算；报窗级 Spearman（jerk vs occ）与每档内 vs V 行 lower 误差。
2. 在 ST 那一折（V8_Cv5ST_fS5S6S11_s1）的 test split 上，对同一批 VT 窗
   分别跑 clean V 和**同会话换窗 V**（复用 train._stale_partner），比较
   配对 jerk；这是 ST 场景（真实污染方式）的直接验证。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/jitter_window_pilot.py --device cuda
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
from anysole.train import (condition_inputs, move_batch,  # noqa: E402
                           _stale_partner)
from anysole.types import CONFIG_VT, FPS  # noqa: E402
from anysole.utils.geometry import fk_pose6d  # noqa: E402
from anysole.utils.metrics import pelvis_align, temporal_metrics  # noqa: E402

CELL = "V8_Cv5_fS5S6S11_s1"
ST_CELL = "V8_Cv5ST_fS5S6S11_s1"
FOLD = "fS5S6S11"
OCC = ("000", "010", "020", "040", "060", "080", "100")
OUT = REPO / "results/diagnostics/20261106_jitter_window_pilot"


def load_window_errors(level: str, sid: str) -> dict:
    if level == "000":
        path = REPO / "results/cells" / CELL / "metrics/window_detail_test.json"
    else:
        path = (REPO / "results/cells" / ("%s_evalocc%s" % (CELL, level))
                / "metrics/window_detail_test.json")
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))["conditions"]["V2M"]
    out = {}
    for s, start, lo in zip(data["session_ids"], data["frame_starts"],
                            data["metrics"]["PA-MPJPE_lower"]):
        if str(s) == sid:
            out[int(start)] = float(lo)
    return out


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        for val in np.unique(v):
            idx = np.where(v == val)[0]
            r[idx] = r[idx].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def window_jerks(out, batch) -> np.ndarray:
    pred_kp = fk_pose6d(out["x0_hat"], out["trans_hat"],
                        batch["offsets"], batch["parents"])
    tw = pred_kp.shape[1]
    times = np.arange(tw, dtype=np.float64) / float(FPS)
    vals = []
    for b in range(pred_kp.shape[0]):
        aligned, _ = pelvis_align(pred_kp[b].detach().cpu().numpy())
        vals.append(float(temporal_metrics(aligned, aligned, times)["jitter_pred_m_s3"]))
    return np.asarray(vals)


def run_occ(net, loader, device, max_batches: int = 0):
    jerks, sids, starts = [], [], []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = move_batch(batch, device)
        cid = torch.full((batch["V_feat"].shape[0],), CONFIG_VT,
                         dtype=torch.long, device=device)
        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
        with torch.no_grad():
            out = net(v_feat, t_raw, t_phys, cid, batch.get("subject_ids"),
                      T_s2m=t_s2m)
        jerks.append(window_jerks(out, batch))
        sids.extend(batch["session_id"])
        starts.extend(batch["frame_start"].detach().cpu().tolist())
    return np.concatenate(jerks), sids, starts


def run_st(net, loader, device, max_batches: int = 0):
    clean_vals, swap_vals, non_vals, n_swap = [], [], [], 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = move_batch(batch, device)
        cid = torch.full((batch["V_feat"].shape[0],), CONFIG_VT,
                         dtype=torch.long, device=device)
        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
        partner = _stale_partner(batch, device)
        if partner is None:
            continue
        n = v_feat.shape[0]
        self_idx = torch.arange(n, device=device)
        eligible = partner != self_idx
        swap = eligible & (torch.rand(n, device=device) < 1.0)
        if not bool(swap.any()):
            continue
        v_swap = v_feat.clone()
        v_swap[swap] = v_feat[partner[swap]]
        with torch.no_grad():
            out_c = net(v_feat, t_raw, t_phys, cid, batch.get("subject_ids"), T_s2m=t_s2m)
            out_s = net(v_swap, t_raw, t_phys, cid, batch.get("subject_ids"), T_s2m=t_s2m)
        jc = window_jerks(out_c, batch)
        js = window_jerks(out_s, batch)
        m = swap.detach().cpu().numpy()
        clean_vals.append(jc[m]); swap_vals.append(js[m])
        non_vals.append(jc[~m])
        n_swap += int(m.sum())
    cat = lambda xs: np.concatenate(xs) if xs else np.asarray([])
    return cat(clean_vals), cat(swap_vals), cat(non_vals), n_swap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--levels", default=",".join(OCC))
    ap.add_argument("--max-batches", type=int, default=0,
                    help=">0 只跑前 N 个 batch（冒烟用）")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    device = torch.device(args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 逐窗 jitter × occ ----
    net, ckpt, t_phys_v2 = build_model(CELL, device)
    print("[occ ckpt] %s" % ckpt)
    fold_yaml = FOLD[1:] if FOLD.startswith("f") else FOLD
    rows = []
    all_j, all_lv = [], []
    for level in levels:
        config = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                  else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
        loader = build_loader(config, "test", args.batch_size, t_phys_v2=t_phys_v2)
        jerks, sids, starts = run_occ(net, loader, device, args.max_batches)
        lv = int(level)
        all_j.append(jerks); all_lv.append(np.full(jerks.size, lv))
        # within-level vs V lower error
        xs, ys = [], []
        for sid, st, j in zip(sids, starts, jerks):
            e = load_window_errors(level, sid)
            if st in e:
                xs.append(j); ys.append(e[st])
        rows.append({"level": level, "n_windows": int(jerks.size),
                     "jerk_mean": float(np.mean(jerks)),
                     "jerk_median": float(np.median(jerks)),
                     "jerk_p05": float(np.percentile(jerks, 5)),
                     "rho_vs_vlower": spearman(xs, ys) if len(xs) >= 3 else float("nan"),
                     "n_corr": len(xs)})
        print("[occ%s] n=%d mean=%.1f median=%.1f rho_vs_vlower=%+.3f ncorr=%d"
              % (level, jerks.size, np.mean(jerks), np.median(jerks),
                 rows[-1]["rho_vs_vlower"], rows[-1]["n_corr"]))
    all_j = np.concatenate(all_j); all_lv = np.concatenate(all_lv)
    level_means = np.array([r["jerk_mean"] for r in rows])
    corr = {"window_rho_vs_occ": spearman(all_j, all_lv),
            "level_mean_rho_vs_occ": spearman(level_means, np.arange(len(rows)))}
    print("[occ] window-level rho(jerk, occ)=%+.3f  level-mean rho=%+.3f"
          % (corr["window_rho_vs_occ"], corr["level_mean_rho_vs_occ"]))

    # ---- 2) ST：clean vs 同会话换窗 ----
    net_st, ckpt_st, t_phys_v2_st = build_model(ST_CELL, device)
    print("[st ckpt] %s" % ckpt_st)
    loader = build_loader("anysole/configs/v1_fold_%s.yaml" % fold_yaml,
                          "test", args.batch_size, t_phys_v2=t_phys_v2_st)
    clean, swap, non, n_swap = run_st(net_st, loader, device, args.max_batches)
    diff = swap - clean
    st = {"cell": ST_CELL, "split": "test", "n_swap": n_swap,
          "clean_mean": float(np.mean(clean)) if clean.size else float("nan"),
          "swap_mean": float(np.mean(swap)) if swap.size else float("nan"),
          "non_mean": float(np.mean(non)) if non.size else float("nan"),
          "paired_diff_mean": float(np.mean(diff)) if diff.size else float("nan"),
          "paired_diff_median": float(np.median(diff)) if diff.size else float("nan"),
          "frac_positive": float((diff > 0).mean()) if diff.size else float("nan")}
    print("[st] n_swap=%d clean=%.1f swap=%.1f non=%.1f diff=%+.1f frac+=%.2f"
          % (n_swap, st["clean_mean"], st["swap_mean"], st["non_mean"],
             st["paired_diff_mean"], st["frac_positive"]))

    with (out_dir / "occ.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out_dir / "report.json").write_text(json.dumps(
        {"cell": CELL, "st_cell": ST_CELL, "fold": FOLD, "occ": rows,
         "occ_corr": corr, "st": st}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("[out] %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
