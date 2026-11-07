#!/usr/bin/env python3
"""Jitter_pred 窗级相关性的混淆变量检验（2026-11-06）。

上一轮得到 ρ(jerk_pred, V_lower_error) 每档 +0.33…+0.71。有两种解释：

  A. V 输入不可靠 → 解码抖动 → jerk 高且误差大（真可靠性信号）；
  B. 窗内动作快/复杂 → GT jerk 与误差同时被「动作难度」推高（混淆）。

本脚本在与上一轮完全相同的窗上，额外计算 GT 的窗级 jerk（同一
temporal_metrics 口径）和 GT 根速度，然后报：

  r_pe  = ρ(jerk_pred, V_lower)
  r_pg  = ρ(jerk_pred, GT_jerk)
  r_eg  = ρ(GT_jerk, V_lower)
  partial_g   = ρ(jerk_pred, V_lower | GT_jerk)          （秩偏相关）
  partial_gs  = ρ(jerk_pred, V_lower | GT_jerk, GT_speed)
  分位 ρ：按 GT_jerk 四分位分箱后箱内的 ρ(jerk_pred, V_lower)

判读：partial_g 若接近 0 → 解释 B 为主，jerk 只是动作难度，门会在快动作
窗系统性地压 V（危险）；若 partial 仍显著为正 → A 有独立成分。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/jitter_confound_pilot.py --device cuda
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
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import CONFIG_VT, FPS  # noqa: E402
from anysole.utils.geometry import fk_pose6d  # noqa: E402
from anysole.utils.metrics import pelvis_align, temporal_metrics  # noqa: E402

CELL = "V8_Cv5_fS5S6S11_s1"
FOLD = "fS5S6S11"
LEVELS = ("000", "020", "040", "060", "080", "100")
OUT = REPO / "results/diagnostics/20261106_jitter_confound_pilot"


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


def rank(v):
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v)
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(len(v), dtype=np.float64)
    for val in np.unique(v):
        idx = np.where(v == val)[0]
        r[idx] = r[idx].mean()
    return r


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    x = x - x.mean(); y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def spearman(x, y) -> float:
    return pearson(rank(x), rank(y))


def partial_spearman(x, y, controls) -> float:
    """秩偏相关：对 rank(x)/rank(y) 分别回归 controls 的秩，取残差相关。"""
    rx, ry = rank(x), rank(y)
    C = np.stack([rank(c) for c in controls], axis=1)
    C = np.concatenate([np.ones((C.shape[0], 1)), C], axis=1)
    bx, *_ = np.linalg.lstsq(C, rx, rcond=None)
    by, *_ = np.linalg.lstsq(C, ry, rcond=None)
    return pearson(rx - C @ bx, ry - C @ by)


def window_pred_jerk(out, batch) -> np.ndarray:
    pred_kp = fk_pose6d(out["x0_hat"], out["trans_hat"],
                        batch["offsets"], batch["parents"])
    tw = pred_kp.shape[1]
    times = np.arange(tw, dtype=np.float64) / float(FPS)
    vals = []
    for b in range(pred_kp.shape[0]):
        aligned, _ = pelvis_align(pred_kp[b].detach().cpu().numpy())
        vals.append(float(temporal_metrics(aligned, aligned, times)["jitter_pred_m_s3"]))
    return np.asarray(vals)


def window_gt_jerk_speed(batch) -> tuple[np.ndarray, np.ndarray]:
    kp = batch["kp_gt"].detach().cpu().numpy()  # (B,tw,J,3)
    tw = kp.shape[1]
    times = np.arange(tw, dtype=np.float64) / float(FPS)
    jerks, speeds = [], []
    for b in range(kp.shape[0]):
        aligned, _ = pelvis_align(kp[b])
        jerks.append(float(temporal_metrics(aligned, aligned, times)["jitter_gt_m_s3"]))
        root = kp[b, :, 0, :]
        speeds.append(float(np.linalg.norm(np.diff(root, axis=0), axis=-1).mean()
                            * float(FPS)))
    return np.asarray(jerks), np.asarray(speeds)


def run_level(net, loader, device, max_batches: int = 0):
    rows = []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = move_batch(batch, device)
        cid = torch.full((batch["V_feat"].shape[0],), CONFIG_VT,
                         dtype=torch.long, device=device)
        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
        with torch.no_grad():
            out = net(v_feat, t_raw, t_phys, cid, batch.get("subject_ids"), T_s2m=t_s2m)
        jp = window_pred_jerk(out, batch)
        jg, sp = window_gt_jerk_speed(batch)
        for sid, st, a, b, c in zip(batch["session_id"],
                                    batch["frame_start"].detach().cpu().tolist(),
                                    jp, jg, sp):
            rows.append({"session": sid, "start": int(st),
                         "jerk_pred": float(a), "jerk_gt": float(b),
                         "speed_gt": float(c)})
    return rows


def analyze(level: str, rows: list[dict]) -> dict:
    valid = []
    for r in rows:
        e = load_window_errors(level, r["session"]).get(r["start"])
        if e is not None:
            r = {**r, "v_lower": float(e)}
            valid.append(r)
    if len(valid) < 8:
        return {"level": level, "n": len(valid)}
    p = np.array([r["jerk_pred"] for r in valid])
    e = np.array([r["v_lower"] for r in valid])
    g = np.array([r["jerk_gt"] for r in valid])
    s = np.array([r["speed_gt"] for r in valid])
    # stratified by GT jerk quartile (within level)
    q = np.quantile(g, [0.25, 0.5, 0.75])
    bins = np.digitize(g, q)
    strat = []
    for k in range(4):
        m = bins == k
        strat.append(spearman(p[m], e[m]) if m.sum() >= 5 else float("nan"))
    return {"level": level, "n": len(valid),
            "r_pe": spearman(p, e), "r_pg": spearman(p, g), "r_eg": spearman(g, e),
            "partial_g": partial_spearman(p, e, [g]),
            "partial_gs": partial_spearman(p, e, [g, s]),
            "strat_q1_q4": strat,
            "mean_speed": float(s.mean()), "mean_gt_jerk": float(g.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    device = torch.device(args.device)
    net, ckpt, t_phys_v2 = build_model(CELL, device)
    print("[ckpt] %s" % ckpt)
    fold_yaml = FOLD[1:] if FOLD.startswith("f") else FOLD
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for level in levels:
        config = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                  else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
        loader = build_loader(config, "test", args.batch_size, t_phys_v2=t_phys_v2)
        rows = run_level(net, loader, device, args.max_batches)
        r = analyze(level, rows)
        results.append(r)
        print("[occ%s] n=%d r_pe=%+.3f r_pg=%+.3f r_eg=%+.3f partial_g=%+.3f partial_gs=%+.3f strat=%s"
              % (level, r.get("n", 0), r.get("r_pe", float("nan")),
                 r.get("r_pg", float("nan")), r.get("r_eg", float("nan")),
                 r.get("partial_g", float("nan")), r.get("partial_gs", float("nan")),
                 " ".join("%+.2f" % v for v in r.get("strat_q1_q4", []))))
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        keys = [k for k in results[0] if k != "strat_q1_q4"]
        w = csv.DictWriter(fh, fieldnames=keys + ["strat_q1_q4"])
        w.writeheader()
        for r in results:
            w.writerow({**r, "strat_q1_q4": ";".join("%.3f" % v for v in r.get("strat_q1_q4", []))})
    (out_dir / "report.json").write_text(json.dumps(
        {"cell": CELL, "fold": FOLD, "levels": list(LEVELS), "results": results},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[out] %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
