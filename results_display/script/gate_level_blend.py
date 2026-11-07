#!/usr/bin/env python3
"""档位级可靠性门 + 输出空间融合（2026-11-08，修改 16 第一版）。

设计（与 11-07 的 VIG/VUG 特征门区分）：
- 信号 = 会话级 IG 中位，IG = 1 − BCE(h(V),c)/BCE(h(∅),c)；
  c 运行时应为 T 行 contact 伪标签（本脚本同时记录 GT 版作对照）。
- **per-session clean 参考**：同一会话的 000 档（clean）IG 中位作分母，
  s = IG_sess / IG_clean_sess。000 档只作无标签 clean 校准，不参与门
  阈值拟合（阈值仍由 val 拟合）。绝对 IG 有强受试者偏移（val clean
  ~0.9，test clean ~0.4），per-session 参考后 clean 档严格 s=1。
- 门 g(s) = 1 (s≥τ+w/2)，线性 ramp，0 (s≤τ−w/2)；会话级常数。
- 输出 y = (1−g)·y_T + g·y_VT；**g=0 严格等于 T 行，g=1 严格等于 VT 行**。
  这是为了绕开特征门 g=0≠T 的问题（VIG ckpt 单 batch 实测 trans_hat
  差可达 51.6mm）。
- **主指标 = 全身体 PA-MPJPE**（与 E5/E6 主表同口径）；lower-body PA
  只作诊断。2026-11-08 修正：用 lower-body 拟合会在 060 选中 g<1，
  下身体变好但全身体 PA 比 VT 差。
- (τ,w) 只在 val 上拟合；test 只评估一次。约束：val 000 税≤+0.5、
  val 020 税≤+2.0（val 的 020 比 test 更严；test 目标 ≤+0.5）、
  val 100 与 T 行差≤+0.5；目标 max gain_100 + .5 gain_080
  + .5 gain_060 + .25 gain_040；可行集内优先取更保守的低阈值。
- clean 参考 <0.10 的会话：IG 不可信，门钉 1（不动作），避免把 clean
  会话误判成高遮挡。
- 参照：g=1(VT)、g=0(T)、val-fit 逐档常数、test 逐窗 oracle。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/gate_level_blend.py --fold fS5S6S11 --device cuda:2
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
from ig_contact_pilot import ContactHead, train_head  # noqa: E402
from ig9_gate_smoke import collect_head_rows, ema_by_session  # noqa: E402
from stage2_gate_pilot import pred_kp, win_lower_pa  # noqa: E402
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import CONFIG_T, CONFIG_VT  # noqa: E402
from anysole.utils.metrics import mean_point_error, pa_mpjpe, pelvis_align  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
SUBJECTS = ("S14", "S13", "S12", "S11", "S8", "S7", "S6", "S5")
G_GRID = np.linspace(0.0, 1.0, 11)
WIDTHS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
EMA_ALPHA = 0.7
REF_FLOOR = 0.05   # clean IG 参考的下限；fS6S7 有 clean 会话 IG≈0，防除爆
REF_MIN = 0.10     # clean 参考低于此值的会话：IG 信号不可信，门钉 1（不动作）
TAX_000 = 0.5
TAX_020 = 2.0
TAU_QUANTILES = np.linspace(0.01, 0.99, 99)
OUT = REPO / "results/diagnostics/20261108_gate_level"


def subject_of(session):
    s = str(session)
    for subj in SUBJECTS:
        if s.startswith(subj):
            return subj
    return s[:3]


def session_median(values, sessions):
    out = np.empty_like(np.asarray(values, dtype=np.float64))
    for sess in np.unique(sessions):
        idx = sessions == sess
        out[idx] = np.median(np.asarray(values)[idx])
    return out


def blend(p_t, p_vt, g):
    return (1.0 - g)[:, None, None, None] * p_t + g[:, None, None, None] * p_vt


def per_window_pa(pred, gt):
    return np.asarray([float(pa_mpjpe(pred[b], gt[b]).mean())
                       for b in range(pred.shape[0])])


def per_window_mpjpe(pred, gt):
    """骨盆对齐 MPJPE（逐窗；不是 GT-anchored Global，只作诊断）。"""
    out = []
    for b in range(pred.shape[0]):
        p, _ = pelvis_align(pred[b])
        g, _ = pelvis_align(gt[b])
        out.append(float(mean_point_error(p, g, scale=1000.0).mean()))
    return np.asarray(out)


def gate_from_s(s, tau, width):
    if width <= 1e-9:
        return (s >= tau).astype(np.float64)
    lo = tau - 0.5 * width
    hi = tau + 0.5 * width
    return np.clip((s - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def precompute_E(p_t, p_vt, gt):
    """E[n, j] = 逐窗 **全身体 PA-MPJPE** at g=G_GRID[j]（主指标）。

    2026-11-08 修正：此前用 lower-body PA，导致 060 选中 g<1（下身体
    增益）但全身体 PA 反而比 VT 差（fS5S6S11 060: 76.8 vs 68.1）。门
    的拟合/选择必须用主表口径的全身体 PA；lower-body 只作诊断。
    """
    E = np.empty((p_t.shape[0], len(G_GRID)), dtype=np.float64)
    for j, g in enumerate(G_GRID):
        E[:, j] = per_window_pa(blend(p_t, p_vt, np.full(p_t.shape[0], g)), gt)
    return E


def collect_gate_rows(net, head, loader, device, split, level):
    rows = {k: [] for k in ("session", "subject", "start", "p_t", "p_vt", "gt",
                            "ig_gt", "ig_ps")}
    for batch in loader:
        batch = move_batch(batch, device)
        n = batch["V_feat"].shape[0]
        cid_t = torch.full((n,), CONFIG_T, dtype=torch.long, device=device)
        cid_vt = torch.full((n,), CONFIG_VT, dtype=torch.long, device=device)
        with torch.no_grad():
            vf, tr, tp, s2 = condition_inputs(batch, cid_t)
            out_t = net(vf, tr, tp, cid_t, batch.get("subject_ids"), T_s2m=s2)
            vf, tr, tp, s2 = condition_inputs(batch, cid_vt)
            out_vt = net(vf, tr, tp, cid_vt, batch.get("subject_ids"), T_s2m=s2)
            v_tok = out_vt["v_tok"]
            c_gt = batch["contact_gt"]
            c_t = torch.sigmoid(out_t["contact_logits"])

            def ig(c):
                e_v = F.binary_cross_entropy_with_logits(
                    head(v_tok), c, reduction="none").mean(dim=(1, 2))
                e_0 = F.binary_cross_entropy_with_logits(
                    head(torch.zeros_like(v_tok)), c, reduction="none").mean(dim=(1, 2))
                return (1.0 - e_v / e_0.clamp(min=1e-6)).cpu().numpy()

            ig_gt = ig(c_gt)
            ig_ps = ig(c_t)
        p_t = pred_kp(out_t, batch)
        p_vt = pred_kp(out_vt, batch)
        gt = batch["kp_gt"].detach().cpu().numpy()
        rows["session"].extend(batch["session_id"])
        rows["subject"].extend(subject_of(s) for s in batch["session_id"])
        rows["start"].extend(batch["frame_start"].detach().cpu().tolist())
        rows["p_t"].append(p_t); rows["p_vt"].append(p_vt); rows["gt"].append(gt)
        rows["ig_gt"].append(ig_gt); rows["ig_ps"].append(ig_ps)
    out = {"session": np.asarray(rows["session"]),
           "subject": np.asarray(rows["subject"]),
           "start": np.asarray(rows["start"]),
           "p_t": np.concatenate(rows["p_t"]),
           "p_vt": np.concatenate(rows["p_vt"]),
           "gt": np.concatenate(rows["gt"]),
           "ig_gt": np.concatenate(rows["ig_gt"]),
           "ig_ps": np.concatenate(rows["ig_ps"]),
           "split": split, "level": level}
    out["ig_gt_s"] = ema_by_session(out["ig_gt"], out["session"])
    out["ig_ps_s"] = ema_by_session(out["ig_ps"], out["session"])
    return out


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 5:
        return float("nan")
    def rank(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
        for val in np.unique(v):
            idx = np.where(v == val)[0]; r[idx] = r[idx].mean()
        return r
    rx, ry = rank(x), rank(y); rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def mean_at_g(E, g):
    idx = np.rint(np.asarray(g) * (len(G_GRID) - 1)).astype(int)
    idx = np.clip(idx, 0, len(G_GRID) - 1)
    return E[np.arange(E.shape[0]), idx]


def fit_gate(data, E):
    s_val = np.concatenate([data[("val", l)]["s_ps"] for l in LEVELS])
    taus = np.unique(np.concatenate([
        np.quantile(s_val, TAU_QUANTILES),
        np.arange(0.0, 1.001, 0.01),
        np.asarray([s_val.min(), s_val.max()])]))
    rows = []
    for tau in taus:
        for width in WIDTHS:
            e_mix = {}
            for level in LEVELS:
                g = gate_from_s(data[("val", level)]["s_ps"], float(tau), width)
                e_mix[level] = float(mean_at_g(E[("val", level)], g).mean())
            e_vt = {l: float(E[("val", l)][:, -1].mean()) for l in LEVELS}
            e_t = {l: float(E[("val", l)][:, 0].mean()) for l in LEVELS}
            tax = {l: e_mix[l] - e_vt[l] for l in LEVELS}
            gain = {l: -tax[l] for l in LEVELS}
            ok = (tax["000"] <= TAX_000 and tax["020"] <= TAX_020
                  and (e_mix["100"] - e_t["100"]) <= 0.5)
            score = (gain["100"] + 0.5 * gain["080"] + 0.5 * gain["060"]
                     + 0.25 * gain["040"])
            rows.append({"tau": float(tau), "width": float(width),
                         "e_mix": e_mix, "e_vt": e_vt, "e_t": e_t,
                         "tax": tax, "gain": gain, "ok": bool(ok),
                         "score": float(score)})
    feasible = [r for r in rows if r["ok"]]
    if feasible:
        # 低 occ 优先：在最高分 1.0 以内的可行点里取最小 tau / 最小 width，
        # 避免为了 val 的 060/080 收益把阈值推到 020 的低尾上。
        top = max(r["score"] for r in feasible)
        near = [r for r in feasible if r["score"] >= top - 1.0]
        best = min(near, key=lambda r: (r["tau"], r["width"]))
    else:
        # 无可行点：取 clean 违反量最小，再按 score
        best = min(rows, key=lambda r: (
            max(0.0, r["tax"]["000"] - TAX_000) + max(0.0, r["tax"]["020"] - TAX_020)
            + max(0.0, r["e_mix"]["100"] - r["e_t"]["100"] - 0.5), -r["score"]))
    # Pareto 诊断：最小 000 税、最大 100 gain
    best_clean = min(rows, key=lambda r: r["tax"]["000"])
    best_high = max(rows, key=lambda r: r["gain"]["100"])
    return best, best_clean, best_high, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--retrain-head", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    fold = args.fold
    fold_yaml = fold[1:] if fold.startswith("f") else fold
    cell = "V8_Cv5_%s_s1" % fold
    net, ckpt, tp = build_model(cell, device)
    print("[ckpt] %s" % ckpt)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    head_path = args.out_dir / ("contact_head_%s.pt" % fold)

    # ---- 1. contact head：train split 训，val 早停 -------------------------
    if head_path.is_file() and not args.retrain_head:
        head = ContactHead(256).to(device)
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
        head.eval(); head.requires_grad_(False)
        print("[head] loaded %s" % head_path)
    else:
        print("[head] training on %s train split (all levels)" % fold)
        tr_parts, va_parts = [], []
        for level in levels:
            cfg = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                   else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            tr_parts.append(collect_head_rows(
                net, build_loader(cfg, "train", args.batch_size, t_phys_v2=tp), device))
            va_parts.append(collect_head_rows(
                net, build_loader(cfg, "val", args.batch_size, t_phys_v2=tp), device))
        tr = {k: np.concatenate([p[k] for p in tr_parts])
              for k in ("v_tok", "contact")}
        va = {k: np.concatenate([p[k] for p in va_parts])
              for k in ("v_tok", "contact")}
        print("[head] train windows=%d val windows=%d" % (len(tr["v_tok"]), len(va["v_tok"])))
        head = train_head(tr, va, device)
        torch.save(head.state_dict(), head_path)
        print("[head] saved %s" % head_path)
    head.eval(); head.requires_grad_(False)

    # ---- 2. 逐窗 T/VT、IG、缓存 -------------------------------------------
    data = {}
    for split in splits:
        for level in levels:
            cfg = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                   else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            loader = build_loader(cfg, split, args.batch_size, t_phys_v2=tp)
            data[(split, level)] = collect_gate_rows(
                net, head, loader, device, split, level)
            r = data[(split, level)]
            print("[%s/%s] n=%d IG_gt med=%+.2f IG_ps med=%+.2f"
                  % (split, level, len(r["ig_ps"]),
                     float(np.median(r["ig_gt_s"])), float(np.median(r["ig_ps_s"]))))

    # ---- 2b. 受试者 clean 归一化 -----------------------------------------
    # 绝对 IG 有强受试者偏移（val clean ~0.9，test clean ~0.4）。用同一受试者
    # 的 000 档中位作 clean 参考，s_rel = IG / clean_median；000 档本身只作
    # 无标签 clean 校准，不参与门阈值拟合（阈值仍由 val 拟合）。
    if "000" not in levels:
        raise ValueError("levels must include 000 for per-subject clean calibration")
    for split in splits:
        for level in levels:
            r = data[(split, level)]
            r["s_sess_raw"] = session_median(r["ig_ps_s"], r["session"])
    clean_ref = {}
    for split in splits:
        r0 = data[(split, "000")]
        for subj in np.unique(r0["subject"]):
            m = r0["subject"] == subj
            clean_ref[(split, str(subj))] = max(float(np.median(r0["s_sess_raw"][m])), REF_FLOOR)
    clean_sess_ref = {}
    for split in splits:
        r0 = data[(split, "000")]
        for sess in np.unique(r0["session"]):
            m = r0["session"] == sess
            clean_sess_ref[(split, str(sess))] = max(float(np.median(r0["s_sess_raw"][m])), REF_FLOOR)
    for split in splits:
        for level in levels:
            r = data[(split, level)]
            ref = np.asarray([clean_ref[(split, str(s))] for s in r["subject"]])
            sess_ref = np.asarray([clean_sess_ref[(split, str(s))] for s in r["session"]])
            r["s_rel"] = np.clip(r["ig_ps_s"] / ref, -10.0, 10.0)
            r["s_sess"] = np.clip(r["s_sess_raw"] / ref, -10.0, 10.0)
            r["s_ps"] = np.clip(r["s_sess_raw"] / sess_ref, -10.0, 10.0)
            # clean 参考太低 = 该会话的 IG 本身不可信；钉 1，避免把 clean
            # 会话误判成高遮挡（fS5S6S8 有 clean IG≈0 的会话）。
            guard = sess_ref < REF_MIN
            r["s_ps"][guard] = 1.0
            r["guarded"] = guard
    print("[clean-ref] " + " | ".join(
        "%s:%s=%.2f" % (split, subj, clean_ref[(split, subj)])
        for split in splits for subj in np.unique(data[(split, "000")]["subject"])))

    # ---- 3. 预计算 E(g) ---------------------------------------------------
    E = {key: precompute_E(r["p_t"], r["p_vt"], r["gt"]) for key, r in data.items()}

    # ---- 4. val 拟合 ------------------------------------------------------
    best, best_clean, best_high, frontier = fit_gate(data, E)
    print("[fit] tau=%.4f width=%.3f ok=%s score=%+.2f | tax000=%+.2f tax020=%+.2f "
          "gain100=%+.2f gain080=%+.2f gain060=%+.2f"
          % (best["tau"], best["width"], best["ok"], best["score"],
             best["tax"]["000"], best["tax"]["020"], best["gain"]["100"],
             best["gain"]["080"], best["gain"]["060"]))
    print("[fit] best_clean tau=%.4f w=%.3f tax000=%+.2f gain100=%+.2f | "
          "best_high tau=%.4f w=%.3f tax000=%+.2f gain100=%+.2f"
          % (best_clean["tau"], best_clean["width"], best_clean["tax"]["000"],
             best_clean["gain"]["100"], best_high["tau"], best_high["width"],
             best_high["tax"]["000"], best_high["gain"]["100"]))

    # ---- 5. test 评估 -----------------------------------------------------
    tau, width = best["tau"], best["width"]
    report = {"fold": fold, "ckpt": str(ckpt), "tau": tau, "width": width,
              "ema_alpha": EMA_ALPHA, "fit_ok": best["ok"],
              "clean_ref": {split: {subj: clean_ref[(split, subj)]
                                    for subj in np.unique(data[(split, "000")]["subject"])}
                            for split in splits},
              "clean_sess_ref": {split: {str(sess): clean_sess_ref[(split, str(sess))]
                                         for sess in np.unique(data[(split, "000")]["session"])}
                                 for split in splits},
              "fit": {k: best[k] for k in ("tau", "width", "score", "tax", "gain",
                                           "e_mix", "e_vt", "e_t")},
              "best_clean": {k: best_clean[k] for k in ("tau", "width", "tax", "gain")},
              "best_high": {k: best_high[k] for k in ("tau", "width", "tax", "gain")},
              "signal": {}, "per_level": {}}
    for split in splits:
        for level in levels:
            r = data[(split, level)]
            report["signal"]["%s/%s" % (split, level)] = {
                "n": int(len(r["ig_ps"])),
                "ig_gt_med": float(np.median(r["ig_gt_s"])),
                "ig_ps_med": float(np.median(r["ig_ps_s"])),
                "ig_gt_p05": float(np.percentile(r["ig_gt_s"], 5)),
                "ig_ps_p05": float(np.percentile(r["ig_ps_s"], 5)),
                "ig_gt_p95": float(np.percentile(r["ig_gt_s"], 95)),
                "ig_ps_p95": float(np.percentile(r["ig_ps_s"], 95)),
                "s_rel_med": float(np.median(r["s_rel"])),
                "s_rel_p05": float(np.percentile(r["s_rel"], 5)),
                "s_rel_p95": float(np.percentile(r["s_rel"], 95)),
                "s_sess_med": float(np.median(r["s_sess"])),
                "s_sess_p05": float(np.percentile(r["s_sess"], 5)),
                "s_sess_p95": float(np.percentile(r["s_sess"], 95)),
                "s_ps_med": float(np.median(r["s_ps"])),
                "s_ps_p05": float(np.percentile(r["s_ps"], 5)),
                "s_ps_p95": float(np.percentile(r["s_ps"], 95)),
                "guarded_frac": float(np.mean(r["guarded"])),
                "rho_gt_ps": spearman(r["ig_gt_s"], r["ig_ps_s"]),
            }
    for level in levels:
        r = data[("test", level)]
        g = gate_from_s(r["s_ps"], tau, width)
        mix = blend(r["p_t"], r["p_vt"], g)
        e_lower = win_lower_pa(mix, r["gt"])
        e_lower_t = win_lower_pa(r["p_t"], r["gt"])
        e_lower_vt = win_lower_pa(r["p_vt"], r["gt"])
        e_pa = per_window_pa(mix, r["gt"])
        e_mpjpe = per_window_mpjpe(mix, r["gt"])
        # 主口径 = 全身体 PA（precompute_E 已用 per_window_pa）
        e_vt = E[("test", level)][:, -1]
        e_t = E[("test", level)][:, 0]
        # val-fit 逐档常数（在 val 的 E 上选 g）
        j_val = int(np.argmin(E[("val", level)].mean(axis=0)))
        g_lc = float(G_GRID[j_val])
        e_lc = E[("test", level)][:, j_val]
        e_or = E[("test", level)].min(axis=1)
        rec = {
            "level": level, "n": int(len(g)), "g_lc": g_lc,
            "g_mean": float(g.mean()), "g_median": float(np.median(g)),
            "g_p05": float(np.percentile(g, 5)),
            "g_p95": float(np.percentile(g, 95)),
            "e_pa_t": float(e_t.mean()), "e_pa_vt": float(e_vt.mean()),
            "e_pa": float(e_pa.mean()),
            "e_lower": float(e_lower.mean()),
            "e_lower_t": float(e_lower_t.mean()),
            "e_lower_vt": float(e_lower_vt.mean()),
            "e_mpjpe": float(e_mpjpe.mean()),
            "e_levelconst": float(e_lc.mean()), "e_oracle": float(e_or.mean()),
            "tax": float(e_pa.mean() - e_vt.mean()),
            "gain_vs_vt": float(e_vt.mean() - e_pa.mean()),
            "gain_vs_t": float(e_t.mean() - e_pa.mean()),
            "gain_vs_vt_lower": float(e_lower_vt.mean() - e_lower.mean()),
            "gain_levelconst": float(e_vt.mean() - e_lc.mean()),
            "gain_oracle": float(e_vt.mean() - e_or.mean()),
            "realization_vs_levelconst": (float((e_vt.mean() - e_pa.mean())
                                                / (e_vt.mean() - e_lc.mean()))
                                          if (e_vt.mean() - e_lc.mean()) > 0.5 else None),
        }
        report["per_level"][level] = rec
        print("[test/%s] g=%.3f [%.2f,%.2f] | PA: t=%.1f vt=%.1f gate=%.1f "
              "| gain_vt=%+.1f gain_t=%+.1f | lc=%.1f oracle=%.1f "
              "| lower: t=%.1f vt=%.1f gate=%.1f | MPJPE=%.1f"
              % (level, rec["g_mean"], rec["g_p05"], rec["g_p95"],
                 rec["e_pa_t"], rec["e_pa_vt"], rec["e_pa"],
                 rec["gain_vs_vt"], rec["gain_vs_t"],
                 rec["e_levelconst"], rec["e_oracle"],
                 rec["e_lower_t"], rec["e_lower_vt"], rec["e_lower"],
                 rec["e_mpjpe"]))

    # ---- 6. 落盘 ----------------------------------------------------------
    report["frontier"] = frontier
    (args.out_dir / ("report_%s.json" % fold)).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split in splits:
        keys = ("level", "session", "start", "ig_gt_s", "ig_ps_s", "e_t", "e_vt")
        arrays = {
            "level": np.concatenate([np.full(len(data[(split, l)]["ig_ps"]), int(l))
                                     for l in levels]),
            "session": np.concatenate([data[(split, l)]["session"] for l in levels]),
            "subject": np.concatenate([data[(split, l)]["subject"] for l in levels]),
            "start": np.concatenate([data[(split, l)]["start"] for l in levels]),
            "ig_gt_s": np.concatenate([data[(split, l)]["ig_gt_s"] for l in levels]),
            "ig_ps_s": np.concatenate([data[(split, l)]["ig_ps_s"] for l in levels]),
            "s_rel": np.concatenate([data[(split, l)]["s_rel"] for l in levels]),
            "s_sess": np.concatenate([data[(split, l)]["s_sess"] for l in levels]),
            "s_ps": np.concatenate([data[(split, l)]["s_ps"] for l in levels]),
            "guarded": np.concatenate([data[(split, l)]["guarded"] for l in levels]),
            "e_t": np.concatenate([E[(split, l)][:, 0] for l in levels]),
            "e_vt": np.concatenate([E[(split, l)][:, -1] for l in levels]),
            "E": np.concatenate([E[(split, l)] for l in levels]),
            "p_t": np.concatenate([data[(split, l)]["p_t"] for l in levels]),
            "p_vt": np.concatenate([data[(split, l)]["p_vt"] for l in levels]),
            "gt": np.concatenate([data[(split, l)]["gt"] for l in levels]),
        }
        np.savez_compressed(args.out_dir / ("rows_%s_%s.npz" % (fold, split)), **arrays)
    print("[out] %s" % (args.out_dir / ("report_%s.json" % fold)))


if __name__ == "__main__":
    main()
