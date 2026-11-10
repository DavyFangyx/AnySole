#!/usr/bin/env python3
"""SAM3DB -> V<T cheap pilot（2026-11-09）。

只做 SAM3DB 侧已有 result.pkl 的信号，不重跑 SAM3DB：
  alignment_rmse、betas、bbox、smpl_joints_2d 下肢可见性
与 V8_Cv5 的 V 行/T 行逐窗 full PA 的 V<T 标签做 within-level Spearman/AUC。
3 会话 × 5 档（000 clean / 020/040/060/080），先看 within-level 是否有信号。
颜色混淆在下一步用 RTMPose 黑 vs 灰重绘遮挡条做。

用法：python results_display/script/sam3db_vt_pilot.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results_display/script"))

from gate_metric_crossfit_export import load_motion  # noqa: E402

GATE_FOLD = "fS5S6S11"
SESSIONS = ("S6073", "S11021", "S5112")
LEVELS = ("000", "020", "040", "060", "080")
LOWER = (7, 8, 10, 11)  # left/right ankle + foot (SMPL-24)
OUT = REPO / "results/diagnostics/20261108_gate_level/sam3db_pilot"


def sam3db_result(level: str, session: str) -> dict:
    if level == "000":
        motion = (REPO / "baselines_deliverables/predictions/SAM3DB/local_gen/"
                  "eval_motion" / ("%s.npz" % session))
    else:
        motion = (REPO / "results/baselines/SAM3DB/predictions"
                  / ("e5_occ_lower%s" % level) / "eval_motion" / ("%s.npz" % session))
    m = np.load(motion, allow_pickle=True)
    native = REPO / str(m["source_native_output"])
    with native.open("rb") as f:
        return pickle.load(f)


def v8_rows(level: str, session: str):
    cell = "V8_Cv5_%s_s1" % GATE_FOLD
    if level != "000":
        cell += "_evalocc%s" % level
    v = load_motion("V8_Cv5", GATE_FOLD, level, session, "V2M")
    t = load_motion("V8_Cv5", GATE_FOLD, level, session, "T2M")
    return v, t


def rows_gt(level: str, session: str):
    rows = np.load(REPO / "results/diagnostics/20261108_gate_level" /
                   ("rows_%s_test.npz" % GATE_FOLD), allow_pickle=True)
    m = (rows["session"] == session) & (rows["level"] == int(level))
    return rows["start"][m].astype(int), rows["gt"][m]


def frame_pa(pred, gt):
    """pred/gt: (N,24,3) per-frame joints -> per-frame full PA."""
    from anysole.utils.metrics import pa_mpjpe
    return np.asarray([float(pa_mpjpe(pred[i][None], gt[i][None]).mean())
                       for i in range(pred.shape[0])])


def auc_score(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos = s[y == 1]; neg = s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])[order]
    ranks = np.arange(1, len(lab) + 1)
    return float((ranks[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 5:
        return float("nan")
    def rank(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
        for val in np.unique(v):
            idx = np.where(v == val)[0]; r[idx] = r[idx].mean()
        return r
    rx, ry = rank(x), rank(y); rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_level = {}
    rows_out = []
    for level in LEVELS:
        sig = {k: [] for k in ("align", "betas_std", "bbox_h", "low_y",
                               "low_spread", "v_pa", "t_pa", "v_lt")}
        for session in SESSIONS:
            v, t = v8_rows(level, session)
            starts, gt_win = rows_gt(level, session)
            if v is None or t is None:
                continue
            frames = v["frame"]
            ind, gt = [], []
            for wi, start in enumerate(starts):
                for j in range(gt_win.shape[1]):
                    idx = np.where(frames == (start + j))[0]
                    if len(idx):
                        ind.append(int(idx[0])); gt.append(gt_win[wi, j])
            if not ind:
                continue
            ind = np.asarray(ind); gt = np.asarray(gt)
            p_v = v["joint"][ind]; p_t = t["joint"][ind]
            e_v = frame_pa(p_v, gt); e_t = frame_pa(p_t, gt)
            res = sam3db_result(level, session)
            arr = res["arrays"]
            sam_frame = np.asarray(arr["frame"])
            align = np.asarray(arr["alignment_rmse"], dtype=float)
            betas = np.asarray(arr["betas"], dtype=float)
            bbox = np.asarray(arr["bbox_xyxy"], dtype=float)
            j2d = np.asarray(arr["smpl_joints_2d"], dtype=float)
            # align SAM3DB frames to V8 frames
            order = {int(f): i for i, f in enumerate(sam_frame)}
            ok = np.asarray([f in order for f in frames[ind]])
            if ok.sum() < 10:
                continue
            si = np.asarray([order[int(f)] for f in frames[ind] if f in order])
            vi = ind[ok]
            e_v = frame_pa(p_v[ok], gt[ok]); e_t = frame_pa(p_t[ok], gt[ok])
            low = j2d[si][:, LOWER, :]
            h = bbox[si][:, 3] - bbox[si][:, 1]
            sig["align"].extend(align[si].tolist())
            sig["betas_std"].extend(betas[si].std(axis=1).tolist())
            sig["bbox_h"].extend(h.tolist())
            sig["low_y"].extend((low[..., 1].mean(axis=1) / np.maximum(h, 1e-6)).tolist())
            sig["low_spread"].extend(low.std(axis=1).mean(axis=1).tolist())
            sig["v_pa"].extend(e_v.tolist()); sig["t_pa"].extend(e_t.tolist())
            sig["v_lt"].extend((e_v > e_t).astype(float).tolist())
            for i in range(len(e_v)):
                rows_out.append({"level": level, "session": session,
                                 "v_pa": float(e_v[i]), "t_pa": float(e_t[i]),
                                 "v_lt": int(e_v[i] > e_t[i]),
                                 "align": float(align[si][i]),
                                 "betas_std": float(betas[si][i].std()),
                                 "bbox_h": float(h[i]),
                                 "low_y": float(low[i, :, 1].mean() / max(h[i], 1e-6)),
                                 "low_spread": float(low[i].std(axis=0).mean())})
        y = np.asarray(sig["v_lt"])
        rec = {"n": int(len(y)), "v_lt_rate": float(y.mean()) if len(y) else None}
        for k in ("align", "betas_std", "bbox_h", "low_y", "low_spread"):
            s = np.asarray(sig[k])
            rec[k] = {"spearman": float(spearman(s, y)),
                      "auc": float(auc_score(y, s))}
        per_level[level] = rec
        print("[%s] n=%d V<T=%.2f | align rho=%+.2f AUC=%.2f | "
              "betas_std rho=%+.2f | bbox_h rho=%+.2f | low_y rho=%+.2f | "
              "low_spread rho=%+.2f"
              % (level, rec["n"], rec["v_lt_rate"] or 0,
                 rec["align"]["spearman"], rec["align"]["auc"],
                 rec["betas_std"]["spearman"], rec["bbox_h"]["spearman"],
                 rec["low_y"]["spearman"], rec["low_spread"]["spearman"]))
    # within-level pooled: center each signal and label by level
    all_rows = rows_out
    y = np.asarray([r["v_lt"] for r in all_rows])
    rec_all = {"n": int(len(y)), "v_lt_rate": float(y.mean())}
    for k in ("align", "betas_std", "bbox_h", "low_y", "low_spread"):
        s = np.asarray([r[k] for r in all_rows])
        # level-centered (remove per-level mean) — controls the level confound
        lv = np.asarray([r["level"] for r in all_rows])
        s_c = s.copy()
        for l in LEVELS:
            m = lv == l
            if m.any():
                s_c[m] -= s[m].mean()
        rec_all[k] = {"raw_spearman": float(spearman(s, y)),
                      "within_level_spearman": float(spearman(s_c, y)),
                      "auc": float(auc_score(y, s))}
    print("[pooled] n=%d V<T=%.2f" % (rec_all["n"], rec_all["v_lt_rate"]))
    for k in ("align", "betas_std", "bbox_h", "low_y", "low_spread"):
        print("  %-10s raw rho=%+.2f within rho=%+.2f AUC=%.2f"
              % (k, rec_all[k]["raw_spearman"], rec_all[k]["within_level_spearman"],
                 rec_all[k]["auc"]))
    (OUT / "report.json").write_text(
        json.dumps({"per_level": per_level, "pooled": rec_all,
                    "rows": rows_out}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("[out] %s" % (OUT / "report.json"))


if __name__ == "__main__":
    main()
