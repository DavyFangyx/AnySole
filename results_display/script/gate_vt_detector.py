#!/usr/bin/env python3
"""V<T 硬检测器（2026-11-09）：会话级二分类 + 硬切换 T。

目标：D(session)=1[PA_V > PA_T]；g=1-D；D=1 → T，D=0 → VT。
与 IG 阈值门控的区别：标签直接是 V 不可信（V<T），不是 IG 低。

单折用法（先 fS5S6S11 冒烟）：
  python results_display/script/gate_vt_detector.py --fold fS5S6S11 --device cuda:2
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

from gate_level_blend import (gate_from_s, session_median)  # noqa: E402
from gate_metric_crossfit_export import (  # noqa: E402
    cell_values, load_motion, session_metrics)
from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402
from ig_contact_pilot import ContactHead  # noqa: E402
from stage2_gate_pilot import pred_kp  # noqa: E402
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import ANKLE_FOOT_JOINTS, CONFIG_T, CONFIG_V, CONFIG_VT  # noqa: E402
from anysole.utils.metrics import mean_point_error, pa_mpjpe, pelvis_align  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
GATE_OUT = REPO / "results/diagnostics/20261108_gate_level"
OUT = GATE_OUT / "vt_detector"
EMA_ALPHA = 0.7
REF_FLOOR = 0.05
REF_MIN = 0.10


def per_window_pa(pred, gt):
    return np.asarray([float(pa_mpjpe(pred[b], gt[b]).mean())
                       for b in range(pred.shape[0])])


def per_window_ankle(pred, gt):
    out = []
    for b in range(pred.shape[0]):
        p, _ = pelvis_align(pred[b])
        g, _ = pelvis_align(gt[b])
        out.append(float(mean_point_error(p[:, ANKLE_FOOT_JOINTS],
                                          g[:, ANKLE_FOOT_JOINTS],
                                          scale=1000.0).mean()))
    return np.asarray(out)


def ema_by_session(sig, sessions):
    out = np.empty_like(np.asarray(sig, dtype=np.float64))
    prev_s, prev_v = None, None
    for i, (s, v) in enumerate(zip(sessions, sig)):
        if s != prev_s:
            prev_v = None
        prev_v = v if prev_v is None else EMA_ALPHA * v + (1 - EMA_ALPHA) * prev_v
        out[i] = prev_v
        prev_s = s
    return out


def collect_split(net, head, loader, device, split, level):
    rows = {k: [] for k in ("session", "start", "v_tok", "t_tok", "ig_ps",
                            "e_v", "e_t", "e_vt", "e_a_v", "e_a_t")}
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
            c_gt = batch["contact_gt"]
            c_t = torch.sigmoid(out_t["contact_logits"])
            e_v = F.binary_cross_entropy_with_logits(
                head(v_tok), c_gt, reduction="none").mean(dim=(1, 2))
            e_0 = F.binary_cross_entropy_with_logits(
                head(torch.zeros_like(v_tok)), c_gt, reduction="none").mean(dim=(1, 2))
            ig = (1.0 - e_v / e_0.clamp(min=1e-6)).cpu().numpy()
        p_v = pred_kp(out_v, batch); p_t = pred_kp(out_t, batch)
        p_vt = pred_kp(out_vt, batch)
        gt = batch["kp_gt"].detach().cpu().numpy()
        rows["session"].extend(batch["session_id"])
        rows["start"].extend(batch["frame_start"].detach().cpu().tolist())
        rows["v_tok"].append(v_tok.mean(dim=1).detach().cpu().numpy())
        rows["t_tok"].append(out_vt["t_tok"].mean(dim=1).detach().cpu().numpy())
        rows["ig_ps"].append(ig)
        rows["e_v"].append(per_window_pa(p_v, gt))
        rows["e_t"].append(per_window_pa(p_t, gt))
        rows["e_vt"].append(per_window_pa(p_vt, gt))
        rows["e_a_v"].append(per_window_ankle(p_v, gt))
        rows["e_a_t"].append(per_window_ankle(p_t, gt))
    out = {"session": np.asarray(rows["session"]),
           "start": np.asarray(rows["start"]),
           "ig_ps": np.concatenate(rows["ig_ps"]),
           "e_v": np.concatenate(rows["e_v"]),
           "e_t": np.concatenate(rows["e_t"]),
           "e_vt": np.concatenate(rows["e_vt"]),
           "e_a_v": np.concatenate(rows["e_a_v"]),
           "e_a_t": np.concatenate(rows["e_a_t"]),
           "v_tok": np.concatenate(rows["v_tok"]),
           "t_tok": np.concatenate(rows["t_tok"]),
           "split": split, "level": level}
    return out


def add_s_ps(data, levels):
    for split in data:
        for level in levels:
            r = data[split][level]
            r["s_raw"] = session_median(r["ig_ps"], r["session"])
    # clean ref per session from level 000
    clean = {}
    for split in data:
        r0 = data[split]["000"]
        for s in np.unique(r0["session"]):
            m = r0["session"] == s
            clean[(split, str(s))] = max(float(np.median(r0["s_raw"][m])), REF_FLOOR)
    for split in data:
        for level in levels:
            r = data[split][level]
            ref = np.asarray([clean[(split, str(s))] for s in r["session"]])
            sr = np.clip(r["s_raw"] / ref, -10.0, 10.0)
            sr[ref < REF_MIN] = 1.0
            r["s_ps"] = sr


def session_features(data, split, levels):
    """session -> dict(features, label V<T, level, split)."""
    rows = {}
    for level in levels:
        r = data[split][level]
        for s in np.unique(r["session"]):
            m = r["session"] == s
            key = (str(s), level)
            rec = rows.setdefault(key, {"session": str(s), "split": split,
                                        "level": level, "v_tok": [], "t_tok": [],
                                        "s_ps": [], "ig": [], "e_v": [], "e_t": [],
                                        "e_a_v": [], "e_a_t": []})
            rec["level"] = level
            rec["v_tok"].append(r["v_tok"][m].mean(0))
            rec["t_tok"].append(r["t_tok"][m].mean(0))
            rec["s_ps"].append(r["s_ps"][m].mean())
            rec["ig"].append(float(np.median(r["ig_ps"][m])))
            rec["e_v"].append(float(np.mean(r["e_v"][m])))
            rec["e_t"].append(float(np.mean(r["e_t"][m])))
            rec["e_a_v"].append(float(np.mean(r["e_a_v"][m])))
            rec["e_a_t"].append(float(np.mean(r["e_a_t"][m])))
    out = []
    for s, rec in rows.items():
        f = np.concatenate([np.asarray(rec["v_tok"]).mean(0),
                            np.asarray(rec["t_tok"]).mean(0),
                            [np.mean(rec["s_ps"]), np.mean(rec["ig"]),
                             np.linalg.norm(np.asarray(rec["v_tok"]).mean(0)),
                             np.linalg.norm(np.asarray(rec["t_tok"]).mean(0))]])
        out.append({"session": s, "split": split,
                    "level": int(rec["level"]), "x": f.astype(np.float32),
                    "e_v": float(np.mean(rec["e_v"])),
                    "e_t": float(np.mean(rec["e_t"])),
                    "e_a_v": float(np.mean(rec["e_a_v"])),
                    "e_a_t": float(np.mean(rec["e_a_t"]))})
    return out


class Detector(torch.nn.Module):
    def __init__(self, d, hidden=128, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_detector(tr, va, device, epochs=300, lr=1e-3):
    d = tr[0]["x"].size
    model = Detector(d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(np.stack([r["x"] for r in tr])).float().to(device)
    yt = torch.from_numpy(np.asarray([1.0 if r["e_v"] > r["e_t"] else 0.0
                                      for r in tr])).float().to(device)
    xv = torch.from_numpy(np.stack([r["x"] for r in va])).float().to(device)
    yv = torch.from_numpy(np.asarray([1.0 if r["e_v"] > r["e_t"] else 0.0
                                      for r in va])).float().to(device)
    # standardize on train
    mu = xt.mean(0, keepdim=True); sd = xt.std(0, keepdim=True).clamp(min=1e-4)
    xt = (xt - mu) / sd; xv = (xv - mu) / sd
    best = (float("inf"), None)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xt), yt)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(F.binary_cross_entropy_with_logits(model(xv), yv))
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1]); model.eval()
    return model, mu, sd


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    fold = args.fold
    fold_yaml = fold[1:] if fold.startswith("f") else fold
    cell = "V8_Cv5_%s_s1" % fold
    net, ckpt, tp = build_model(cell, device)
    head = ContactHead(256).to(device)
    head.load_state_dict(torch.load(GATE_OUT / ("contact_head_%s.pt" % fold),
                                    map_location="cpu"))
    head.eval(); head.requires_grad_(False)
    print("[ckpt] %s" % ckpt)

    data = {"train": {}, "val": {}, "test": {}}
    for split in ("train", "val", "test"):
        for level in levels:
            cfg = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                   else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            loader = build_loader(cfg, split, args.batch_size, t_phys_v2=tp)
            data[split][level] = collect_split(net, head, loader, device, split, level)
            print("[%s/%s] n=%d" % (split, level, len(data[split][level]["ig_ps"])))
    add_s_ps(data, levels)
    tr = session_features(data, "train", levels)
    va = session_features(data, "val", levels)
    te = session_features(data, "test", levels)
    print("[sessions] train=%d val=%d test=%d" % (len(tr), len(va), len(te)))
    model, mu, sd = train_detector(tr, va, device, epochs=args.epochs)

    def predict(rows):
        x = torch.from_numpy(np.stack([r["x"] for r in rows])).float().to(device)
        with torch.no_grad():
            p = torch.sigmoid(model((x - mu) / sd)).cpu().numpy()
        return p

    y_te = np.asarray([1.0 if r["e_v"] > r["e_t"] else 0.0 for r in te])
    p_te = predict(te)
    print("[detector] test AUC=%.3f acc=%.3f" % (auc_score(y_te, p_te),
                                                 float(((p_te >= 0.5) == y_te).mean())))
    per_level = {}
    for lv in levels:
        idx = [i for i, r in enumerate(te) if r["level"] == int(lv)]
        if not idx:
            continue
        yy = y_te[idx]; pp = p_te[idx]
        per_level[lv] = {"n": len(idx), "pos_rate": float(yy.mean()),
                         "d_rate": float((pp >= 0.5).mean()),
                         "auc": auc_score(yy, pp)}
        print("[test/%s] n=%d V<T rate=%.2f D=1 rate=%.2f AUC=%s"
              % (lv, len(idx), yy.mean(), (pp >= 0.5).mean(),
                 "%.3f" % per_level[lv]["auc"] if np.isfinite(per_level[lv]["auc"]) else "n/a"))

    # gate on test sessions: hard g=1-D
    rows = np.load(GATE_OUT / ("rows_%s_test.npz" % fold), allow_pickle=True)
    ses = rows["session"]; lev = rows["level"]
    s_ps = rows["s_ps"]
    rep = json.loads((GATE_OUT / ("report_%s.json" % fold)).read_text(encoding="utf-8"))
    tau, width = float(rep["tau"]), float(rep["width"])
    p_by_sess = {(r["session"], str(r["level"])): float(p)
                 for r, p in zip(te, p_te)}
    out_gate = {"hard": {}, "current": {}}
    for mode in ("hard", "current"):
        vals = {}
        for level in levels:
            vals[level] = {m: [] for m in ("PA-MPJPE", "ankle_foot_mpjpe",
                                           "GlobalMPJPE", "FootSliding",
                                           "GroundPen", "traj_ATE")}
        for session in np.unique(ses):
            m = ses == session
            level = str(int(lev[m][0]))
            if mode == "hard":
                p = p_by_sess.get((str(session), level), 1.0)
                g_val = 0.0 if p >= 0.5 else 1.0
            else:
                g_val = float(gate_from_s(s_ps[m], tau, width)[0])
            base = load_motion("V8_Cv5", fold, level, str(session), "VT2M")
            t_m = load_motion("V8_Cv5", fold, level, str(session), "T2M")
            if base is None or t_m is None:
                continue
            frames = base["frame"]
            starts = rows["start"][m].astype(int)
            gt_win = rows["gt"][m]
            ind, gt, times = [], [], []
            for wi, start in enumerate(starts):
                for j in range(gt_win.shape[1]):
                    idx = np.where(frames == (start + j))[0]
                    if len(idx) == 0:
                        continue
                    ind.append(int(idx[0])); gt.append(gt_win[wi, j])
                    times.append(t_m["time"][idx[0]])
            if not ind:
                continue
            ind = np.asarray(ind); gt = np.asarray(gt)
            pred = (g_val * base["joint"][ind] + (1.0 - g_val) * t_m["joint"][ind])
            floor_y = float(np.percentile(gt[:, :, 1], 5))
            for metric, v in session_metrics(pred, gt, np.arange(len(gt)) / 40.0,
                                             floor_y).items():
                vals[level][metric].append(v)
        out_gate[mode] = {l: {m: float(np.mean(v)) for m, v in vals[l].items()
                              if v} for l in levels}
        print("== %s gate ==" % mode)
        for level in levels:
            print(" ", level, {m: round(v, 1) for m, v in out_gate[mode][level].items()})

    # cells for V8 / Cv7 / T / V
    refs = {}
    for group, fam, cond in (("V8_Cv5", "V8_Cv5", "VT2M"),
                             ("V8_Cv7", "V8_Cv7", "VT2M"),
                             ("T_only", "V8_Cv5", "T2M"),
                             ("V_only", "V8_Cv5", "V2M")):
        refs[group] = {l: cell_values(fam, fold, l, cond) for l in levels}
    report = {"fold": fold, "detector": {"test_auc": auc_score(y_te, p_te),
                                         "test_acc": float(((p_te >= 0.5) == y_te).mean()),
                                         "per_level": per_level},
              "gate": out_gate, "refs": refs,
              "sessions": {"train": len(tr), "val": len(va), "test": len(te)}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ("report_%s.json" % fold)).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[out] %s" % (args.out_dir / ("report_%s.json" % fold)))


if __name__ == "__main__":
    main()
