#!/usr/bin/env python3
"""IG（视觉信息增益）pilot：V 预测 T 的接触状态，用 IG 判断 V 是否可信。

定义（每窗 w）：
  h       : V token (B,tw,d) → 左右脚接触 logits (B,tw,2) 的逐帧小头
  e_V(w)  : BCE(h(V_w), contact_gt_w)            窗内均值
  e_0(w)  : BCE(h(∅),    contact_gt_w)            同窗、V token 置零
  IG(w)   = 1 − e_V(w) / e_0(w)
训练时对 V token 做 modality dropout（p=--dropout），保证 h(∅) 有定义；
head 在 train split × clean+occ 上训练，val 早停，test（held-out 受试者）
只验证。

输出：逐窗 (IG, e_V, e_0, V 行 lower PA, GT jerk/speed)；按档汇总
IG 中位数、IG~Vpose 的 within-level Spearman（控动作难度）、IG 与接触
误差的关系。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/ig_contact_pilot.py \
      --cell V8_Cv5_fS5S6S11_s1 --fold fS5S6S11 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "results_display/script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import (CONFIG_V, CONFIG_VT, FPS,  # noqa: E402
                           JOINT_HIERARCHY)
from anysole.utils.geometry import fk_pose6d  # noqa: E402
from anysole.utils.metrics import (pa_mpjpe, pelvis_align,  # noqa: E402
                                   temporal_metrics)

LOWER = list(JOINT_HIERARCHY["lower"])
LEVELS = ("000", "020", "040", "060", "080", "100")
OUT = REPO / "results/diagnostics/20261107_ig_contact_pilot"


def per_window_lower_pa(out, batch) -> np.ndarray:
    pred = fk_pose6d(out["x0_hat"], out["trans_hat"],
                     batch["offsets"], batch["parents"])
    p = pred.detach().cpu().numpy(); g = batch["kp_gt"].detach().cpu().numpy()
    return np.asarray([float(pa_mpjpe(p[b][:, LOWER], g[b][:, LOWER]).mean())
                       for b in range(p.shape[0])])


def gt_jerk_speed(batch):
    kp = batch["kp_gt"].detach().cpu().numpy()
    tw = kp.shape[1]; times = np.arange(tw, dtype=np.float64) / float(FPS)
    jerks, speeds = [], []
    for b in range(kp.shape[0]):
        aligned, _ = pelvis_align(kp[b])
        jerks.append(float(temporal_metrics(aligned, aligned, times)["jitter_gt_m_s3"]))
        root = kp[b, :, 0, :]
        speeds.append(float(np.linalg.norm(np.diff(root, axis=0), axis=-1).mean() * float(FPS)))
    return np.asarray(jerks), np.asarray(speeds)


def run_split(net, loader, device, split, level, max_batches=0):
    rows = {k: [] for k in ("session", "start", "v_tok", "contact",
                            "e_pose", "jerk_gt", "speed_gt")}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = move_batch(batch, device)
        n = batch["V_feat"].shape[0]
        with torch.no_grad():
            cid = torch.full((n,), CONFIG_VT, dtype=torch.long, device=device)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
            out_vt = net(v_feat, t_raw, t_phys, cid, batch.get("subject_ids"), T_s2m=t_s2m)
            v_tok = out_vt["v_tok"].detach().cpu().numpy()
            cid_v = torch.full((n,), CONFIG_V, dtype=torch.long, device=device)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid_v)
            out_v = net(v_feat, t_raw, t_phys, cid_v, batch.get("subject_ids"), T_s2m=t_s2m)
        e_pose = per_window_lower_pa(out_v, batch)
        jg, sp = gt_jerk_speed(batch)
        rows["session"].extend(batch["session_id"])
        rows["start"].extend(batch["frame_start"].detach().cpu().tolist())
        rows["v_tok"].append(v_tok)
        rows["contact"].append(batch["contact_gt"].detach().cpu().numpy())
        rows["e_pose"].append(e_pose)
        rows["jerk_gt"].append(jg); rows["speed_gt"].append(sp)
    return {"session": np.asarray(rows["session"]),
            "start": np.asarray(rows["start"]),
            "v_tok": np.concatenate(rows["v_tok"]),
            "contact": np.concatenate(rows["contact"]),
            "e_pose": np.concatenate(rows["e_pose"]),
            "jerk_gt": np.concatenate(rows["jerk_gt"]),
            "speed_gt": np.concatenate(rows["speed_gt"]),
            "split": split, "level": level}


class ContactHead(nn.Module):
    def __init__(self, d, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def head_bce(head, tok, contact, v_drop=0.0, device=None):
    """tok (B,tw,d), contact (B,tw,2) → 逐窗 BCE (B,)。v_drop 概率把 token 置零。"""
    x = torch.from_numpy(tok).float().to(device)
    c = torch.from_numpy(contact).float().to(device)
    if v_drop > 0:
        keep = (torch.rand(x.shape[:2] + (1,), device=device) >= v_drop).float()
        x = x * keep
    logits = head(x)
    return F.binary_cross_entropy_with_logits(logits, c, reduction="none").mean(dim=(1, 2))


def train_head(tr, va, device, epochs=200, lr=1e-3, v_drop=0.25):
    head = ContactHead(tr["v_tok"].shape[-1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(tr["v_tok"]).float().to(device)
    ct = torch.from_numpy(tr["contact"]).float().to(device)
    xv = torch.from_numpy(va["v_tok"]).float().to(device)
    cv = torch.from_numpy(va["contact"]).float().to(device)
    best = (float("inf"), None)
    for _ in range(epochs):
        head.train(); opt.zero_grad()
        keep = (torch.rand(xt.shape[:2] + (1,), device=device) >= v_drop).float()
        loss = F.binary_cross_entropy_with_logits(head(xt * keep), ct)
        loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = float(F.binary_cross_entropy_with_logits(head(xv), cv))
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1]); head.eval()
    return head


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


def partial_spearman(x, y, controls):
    rx, ry = None, None
    def rank(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
        for val in np.unique(v):
            idx = np.where(v == val)[0]; r[idx] = r[idx].mean()
        return r
    rx, ry = rank(x), rank(y)
    C = np.stack([rank(c) for c in controls], axis=1)
    C = np.concatenate([np.ones((C.shape[0], 1)), C], axis=1)
    bx, *_ = np.linalg.lstsq(C, rx, rcond=None)
    by, *_ = np.linalg.lstsq(C, ry, rcond=None)
    ex, ey = rx - C @ bx, ry - C @ by
    ex -= ex.mean(); ey -= ey.mean()
    den = np.sqrt((ex * ex).sum() * (ey * ey).sum())
    return float((ex * ey).sum() / den) if den > 0 else float("nan")


def ig_for(head, rows, device):
    x = torch.from_numpy(rows["v_tok"]).float().to(device)
    c = torch.from_numpy(rows["contact"]).float().to(device)
    with torch.no_grad():
        e_v = F.binary_cross_entropy_with_logits(head(x), c, reduction="none").mean(dim=(1, 2))
        e_0 = F.binary_cross_entropy_with_logits(head(torch.zeros_like(x)), c,
                                                 reduction="none").mean(dim=(1, 2))
    ev = e_v.cpu().numpy(); e0 = e_0.cpu().numpy()
    ig = 1.0 - ev / np.maximum(e0, 1e-6)
    return ig, ev, e0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="V8_Cv5_fS5S6S11_s1")
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--max-batches", type=int, default=0)
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
            data[(split, level)] = run_split(net, loader, device, split, level,
                                             args.max_batches)
            print("[%s/%s] n=%d" % (split, level, len(data[(split, level)]["e_pose"])))

    if "train" not in splits:
        raise SystemExit("need train split")
    tr = {k: np.concatenate([data[("train", l)][k] for l in levels])
          for k in ("v_tok", "contact", "e_pose", "jerk_gt", "speed_gt")}
    va = ({k: np.concatenate([data[("val", l)][k] for l in levels])
           for k in ("v_tok", "contact", "e_pose", "jerk_gt", "speed_gt")}
          if "val" in splits else tr)
    head = train_head(tr, va, device, v_drop=args.dropout)
    torch.save(head.state_dict(), out_dir / "contact_head.pt")

    metrics = {"cell": args.cell, "fold": args.fold, "levels": list(levels),
               "dropout": args.dropout, "per_level": []}
    for split in [s for s in ("val", "test") if s in splits]:
        for level in levels:
            r = data[(split, level)]
            ig, ev, e0 = ig_for(head, r, device)
            ok = e0 > 1e-4
            metrics["per_level"].append({
                "split": split, "level": level, "n": int(ok.sum()),
                "ig_median": float(np.median(ig[ok])),
                "ig_mean": float(np.mean(ig[ok])),
                "ig_p05": float(np.percentile(ig[ok], 5)) if ok.any() else float("nan"),
                "frac_ig_neg": float((ig[ok] < 0).mean()) if ok.any() else float("nan"),
                "e_v_med": float(np.median(ev[ok])) if ok.any() else float("nan"),
                "e_0_med": float(np.median(e0[ok])) if ok.any() else float("nan"),
                "rho_ig_vpose": spearman(ig[ok], r["e_pose"][ok]),
                "rho_ig_vpose_partial": partial_spearman(
                    ig[ok], r["e_pose"][ok], [r["jerk_gt"][ok], r["speed_gt"][ok]])})
            print("[%s/%s] IG med=%+.3f neg=%.2f e_v=%.4f e_0=%.4f rho~Vpose=%+.3f (partial %+.3f)"
                  % (split, level, metrics["per_level"][-1]["ig_median"],
                     metrics["per_level"][-1]["frac_ig_neg"],
                     metrics["per_level"][-1]["e_v_med"], metrics["per_level"][-1]["e_0_med"],
                     metrics["per_level"][-1]["rho_ig_vpose"],
                     metrics["per_level"][-1]["rho_ig_vpose_partial"]))
    for split in splits:
        np.savez_compressed(
            out_dir / ("rows_%s.npz" % split),
            session=np.concatenate([data[(split, l)]["session"] for l in levels]),
            start=np.concatenate([data[(split, l)]["start"] for l in levels]),
            level=np.concatenate([np.full(len(data[(split, l)]["e_pose"]), int(l))
                                  for l in levels]),
            v_tok=np.concatenate([data[(split, l)]["v_tok"] for l in levels]),
            contact=np.concatenate([data[(split, l)]["contact"] for l in levels]),
            e_pose=np.concatenate([data[(split, l)]["e_pose"] for l in levels]),
            jerk_gt=np.concatenate([data[(split, l)]["jerk_gt"] for l in levels]),
            speed_gt=np.concatenate([data[(split, l)]["speed_gt"] for l in levels]))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print("[out] %s" % out_dir)


if __name__ == "__main__":
    main()
