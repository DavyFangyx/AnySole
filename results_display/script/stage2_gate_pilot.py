#!/usr/bin/env python3
"""阶段 2 门 pilot：逐窗 oracle g* + IG/token → V 信任头（冻结主干）。

对一折、Cv5 基座：
1. 跑 VT / V / T 三行，取 V/T token、contact、逐窗 lower PA；
2. IG = 1 − e(h(V), T) / e(h(∅), T)，h 用 IG pilot 的冻结 contact head；
3. oracle g*(w)：用 V/T 两行预测对 GT 求最小二乘插值权重
   w* = <V−T, GT−T> / ||V−T||²（clamp [0,1]，逐窗）；
   VT 行只作 oracle 校验（不进入 w* 计算）。
4. 门头输入 cat(v_tok, t_tok, IG)，输出 ĝ；train 训练、val 早停、
   test（held-out）验证 ĝ~g*。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/stage2_gate_pilot.py \
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
from ig_contact_pilot import ContactHead  # noqa: E402
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import (CONFIG_T, CONFIG_V, CONFIG_VT, FPS,  # noqa: E402
                           JOINT_HIERARCHY)
from anysole.utils.geometry import fk_pose6d  # noqa: E402
from anysole.utils.metrics import (pa_mpjpe, pelvis_align,  # noqa: E402
                                   temporal_metrics)

LOWER = list(JOINT_HIERARCHY["lower"])
LEVELS = ("000", "020", "040", "060", "080", "100")
IG_HEAD = REPO / "results/diagnostics/20261107_ig_contact_pilot/contact_head.pt"
OUT = REPO / "results/diagnostics/20261107_stage2_gate_pilot"


def pred_kp(out, batch):
    return fk_pose6d(out["x0_hat"], out["trans_hat"],
                     batch["offsets"], batch["parents"]).detach().cpu().numpy()


def win_lower_pa(pred, gt):
    return np.asarray([float(pa_mpjpe(pred[b][:, LOWER], gt[b][:, LOWER]).mean())
                       for b in range(pred.shape[0])])


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


def oracle_g(pred_v, pred_t, gt):
    """逐窗最小二乘 V 信任权重 w*（V/T 两行插值到 GT）。"""
    d = pred_v - pred_t
    r = gt - pred_t
    num = (d * r).sum(axis=(2, 3))
    den = (d * d).sum(axis=(2, 3))
    return np.clip(num / np.maximum(den, 1e-8), 0.0, 1.0)


def run_split(net, head, loader, device, split, level, max_batches=0):
    rows = {k: [] for k in ("session", "start", "v_tok", "t_tok", "ig",
                            "g_star", "e_v", "e_t", "e_vt", "jerk_gt", "speed_gt")}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = move_batch(batch, device)
        n = batch["V_feat"].shape[0]
        cid_vt = torch.full((n,), CONFIG_VT, dtype=torch.long, device=device)
        cid_v = torch.full((n,), CONFIG_V, dtype=torch.long, device=device)
        cid_t = torch.full((n,), CONFIG_T, dtype=torch.long, device=device)
        with torch.no_grad():
            vf, tr, tp, s2 = condition_inputs(batch, cid_vt)
            out_vt = net(vf, tr, tp, cid_vt, batch.get("subject_ids"), T_s2m=s2)
            vf, tr, tp, s2 = condition_inputs(batch, cid_v)
            out_v = net(vf, tr, tp, cid_v, batch.get("subject_ids"), T_s2m=s2)
            vf, tr, tp, s2 = condition_inputs(batch, cid_t)
            out_t = net(vf, tr, tp, cid_t, batch.get("subject_ids"), T_s2m=s2)
            # IG（contact head 冻结）
            v_tok = out_vt["v_tok"]
            c = batch["contact_gt"]
            e_v = F.binary_cross_entropy_with_logits(head(v_tok), c, reduction="none").mean(dim=(1, 2))
            e_0 = F.binary_cross_entropy_with_logits(head(torch.zeros_like(v_tok)), c,
                                                     reduction="none").mean(dim=(1, 2))
            ig = (1.0 - e_v / e_0.clamp(min=1e-6)).cpu().numpy()
        p_v = pred_kp(out_v, batch); p_t = pred_kp(out_t, batch)
        gt = batch["kp_gt"].detach().cpu().numpy()
        g_star = oracle_g(p_v, p_t, gt)
        jg, sp = gt_jerk_speed(batch)
        rows["session"].extend(batch["session_id"])
        rows["start"].extend(batch["frame_start"].detach().cpu().tolist())
        rows["v_tok"].append(out_vt["v_tok"].mean(dim=1).detach().cpu().numpy())
        rows["t_tok"].append(out_vt["t_tok"].mean(dim=1).detach().cpu().numpy())
        rows["ig"].append(ig)
        rows["g_star"].append(g_star.mean(axis=1))          # 窗级标量
        rows["e_v"].append(win_lower_pa(p_v, gt))
        rows["e_t"].append(win_lower_pa(p_t, gt))
        rows["e_vt"].append(win_lower_pa(pred_kp(out_vt, batch), gt))
        rows["jerk_gt"].append(jg); rows["speed_gt"].append(sp)
    out = {k: (np.concatenate(v) if k in ("v_tok", "t_tok", "ig", "g_star",
                                          "e_v", "e_t", "e_vt", "jerk_gt",
                                          "speed_gt") else np.asarray(v))
           for k, v in rows.items()}
    out["split"] = split; out["level"] = level
    return out


class GateHead(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d + 1, hidden), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)


def feats(r):
    return np.concatenate([r["v_tok"], r["t_tok"], r["ig"][:, None]], axis=1)


def train_gate(tr, va, device, epochs=300, lr=1e-3):
    g = GateHead(tr["v_tok"].shape[1]).to(device)
    opt = torch.optim.Adam(g.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(feats(tr)).float().to(device)
    yt = torch.from_numpy(tr["g_star"]).float().to(device)
    xv = torch.from_numpy(feats(va)).float().to(device)
    yv = torch.from_numpy(va["g_star"]).float().to(device)
    best = (float("inf"), None)
    for _ in range(epochs):
        g.train(); opt.zero_grad()
        loss = F.mse_loss(g(xt), yt)
        loss.backward(); opt.step()
        g.eval()
        with torch.no_grad():
            vl = float(F.mse_loss(g(xv), yv))
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in g.state_dict().items()})
    g.load_state_dict(best[1]); g.eval()
    return g


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="V8_Cv5_fS5S6S11_s1")
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    net, ckpt, tp = build_model(args.cell, device)
    print("[ckpt] %s" % ckpt)
    head = ContactHead(256).to(device)
    head.load_state_dict(torch.load(IG_HEAD, map_location="cpu"))
    head.eval(); head.requires_grad_(False)
    fold_yaml = args.fold[1:] if args.fold.startswith("f") else args.fold
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for split in splits:
        for level in levels:
            cfg = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                   else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            loader = build_loader(cfg, split, args.batch_size, t_phys_v2=tp)
            data[(split, level)] = run_split(net, head, loader, device, split, level,
                                             args.max_batches)
            print("[%s/%s] n=%d" % (split, level, len(data[(split, level)]["g_star"])))

    tr = {k: np.concatenate([data[("train", l)][k] for l in levels])
          for k in ("v_tok", "t_tok", "ig", "g_star")}
    va = ({k: np.concatenate([data[("val", l)][k] for l in levels])
           for k in ("v_tok", "t_tok", "ig", "g_star")} if "val" in splits else tr)
    gate = train_gate(tr, va, device)
    torch.save(gate.state_dict(), out_dir / "gate_head.pt")

    metrics = {"cell": args.cell, "fold": args.fold, "levels": list(levels),
               "g_star_mean": float(tr["g_star"].mean()), "per_level": []}
    for split in [s for s in ("val", "test") if s in splits]:
        for level in levels:
            r = data[(split, level)]
            x = torch.from_numpy(feats(r)).float().to(device)
            with torch.no_grad():
                p = gate(x).cpu().numpy()
            metrics["per_level"].append({
                "split": split, "level": level, "n": len(p),
                "g_star_mean": float(r["g_star"].mean()),
                "g_star_p05": float(np.percentile(r["g_star"], 5)),
                "g_star_p95": float(np.percentile(r["g_star"], 95)),
                "ghat_vs_g": spearman(p, r["g_star"]),
                "ig_vs_g": spearman(r["ig"], r["g_star"]),
                "y_oracle_mae": float(np.abs(r["e_vt"] - (r["g_star"] * r["e_v"] +
                                                          (1 - r["g_star"]) * r["e_t"])).mean())})
            print("[%s/%s] g*=%.2f [p05 %.2f p95 %.2f] ghat~g*=%+.3f IG~g*=%+.3f"
                  % (split, level, metrics["per_level"][-1]["g_star_mean"],
                     metrics["per_level"][-1]["g_star_p05"],
                     metrics["per_level"][-1]["g_star_p95"],
                     metrics["per_level"][-1]["ghat_vs_g"],
                     metrics["per_level"][-1]["ig_vs_g"]))
    for split in splits:
        np.savez_compressed(
            out_dir / ("rows_%s.npz" % split),
            session=np.concatenate([data[(split, l)]["session"] for l in levels]),
            start=np.concatenate([data[(split, l)]["start"] for l in levels]),
            level=np.concatenate([np.full(len(data[(split, l)]["g_star"]), int(l))
                                  for l in levels]),
            v_tok=np.concatenate([data[(split, l)]["v_tok"] for l in levels]),
            t_tok=np.concatenate([data[(split, l)]["t_tok"] for l in levels]),
            ig=np.concatenate([data[(split, l)]["ig"] for l in levels]),
            g_star=np.concatenate([data[(split, l)]["g_star"] for l in levels]),
            e_v=np.concatenate([data[(split, l)]["e_v"] for l in levels]),
            e_t=np.concatenate([data[(split, l)]["e_t"] for l in levels]),
            e_vt=np.concatenate([data[(split, l)]["e_vt"] for l in levels]))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print("[out] %s" % out_dir)


if __name__ == "__main__":
    main()
