#!/usr/bin/env python3
"""A：双向残差可靠性 head —— 阶段 1（标签/token 抽取 + head 训练/验证）。

对一折、一个基座（默认 V8_Cv5VG_fS5S6S11_s1）：

1. 在 split × occ 档上跑模型的 VT / V / T 三行：
   - VT 行：取 `v_tok` / `t_tok` 的窗均值（门的输入）；
   - V 行 / T 行：`fk_pose6d` → 逐窗 `PA-MPJPE_lower`（行误差 e_V / e_T）；
   - GT 动作难度：逐窗 GT jerk + GT 根速度。
2. 在 **train** 行上拟合 `log e_row ~ [1, log jerk_gt, speed_gt]`，得到残差
   `z_row = (log e_row − f_row) / std_train`（就是「比同动作难度预期好/坏多少」）。
3. 在 train 行上训练两个小 MLP：`v_tok → z_V`、`t_tok → z_T`；val 早停；
   test（held-out 受试者）只验证。
4. 落盘 head 权重 + 逐窗 rows（token/残差/误差/难度）+ 验证指标。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/a_residual_head.py \
      --cell V8_Cv5VG_fS5S6S11_s1 --fold fS5S6S11 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "results_display/script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import (CONFIG_T, CONFIG_V, CONFIG_VT, FPS,  # noqa: E402
                           JOINT_HIERARCHY)
from anysole.utils.geometry import fk_pose6d  # noqa: E402
from anysole.utils.metrics import (pa_mpjpe, pelvis_align,  # noqa: E402
                                   temporal_metrics)

LOWER = list(JOINT_HIERARCHY["lower"])
LEVELS = ("000", "020", "040", "060", "080", "100")
OUT = REPO / "results/diagnostics/20261107_a_residual_head"


def per_window_errors(out, batch) -> np.ndarray:
    """逐窗 V/T 行 lower PA-MPJPE（mm）。out 为该行的输出。"""
    pred = fk_pose6d(out["x0_hat"], out["trans_hat"],
                     batch["offsets"], batch["parents"])
    gt = batch["kp_gt"]
    p = pred.detach().cpu().numpy()
    g = gt.detach().cpu().numpy()
    vals = []
    for b in range(p.shape[0]):
        err = pa_mpjpe(p[b][:, LOWER], g[b][:, LOWER])  # (tw,)
        vals.append(float(err.mean()))
    return np.asarray(vals)


def gt_jerk_speed(batch) -> tuple[np.ndarray, np.ndarray]:
    kp = batch["kp_gt"].detach().cpu().numpy()
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


def run_split(net, loader, device, split: str, level: str, max_batches: int = 0) -> dict:
    rows = {k: [] for k in ("session", "start", "v_tok", "t_tok",
                            "e_v", "e_t", "jerk_gt", "speed_gt")}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = move_batch(batch, device)
        n = batch["V_feat"].shape[0]
        with torch.no_grad():
            # VT 行：token 与门输入
            cid = torch.full((n,), CONFIG_VT, dtype=torch.long, device=device)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
            out_vt = net(v_feat, t_raw, t_phys, cid, batch.get("subject_ids"), T_s2m=t_s2m)
            v_tok = out_vt["v_tok"].mean(dim=1).detach().cpu().numpy()
            t_tok = out_vt["t_tok"].mean(dim=1).detach().cpu().numpy()
            # V 行
            cid_v = torch.full((n,), CONFIG_V, dtype=torch.long, device=device)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid_v)
            out_v = net(v_feat, t_raw, t_phys, cid_v, batch.get("subject_ids"), T_s2m=t_s2m)
            # T 行
            cid_t = torch.full((n,), CONFIG_T, dtype=torch.long, device=device)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid_t)
            out_t = net(v_feat, t_raw, t_phys, cid_t, batch.get("subject_ids"), T_s2m=t_s2m)
        e_v = per_window_errors(out_v, batch)
        e_t = per_window_errors(out_t, batch)
        jg, sp = gt_jerk_speed(batch)
        rows["session"].extend(batch["session_id"])
        rows["start"].extend(batch["frame_start"].detach().cpu().tolist())
        rows["v_tok"].append(v_tok); rows["t_tok"].append(t_tok)
        rows["e_v"].append(e_v); rows["e_t"].append(e_t)
        rows["jerk_gt"].append(jg); rows["speed_gt"].append(sp)
    out = {k: (np.concatenate(v) if k in ("v_tok", "t_tok", "e_v", "e_t",
                                          "jerk_gt", "speed_gt") else np.asarray(v))
           for k, v in rows.items()}
    out["split"] = split; out["level"] = level
    return out


def fit_residual(rows: dict, key: str, stats: dict | None = None):
    """log e ~ [1, log jerk, speed] 的 train 拟合；返回残差与拟合参数。"""
    e = np.asarray(rows[key], dtype=np.float64)
    j = np.log(np.asarray(rows["jerk_gt"], dtype=np.float64) + 1e-6)
    s = np.asarray(rows["speed_gt"], dtype=np.float64)
    X = np.stack([np.ones_like(j), j, s], axis=1)
    y = np.log(np.clip(e, 1e-6, None))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    if stats is None:
        stats = {"coef": coef.tolist(), "std": float(np.std(resid) + 1e-8)}
    z = resid / stats["std"]
    return z, stats


def apply_residual(rows: dict, key: str, stats: dict) -> np.ndarray:
    e = np.asarray(rows[key], dtype=np.float64)
    j = np.log(np.asarray(rows["jerk_gt"], dtype=np.float64) + 1e-6)
    s = np.asarray(rows["speed_gt"], dtype=np.float64)
    X = np.stack([np.ones_like(j), j, s], axis=1)
    return (np.log(np.clip(e, 1e-6, None)) - X @ np.asarray(stats["coef"])) / stats["std"]


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 5:
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


class Head(nn.Module):
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(x_tr, z_tr, x_va, z_va, device, epochs=300, lr=1e-3):
    d = x_tr.shape[1]
    head = Head(d).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(x_tr).float().to(device)
    zt = torch.from_numpy(z_tr).float().to(device)
    xv = torch.from_numpy(x_va).float().to(device)
    zv = torch.from_numpy(z_va).float().to(device)
    best = (float("inf"), None)
    for _ in range(epochs):
        head.train(); opt.zero_grad()
        loss = nn.functional.mse_loss(head(xt), zt)
        loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = float(nn.functional.mse_loss(head(xv), zv))
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1]); head.eval()
    return head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="V8_Cv5VG_fS5S6S11_s1")
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
    net, ckpt, t_phys_v2 = build_model(args.cell, device)
    print("[ckpt] %s" % ckpt)
    fold_yaml = args.fold[1:] if args.fold.startswith("f") else args.fold
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for split in splits:
        for level in levels:
            config = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                      else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
            loader = build_loader(config, split, args.batch_size, t_phys_v2=t_phys_v2)
            r = run_split(net, loader, device, split, level, args.max_batches)
            data[(split, level)] = r
            print("[%s/%s] windows=%d" % (split, level, len(r["e_v"])))

    # 残差拟合：train 全档合并
    tr = {k: np.concatenate([data[("train", l)][k] for l in levels])
          for k in ("e_v", "e_t", "jerk_gt", "speed_gt", "v_tok", "t_tok")}
    zV_tr, statsV = fit_residual(tr, "e_v")
    zT_tr, statsT = fit_residual(tr, "e_t")
    zV = {l: apply_residual(data[("train", l)], "e_v", statsV) for l in levels}
    zT = {l: apply_residual(data[("train", l)], "e_t", statsT) for l in levels}

    # head：train 训练，val 早停
    xV_tr = tr["v_tok"]; xT_tr = tr["t_tok"]
    va = {k: np.concatenate([data[("val", l)][k] for l in levels])
          for k in ("v_tok", "t_tok")}
    zV_va = np.concatenate([apply_residual(data[("val", l)], "e_v", statsV) for l in levels])
    zT_va = np.concatenate([apply_residual(data[("val", l)], "e_t", statsT) for l in levels])
    headV = train_head(xV_tr, zV_tr, va["v_tok"], zV_va, device)
    headT = train_head(xT_tr, zT_tr, va["t_tok"], zT_va, device)

    # 验证：val/test 逐档
    metrics = {"cell": args.cell, "fold": args.fold, "levels": list(levels),
               "statsV": statsV, "statsT": statsT, "per_level": []}
    for split in [s for s in ("val", "test") if s in splits]:
        for level in levels:
            r = data[(split, level)]
            zV_true = apply_residual(r, "e_v", statsV)
            zT_true = apply_residual(r, "e_t", statsT)
            with torch.no_grad():
                pV = headV(torch.from_numpy(r["v_tok"]).float().to(device)).cpu().numpy()
                pT = headT(torch.from_numpy(r["t_tok"]).float().to(device)).cpu().numpy()
            metrics["per_level"].append({
                "split": split, "level": level, "n": len(zV_true),
                "V_head_vs_z": spearman(pV, zV_true),
                "T_head_vs_z": spearman(pT, zT_true),
                "V_head_vs_e": spearman(pV, r["e_v"]),
                "T_head_vs_e": spearman(pT, r["e_t"])})
            print("[%s/%s] V: head~z=%+.3f head~e=%+.3f | T: head~z=%+.3f head~e=%+.3f"
                  % (split, level, metrics["per_level"][-1]["V_head_vs_z"],
                     metrics["per_level"][-1]["V_head_vs_e"],
                     metrics["per_level"][-1]["T_head_vs_z"],
                     metrics["per_level"][-1]["T_head_vs_e"]))
    # 合并 test 的逐窗 rows 供门阶段使用
    for split in splits:
        np.savez_compressed(
            out_dir / ("rows_%s.npz" % split),
            session=np.concatenate([data[(split, l)]["session"] for l in levels]),
            start=np.concatenate([data[(split, l)]["start"] for l in levels]),
            level=np.concatenate([np.full(len(data[(split, l)]["e_v"]), int(l))
                                  for l in levels]),
            v_tok=np.concatenate([data[(split, l)]["v_tok"] for l in levels]),
            t_tok=np.concatenate([data[(split, l)]["t_tok"] for l in levels]),
            e_v=np.concatenate([data[(split, l)]["e_v"] for l in levels]),
            e_t=np.concatenate([data[(split, l)]["e_t"] for l in levels]),
            jerk_gt=np.concatenate([data[(split, l)]["jerk_gt"] for l in levels]),
            speed_gt=np.concatenate([data[(split, l)]["speed_gt"] for l in levels]))
    torch.save(headV.state_dict(), out_dir / "headV.pt")
    torch.save(headT.state_dict(), out_dir / "headT.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print("[out] %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
