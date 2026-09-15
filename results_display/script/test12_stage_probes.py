"""Pinpoint where the V pose-info dies: v_tok -> F -> head, ridge probes each.

Baseline: ridge V_feat(2051) -> pose = 70.1mm; trained head condition-only
output = 149.4mm. Probe each stage of the trained model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.geometry import fk_pose6d
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1

CKPT = REPO / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt"
device = torch.device("cuda")

ckpt = torch.load(CKPT, map_location="cpu")
cfg = ckpt["config"]
model = AnySoleModel(
    d=int(cfg["d_model"]), tw=int(cfg["tw"]), modal=MODEL_ANYSOLEV1,
    dropout=float(cfg["dropout"]), pose_layers=int(cfg.get("pose_layers", 6)),
).to(device)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()


def ridge_fit(X, Y, lam=1.0):
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, device=device)], dim=1)
    A = Xb.T @ Xb + lam * torch.eye(d + 1, device=device)
    return torch.linalg.solve(A, Xb.T @ Y)


def ridge_apply(W, X):
    return torch.cat([X, torch.ones(X.shape[0], 1, device=device)], dim=1) @ W


# collect v_tok (20,256), F (40,256), pose per frame from TRAIN, then fit
def collect(mode):
    ds = AnySoleDataset(
        mode=mode, seq_root=Path(cfg["seq_root"]), split_csv=Path(cfg["split_csv"]),
        cache_root=Path(cfg["cache_root"]), window_length=int(cfg["tw"]),
        session_ids=(load_split_ids(Path(cfg["split_csv"]), "test") if mode == "eval" else None),
        contact_method=str(cfg.get("contact_method", "joint_and")),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_windows)
    vtok, fus, pos = [], [], []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
            B = batch["pose_gt"].shape[0]
            config_id = torch.zeros(B, dtype=torch.long, device=device)
            v_tok, t_tok = model.encoders(batch["V_feat"], batch["T_raw"], batch["T_phys"], config_id)
            F = model.fusion(v_tok, t_tok)
            vtok.append(v_tok)
            fus.append(F)
            pos.append(batch["pose_gt"])
    return (torch.cat(vtok).reshape(-1, 256), torch.cat(fus).view(-1, 20, 2, 256).reshape(-1, 512),
            torch.cat(pos).reshape(-1, 138)), ds


(train_vtok, train_F, train_p), train_ds = collect("train")
(val_vtok, val_F, val_p), val_ds = collect("eval")
print("train %d, val %d frames" % (train_p.shape[0], val_p.shape[0]))

val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_windows)


def mpjpe(pose_pred):
    idx = 0
    sum_err, n = torch.zeros(()).to(device), 0
    with torch.no_grad():
        for batch in val_loader:
            B, TW = batch["pose_gt"].shape[0], batch["pose_gt"].shape[1]
            pred = pose_pred[idx: idx + B * TW].reshape(B, TW, -1).to(device)
            idx += B * TW
            gt_trans = batch["trans_gt"].to(device) + batch["trans_anchor"].to(device)[:, None, :]
            err = torch.linalg.vector_norm(
                fk_pose6d(pred, gt_trans, batch["offsets"].to(device), batch["parents"].to(device))
                - batch["kp_gt"].to(device), dim=-1) * 1000.0
            sum_err += err.sum()
            n += err.numel()
    return float(sum_err / n)


for name, Xtr, Ytr, Xva in (("v_tok (256/frame)", train_vtok, train_p, val_vtok),
                            ("F (512/frame)", train_F, train_p, val_F)):
    W = ridge_fit(Xtr, Ytr)
    pred = ridge_apply(W, Xva)
    print("ridge %s -> pose: val MPJPE = %.1f mm, L_pose = %.4f" % (name, mpjpe(pred), float(((pred - val_p) ** 2).mean())))
print("reference: raw V_feat=70.1 | B1=153.2 | trained head tau999 output=149.4 (L_pose=0.051)")

# per-joint MPJPE: ridge-F prediction vs head tau999 output
from anysole.types import JOINT_NAMES
W_F = ridge_fit(train_F, train_p)
pred_F = ridge_apply(W_F, val_F)
sum_r, sum_h, n = {}, {}, 0
with torch.inference_mode():
    idx = 0
    for batch in val_loader:
        B, TW = batch["pose_gt"].shape[0], batch["pose_gt"].shape[1]
        x = batch["pose_gt"].to(device)
        # head tau999 output on this batch
        noise = torch.randn_like(x)
        tau = torch.full((B,), 999, dtype=torch.long, device=device)
        config_id = torch.zeros(B, dtype=torch.long, device=device)
        with torch.no_grad():
            y_h = model(batch["V_feat"].to(device), batch["T_raw"].to(device),
                        batch["T_phys"].to(device), noise, tau, config_id)["x0_hat"]
        y_r = pred_F[idx: idx + B * TW].reshape(B, TW, -1).to(device)
        idx += B * TW
        gt_trans = batch["trans_gt"].to(device) + batch["trans_anchor"].to(device)[:, None, :]
        kp_gt = batch["kp_gt"].to(device)
        offs, pars = batch["offsets"].to(device), batch["parents"].to(device)
        err_r = torch.linalg.vector_norm(fk_pose6d(y_r, gt_trans, offs, pars) - kp_gt, dim=-1) * 1000.0
        err_h = torch.linalg.vector_norm(fk_pose6d(y_h, gt_trans, offs, pars) - kp_gt, dim=-1) * 1000.0
        for j, name_j in enumerate(JOINT_NAMES):
            sum_r[name_j] = sum_r.get(name_j, 0.0) + float(err_r[:, :, j].sum())
            sum_h[name_j] = sum_h.get(name_j, 0.0) + float(err_h[:, :, j].sum())
        n += err_r.shape[0] * err_r.shape[1]
print("\njoint       ridge-F   head(t999)  B1")
b1 = {"Hips": 0.0, "LeftArm": 170.2, "LeftFoot": 207.7, "LeftForeArm": 235.0, "LeftHand": 318.7,
      "LeftLeg": 205.0, "LeftShoulder": 99.8, "LeftToeBase": 234.1, "LeftUpLeg": 66.8, "Neck": 111.6,
      "RightArm": 172.7, "RightFoot": 204.0, "RightForeArm": 240.0, "RightHand": 313.2, "RightLeg": 198.1,
      "RightShoulder": 100.4, "RightToeBase": 227.3, "RightUpLeg": 67.0, "Spine": 22.2, "Spine1": 39.9,
      "Spine2": 64.1, "Spine3": 91.3, "Head": 134.7}
for name_j in JOINT_NAMES:
    print("%-12s %8.1f %8.1f %8.1f" % (name_j, sum_r[name_j] / n, sum_h[name_j] / n, b1.get(name_j, float("nan"))))
