"""Information probes: how much pose info survives in the raw modality features?

Ridge regression per-frame: V_feat (2051) -> native SMPL-24 pose_gt (144) and T (108) -> pose.
Fitted on the train split, MPJPE on the val split with GT trajectory.
Compare with B1 (mean pose) = 153.2mm and the model's V2M/T2M (180.6 / 195.5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.geometry import fk_pose6d
from anysole.types import POSE_DIM, T_PHYS_DIM, T_RAW_DIM, V_FEAT_DIM

device = torch.device("cuda")
CFG = dict(seq_root=Path("AnysoleWorkspace/derived/MotionPRO/sequences/cam3"),
           split_csv=Path("AnysoleWorkspace/splits/default/splits.csv"),
           cache_root=Path("AnysoleWorkspace/derived/AnySole/hrnet_cache/cam3"),
           window_length=20, contact_method="joint_and")


def load_split(mode):
    ds = AnySoleDataset(mode=mode, **CFG)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_windows)
    vfeats, tf, poses = [], [], []
    with torch.no_grad():
        for batch in loader:
            vfeats.append(batch["V_feat"])
            tf.append(torch.cat([batch["T_raw"], batch["T_phys"]], dim=-1))
            poses.append(batch["pose_gt"])
    return (torch.cat(vfeats).to(device), torch.cat(tf).to(device), torch.cat(poses).to(device)), ds


def ridge_fit(X, Y, lam=1.0):
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, device=device)], dim=1)
    A = Xb.T @ Xb + lam * torch.eye(d + 1, device=device)
    W = torch.linalg.solve(A, Xb.T @ Y)
    return W


def mpjpe_from_pose(pose_pred, ds, split_loader):
    """MPJPE of predicted poses using each window's GT trans + offsets/parents."""
    idx = 0
    sum_err, n = torch.zeros(()).to(device), 0
    with torch.no_grad():
        for batch in split_loader:
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


(train_v, train_t, train_p), train_ds = load_split("train")
(val_v, val_t, val_p), val_ds = load_split("eval")
train_v = train_v.reshape(-1, V_FEAT_DIM)
train_t = train_t.reshape(-1, T_RAW_DIM + T_PHYS_DIM)
train_p = train_p.reshape(-1, POSE_DIM)
val_v = val_v.reshape(-1, V_FEAT_DIM)
val_t = val_t.reshape(-1, T_RAW_DIM + T_PHYS_DIM)
val_p = val_p.reshape(-1, POSE_DIM)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_windows)
print("train samples %d, val %d" % (train_v.shape[0], val_v.shape[0]))

for name, Xtr, Ytr, Xva in (("V_feat(2051)", train_v, train_p, val_v),
                            ("T(108)", train_t, train_p, val_t),
                            ("V+T(2159)", torch.cat([train_v, train_t], dim=1), train_p,
                             torch.cat([val_v, val_t], dim=1))):
    W = ridge_fit(Xtr, Ytr)
    pred = torch.cat([Xva, torch.ones(Xva.shape[0], 1, device=device)], dim=1) @ W
    m = mpjpe_from_pose(pred, val_ds, val_loader)
    print("ridge %s -> pose: val MPJPE = %.1f mm (B1=153.2, model V2M=180.6 T2M=195.5)" % (name, m))
