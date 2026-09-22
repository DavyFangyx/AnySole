"""Decompose full-split MPJPE: pose error vs trajectory error contribution.

Per window: MPJPE with pred_trans (eval.py exact) vs MPJPE with GT trans.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.utils.diffusion import GaussianDiffusion
from anysole.utils.geometry import fk_pose6d
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1
from anysole.train import condition_inputs
from anysole.types import POSE_DIM

CKPT = REPO / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt"
device = torch.device("cuda")
torch.manual_seed(0)

ckpt = torch.load(CKPT, map_location="cpu")
cfg = ckpt["config"]
model = AnySoleModel(
    d=int(cfg["d_model"]), tw=int(cfg["tw"]), modal=MODEL_ANYSOLEV1,
    dropout=float(cfg["dropout"]), pose_layers=int(cfg.get("pose_layers", 6)),
).to(device)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
diffusion = GaussianDiffusion(n_train_steps=int(cfg["diffusion_train_steps"]))

session_ids = load_split_ids(Path(cfg["split_csv"]), "test")
ds = AnySoleDataset(
    mode="eval", seq_root=Path(cfg["seq_root"]), split_csv=Path(cfg["split_csv"]),
    cache_root=Path(cfg["cache_root"]), window_length=int(cfg["tw"]),
    session_ids=session_ids, contact_method=str(cfg.get("contact_method", "joint_and")),
)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_windows)
sum_pred = torch.zeros(()).to(device)  # MPJPE with predicted trajectory
sum_gt = torch.zeros(()).to(device)    # MPJPE with GT trajectory
n_joints_total = 0
per_session_gt = {}
with torch.inference_mode():
    for raw_batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
        B = batch["pose_gt"].shape[0]
        config_id = torch.zeros(B, dtype=torch.long, device=device)  # VT2M
        v_feat, t_raw, t_phys = condition_inputs(batch, config_id)
        pred_pose = diffusion.ddim_sample_loop(
            model,
            tau_related_kwargs={"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                                "config_id": config_id, "session_id": batch.get("session_id")},
            shape=(B, int(cfg["tw"]), POSE_DIM), steps=int(cfg["diffusion_sample_steps"]),
            eta=0.0, device=device,
        )
        tau_zero = torch.zeros(B, device=device, dtype=torch.long)
        out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id, batch.get("session_id"))
        anchor = batch["trans_anchor"][:, None, :]
        pred_trans_world = out["trans_hat"] + anchor
        gt_trans_world = batch["trans_gt"] + anchor
        err_pred = torch.linalg.vector_norm(
            fk_pose6d(pred_pose, pred_trans_world, batch["offsets"], batch["parents"])
            - batch["kp_gt"], dim=-1) * 1000.0  # (B,20,23)
        err_gt = torch.linalg.vector_norm(
            fk_pose6d(pred_pose, gt_trans_world, batch["offsets"], batch["parents"])
            - batch["kp_gt"], dim=-1) * 1000.0
        sum_pred += err_pred.sum()
        sum_gt += err_gt.sum()
        n_joints_total += err_pred.numel()
        for i, sid in enumerate(raw_batch["session_id"]):
            per_session_gt.setdefault(sid, []).append(float(err_gt[i].mean().item()))

print("windows=%d, MPJPE pred_trans=%.1f mm | GT_trans=%.1f mm | B1 baseline=153.2 mm"
      % (len(ds), float(sum_pred / n_joints_total), float(sum_gt / n_joints_total)))
sess = sorted((sum(v) / len(v), k) for k, v in per_session_gt.items())
print("per-session GT-trans MPJPE: min %.0f  q25 %.0f  med %.0f  q75 %.0f  max %.0f"
      % (sess[0][0], sess[len(sess) // 4][0], sess[len(sess) // 2][0],
         sess[3 * len(sess) // 4][0], sess[-1][0]))
print("best  :", [(k, round(m)) for m, k in sess[:3]])
print("worst :", [(k, round(m)) for m, k in sess[-3:]])
