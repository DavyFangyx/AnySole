"""Condition ceiling: MPJPE of the pure condition-only prediction (no sampling).

model(noise, tau=999) = E[x0|F], the best any denoiser could do without x_tau
info. Compare with B1 (mean pose, 153.2mm) and the DDIM endpoint (150.4mm).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.utils.geometry import fk_pose6d
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1
from anysole.train import condition_inputs

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

session_ids = load_split_ids(Path(cfg["split_csv"]), "test")
ds = AnySoleDataset(
    mode="eval", seq_root=Path(cfg["seq_root"]), split_csv=Path(cfg["split_csv"]),
    cache_root=Path(cfg["cache_root"]), window_length=int(cfg["tw"]),
    session_ids=session_ids, contact_method=str(cfg.get("contact_method", "joint_and")),
)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_windows)
sum_err = torch.zeros(()).to(device)
n = 0
with torch.inference_mode():
    for raw_batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
        B = batch["pose_gt"].shape[0]
        config_id = torch.zeros(B, dtype=torch.long, device=device)
        v_feat, t_raw, t_phys = condition_inputs(batch, config_id)
        noise = torch.randn_like(batch["pose_gt"])
        tau = torch.full((B,), 999, dtype=torch.long, device=device)
        y = model(v_feat, t_raw, t_phys, noise, tau, config_id)["x0_hat"]
        gt_trans_world = batch["trans_gt"] + batch["trans_anchor"][:, None, :]
        err = torch.linalg.vector_norm(
            fk_pose6d(y, gt_trans_world, batch["offsets"], batch["parents"])
            - batch["kp_gt"], dim=-1) * 1000.0
        sum_err += err.sum()
        n += err.numel()

print("condition-only (tau=999, one shot) MPJPE = %.1f mm" % float(sum_err / n))
print("  vs DDIM endpoint (GT trans) 150.4 mm | B1 mean-pose 153.2 mm")
