"""Probe the retrained (new-head) checkpoint: D1/D2/tau0/DDIM on a fixed eval batch.

Compares against the old-head probe record (D1 ~2% of output std, D2 22%,
trained tau0 L_pose 0.208) and the B1 mean-pose baseline (153.2 mm).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.utils.diffusion import GaussianDiffusion
from anysole.utils.geometry import fk_pose6d
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1

CKPT = REPO / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print("loaded %s (epoch %s, pose_layers=%s)" % (CKPT, ckpt["epoch"], cfg.get("pose_layers")))

ds = AnySoleDataset(
    mode="eval", seq_root=Path(cfg["seq_root"]), split_csv=Path(cfg["split_csv"]),
    cache_root=Path(cfg["cache_root"]), window_length=int(cfg["tw"]),
    contact_method=str(cfg.get("contact_method", "joint_and")),
)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_windows)
batch = next(iter(loader))
batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
B = batch["pose_gt"].shape[0]
config_id = torch.zeros(B, dtype=torch.long, device=device)  # VT-only
x0 = batch["pose_gt"]


def run_model(x, tau):
    with torch.no_grad():
        return model(batch["V_feat"], batch["T_raw"], batch["T_phys"], x, tau, config_id)["x0_hat"]


def mse(a, b):
    return float(((a - b) ** 2).mean().item())


noise = torch.randn_like(x0)
y_std = float(run_model(x0, torch.zeros(B, dtype=torch.long, device=device)).std().item())

# tau0 clean-input reconstruction
y_tau0 = run_model(x0, torch.zeros(B, dtype=torch.long, device=device))
print("tau0 clean-input: L_pose=%.4f (old head trained: 0.208)" % mse(y_tau0, x0))

# D1: x_tau sensitivity at tau=0 (GT vs noise input)
y_noise_tau0 = run_model(noise, torch.zeros(B, dtype=torch.long, device=device))
d1 = float((y_tau0 - y_noise_tau0).abs().mean().item())
print("D1 x_tau sensitivity at tau=0: %.4f (%.0f%% of output std %.4f; old: 0.010 ~2%%)"
      % (d1, 100 * d1 / y_std, y_std))

# D2: tau sensitivity (same x_tau=GT, tau 0 vs 999)
y_999 = run_model(x0, torch.full((B,), 999, dtype=torch.long, device=device))
d2 = float((y_tau0 - y_999).abs().mean().item())
print("D2 tau sensitivity (0 vs 999): %.4f (%.0f%% of output std; old: 0.108 = 22%%)"
      % (d2, 100 * d2 / y_std))

# Per-tau L_pose curve vs condition-only floor and data variance
print("\ntau       L_pose      (floor: condition-only=%.4f, x0 var=%.4f)"
      % (mse(y_999, x0), float(x0.var().item())))
for t in (0, 10, 30, 60, 110, 210, 400, 700, 999):
    tau = torch.full((B,), t, dtype=torch.long, device=device)
    x_tau = diffusion.q_sample(x0, tau)
    err = mse(run_model(x_tau, tau), x0)
    print("%4d   %.4f" % (t, err))

# DDIM sample on this batch -> fk MPJPE vs kp_gt
with torch.no_grad():
    sample = diffusion.ddim_sample_loop(
        model,
        tau_related_kwargs={"V_feat": batch["V_feat"], "T_raw": batch["T_raw"],
                            "T_phys": batch["T_phys"], "config_id": config_id},
        steps=int(cfg["diffusion_sample_steps"]), eta=0.0, device=device,
    )
    anchor = batch.get("trans_anchor")
    if anchor is None:
        anchor = torch.zeros(B, 3, device=device)
    pred_kp = fk_pose6d(sample, batch["trans_gt"] + anchor[:, None, :], batch["offsets"], batch["parents"])
    mpjpe = float(torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1).mean().item() * 1000.0)
    kp_std = float(batch["kp_gt"].std(dim=(0, 1)).norm().item())
    print("\nDDIM sample MPJPE on this batch: %.1f mm  (B1 mean-pose baseline: 153.2, kp std %.1f)"
          % (mpjpe, kp_std))
