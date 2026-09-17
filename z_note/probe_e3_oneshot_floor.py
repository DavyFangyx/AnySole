"""E3 one-shot g(F) floor over the full test split (the "baseline of baselines").

Deterministic (no sampling, no chain): feed pure noise at tau=999 (head echo
slope is ~0 there, so the output is the pure condition readout g(F)), then FK
with the traj head's own trans (same metric path as eval.py chains).  Reports
MPJPE / per-joint MPJPE / frame jitter / per-joint jitter for VT/V/T.

Chain sanity gate for any future model: chain MPJPE must be <= one-shot MPJPE,
and chain jitter must be close to one-shot jitter.  Anything worse means the
sampling path is destroying the readout.
"""
import sys
from pathlib import Path

import numpy as np
import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.diffusion import GaussianDiffusion
from anysole.geometry import fk_pose6d
from anysole.models import AnySoleModel
from anysole.train import condition_inputs, load_config, move_batch
from anysole.types import CONFIG_NAMES, POSE_DIM

DEVICE = torch.device("cuda:6")
CKPT = GAIT / "results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt"


def main():
    config = load_config(GAIT / "configs/v1.yaml")
    config["contact_method"] = "joint_and"
    ckpt = torch.load(CKPT, map_location="cpu")
    saved = ckpt["config"]
    model = AnySoleModel(
        d=int(saved["d_model"]), tw=int(saved["tw"]), modal=str(saved["modal"]),
        dropout=float(saved["dropout"]), pose_layers=int(saved["pose_layers"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.noise_scaled = bool(saved.get("noise_scaled", False))
    model.eval()
    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=20, contact_method="joint_and",
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                         num_workers=4, collate_fn=collate_windows)

    n = 0
    err_sum = torch.zeros(23, device=DEVICE, dtype=torch.float64)
    jit_sum = torch.zeros(23, device=DEVICE, dtype=torch.float64)
    n_frames = 0
    with torch.inference_mode():
        for config_value, name in zip((0, 1, 2), CONFIG_NAMES):
            err_sum.zero_(); jit_sum.zero_(); n = 0; n_frames = 0
            for raw in loader:
                batch = move_batch(raw, DEVICE)
                bsz = batch["pose_gt"].shape[0]
                cid = torch.full((bsz,), config_value, device=DEVICE, dtype=torch.long)
                v, tr, tp = condition_inputs(batch, cid)
                tau = torch.full((bsz,), 999, device=DEVICE, dtype=torch.long)
                noise = torch.randn(bsz, 20, POSE_DIM, device=DEVICE)
                out = model(v, tr, tp, noise, tau, cid, batch.get("session_id"))
                anchor = batch["trans_anchor"][:, None, :]
                trans_world = out["trans_hat"] + anchor
                kp = fk_pose6d(out["x0_hat"], trans_world, batch["offsets"], batch["parents"])
                err = torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1)  # (B,T,J)
                jit = torch.linalg.vector_norm(kp[:, 1:] - kp[:, :-1], dim=-1)  # (B,T-1,J)
                err_sum += err.double().sum(dim=(0, 1))
                jit_sum += jit.double().sum(dim=(0, 1))
                n += bsz * 20
                n_frames += bsz * 19
            mpjpe = err_sum / n
            jit_mean = jit_sum / n_frames
            print("%s one-shot: MPJPE %.1f mm | jitter %.1f mm/frame" % (name, mpjpe.mean() * 1000, jit_mean.mean() * 1000))
            print("  per-joint MPJPE:", np.round(mpjpe.cpu().numpy() * 1000, 1))
            print("  per-joint jitter:", np.round(jit_mean.cpu().numpy() * 1000, 1))


if __name__ == "__main__":
    main()
