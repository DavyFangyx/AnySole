"""E3 jitter source decomposition: condition-readout noise vs chain noise.

On the E3_nocontact checkpoint, for 8 consecutive windows of one session:
1. tau0 path: feed GT pose at tau=0, measure x0_hat FK kp frame-to-frame jitter.
   Small => readout path smooth, chain is the jitter source. Large => per-frame
   condition noise dominates.
2. DDIM chain (10/20/50 steps) kp jitter on the same windows.
3. Echo slope at high tau: regress x0_hat on the input noise (pure-noise input)
   vs Bayes-optimal slope sqrt(abar).
4. Per-joint jitter spectrum to see where jitter concentrates (ankle/feet?).
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
from anysole.types import CONFIG_VT, N_JOINTS, POSE_DIM

DEVICE = torch.device("cuda:6")
CKPT = GAIT / "results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt"


def jitter_mm(kp):
    """Mean frame-to-frame keypoint displacement in mm, per joint."""
    d = torch.linalg.vector_norm(kp[:, 1:] - kp[:, :-1], dim=-1)  # (B,T-1,J)
    return d.mean(dim=(0, 1)).cpu().numpy() * 1000.0


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
    diffusion = GaussianDiffusion(n_train_steps=int(saved["diffusion_train_steps"]))

    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=20,
        contact_method="joint_and",
    )
    idx = [i for i in range(len(ds)) if ds[i]["session_id"] == "S10103"][:8]
    batch = move_batch(collate_windows([ds[i] for i in idx]), DEVICE)
    bsz = batch["pose_gt"].shape[0]
    c = torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long)
    v, tr, tp = condition_inputs(batch, c)
    cond = {"V_feat": v, "T_raw": tr, "T_phys": tp, "config_id": c, "session_id": batch.get("session_id")}
    anchor = batch["trans_anchor"][:, None, :]
    trans_world = batch["trans_gt"] + anchor

    # 1. tau0: clean GT pose in, readout jitter
    with torch.inference_mode():
        out0 = model(v, tr, tp, batch["pose_gt"], torch.zeros(bsz, device=DEVICE, dtype=torch.long), c, batch.get("session_id"))
        kp0 = fk_pose6d(out0["x0_hat"], trans_world, batch["offsets"], batch["parents"])
        j0 = jitter_mm(kp0)
        print("tau0   readout jitter: mean %.1f mm/frame, per-joint:" % j0.mean())
        print("   ", np.round(j0, 1))

    # 2. DDIM chains
    for steps in (10, 20, 50):
        with torch.inference_mode():
            p = diffusion.ddim_sample_loop(model, tau_related_kwargs=cond, shape=(bsz, 20, POSE_DIM),
                                           steps=steps, eta=0.0, device=DEVICE)
            out = model(v, tr, tp, p, torch.zeros(bsz, device=DEVICE, dtype=torch.long), c, batch.get("session_id"))
            kp = fk_pose6d(p, out["trans_hat"] + anchor, batch["offsets"], batch["parents"])
            j = jitter_mm(kp)
            err = torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1).mean().item() * 1000
        print("chain %2d steps: MPJPE %.1f mm, jitter mean %.1f mm/frame" % (steps, err, j.mean()))

    # 3. Echo slope at high tau (pure-noise input, tau=999)
    with torch.inference_mode():
        tau = torch.full((bsz,), 999, device=DEVICE, dtype=torch.long)
        noise = torch.randn(bsz, 20, POSE_DIM, device=DEVICE)
        outn = model(v, tr, tp, noise, tau, c, batch.get("session_id"))["x0_hat"]
        # slope per dim: regress out on noise (no intercept): <out,noise>/<noise,noise>
        slope = (outn * noise).sum() / (noise * noise).sum()
        abar = float(diffusion.alphas_cumprod[999])
        print("echo slope on pure noise @tau=999: %.3f (Bayes-optimal sqrt(abar)=%.3f)" % (slope, np.sqrt(abar)))
        # how much of the input does the output explain
        outn2 = outn.reshape(bsz, 20, N_JOINTS, 6)
        noise2 = noise.reshape(bsz, 20, N_JOINTS, 6)
        per_joint = ((outn2 * noise2).sum(dim=(0, 1, 3)) / (noise2 * noise2).sum(dim=(0, 1, 3)))
        print("echo slope per joint:", np.round(per_joint.cpu().numpy(), 2))

    # 4. per-joint jitter spectrum of the 50-step chain
    with torch.inference_mode():
        p = diffusion.ddim_sample_loop(model, tau_related_kwargs=cond, shape=(bsz, 20, POSE_DIM),
                                       steps=50, eta=0.0, device=DEVICE)
        out = model(v, tr, tp, p, torch.zeros(bsz, device=DEVICE, dtype=torch.long), c, batch.get("session_id"))
        kp = fk_pose6d(p, out["trans_hat"] + anchor, batch["offsets"], batch["parents"])
        j = jitter_mm(kp)
        print("chain 50 per-joint jitter:", np.round(j, 1))
        err_j = torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1).mean(dim=(0, 1)).cpu().numpy() * 1000
        print("chain 50 per-joint MPJPE:", np.round(err_j, 1))


if __name__ == "__main__":
    main()
