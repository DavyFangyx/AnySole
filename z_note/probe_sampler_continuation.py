"""Verify E5 head under Step2Motion-style sampling (code-level comparison).

Runs the SAME E5 checkpoint under:
  (a) current eval: 10-step DDIM, independent windows
  (b) Step2Motion sampler: T=200 DDPM posterior (stochastic), independent
  (c) Step2Motion sampler + continuation inpainting (metrics.py prev_x_t)
  (d) 50-step DDIM (the note claims more steps = worse)
Plus head characterization: single-step denoising curve vs tau, and the
echo factor (slope of x0_hat on x_tau) per tau.

Posterior noise is scaled by pose_std: training noise is randn*pose_std in
raw 6D space, so the DDPM posterior noise in raw space is pstd*pose_std.
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

DEVICE = torch.device("cuda:4")
CKPT = GAIT / "results/AnySole/E5_sched200/checkpoints/ckpt_last.pt"
SEED = 0


def build_ddpm(diffusion):
    """Step2Motion posterior coefficients for the T=200 linear schedule."""
    betas = np.concatenate([[0.0], diffusion.betas])          # (201,) leading 0
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)                                  # abar[t], t in [0,200]
    abar_prev = np.concatenate([[1.0], abar[:-1]])
    one_minus = np.where(1.0 - abar > 0, 1.0 - abar, 1.0)
    coeff1 = betas * np.sqrt(abar_prev) / one_minus            # on x0_hat
    coeff2 = np.sqrt(alphas) * (1.0 - abar_prev) / one_minus   # on x_t
    pstd = np.sqrt(np.where(1.0 - abar > 0, betas * (1.0 - abar_prev) / one_minus, 0.0))
    return {k: torch.from_numpy(v).float() for k, v in
            dict(coeff1=coeff1, coeff2=coeff2, pstd=pstd).items()}


def _posterior_step(x_t, t, x0_hat, ddpm, pose_std):
    c1 = ddpm["coeff1"][t].to(x_t.device).view(1, 1, 1)
    c2 = ddpm["coeff2"][t].to(x_t.device).view(1, 1, 1)
    s = ddpm["pstd"][t].to(x_t.device).view(1, 1, 1)
    z = torch.randn_like(x_t) * pose_std.view(1, 1, -1)
    return c1 * x0_hat + c2 * x_t + s * z


def sample_window(model, diffusion, ddpm, cond, bsz, tw, pose_std):
    x = torch.randn(bsz, tw, POSE_DIM, device=DEVICE) * pose_std.view(1, 1, -1)
    for t in range(200, 0, -1):
        tau = torch.full((bsz,), min(t, 199), device=DEVICE, dtype=torch.long)
        out = diffusion._call_model(model, x, tau, cond)
        x = _posterior_step(x, t, out["x0_hat"], ddpm, pose_std)
    return x


def sample_continue(model, diffusion, ddpm, cond, bsz, tw, pose_std, half):
    """Step2Motion metrics.py mechanism over B consecutive windows (one batch).

    At every step t, window i's first half is seeded with window i-1's second
    half at the same noise level (pre-update), so the chain continues across
    windows. Window 0 keeps its own chain (fresh noise at t=200).
    """
    x = torch.randn(bsz, tw, POSE_DIM, device=DEVICE) * pose_std.view(1, 1, -1)
    for t in range(200, 0, -1):
        x[1:, :half] = x[:-1, half:].clone()
        tau = torch.full((bsz,), min(t, 199), device=DEVICE, dtype=torch.long)
        out = diffusion._call_model(model, x, tau, cond)
        x = _posterior_step(x, t, out["x0_hat"], ddpm, pose_std)
    return x


def fk_err(model, batch, pred_pose, use_pred_traj=True):
    bsz = batch["pose_gt"].shape[0]
    c = batch["config_id"]
    t0 = torch.zeros(bsz, device=DEVICE, dtype=torch.long)
    out = model(batch["V_feat"], batch["T_raw"], batch["T_phys"], pred_pose, t0, c,
                batch.get("session_id"))
    anchor = batch["trans_anchor"][:, None, :]
    trans = (out["trans_hat"] + anchor) if use_pred_traj else (batch["trans_gt"] + anchor)
    kp = fk_pose6d(pred_pose, trans, batch["offsets"], batch["parents"])
    return torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1), trans


def jitter_mm(kp):
    return torch.linalg.vector_norm(kp[:, 1:] - kp[:, :-1], dim=-1).mean().item() * 1000.0


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    config = load_config(GAIT / "configs/v1.yaml")
    config["contact_method"] = "joint_and"
    ckpt = torch.load(CKPT, map_location="cpu")
    saved = ckpt["config"]
    print("ckpt epoch=%d contact_method=%s noise_scaled=%s" %
          (ckpt["epoch"], saved.get("contact_method"), saved.get("noise_scaled")))
    model = AnySoleModel(
        d=int(saved["d_model"]), tw=int(saved["tw"]), modal=str(saved["modal"]),
        dropout=float(saved["dropout"]), pose_layers=int(saved["pose_layers"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.noise_scaled = bool(saved.get("noise_scaled", False))
    model.eval()
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    ddpm = build_ddpm(diffusion)
    tw = int(config["tw"])
    half = tw // 2
    pose_std = model.pose_head.pose_std
    pose_mean = model.pose_head.pose_mean

    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=tw, contact_method="joint_and",
    )
    print("val windows: %d" % len(ds))

    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(ds), size=64, replace=False).tolist()
    batch = move_batch(collate_windows([ds[i] for i in idxs]), DEVICE)
    bsz = batch["pose_gt"].shape[0]
    cid = torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long)
    batch["config_id"] = cid
    v, tr, tp = condition_inputs(batch, cid)
    cond = {"V_feat": v, "T_raw": tr, "T_phys": tp, "config_id": cid,
            "session_id": batch.get("session_id")}
    gt_kp = batch["kp_gt"]
    gt_trans = batch["trans_gt"] + batch["trans_anchor"][:, None, :]

    # ---------- head characterization: denoising curve + echo ----------
    with torch.inference_mode():
        x0 = batch["pose_gt"]
        xn = (x0 - pose_mean) / pose_std
        print("\n[head] single-step denoising (64 windows, GT-noised input):")
        for tau_v in (0, 10, 50, 100, 150, 199):
            tau = torch.full((bsz,), tau_v, device=DEVICE, dtype=torch.long)
            noise = torch.randn_like(x0) * pose_std.view(1, 1, -1)
            x_tau = diffusion.q_sample(x0, tau, noise=noise)
            out = model(v, tr, tp, x_tau, tau, cid, batch.get("session_id"))
            xh = (out["x0_hat"] - pose_mean) / pose_std
            deno_err = (xh - xn).square().mean().item()
            # echo: slope of x0_hat on x_tau in normalized space
            x_tn = (x_tau - pose_mean) / pose_std
            xc = xh - xh.mean(dim=(0, 1), keepdim=True)
            tc = x_tn - x_tn.mean(dim=(0, 1), keepdim=True)
            slope = (xc * tc).sum() / tc.square().sum().clamp_min(1e-8)
            print("  tau=%3d  mse(x0_hat,x0)=%.5f (x0 var=%.4f)  echo slope=%.3f"
                  % (tau_v, deno_err, xn.var().item(), float(slope)))

    # ---------- (a) 10-step DDIM ----------
    with torch.inference_mode():
        pred = diffusion.ddim_sample_loop(model, tau_related_kwargs=cond,
            shape=(bsz, tw, POSE_DIM), steps=10, eta=0.0, device=DEVICE)
        err, _ = fk_err(model, batch, pred)
        kp = fk_pose6d(pred, gt_trans, batch["offsets"], batch["parents"])
        print("\n[a] DDIM 10 steps : MPJPE %.1f mm  jitter %.1f mm/frame"
              % (err.mean().item() * 1000, jitter_mm(kp)))

    # ---------- (d) 50-step DDIM ----------
    with torch.inference_mode():
        pred = diffusion.ddim_sample_loop(model, tau_related_kwargs=cond,
            shape=(bsz, tw, POSE_DIM), steps=50, eta=0.0, device=DEVICE)
        err, _ = fk_err(model, batch, pred)
        kp = fk_pose6d(pred, gt_trans, batch["offsets"], batch["parents"])
        print("[d] DDIM 50 steps : MPJPE %.1f mm  jitter %.1f mm/frame"
              % (err.mean().item() * 1000, jitter_mm(kp)))

    # ---------- (b) 200-step DDPM posterior ----------
    with torch.inference_mode():
        pred = sample_window(model, diffusion, ddpm, cond, bsz, tw, pose_std)
        err, _ = fk_err(model, batch, pred)
        kp = fk_pose6d(pred, gt_trans, batch["offsets"], batch["parents"])
        print("[b] DDPM 200 post : MPJPE %.1f mm  jitter %.1f mm/frame"
              % (err.mean().item() * 1000, jitter_mm(kp)))
        per_joint_b = err.mean(dim=(0, 1)) * 1000

    # ---------- (c) continuation on 8 consecutive windows ----------
    from collections import Counter
    counts = Counter(ds[i]["session_id"] for i in range(len(ds)))
    session = next(s for s, n in counts.items() if n >= 8)
    idxs = [i for i in range(len(ds)) if ds[i]["session_id"] == session][:8]
    wins = [ds[i] for i in idxs]
    batch8 = move_batch(collate_windows(wins), DEVICE)
    bsz8 = batch8["pose_gt"].shape[0]
    cid8 = torch.full((bsz8,), CONFIG_VT, device=DEVICE, dtype=torch.long)
    batch8["config_id"] = cid8
    v8, tr8, tp8 = condition_inputs(batch8, cid8)
    cond8 = {"V_feat": v8, "T_raw": tr8, "T_phys": tp8, "config_id": cid8,
             "session_id": batch8.get("session_id")}
    with torch.inference_mode():
        pred_cont = sample_continue(model, diffusion, ddpm, cond8, bsz8, tw, pose_std, half)
        err8, _ = fk_err(model, batch8, pred_cont)
        kp_cont = fk_pose6d(pred_cont, batch8["trans_gt"] + batch8["trans_anchor"][:, None, :],
                            batch8["offsets"], batch8["parents"])
    print("[c] DDPM 200 + continuation (8 consecutive windows, %s): MPJPE %.1f mm  jitter %.1f mm/frame"
          % (session, err8.mean().item() * 1000, jitter_mm(kp_cont)))
    jumps_cont = [torch.linalg.vector_norm(kp_cont[k, -1] - kp_cont[k + 1, 0], dim=-1).mean().item()
                  for k in range(bsz8 - 1)]
    jumps_gt = [torch.linalg.vector_norm(batch8["kp_gt"][k, -1] - batch8["kp_gt"][k + 1, 0],
                 dim=-1).mean().item() for k in range(bsz8 - 1)]
    print("    stitch jump: continuation %.1f mm vs GT %.1f mm"
          % (1000 * np.mean(jumps_cont), 1000 * np.mean(jumps_gt)))


if __name__ == "__main__":
    main()
