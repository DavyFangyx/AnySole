"""Full-val continuation evaluation + per-joint profile + sampler interaction.

Follows probe_sampler_continuation.py: the E5 head under Step2Motion's
continuation inpainting (seed first half of each window with the previous
window's noisy latents at the same t). Runs every val session, windows
sequential per session, and reports:
  - fresh-half MPJPE (second half of each window = the genuinely generated part)
  - seam jump (last frame of window k vs first frame of window k+1)
  - per-joint profile vs DDIM-10
  - DDIM-10 + continuation variant (sampler x continuation interaction)
Read-only. GPU 4.
"""
import sys
from collections import defaultdict
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
N_SESSIONS = 34  # all val sessions


def build_ddpm(diffusion):
    betas = np.concatenate([[0.0], diffusion.betas])
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    abar_prev = np.concatenate([[1.0], abar[:-1]])
    one_minus = np.where(1.0 - abar > 0, 1.0 - abar, 1.0)
    coeff1 = betas * np.sqrt(abar_prev) / one_minus
    coeff2 = np.sqrt(alphas) * (1.0 - abar_prev) / one_minus
    pstd = np.sqrt(np.where(1.0 - abar > 0, betas * (1.0 - abar_prev) / one_minus, 0.0))
    return {k: torch.from_numpy(v).float() for k, v in
            dict(coeff1=coeff1, coeff2=coeff2, pstd=pstd).items()}


def posterior_step(x_t, t, x0_hat, ddpm, pose_std):
    c1 = ddpm["coeff1"][t].to(x_t.device).view(1, 1, 1)
    c2 = ddpm["coeff2"][t].to(x_t.device).view(1, 1, 1)
    s = ddpm["pstd"][t].to(x_t.device).view(1, 1, 1)
    return c1 * x0_hat + c2 * x_t + s * torch.randn_like(x_t) * pose_std.view(1, 1, -1)


def run_chain(model, diffusion, ddpm, cond, x, pose_std, ddpm_mode=True, ddim_steps=10):
    """x: (B, tw, 138). Continuation seeding happens inside (x[1:, :half] = x[:-1, half:])."""
    bsz, tw, _ = x.shape
    half = tw // 2
    if ddpm_mode:
        for t in range(200, 0, -1):
            x[1:, :half] = x[:-1, half:].clone()
            tau = torch.full((bsz,), min(t, 199), device=DEVICE, dtype=torch.long)
            out = diffusion._call_model(model, x, tau, cond)
            x = posterior_step(x, t, out["x0_hat"], ddpm, pose_std)
        return x
    timesteps = diffusion._timestep_schedule(diffusion.n_train_steps, ddim_steps)
    for i, t in enumerate(timesteps):
        if i > 0:
            x[1:, :half] = x[:-1, half:].clone()
        tau = torch.full((bsz,), int(t), device=DEVICE, dtype=torch.long)
        out = diffusion._call_model(model, x, tau, cond)
        x0_hat = out["x0_hat"]
        if int(t) == 0:
            return x0_hat
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
        tau_prev = torch.full((bsz,), int(t_prev), device=DEVICE, dtype=torch.long)
        x = diffusion.ddim_step(x, tau, tau_prev, x0_hat, eta=0.0)
    return x


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    ddpm = build_ddpm(diffusion)
    tw = int(config["tw"])
    half = tw // 2
    pose_std = model.pose_head.pose_std

    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=tw, contact_method="joint_and",
    )
    # windows per session, in dataset order (consecutive)
    per_session = defaultdict(list)
    for i in range(len(ds)):
        per_session[ds[i]["session_id"]].append(i)
    sessions = sorted(per_session)[:N_SESSIONS]
    print("sessions: %d, windows: %d" % (len(sessions), sum(len(per_session[s]) for s in sessions)))

    JOINT_NAMES = ("Hips","Spine","Spine1","Spine2","Spine3","Neck","Head","LShoulder","LArm","LForeArm","LHand",
                   "RShoulder","RArm","RForeArm","RHand","LUpLeg","LLeg","LFoot","LToe","RUpLeg","RLeg","RFoot","RToe")

    def eval_session(session, ddpm_mode, ddim_steps=10):
        idxs = per_session[session]
        batch = move_batch(collate_windows([ds[i] for i in idxs]), DEVICE)
        bsz = batch["pose_gt"].shape[0]
        cid = torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long)
        batch["config_id"] = cid
        v, tr, tp = condition_inputs(batch, cid)
        cond = {"V_feat": v, "T_raw": tr, "T_phys": tp, "config_id": cid,
                "session_id": batch.get("session_id")}
        x = torch.randn(bsz, tw, POSE_DIM, device=DEVICE) * pose_std.view(1, 1, -1)
        with torch.inference_mode():
            pred = run_chain(model, diffusion, ddpm, cond, x, pose_std, ddpm_mode, ddim_steps)
            t0 = torch.zeros(bsz, device=DEVICE, dtype=torch.long)
            out = model(v, tr, tp, pred, t0, cid, batch.get("session_id"))
            anchor = batch["trans_anchor"][:, None, :]
            trans = out["trans_hat"] + anchor
            gt_trans = batch["trans_gt"] + anchor
            kp = fk_pose6d(pred, trans, batch["offsets"], batch["parents"])
            kp_poseonly = fk_pose6d(pred, gt_trans, batch["offsets"], batch["parents"])
            err = torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1)      # (B,20,23)
            err_po = torch.linalg.vector_norm(kp_poseonly - batch["kp_gt"], dim=-1)
            # fresh half only (frames 10..19 of each window = the generated part)
            fresh_err = err[:, half:].mean().item() * 1000
            fresh_po = err_po[:, half:].mean().item() * 1000
            full_err = err.mean().item() * 1000
            per_joint = err.mean(dim=(0, 1)).cpu() * 1000
            jitter = torch.linalg.vector_norm(kp[:, 1:] - kp[:, :-1], dim=-1).mean().item() * 1000
            jumps = []
            for k in range(bsz - 1):
                jumps.append(torch.linalg.vector_norm(kp[k, -1] - kp[k + 1, 0], dim=-1).mean().item() * 1000)
            return full_err, fresh_err, fresh_po, per_joint, jitter, float(np.mean(jumps))

    results = {}
    for ddpm_mode, name in ((True, "DDPM200+cont"), (False, "DDIM10+cont")):
        sums = {"full": 0.0, "fresh": 0.0, "fresh_po": 0.0, "jitter": 0.0, "jump": 0.0, "n": 0}
        per_joint_sum = torch.zeros(N_JOINTS)
        for s in sessions:
            full_err, fresh_err, fresh_po, pj, jitter, jump = eval_session(s, ddpm_mode)
            w = len(per_session[s])
            sums["full"] += full_err * w
            sums["fresh"] += fresh_err * w
            sums["fresh_po"] += fresh_po * w
            sums["jitter"] += jitter * w
            sums["jump"] += jump * w
            per_joint_sum += pj * w
            sums["n"] += w
        n = sums["n"]
        print("\n== %s (all %d sessions) ==" % (name, len(sessions)))
        print("  MPJPE full-window %.1f mm | fresh-half %.1f mm | fresh-half pose-only %.1f mm"
              % (sums["full"] / n, sums["fresh"] / n, sums["fresh_po"] / n))
        print("  jitter %.1f mm/frame | seam jump %.1f mm" % (sums["jitter"] / n, sums["jump"] / n))
        results[name] = per_joint_sum / n
    print("\nper-joint fresh... (full-window MPJPE mm):")
    print("  joint        DDPM200+cont   DDIM10+cont")
    for j in range(N_JOINTS):
        print("  %-10s   %7.1f     %7.1f" % (JOINT_NAMES[j], results["DDPM200+cont"][j], results["DDIM10+cont"][j]))


if __name__ == "__main__":
    main()
