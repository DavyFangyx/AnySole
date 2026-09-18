"""E6.x decoupling smoke: each gate independently on/off on the E3 basis.

1. E6.1 alone (pos modal, tw=20, E3 regime): one training step + loss values
2. E6.2 on top of E6.1: lambda_pose_vel/lambda_bone active
3. E6.3: tw=100 stride=1 dataset + one step (pos)
4. E6.5 warm_start on the E3 6D checkpoint (repr-agnostic check)
5. E6.4 continuation eval wiring (pos, tiny 2-epoch model)
"""
import sys
from pathlib import Path

import numpy as np
import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.data.dataset import AnySoleDataset, collate_windows, sample_config_ids
from anysole.diffusion import GaussianDiffusion
from anysole.geometry import fk_pose6d
from anysole.losses import compute_losses
from anysole.models import AnySoleModel
from anysole.train import condition_inputs, fit_pose_stats, load_config, move_batch
from anysole.types import CONFIG_VT, POSE_DIM

DEVICE = torch.device("cuda:4")
E3_CKPT = GAIT / "results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt"


def build_batch(ds, idxs):
    batch = move_batch(collate_windows([ds[i] for i in idxs]), DEVICE)
    cid = sample_config_ids(batch["pose_gt"].shape[0], probs=[0.5, 0.25, 0.25]).to(DEVICE)
    batch["config_id"] = cid
    return batch


def one_step(model, batch, diffusion, config, repr_key="pose_gt"):
    bsz = batch["pose_gt"].shape[0]
    cid = batch["config_id"]
    v, tr, tp, _, _ = condition_inputs(batch, cid)
    tau = torch.randint(0, 1000, (bsz,), device=DEVICE)
    target = batch[repr_key]
    noise = torch.randn_like(target)
    if model.noise_scaled:
        noise = noise * model.pose_head.pose_std.view(1, 1, -1)
    x_tau = diffusion.q_sample(target, tau, noise=noise)
    out = model(v, tr, tp, x_tau, tau, cid, batch.get("session_id"))
    losses = compute_losses(out, batch, cid, config)
    return losses


def main():
    config = load_config(GAIT / "configs/v1.yaml")  # E3 base, gates off
    config["contact_method"] = "joint_and"

    # ---------- 1. E6.1: pos modal, E3 regime, one step ----------
    print("== 1. E6.1 pos (tw=20, noise_scaled=false, E3 regime) ==")
    ds = AnySoleDataset(mode="train", seq_root=Path(config["seq_root"]),
                        split_csv=Path(config["split_csv"]), cache_root=Path(config["cache_root"]),
                        window_length=20, contact_method="joint_and")
    model = AnySoleModel(d=256, tw=20, modal="anysolev1_pos", pose_layers=6).to(DEVICE)
    model.noise_scaled = False
    # stats on a subset
    from torch.utils.data import DataLoader, Subset
    sub = Subset(ds, list(range(0, len(ds), 10))[:64])
    loader = DataLoader(sub, batch_size=16, shuffle=True, collate_fn=collate_windows)
    pm, ps = fit_pose_stats(loader, DEVICE, key="pose_gt_pos")
    model.pose_head.set_stats(pm, ps)
    diffusion = GaussianDiffusion(n_train_steps=1000)
    batch = build_batch(ds, list(range(8)))
    cfg1 = dict(config)
    cfg1["pose_repr"] = "pos"
    losses = one_step(model, batch, diffusion, cfg1, "pose_gt_pos")
    print("  x0_hat", tuple(model(batch["V_feat"], batch["T_raw"], batch["T_phys"], batch["pose_gt_pos"],
          torch.zeros(8, device=DEVICE, dtype=torch.long), batch["config_id"],
          batch.get("session_id"))["x0_hat"].shape),
          "| losses:", {k: round(float(v), 4) for k, v in losses.items() if k in ("L_pose", "L_kp", "L_pose_vel", "L_bone", "loss")})

    # ---------- 2. E6.2: weights on ----------
    print("\n== 2. E6.2 lambda_pose_vel=10, lambda_bone=1 (pos) ==")
    config2 = dict(config)
    config2["pose_repr"] = "pos"
    config2["lambda_pose_vel"] = 10.0
    config2["lambda_bone"] = 1.0
    losses2 = one_step(model, batch, diffusion, config2, "pose_gt_pos")
    print("  L_pose=%.4f L_pose_vel=%.4f L_bone=%.4f loss=%.4f"
          % (losses2["L_pose"], losses2["L_pose_vel"], losses2["L_bone"], losses2["loss"]))
    losses2["loss"].backward()
    print("  backward OK")

    # ---------- 3. E6.3: tw=100 stride=1 ----------
    print("\n== 3. E6.3 tw=100 stride=1 ==")
    ds100 = AnySoleDataset(mode="train", seq_root=Path(config["seq_root"]),
                           split_csv=Path(config["split_csv"]), cache_root=Path(config["cache_root"]),
                           window_length=100, contact_method="joint_and", stride=1)
    print("  train windows:", len(ds100))
    model100 = AnySoleModel(d=256, tw=100, modal="anysolev1_pos", pose_layers=6).to(DEVICE)
    model100.noise_scaled = False
    sub100 = Subset(ds100, list(range(0, len(ds100), 100))[:64])
    loader100 = DataLoader(sub100, batch_size=8, shuffle=True, collate_fn=collate_windows)
    pm100, ps100 = fit_pose_stats(loader100, DEVICE, key="pose_gt_pos")
    model100.pose_head.set_stats(pm100, ps100)
    batch100 = build_batch(ds100, [0, 1, 2, 3])
    losses100 = one_step(model100, batch100, diffusion, cfg1, "pose_gt_pos")
    print("  step OK, loss=%.4f" % float(losses100["loss"]))
    del model100, batch100, ds100
    torch.cuda.empty_cache()

    # ---------- 4. E6.5 warm_start on E3 6D ckpt ----------
    print("\n== 4. E6.5 warm_start (E3 6D ckpt) ==")
    ckpt = torch.load(E3_CKPT, map_location="cpu")
    saved = ckpt["config"]
    m6 = AnySoleModel(d=int(saved["d_model"]), tw=int(saved["tw"]), modal="anysolev1",
                      pose_layers=int(saved["pose_layers"])).to(DEVICE)
    m6.load_state_dict(ckpt["model"], strict=True)
    m6.noise_scaled = bool(saved.get("noise_scaled", False))
    m6.eval()
    diff6 = GaussianDiffusion(n_train_steps=int(saved["diffusion_train_steps"]))
    m6.diffusion_abar_top = float(diff6.alphas_cumprod[diff6.n_train_steps - 1])
    ds6 = AnySoleDataset(mode="eval", seq_root=Path(config["seq_root"]),
                         split_csv=Path(config["split_csv"]), cache_root=Path(config["cache_root"]),
                         window_length=20, contact_method="joint_and")
    b6 = build_batch(ds6, list(range(16)))
    bsz = b6["pose_gt"].shape[0]
    cid = b6["config_id"]
    v, tr, tp, _, _ = condition_inputs(b6, cid)
    cond = {"V_feat": v, "T_raw": tr, "T_phys": tp, "config_id": cid, "session_id": b6.get("session_id")}
    with torch.inference_mode():
        p0 = diff6.ddim_sample_loop(m6, tau_related_kwargs=cond, shape=(bsz, 20, POSE_DIM),
                                    steps=10, eta=0.0, device=DEVICE)
        prior = m6.pose_head.pose_mean.view(1, 1, -1).expand(bsz, 20, -1)
        pw = diff6.ddim_sample_loop(m6, tau_related_kwargs=cond, shape=(bsz, 20, POSE_DIM),
                                    steps=10, eta=0.0, device=DEVICE, prior=prior)
        t0 = torch.zeros(bsz, device=DEVICE, dtype=torch.long)
        o0 = m6(v, tr, tp, p0, t0, cid, b6.get("session_id"))
        ow = m6(v, tr, tp, pw, t0, cid, b6.get("session_id"))
        anchor = b6["trans_anchor"][:, None, :]
        gt = b6["trans_gt"] + anchor
        k0 = fk_pose6d(p0, o0["trans_hat"] + anchor, b6["offsets"], b6["parents"])
        kw = fk_pose6d(pw, ow["trans_hat"] + anchor, b6["offsets"], b6["parents"])
        e0 = torch.linalg.vector_norm(k0 - b6["kp_gt"], dim=-1).mean().item() * 1000
        ew = torch.linalg.vector_norm(kw - b6["kp_gt"], dim=-1).mean().item() * 1000
    print("  10-step chain MPJPE: cold %.1f mm -> warm %.1f mm" % (e0, ew))

    print("\nall decoupling smokes done")


if __name__ == "__main__":
    main()
