"""Decompose E6.x pos-model chaos: floor (one-shot) vs chain vs export inverse-FK.

Same structure as probe_e3_oneshot_floor but for a pos checkpoint:
1. tau0 identity reconstruction (pos space, like train.py dashboard)
2. one-shot g(F) @tau=999 pure noise, in BOTH pos space and world-FK space
   (world-FK uses the eval export path _positions_to_6d_batch -> FK)
3. DDIM chain 10/50 steps, world-FK space (what the BVH shows)
4. per-joint jitter in pos space vs world-FK space (isolates inverse-FK blowup)
"""
import sys
from pathlib import Path

import numpy as np
import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.diffusion import GaussianDiffusion
from anysole.eval import _positions_to_6d_batch
from anysole.geometry import fk_pose6d
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1_POS
from anysole.train import condition_inputs, load_config, move_batch
from anysole.types import CONFIG_VT

DEVICE = torch.device("cuda:6")
CKPT = GAIT / "results/AnySole/E6.2_smooth/checkpoints/ckpt_last.pt"


def kp_metrics(pred6d, trans_world, batch):
    kp = fk_pose6d(pred6d, trans_world, batch["offsets"], batch["parents"])
    err = torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1)
    jit = torch.linalg.vector_norm(kp[:, 1:] - kp[:, :-1], dim=-1)
    return err, jit


def main():
    config = load_config(GAIT / "configs/v1.yaml")
    config["contact_method"] = "joint_and"
    ckpt = torch.load(CKPT, map_location="cpu")
    saved = ckpt["config"]
    print("ckpt config: modal=%s tw=%s epoch=%s noise_scaled=%s lambda_pose_vel=%s lambda_bone=%s stride=%s"
          % (saved.get("modal"), saved.get("tw"), ckpt.get("epoch"),
             saved.get("noise_scaled"), saved.get("lambda_pose_vel"), saved.get("lambda_bone"), saved.get("stride")))
    model = AnySoleModel(
        d=int(saved["d_model"]), tw=int(saved["tw"]), modal=str(saved["modal"]),
        dropout=float(saved["dropout"]), pose_layers=int(saved["pose_layers"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.noise_scaled = bool(saved.get("noise_scaled", False))
    model.eval()
    tw = int(saved["tw"])
    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=tw, contact_method="joint_and",
    )
    idx = [i for i in range(len(ds)) if ds[i]["session_id"] == "S10103"][:8]
    batch = move_batch(collate_windows([ds[i] for i in idx]), DEVICE)
    bsz = batch["pose_gt"].shape[0]
    c = torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long)
    v, tr, tp = condition_inputs(batch, c)
    cond = {"V_feat": v, "T_raw": tr, "T_phys": tp, "config_id": c, "session_id": batch.get("session_id")}
    anchor = batch["trans_anchor"][:, None, :]
    gt_trans = batch["trans_gt"] + anchor
    pos_dim = model.pose_head.pose_dim
    n_joints = 22

    with torch.inference_mode():
        # 1. tau0 identity (pos space, dashboard口径)
        out0 = model(v, tr, tp, batch["pose_gt_pos"], torch.zeros(bsz, device=DEVICE, dtype=torch.long), c, batch.get("session_id"))
        e0 = torch.linalg.vector_norm(out0["x0_hat"].reshape(bsz, tw, n_joints, 3) - batch["pose_gt_pos"].reshape(bsz, tw, n_joints, 3), dim=-1)
        print("tau0 pos-space error: %.1f mm" % (e0.mean() * 1000))

        # 2. one-shot g(F) @tau=999, pos space + world-FK space
        tau = torch.full((bsz,), 999, device=DEVICE, dtype=torch.long)
        noise = torch.randn(bsz, tw, pos_dim, device=DEVICE)
        outn = model(v, tr, tp, noise, tau, c, batch.get("session_id"))
        ep = torch.linalg.vector_norm(outn["x0_hat"].reshape(bsz, tw, n_joints, 3) - batch["pose_gt_pos"].reshape(bsz, tw, n_joints, 3), dim=-1)
        jp = torch.linalg.vector_norm(outn["x0_hat"][:, 1:].reshape(bsz, tw - 1, n_joints, 3) - outn["x0_hat"][:, :-1].reshape(bsz, tw - 1, n_joints, 3), dim=-1)
        print("one-shot pos-space: MPJPE %.1f mm, jitter %.1f mm/frame (root-local)" % (ep.mean() * 1000, jp.mean() * 1000))
        pred6d_n = _positions_to_6d_batch(outn["x0_hat"], batch, DEVICE)
        err_n, jit_n = kp_metrics(pred6d_n, gt_trans, batch)
        print("one-shot world-FK:   MPJPE %.1f mm, jitter %.1f mm/frame (BVH口径)" % (err_n.mean() * 1000, jit_n.mean() * 1000))
        jn = jit_n.mean(dim=(0, 1)).cpu().numpy() * 1000
        print("  world-FK per-joint jitter:", np.round(jn, 1))

        # 3. chains
        for steps in (10, 50):
            p = GaussianDiffusion(n_train_steps=1000).ddim_sample_loop(
                model, tau_related_kwargs=cond, shape=(bsz, tw, pos_dim),
                steps=steps, eta=0.0, device=DEVICE)
            out = model(v, tr, tp, p, torch.zeros(bsz, device=DEVICE, dtype=torch.long), c, batch.get("session_id"))
            pred6d = _positions_to_6d_batch(p, batch, DEVICE)
            err, jit = kp_metrics(pred6d, out["trans_hat"] + anchor, batch)
            print("chain %2d steps world-FK: MPJPE %.1f mm, jitter %.1f mm/frame" % (steps, err.mean() * 1000, jit.mean() * 1000))
            if steps == 50:
                print("  per-joint jitter:", np.round(jit.mean(dim=(0, 1)).cpu().numpy() * 1000, 1))

        # 4. chain pos-space jitter (before inverse FK) to isolate export blowup
        p = GaussianDiffusion(n_train_steps=1000).ddim_sample_loop(
            model, tau_related_kwargs=cond, shape=(bsz, tw, pos_dim), steps=50, eta=0.0, device=DEVICE)
        jp_c = torch.linalg.vector_norm(p[:, 1:].reshape(bsz, tw - 1, n_joints, 3) - p[:, :-1].reshape(bsz, tw - 1, n_joints, 3), dim=-1)
        print("chain 50 pos-space jitter: %.1f mm/frame (before inverse FK)" % (jp_c.mean() * 1000))


if __name__ == "__main__":
    main()
