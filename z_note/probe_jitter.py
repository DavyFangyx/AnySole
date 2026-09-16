"""Quantify motion jitter of E3 model vs GT.

(a) window-stitch jump: the BVH export concatenates NON-overlapping 20-frame
    windows (dataset stride = window_length), so frame 19 of window k and
    frame 0 of window k+1 are two independent predictions of consecutive GT
    frames. Measure the per-joint keypoint jump at the stitch vs the GT
    frame-to-frame delta there.
(b) within-window frame-to-frame keypoint deltas, predicted vs GT.
(c) DDIM chain version (what the BVH actually contains).
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
from anysole.types import CONFIG_VT, POSE_DIM

DEVICE = torch.device("cuda:6")
CKPT = GAIT / "results/AnySole/E4_normdiff/checkpoints/ckpt_last.pt"


def mpjpe_mm(a, b):
    return float(torch.linalg.vector_norm(a - b, dim=-1).mean().item()) * 1000.0


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
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))

    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=int(config["tw"]),
        contact_method="joint_and",
    )
    # windows of one session are contiguous in the dataset (stride = tw):
    # grab 8 consecutive windows (160 GT frames) of session S10103.
    idx = [i for i in range(len(ds)) if ds[i]["session_id"] == "S10103"][:8]
    windows = [ds[i] for i in idx]
    batch = move_batch(collate_windows(windows), DEVICE)
    bsz = batch["pose_gt"].shape[0]
    c = torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long)
    v, tr, tp = condition_inputs(batch, c)
    anchor = batch["trans_anchor"][:, None, :]
    gt_trans = batch["trans_gt"] + anchor
    kp_gt = batch["kp_gt"]
    with torch.inference_mode():
        pred_pose = diffusion.ddim_sample_loop(
            model, tau_related_kwargs={"V_feat": v, "T_raw": tr, "T_phys": tp,
                                       "config_id": c, "session_id": batch.get("session_id")},
            shape=(bsz, int(config["tw"]), POSE_DIM), steps=50, eta=0.0, device=DEVICE)
        t0 = torch.zeros(bsz, device=DEVICE, dtype=torch.long)
        out = model(v, tr, tp, pred_pose, t0, c, batch.get("session_id"))
        pred_trans = out["trans_hat"] + anchor
        kp_pred = fk_pose6d(pred_pose, pred_trans, batch["offsets"], batch["parents"])
        kp_pose_only = fk_pose6d(pred_pose, gt_trans, batch["offsets"], batch["parents"])

    kp_gt = kp_gt.cpu().numpy()      # (B,20,23,3) meters
    kp_pred = kp_pred.cpu().numpy()
    kp_po = kp_pose_only.cpu().numpy()
    n = len(windows)

    # (a) stitch jump: |pred[k,19] - pred[k+1,0]| vs |gt[k,19] - gt[k+1,0]|
    jumps_pred, jumps_gt = [], []
    for k in range(n - 1):
        jumps_pred.append(np.linalg.norm(kp_pred[k, -1] - kp_pred[k + 1, 0], axis=-1).mean())
        jumps_gt.append(np.linalg.norm(kp_gt[k, -1] - kp_gt[k + 1, 0], axis=-1).mean())
    print("[stitch] mean per-joint jump at window boundary:")
    print("  predicted: %.1f mm      GT (frame-to-frame): %.1f mm"
          % (1000 * np.mean(jumps_pred), 1000 * np.mean(jumps_gt)))

    # (b) within-window frame-to-frame delta
    d_pred = np.linalg.norm(kp_pred[:, 1:] - kp_pred[:, :-1], axis=-1).mean()
    d_po = np.linalg.norm(kp_po[:, 1:] - kp_po[:, :-1], axis=-1).mean()
    d_gt = np.linalg.norm(kp_gt[:, 1:] - kp_gt[:, :-1], axis=-1).mean()
    print("[within-window] mean per-joint frame-to-frame displacement:")
    print("  predicted (pose+traj): %.1f mm   predicted (pose only, GT traj): %.1f mm   GT: %.1f mm"
          % (1000 * d_pred, 1000 * d_po, 1000 * d_gt))

    # (c) baseline static error for reference
    print("[static] MPJPE of stitched prediction vs GT: %.1f mm" % mpjpe_mm(
        torch.from_numpy(kp_pred), torch.from_numpy(kp_gt)))


if __name__ == "__main__":
    main()
