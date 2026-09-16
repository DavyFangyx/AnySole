"""Probe why AnySole V1 head output ignores inputs (run yg01hyju, epoch 2000).

Checks (on one val batch):
  1. tau=0 readout with clean GT pose input (reproduce ~145mm from wandb)
  2. mean-pose baseline MPJPE (is the head just outputting the mean pose?)
  3. input sensitivity: GT pose vs mean pose as x_tau at tau=0
  4. F sensitivity: VT vs V-only vs T-only conditioning at tau=0
  5. temporal structure of the output vs GT
  6. cross-attention usage per decoder layer (self + cross, avg over heads)
  7. denoising curve: x0_hat error vs tau
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
from anysole.types import CONFIG_T, CONFIG_V, CONFIG_VT

CKPT = GAIT / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt"
DEVICE = torch.device("cuda:6")


def mpjpe_mm(pred_kp, kp_gt):
    return float(torch.linalg.vector_norm(pred_kp - kp_gt, dim=-1).mean().item()) * 1000.0


def main():
    config = load_config(GAIT / "configs/v1.yaml")
    config["contact_method"] = "joint_and"
    ckpt = torch.load(CKPT, map_location="cpu")
    saved = ckpt["config"]
    print("ckpt epoch=%s modal=%s" % (ckpt.get("epoch"), saved.get("modal")))

    model = AnySoleModel(
        d=int(saved["d_model"]), tw=int(saved["tw"]), modal=str(saved["modal"]),
        dropout=float(saved["dropout"]), pose_layers=int(saved["pose_layers"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))

    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=int(config["tw"]),
        contact_method="joint_and",
    )
    raw = [ds[i] for i in range(8)]
    batch = collate_windows(raw)
    batch = move_batch(batch, DEVICE)
    bsz = batch["pose_gt"].shape[0]
    pose_gt = batch["pose_gt"]
    anchor = batch["trans_anchor"][:, None, :]
    gt_trans_world = batch["trans_gt"] + anchor

    # ---- baselines ----
    pm = model.pose_head.pose_mean  # (138,) fitted mean pose (raw 6D space)
    mean_pose = pm.view(1, 1, -1).expand(bsz, pose_gt.shape[1], -1)
    kp_mean = fk_pose6d(mean_pose, gt_trans_world, batch["offsets"], batch["parents"])
    print("\n[baseline] mean-pose MPJPE = %.1f mm" % mpjpe_mm(kp_mean, batch["kp_gt"]))

    def run(v_feat, t_raw, t_phys, x_tau, tau):
        with torch.inference_mode():
            out = model(v_feat, t_raw, t_phys, x_tau, tau,
                        torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long),
                        batch.get("session_id"))
        return out["x0_hat"]

    def kp_of(pose):
        return fk_pose6d(pose, gt_trans_world, batch["offsets"], batch["parents"])

    zero_t = torch.zeros_like(batch["T_raw"])
    zero_t_phys = torch.zeros_like(batch["T_phys"])
    zero_v = torch.zeros_like(batch["V_feat"])

    # ---- 1/3/4: tau=0 readout variants ----
    t0 = torch.zeros(bsz, device=DEVICE, dtype=torch.long)
    out_gt_vt = run(batch["V_feat"], batch["T_raw"], batch["T_phys"], pose_gt, t0)
    out_gt_v = run(batch["V_feat"], zero_t, zero_t_phys, pose_gt, t0)      # F from V only
    out_gt_t = run(zero_v, batch["T_raw"], batch["T_phys"], pose_gt, t0)  # F from T only
    out_mean_vt = run(batch["V_feat"], batch["T_raw"], batch["T_phys"], mean_pose, t0)

    print("\n[tau=0 readout] MPJPE  GT-input/VT-cond = %.1f mm" % mpjpe_mm(kp_of(out_gt_vt), batch["kp_gt"]))
    print("[tau=0 readout] MPJPE  GT-input/V-cond  = %.1f mm" % mpjpe_mm(kp_of(out_gt_v), batch["kp_gt"]))
    print("[tau=0 readout] MPJPE  GT-input/T-cond  = %.1f mm" % mpjpe_mm(kp_of(out_gt_t), batch["kp_gt"]))
    print("[tau=0 readout] MPJPE  mean-pose input  = %.1f mm" % mpjpe_mm(kp_of(out_mean_vt), batch["kp_gt"]))
    print("[input sensitivity] |out(GT)-out(meanPose)| mean abs = %.4f (raw 6D)"
          % (out_gt_vt - out_mean_vt).abs().mean().item())
    print("[F sensitivity] |out(VT)-out(V)| = %.4f   |out(VT)-out(T)| = %.4f"
          % ((out_gt_vt - out_gt_v).abs().mean().item(),
             (out_gt_vt - out_gt_t).abs().mean().item()))

    # ---- 5: temporal structure ----
    gt_std_t = pose_gt.std(dim=1).mean().item()
    out_std_t = out_gt_vt.std(dim=1).mean().item()
    print("\n[temporal std over 20 frames, raw 6D] GT=%.4f  head-out(tau0)=%.4f"
          % (gt_std_t, out_std_t))

    # sanity check the pose representation: per-joint rotation change over the
    # window (degrees), for a few joints
    from anysole.geometry import rot6d_to_rotmat
    r = rot6d_to_rotmat(pose_gt.reshape(bsz, 20, 23, 6))
    rel = r[:, :1].transpose(-1, -2) @ r[:, -1:]
    cos_ang = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    deg = torch.rad2deg(torch.acos(cos_ang))[0, 0]
    names = ["pelvis", "spine", "head", "Lknee", "Lankle", "Rknee", "Rankle"]
    idx = [0, 1, 4, 16, 17, 20, 21]
    print("  per-joint rotation change over window (deg): " +
          ", ".join("%s=%.1f" % (n, deg[i].item()) for n, i in zip(names, idx)))

    # ---- 7: denoising curve ----
    print("\n[denoising curve: MPJPE of x0_hat vs GT]")
    for tau_val in (0, 1, 2, 5, 10, 50, 200, 500, 999):
        t = torch.full((bsz,), tau_val, device=DEVICE, dtype=torch.long)
        x_tau = diffusion.q_sample(pose_gt, t)
        out = run(batch["V_feat"], batch["T_raw"], batch["T_phys"], x_tau, t)
        print("  tau=%4d -> %.1f mm" % (tau_val, mpjpe_mm(kp_of(out), batch["kp_gt"])))

    # ---- 6: attention usage ----
    print("\n[attention usage per decoder layer]")
    import torch.nn as nn

    class MHAWrap(nn.Module):
        """Force need_weights=True and print the avg attention pattern."""
        def __init__(self, mha, layer_idx):
            super().__init__()
            self.mha = mha
            self.layer_idx = layer_idx

        def forward(self, *args, **kwargs):
            kwargs["need_weights"] = True
            out = self.mha(*args, **kwargs)
            w = out[1]
            key = args[1]
            tag = "cross" if key.shape[1] == 40 else "self"
            print("  L%d %s: shape %s, mean=%.4f max=%.4f"
                  % (self.layer_idx, tag, tuple(w.shape), w.mean().item(), w.max().item()))
            if tag == "cross":
                print("    timestep-token attn: V-half sum=%.3f T-half sum=%.3f"
                      % (w[0, 0, :20].sum().item(), w[0, 0, 20:].sum().item()))
                for t in (0, 5, 10, 19):
                    q = 1 + t * 23
                    row = w[0, q]
                    print("    query frame t=%2d: v_tok[t]=%.3f t_tok[t]=%.3f other=%.3f"
                          % (t, row[t].item(), row[20 + t].item(),
                             (1 - row[t] - row[20 + t]).item()))
            else:
                # self-attention: how much does the timestep token attend to
                # pose tokens, and do pose tokens attend to it
                print("    timestep-token attn to pose tokens: sum=%.3f (1/token=%.4f)"
                      % (w[0, 0, 1:].sum().item(), w[0, 0, 1:].sum().item() / 460.0))
                print("    pose tokens attn to timestep token: mean=%.4f"
                      % w[0, 1:, 0].mean().item())
            return out

    for li, layer in enumerate(model.pose_head.decoder.layers):
        layer.multihead_attn = MHAWrap(layer.multihead_attn, li)

    print("\n--- attention at tau=0, clean GT input ---")
    with torch.inference_mode():
        model(batch["V_feat"], batch["T_raw"], batch["T_phys"], pose_gt, t0,
              torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long), batch.get("session_id"))
    print("\n--- attention at tau=999, noise input ---")
    t999 = torch.full((bsz,), 999, device=DEVICE, dtype=torch.long)
    x999 = diffusion.q_sample(pose_gt, t999)
    with torch.inference_mode():
        model(batch["V_feat"], batch["T_raw"], batch["T_phys"], x999, t999,
              torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long), batch.get("session_id"))
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
