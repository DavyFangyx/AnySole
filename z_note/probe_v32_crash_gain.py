"""V3_2 crash mechanism probe #1: readout gain + data extremes (2026-09-19).

Two diagnostics for the V3_2 late-training explosion (xd6kcdk2 ep271,
1s0yeuwn ep531, both traj-loss-led, finite gradients, nonfinite guard silent):

1) Gain probe: how strongly do the pose / traj / aux readouts amplify a small
   perturbation of the fused memory F?  d(head)/dF measured as
   ||out(F+eps*noise) - out(F)|| / eps (averaged directions), for the
   V3_2 (9-part) and F4a (3-group) checkpoints.  A high-gain readout makes
   the F -> head -> L -> F gradient loop more prone to transient divergence.

2) Data scan: extreme windows of the TRAIN split by target magnitude
   (max |vel_gt| per window, pose_gt norm deviation, kp_gt extent) — hard
   batches are the candidate trigger for the single-step kick.

Usage (touch_gait env):
  python z_note/probe_v32_crash_gain.py --device cuda:1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.eval import _load_model, load_config, resolve_device

F4A_CKPT = REPO / "results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt"
V32_CKPT = REPO / "results/AnySole/V3_2_part9/checkpoints/ckpt_last.pt"
SPLIT_CSV = Path("/data/fangyuxuan/projects/gait/AnysoleWorkspace/splits/default/splits.csv")


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    base_cfg = load_config(Path("/data/fangyuxuan/projects/gait/configs/v1.yaml"))
    model = _load_model(ck, base_cfg, device).eval()
    return model, cfg


def make_ds(mode, cfg):
    return AnySoleDataset(
        mode=mode, seq_root=cfg["seq_root"], split_csv=SPLIT_CSV,
        cache_root=cfg["cache_root"], window_length=int(cfg.get("tw", 20)),
        contact_method=cfg.get("contact_method", "tactile_abs"),
        no_imu=bool(cfg.get("no_imu", False)),
        v_input=cfg.get("v_input", "hrnet"),
        f2_repr=False,
    )


def gain_probe(model, dataset, device, batch_size=32, n_batches=8, eps=1e-3, n_dir=4):
    """||out(F+eps*n) - out(F)||/eps per head, averaged over directions/batches."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_windows)
    # |F| reference
    f_norms = []
    gains = {"pose": [], "traj_v": [], "traj_x": [], "trec": [], "vrec": []}
    n_run = 0
    with torch.inference_mode():
        for raw_batch in loader:
            if n_run >= n_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
            B = batch["pose_gt"].shape[0]
            config_id = torch.zeros(B, dtype=torch.long, device=device)
            out0 = model(batch["V_feat"], batch["T_raw"], batch["T_phys"], config_id)
            F = out0["F"]
            f_norms.append(float(F.norm(dim=-1).mean()))
            for _ in range(n_dir):
                noise = torch.randn_like(F)
                noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-12) * eps
                model_copy = None
                # perturb F by recomputing forward with F detached+perturbed:
                # simplest robust route: hook the fusion output.
                with torch.no_grad():
                    handle_out = {}

                    def hk(module, inp, outp):
                        handle_out["F"] = outp

                    h = model.fusion.register_forward_hook(hk)
                    model(batch["V_feat"], batch["T_raw"], batch["T_phys"], config_id)
                    h.remove()
                    Fp = handle_out["F"] + noise

                    def hk2(module, inp, outp):
                        return Fp

                    h = model.fusion.register_forward_hook(hk2)
                    out1 = model(batch["V_feat"], batch["T_raw"], batch["T_phys"], config_id)
                    h.remove()
                gains["pose"].append(float((out1["x0_hat"] - out0["x0_hat"]).norm() / eps / np.sqrt(out0["x0_hat"].numel())))
                gains["traj_v"].append(float((out1["v_hat"] - out0["v_hat"]).norm() / eps / np.sqrt(out0["v_hat"].numel())))
                gains["traj_x"].append(float((out1["trans_hat"] - out0["trans_hat"]).norm() / eps / np.sqrt(out0["trans_hat"].numel())))
                gains["trec"].append(float((out1["pressure_hat"] - out0["pressure_hat"]).norm() / eps / np.sqrt(out0["pressure_hat"].numel())))
                gains["vrec"].append(float((out1["vfeat_hat"] - out0["vfeat_hat"]).norm() / eps / np.sqrt(out0["vfeat_hat"].numel())))
            n_run += 1
    return f_norms, {k: float(np.mean(v)) for k, v in gains.items()}


def data_scan(cfg, device):
    ds = make_ds("train", cfg)
    print("train windows:", len(ds))
    vel_max, pose_norm, kp_extent = [], [], []
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_windows)
    for raw_batch in loader:
        v = raw_batch["vel_gt"].float().norm(dim=-1)  # (B, 20)
        vel_max.append(v.max(dim=1).values)
        p = raw_batch["pose_gt"].float()
        pose_norm.append(p.norm(dim=-1).mean(dim=1))
        k = raw_batch["kp_gt"].float()
        kp_extent.append((k.amax(dim=(1, 2, 3)) - k.amin(dim=(1, 2, 3))))
    vel_max = torch.cat(vel_max).numpy()
    pose_norm = torch.cat(pose_norm).numpy()
    kp_extent = torch.cat(kp_extent).numpy()
    for name, arr in [("max|vel_gt| m/s", vel_max), ("mean|pose_6d|", pose_norm), ("kp extent m", kp_extent)]:
        qs = np.percentile(arr, [50, 90, 99, 99.9, 100])
        print(f"  {name:16s} p50={qs[0]:.3f} p90={qs[1]:.3f} p99={qs[2]:.3f} p99.9={qs[3]:.3f} max={qs[4]:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--n-batches", type=int, default=8)
    args = ap.parse_args(argv)
    device = resolve_device(args.device)

    print("== data scan (train split) ==")
    f4a_model, cfg = load(F4A_CKPT, device)
    data_scan(cfg, device)

    for name, ckpt in [("F4a (3-group)", F4A_CKPT), ("V3_2 (9-part)", V32_CKPT)]:
        model, cfg = load(ckpt, device)
        print(f"\n== gain probe: {name} ==")
        f_norms, gains = gain_probe(model, make_ds("eval", cfg), device,
                                    batch_size=32, n_batches=args.n_batches, eps=args.eps)
        print("  mean |F| per dim: %.3f" % (np.mean(f_norms) / np.sqrt(model.d)))
        print("  rms gain d(head)/d(F) (per-dim):")
        for k, v in gains.items():
            print(f"    {k:8s} {v:8.3f}")


if __name__ == "__main__":
    main()
