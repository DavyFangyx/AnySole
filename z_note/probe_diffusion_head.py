"""Probe: how much does the trained pose head respond to x_tau and tau?

Loads a checkpoint, builds one eval batch, and measures output sensitivity.
Run: /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probe_diffusion_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1  # noqa: E402
from anysole.train import condition_inputs, load_config, move_batch  # noqa: E402
from anysole.types import CONFIG_VT  # noqa: E402


def main() -> None:
    device = torch.device("cuda:0")
    ckpt = REPO / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt"
    config = load_config(REPO / "configs/v1.yaml")
    checkpoint = torch.load(ckpt, map_location="cpu")
    model = AnySoleModel(
        d=int(checkpoint["config"].get("d_model", 256)),
        tw=int(checkpoint["config"].get("tw", 20)),
        dropout=float(checkpoint["config"].get("dropout", 0.1)),
        modal=MODEL_ANYSOLEV1,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    dataset = AnySoleDataset(
        mode="eval",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=20,
        contact_method="joint_and",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_windows)
    batch = move_batch(next(iter(loader)), device)
    cid = torch.full((8,), CONFIG_VT, device=device, dtype=torch.long)
    v_feat, t_raw, t_phys, _, _ = condition_inputs(batch, cid)
    pose_gt = batch["pose_gt"]

    # mean pose per frame (over this batch)
    mean_pose = pose_gt.mean(dim=0, keepdim=True).expand_as(pose_gt)
    noise = torch.randn_like(pose_gt)

    def run(x_tau, tau):
        with torch.inference_mode():
            out = model(v_feat, t_raw, t_phys, x_tau, tau, cid, batch.get("session_id"))
        return out

    def diff(a, b):
        return float((a - b).square().mean().sqrt().item())

    print("=== 1) tau sensitivity (same x_tau=GT, different tau) ===")
    o0 = run(pose_gt, torch.zeros(8, device=device, dtype=torch.long))
    o500 = run(pose_gt, torch.full((8,), 500, device=device, dtype=torch.long))
    o999 = run(pose_gt, torch.full((8,), 999, device=device, dtype=torch.long))
    print("x0_hat RMS diff tau 0 vs 500 :", diff(o0["x0_hat"], o500["x0_hat"]))
    print("x0_hat RMS diff tau 0 vs 999 :", diff(o0["x0_hat"], o999["x0_hat"]))
    print("x0_hat std over batch        :", float(o0["x0_hat"].std()))

    print("=== 2) x_tau sensitivity (same tau=0, different pose input) ===")
    om = run(mean_pose, torch.zeros(8, device=device, dtype=torch.long))
    on = run(noise, torch.zeros(8, device=device, dtype=torch.long))
    print("x0_hat RMS diff GT vs mean x :", diff(o0["x0_hat"], om["x0_hat"]))
    print("x0_hat RMS diff GT vs noise x:", diff(o0["x0_hat"], on["x0_hat"]))
    print("x0_hat RMS error vs GT pose  :", diff(o0["x0_hat"], pose_gt))

    print("=== 3) x_tau sensitivity at tau=999 (same condition) ===")
    og999 = run(pose_gt, torch.full((8,), 999, device=device, dtype=torch.long))
    on999 = run(noise, torch.full((8,), 999, device=device, dtype=torch.long))
    print("x0_hat RMS diff GT vs noise at tau=999:", diff(og999["x0_hat"], on999["x0_hat"]))

    print("=== 4) timestep embedding magnitude vs h (post self-attn) ===")
    captured = {}

    def hook(module, inp, outp):
        captured["h"] = outp.detach()

    handle = model.pose_head.self_attn.register_forward_hook(hook)
    run(pose_gt, torch.zeros(8, device=device, dtype=torch.long))
    handle.remove()
    h = captured["h"]
    t_emb = model.pose_head.embeddings.timestep(torch.arange(0, 1000, device=device))
    print("h std                 :", float(h.std()))
    print("h mean abs            :", float(h.abs().mean()))
    print("t_emb std (avg tau)   :", float(t_emb.std()))
    print("t_emb abs mean (avg t):", float(t_emb.abs().mean()))
    print("ratio t_emb/h (std)   :", float(t_emb.std() / h.std()))

    print("=== 5) fresh model sanity: does x_tau matter at init? ===")
    fresh = AnySoleModel(d=256, tw=20, dropout=0.0, modal=MODEL_ANYSOLEV1).to(device)
    fresh.eval()
    with torch.inference_mode():
        f0 = fresh(v_feat, t_raw, t_phys, pose_gt, torch.zeros(8, device=device, dtype=torch.long), cid, None)
        fm = fresh(v_feat, t_raw, t_phys, mean_pose, torch.zeros(8, device=device, dtype=torch.long), cid, None)
        fn = fresh(v_feat, t_raw, t_phys, noise, torch.zeros(8, device=device, dtype=torch.long), cid, None)
    print("fresh x0_hat RMS diff GT vs mean:", diff(f0["x0_hat"], fm["x0_hat"]))
    print("fresh x0_hat RMS diff GT vs noise:", diff(f0["x0_hat"], fn["x0_hat"]))

    print("=== 6) trained head: direct check of per-input partial influence ===")
    # Replace F with the same value while varying x_tau -> pure x_tau path
    with torch.inference_mode():
        fv = model(v_feat, t_raw, t_phys, pose_gt, torch.zeros(8, device=device, dtype=torch.long), cid, None)
    h_fixed = fv["F"].clone()
    # bypass fusion: call pose_head directly with fixed F
    import torch.nn as nn  # noqa: F401

    with torch.inference_mode():
        p_gt = model.pose_head(pose_gt, torch.zeros(8, device=device, dtype=torch.long), h_fixed)
        p_mean = model.pose_head(mean_pose, torch.zeros(8, device=device, dtype=torch.long), h_fixed)
        p_noise = model.pose_head(noise, torch.zeros(8, device=device, dtype=torch.long), h_fixed)
    print("pose_head(x=GT,F)  vs pose_head(x=mean,F):", diff(p_gt, p_mean))
    print("pose_head(x=GT,F)  vs pose_head(x=noise,F):", diff(p_gt, p_noise))
    print("pose_head(x=GT,F)  vs GT pose            :", diff(p_gt, pose_gt))
    print("pose_head(x=mean,F) vs GT pose           :", diff(p_mean, pose_gt))
    # And the pure-tau path with fixed x and F
    with torch.inference_mode():
        q0 = model.pose_head(pose_gt, torch.zeros(8, device=device, dtype=torch.long), h_fixed)
        q999 = model.pose_head(pose_gt, torch.full((8,), 999, device=device, dtype=torch.long), h_fixed)
    print("pose_head(tau=0) vs pose_head(tau=999), same x,F:", diff(q0, q999))


if __name__ == "__main__":
    main()
