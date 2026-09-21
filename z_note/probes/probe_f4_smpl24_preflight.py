"""Real-data preflight for the F4a foot-conv SMPL-24 training command.

This is not a replacement for a full run.  It exercises the exact dataset,
native SMPL batch contract, F4a encoder, legacy-F0b warm-start guard, losses,
FK metrics and a short fixed-batch overfit.  The initial random-model numbers
are labelled as diagnostics, not as a trained baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.geometry import fk_pose6d
from anysole.losses import compute_losses
from anysole.models import AnySoleModelV2
from anysole.train import (
    condition_inputs,
    fit_pose_stats,
    init_from_checkpoint,
    load_config,
    move_batch,
    resolve_device,
)
from anysole.types import (
    CONFIG_VT,
    JOINT_PROTOCOL_CHECKSUM,
    JOINT_PARENTS,
    MOTION_PROTOCOL,
    N_JOINTS,
    POSE_DIM,
    assert_batch_shapes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _predict(model, batch):
    cid = batch["config_id"]
    v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
    return model(
        v_feat,
        t_raw,
        t_phys,
        cid,
        batch.get("session_id"),
        T_s2m=t_s2m,
    )


def _mpjpe_mm(pose, trans, batch) -> float:
    kp = fk_pose6d(pose, trans, batch["offsets"], batch["parents"])
    return float(torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1).mean()) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "anysole/configs/v1.yaml")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=REPO_ROOT / "results/AnySole/F0b_seed2/checkpoints/ckpt_last.pt",
    )
    parser.add_argument("--session", default="S10101")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--warmup-steps", type=int, default=0,
        help="Linear warmup for the fixed-batch diagnostic (step 1 starts at lr/N).",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(20260920)
    device = resolve_device(args.device)
    config = load_config(args.config)
    config.update({
        "modal": "anysolev2",
        "contact_method": "joint_and",
        "t_encoder": "foot_conv",
        "motion_protocol": MOTION_PROTOCOL,
        "motion_n_joints": N_JOINTS,
        "motion_pose_dim": POSE_DIM,
        "joint_protocol_checksum": JOINT_PROTOCOL_CHECKSUM,
    })
    if args.lr is not None:
        config["lr"] = float(args.lr)
    dataset = AnySoleDataset(
        mode="train",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=int(config["tw"]),
        session_ids=[args.session],
        contact_method="joint_and",
        stride=int(config.get("stride", config["tw"])),
        smpl_roots=config.get("smpl_roots"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
    )
    raw_batch = next(iter(loader))
    batch = move_batch(raw_batch, device)
    bsz = int(batch["pose_gt"].shape[0])
    batch["config_id"] = torch.full((bsz,), CONFIG_VT, dtype=torch.long, device=device)
    assert_batch_shapes(batch, bsz, tw=int(config["tw"]))
    if tuple(batch["parents"][0].cpu().tolist()) != tuple(JOINT_PARENTS):
        raise AssertionError("batch does not carry the standard SMPL-24 parent tree")
    if batch["pose_gt"].shape[-1] != POSE_DIM or batch["kp_gt"].shape[-2] != N_JOINTS:
        raise AssertionError("batch is not native SMPL-24")

    fresh = AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=0.0,
        pose_layers=int(config.get("pose_layers", 6)),
        t_encoder="foot_conv",
    ).to(device)
    warm = AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=0.0,
        pose_layers=int(config.get("pose_layers", 6)),
        t_encoder="foot_conv",
    ).to(device)
    warm_pose_before = {
        key: value.detach().cpu().clone()
        for key, value in warm.state_dict().items()
        if (
            (key.startswith("pose_head.") and not key.startswith("pose_head.embeddings."))
            or key.startswith("traj_head.")
        )
    }
    init_from_checkpoint(warm, args.init_from)
    for key, before in warm_pose_before.items():
        if not torch.equal(warm.state_dict()[key].detach().cpu(), before):
            raise AssertionError(f"legacy warm-start contaminated {key}")

    # This one-session fit is sufficient for numerical preflight. Production
    # train.py refits the same buffers on all 92 training sessions.
    pose_mean, pose_std = fit_pose_stats(loader, device)
    fresh.pose_head.set_stats(pose_mean, pose_std)
    warm.pose_head.set_stats(pose_mean, pose_std)
    anchor = batch["trans_anchor"][:, None]
    gt_world_trans = batch["trans_gt"] + anchor
    mean_pose = pose_mean.view(1, 1, POSE_DIM).expand_as(batch["pose_gt"])
    mean_mpjpe = _mpjpe_mm(mean_pose, gt_world_trans, batch)

    fresh.eval()
    warm.eval()
    with torch.inference_mode():
        out_fresh = _predict(fresh, batch)
        out_warm = _predict(warm, batch)
        fresh_pose_mpjpe = _mpjpe_mm(out_fresh["x0_hat"], gt_world_trans, batch)
        warm_pose_mpjpe = _mpjpe_mm(out_warm["x0_hat"], gt_world_trans, batch)
        fresh_full_mpjpe = _mpjpe_mm(
            out_fresh["x0_hat"], out_fresh["trans_hat"] + anchor, batch
        )
        warm_full_mpjpe = _mpjpe_mm(
            out_warm["x0_hat"], out_warm["trans_hat"] + anchor, batch
        )

    warm.train()
    base_lr = float(config["lr"])
    optimizer = torch.optim.Adam(warm.parameters(), lr=base_lr)
    loss_values = []
    component_values = []
    for step in range(max(int(args.steps), 1)):
        if int(args.warmup_steps) > 0:
            step_lr = base_lr * min(1.0, float(step + 1) / float(args.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = step_lr
        optimizer.zero_grad(set_to_none=True)
        out = _predict(warm, batch)
        losses = compute_losses(out, batch, batch["config_id"], config)
        if not bool(torch.isfinite(losses["loss"])):
            raise AssertionError("F4a real-batch loss is non-finite")
        losses["loss"].backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in warm.parameters()):
            raise AssertionError("F4a real-batch gradient is non-finite")
        grad_norm = torch.nn.utils.clip_grad_norm_(warm.parameters(), float(args.grad_clip))
        if not bool(torch.isfinite(grad_norm)):
            raise AssertionError("F4a real-batch gradient norm is non-finite")
        optimizer.step()
        loss_values.append(float(losses["loss"].detach()))
        component_values.append({
            key: float(losses[key].detach())
            for key in ("L_pose", "L_traj", "L_kp", "L_Trec", "L_Vrec", "L_con")
        })

    warm.eval()
    with torch.inference_mode():
        out_after = _predict(warm, batch)
        after_pose_mpjpe = _mpjpe_mm(out_after["x0_hat"], gt_world_trans, batch)
        after_full_mpjpe = _mpjpe_mm(
            out_after["x0_hat"], out_after["trans_hat"] + anchor, batch
        )

    print(
        f"contract: protocol=smpl24 pose={tuple(batch['pose_gt'].shape)} "
        f"kp={tuple(batch['kp_gt'].shape)} parents=SMPL24 "
        f"contact_source=BVH-derived-file floor_y={tuple(batch['floor_y'].shape)}"
    )
    print(
        "contact supervision: lambda_con=%.3g (%s)" % (
            float(config.get("lambda_con", 0.0)),
            "labels are metrics-only in this command" if float(config.get("lambda_con", 0.0)) == 0.0
            else "labels affect the optimization",
        )
    )
    print(f"mean-pose + GT-root baseline: {mean_mpjpe:.2f} mm")
    print(
        "random initialization diagnostics (NOT trained baselines): "
        f"fresh pose/full={fresh_pose_mpjpe:.2f}/{fresh_full_mpjpe:.2f} mm; "
        f"legacy-warm pose/full={warm_pose_mpjpe:.2f}/{warm_full_mpjpe:.2f} mm"
    )
    print(
        f"fixed-batch overfit {max(int(args.steps), 1)} steps: "
        f"loss {loss_values[0]:.6f} -> {loss_values[-1]:.6f}; "
        f"pose/full MPJPE -> {after_pose_mpjpe:.2f}/{after_full_mpjpe:.2f} mm"
    )
    print(
        "loss components first -> last: "
        + ", ".join(
            f"{key}={component_values[0][key]:.6g}->{component_values[-1][key]:.6g}"
            for key in component_values[0]
        )
    )
    if loss_values[-1] >= loss_values[0] or after_full_mpjpe >= warm_full_mpjpe:
        raise AssertionError(
            "short fixed-batch optimization did not improve both final loss and full MPJPE"
        )
    print("F4 SMPL-24 PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
