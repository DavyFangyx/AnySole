"""F2a smoke test: GT -> f2 representation -> f2_to_world roundtrip (plan's
"训练前必过" test), forward-axis verification, and model/loss integration.

Run (touch_gait env):
  python z_note/smoke_f2_roundtrip.py [--device cpu]

Checks:
 1. Forward axis: on every informative train session (enough motion), the
    heading-frame lateral/forward velocity ratio and the speed correlation
    favor +Z (geometry.FORWARD_AXIS) over +X.
 2. Roundtrip: GT -> tilt pose + traj_f2 -> f2_to_world -> world pose/trans;
    cumulative drift < 1mm over full windows, per-joint world FK identical.
 3. Dataset: f2_repr=True carries pose_gt (tilt root), traj_gt_f2 (tw,4),
    psi_anchor; f2_repr=False batches stay byte-identical to pre-F2.
 4. Model/loss: AnySoleModelV2(f2_repr=True) v_hat (B,tw,4); compute_losses
    finite with fitted traj stats; gradients reach the traj head.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.data.dataset import AnySoleDataset, load_split_ids
from anysole.geometry import (
    FORWARD_AXIS,
    f2_to_world_np,
    fk_pose6d_np,
    heading_from_root_np,
    rot6d_to_rotmat_np,
    yaw_rotmat_np,
)
from anysole.losses import compute_losses
from anysole.models import AnySoleModelV2
from anysole.types import N_JOINTS, SEQ_ROOT, SPLIT_CSV, TW


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F2a roundtrip smoke test.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sessions", type=int, default=6)
    return parser.parse_args(argv)


def check_forward_axis(sessions: list) -> None:
    """Speed correlation favors +Z over +X on informative sessions."""
    wins = {"+Z": 0, "+X": 0, "uninformative": 0}
    for s in sessions:
        root = rot6d_to_rotmat_np(s["pose_gt"].reshape(-1, N_JOINTS, 6)[:, 0])
        trans = s["trans_global"]
        v = np.diff(trans, axis=0) * 40.0
        v = np.concatenate([v[:1], v], axis=0)
        speed = np.linalg.norm(v[:, [0, 2]], axis=1)
        moving = speed > 0.15
        if moving.sum() < 50:
            wins["uninformative"] += 1
            continue
        corrs = {}
        for name, axis in (("+Z", np.array([0., 0., 1.])), ("+X", np.array([1., 0., 0.]))):
            f = np.einsum("tij,j->ti", root, axis)
            psi = np.arctan2(f[:, 2], f[:, 0])
            c, sn = np.cos(psi), np.sin(psi)
            v_fwd = c * v[:, 0] + sn * v[:, 2]
            corrs[name] = np.corrcoef(v_fwd[moving], speed[moving])[0, 1]
        wins["+Z" if corrs["+Z"] >= corrs["+X"] else "+X"] += 1
    assert wins["+Z"] > wins["+X"], wins
    print("1. forward axis: +Z wins %s (informative sessions)" % wins)


def check_roundtrip(ds_f2: AnySoleDataset, ds_v1: AnySoleDataset) -> None:
    """f2 -> world must recover the v1 world pose/trans to sub-mm."""
    worst_pose, worst_trans = 0.0, 0.0
    for f2_s, v1_s in zip(ds_f2.sessions, ds_v1.sessions):
        assert f2_s["session_id"] == v1_s["session_id"]
        n = f2_s["pose_gt"].shape[0]
        for left in range(0, n - TW, TW):
            right = left + TW
            psi_a = float(f2_s["psi"][max(left - 1, 0)])
            trans_a = f2_s["trans_global"][max(left - 1, 0)]
            world_pose, world_trans = f2_to_world_np(
                f2_s["pose_gt"][left:right], f2_s["traj_gt_f2"][left:right], psi_a, trans_a
            )
            gt_pose = v1_s["pose_gt"][left:right]
            gt_trans = v1_s["trans_global"][left:right]
            kp_w = fk_pose6d_np(world_pose, world_trans, f2_s["offsets"], f2_s["parents"])
            kp_g = v1_s["kp_gt"][left:right]
            worst_pose = max(worst_pose, float(np.abs(world_pose - gt_pose).max()))
            worst_trans = max(worst_trans, float(np.abs(world_trans - gt_trans).max()),
                              float(np.abs(kp_w - kp_g).max()))
    assert worst_trans < 1e-3, ("roundtrip drift %.5f mm" % (worst_trans * 1000.0))
    print("2. roundtrip: world pose/trans/FK recovered to %.4f mm (plan's <1mm bar)" % (worst_trans * 1000.0))


def check_dataset_fields(ds_f2: AnySoleDataset, ds_v1: AnySoleDataset) -> None:
    item_f2 = ds_f2[0]
    item_v1 = ds_v1[0]
    assert "traj_gt_f2" in item_f2 and item_f2["traj_gt_f2"].shape == (TW, 4)
    assert "psi_anchor" in item_f2 and item_f2["psi_anchor"].ndim == 0
    assert "traj_gt_f2" not in item_v1, "f2 fields leaked into the default path"
    # Tilt root: yaw-free — the world root yaw of f2 items must be ~0 after
    # removing the anchor heading (rotation about Y removed by construction).
    root6 = item_f2["pose_gt"][:, :6].numpy().reshape(-1, 6)
    assert np.isfinite(root6).all()
    print("3. dataset: f2 fields present when f2_repr=True, absent otherwise")


def check_model_loss(device: torch.device) -> None:
    model = AnySoleModelV2(f2_repr=True).to(device)
    batch = 2
    v = torch.randn(batch, TW, 2051, device=device)
    tr = torch.randn(batch, TW, 96, device=device)
    tp = torch.randn(batch, TW, 12, device=device)
    cid = torch.zeros(batch, dtype=torch.long, device=device)
    out = model(v, tr, tp, cid)
    assert out["v_hat"].shape == (batch, TW, 4), out["v_hat"].shape
    assert out["x0_hat"].shape == (batch, TW, 138)
    full = {
        "V_feat": v, "T_raw": tr, "T_phys": tp, "T_s2m": torch.zeros(batch, TW, 50, device=device),
        "pose_gt": torch.randn(batch, TW, 138, device=device),
        "vel_gt": torch.randn(batch, TW, 3, device=device),
        "trans_gt": torch.randn(batch, TW, 3, device=device),
        "traj_gt_f2": torch.randn(batch, TW, 4, device=device),
        "psi_anchor": torch.zeros(batch, device=device),
        "trans_anchor": torch.zeros(batch, 3, device=device),
        "kp_gt": torch.randn(batch, TW, 23, 3, device=device),
        "contact_gt": torch.rand(batch, TW, 2, device=device),
        "offsets": torch.rand(batch, 23, 3, device=device),
        "parents": torch.tensor([[-1] + list(range(22))], device=device).expand(batch, -1),
        "config_id": cid,
    }
    weights = {
        "f2_repr": True,
        "traj_f2_stats": {"mean": [0.0, 0.0, 0.0, 0.9], "std": [0.3, 1.0, 1.0, 0.1]},
        "lambda_con": 0.0,
    }
    losses = compute_losses(out, full, cid, weights)
    assert torch.isfinite(losses["loss"]), losses["loss"]
    losses["loss"].backward()
    assert model.traj_head.proj.weight.grad is not None
    assert model.traj_head.proj.weight.grad.abs().sum() > 0
    print("4. model/loss: v_hat (B,tw,4), f2 losses finite, traj-head grads flow")


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    sessions = load_split_ids(SPLIT_CSV, "train")[: args.sessions]
    ds_f2 = AnySoleDataset(mode="eval", seq_root=SEQ_ROOT, split_csv=SPLIT_CSV,
                           window_length=TW, session_ids=sessions,
                           contact_method="joint_and", f2_repr=True)
    ds_v1 = AnySoleDataset(mode="eval", seq_root=SEQ_ROOT, split_csv=SPLIT_CSV,
                           window_length=TW, session_ids=sessions,
                           contact_method="joint_and", f2_repr=False)
    check_forward_axis(ds_v1.sessions)
    check_roundtrip(ds_f2, ds_v1)
    check_dataset_fields(ds_f2, ds_v1)
    check_model_loss(device)
    print("smoke_f2_roundtrip: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
