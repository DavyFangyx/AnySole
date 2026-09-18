"""F0b smoke test (fix_plan_v2.md §3 F0b 单测): regression head shapes, loss
backprop, and the "regress 关 = 旧行为" (diffusion-mode byte-parity) guard.

No data files needed — everything runs on synthetic batches.

Run (touch_gait env):
  python z_note/smoke_f0b_regress.py [--device cuda]

Checks:
 1. AnySoleModelV2 forward shapes: x0_hat (B,tw,138), v_hat/trans_hat
    (B,tw,3), pressure_hat (B,tw,96), vfeat_hat (B,tw,2051), F (B,40,256).
 2. compute_losses on the regression output (E3 weights): finite, backward,
    and the regression query parameter receives gradients.
 3. PoseHead state-dict layout: diffusion mode keeps exactly the pre-F0b
    keys (proj_*/timestep_token); regress mode has `query` and neither.
 4. Diffusion forward(x_tau, tau, F) still returns (B,tw,138); regress
    forward rejects positional diffusion-style calls (guard against silently
    feeding x_tau in as the memory).
 5. Optional: load an existing E3-era ckpt into AnySoleModel (anysolev1)
    strict=True and run one diffusion forward — guards that the F0b edits did
    not change the V1 module layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.models import AnySoleModel, AnySoleModelV2, MODEL_ANYSOLEV1
from anysole.models.pose_head import PoseHead
from anysole.losses import compute_losses
from anysole.train import load_config, resolve_device
from anysole.types import (
    JOINT_PARENTS,
    N_CONTACT,
    N_JOINTS,
    POSE_DIM,
    POSE_POS_DIM,
    T_PHYS_DIM,
    T_RAW_DIM,
    T_S2M_DIM,
    TW,
    V_FEAT_DIM,
)

# E3-era checkpoints whose strict load guards the V1 layout (first hit wins).
E3_CKPT_CANDIDATES = (
    GAIT / "results/AnySole/E3_base/checkpoints/ckpt_last.pt",
    GAIT / "results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt",
    GAIT / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt",
)


def synthetic_batch(batch_size: int, tw: int, device: torch.device) -> dict:
    batch = {
        "V_feat": torch.randn(batch_size, tw, V_FEAT_DIM),
        "T_raw": torch.rand(batch_size, tw, T_RAW_DIM),
        "T_phys": torch.randn(batch_size, tw, T_PHYS_DIM),
        "T_s2m": torch.randn(batch_size, tw, T_S2M_DIM),
        "pose_gt": torch.randn(batch_size, tw, POSE_DIM),
        "pose_gt_pos": torch.randn(batch_size, tw, POSE_POS_DIM),
        "trans_gt": torch.randn(batch_size, tw, 3) * 0.01,
        "vel_gt": torch.randn(batch_size, tw, 3) * 0.4,
        "trans_anchor": torch.randn(batch_size, 3),
        "root_rot_init": torch.eye(3).repeat(batch_size, 1, 1),
        "kp_gt": torch.randn(batch_size, tw, N_JOINTS, 3) * 0.1,
        "contact_gt": torch.rand(batch_size, tw, N_CONTACT),
        "offsets": torch.randn(batch_size, N_JOINTS, 3) * 0.05,
        "parents": torch.tensor(JOINT_PARENTS).repeat(batch_size, 1),
    }
    return {k: v.to(device) for k, v in batch.items()}


def check_v2_forward(device: torch.device, tw: int) -> None:
    print("== 1. AnySoleModelV2 forward shapes ==")
    model = AnySoleModelV2(d=256, tw=tw).to(device).eval()
    batch = synthetic_batch(4, tw, device)
    config_id = torch.zeros(4, dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(batch["V_feat"], batch["T_raw"], batch["T_phys"], config_id, T_s2m=batch["T_s2m"])
    assert out["x0_hat"].shape == (4, tw, POSE_DIM), out["x0_hat"].shape
    assert out["v_hat"].shape == (4, tw, 3), out["v_hat"].shape
    assert out["trans_hat"].shape == (4, tw, 3), out["trans_hat"].shape
    assert out["pressure_hat"].shape == (4, tw, T_RAW_DIM), out["pressure_hat"].shape
    assert out["vfeat_hat"].shape == (4, tw, V_FEAT_DIM), out["vfeat_hat"].shape
    assert out["F"].shape == (4, 2 * tw, 256), out["F"].shape
    print("  shapes OK: x0_hat %s, F %s" % (tuple(out["x0_hat"].shape), tuple(out["F"].shape)))


def check_loss_backward(device: torch.device, tw: int) -> None:
    print("== 2. compute_losses + backward (regress query gets gradients) ==")
    model = AnySoleModelV2(d=256, tw=tw).to(device).train()
    batch = synthetic_batch(4, tw, device)
    config_id = torch.zeros(4, dtype=torch.long, device=device)
    config = load_config(GAIT / "configs" / "v1.yaml")
    config["modal"] = "anysolev2"
    out = model(batch["V_feat"], batch["T_raw"], batch["T_phys"], config_id, T_s2m=batch["T_s2m"])
    losses = compute_losses(out, batch, config_id, config)
    assert torch.isfinite(losses["loss"]), losses["loss"]
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()
    assert model.pose_head.query.grad is not None, "regress query received no gradient"
    assert model.pose_head.query.grad.abs().sum().item() > 0.0, "regress query gradient is zero"
    print("  loss %.6f finite, |grad(query)| %.4f" % (losses["loss"].item(),
                                                      model.pose_head.query.grad.abs().sum().item()))


def check_head_modes(tw: int) -> None:
    print("== 3. PoseHead diffusion/regress layout parity ==")
    diff = PoseHead(None, dim=256, tw=tw, head_mode="diffusion")
    reg = PoseHead(None, dim=256, tw=tw, head_mode="regress")
    diff_keys = set(diff.state_dict().keys())
    # The pre-F0b parameter layout must be exactly these top-level names
    # (plus decoder internals); regress swaps the input projections and the
    # timestep token for one query parameter.
    for key in ("proj_body.weight", "proj_left.weight", "proj_right.weight", "timestep_token"):
        assert key in diff_keys, "diffusion head lost %s" % key
        assert key not in reg.state_dict(), "regress head must not keep %s" % key
    assert "query" in reg.state_dict(), "regress head missing query"
    assert "query" not in diff_keys, "diffusion head must not carry the regress query"
    print("  layout OK: diffusion keeps proj_*/timestep_token, regress has query only")

    print("== 4. forward signatures ==")
    x_tau = torch.randn(2, tw, POSE_DIM)
    tau = torch.zeros(2, dtype=torch.long)
    F = torch.randn(2, 2 * tw, 256)
    with torch.inference_mode():
        out_diff = diff(x_tau, tau, F)
        out_reg = reg(F=F)
    assert out_diff.shape == (2, tw, POSE_DIM), out_diff.shape
    assert out_reg.shape == (2, tw, POSE_DIM), out_reg.shape
    try:
        reg(F, x_tau, tau)  # positional diffusion-style call must be rejected
        raise AssertionError("regress head accepted positional (x_tau, tau, F)")
    except ValueError:
        pass
    print("  diffusion (x_tau, tau, F) -> %s; regress F=... -> %s; positional guard OK"
          % (tuple(out_diff.shape), tuple(out_reg.shape)))


def check_e3_ckpt_loads(device: torch.device) -> None:
    print("== 5. E3-era ckpt strict load (V1 layout unchanged) ==")
    ckpt_path = next((p for p in E3_CKPT_CANDIDATES if p.is_file()), None)
    if ckpt_path is None:
        print("  skipped: no E3-era checkpoint found under results/AnySole")
        return
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved = checkpoint.get("config", {})
    tw = int(saved.get("tw", TW))
    model = AnySoleModel(
        d=int(saved.get("d_model", 256)), tw=tw, modal=MODEL_ANYSOLEV1,
        dropout=float(saved.get("dropout", 0.1)),
        pose_layers=int(saved.get("pose_layers", 6)),
        tactile_input=str(saved.get("tactile_input", "raw108")),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    batch = synthetic_batch(2, tw, device)
    config_id = torch.zeros(2, dtype=torch.long, device=device)
    tau = torch.zeros(2, dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(batch["V_feat"], batch["T_raw"], batch["T_phys"],
                    batch["pose_gt"], tau, config_id, T_s2m=batch["T_s2m"])
    assert out["x0_hat"].shape == (2, tw, POSE_DIM)
    print("  %s strict-loaded and ran: x0_hat %s" % (ckpt_path.name, tuple(out["x0_hat"].shape)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="F0b regression smoke test.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    tw = TW
    check_v2_forward(device, tw)
    check_loss_backward(device, tw)
    check_head_modes(tw)
    check_e3_ckpt_loads(device)
    print("smoke_f0b_regress: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
