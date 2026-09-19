"""V3-2 smoke test: 9-part pose-head migration (fix_plan_v3.md §V3-2).

Checks (no data files needed except the F4a checkpoint for the warm-start):
 1. n_parts=3 (default) and n_parts=9 both produce x0_hat (B,tw,138) and run
    the shared decoder; n_parts=9 rejects diffusion/pos combinations.
 2. Part scatter: with per-part biases set to constant markers, _unembed maps
    each joint's 6D output to its own part marker — proves the part-order ->
    joint-order scatter (part_place_idx) is correct (r_leg/r_foot swap).
 3. Warm-start from F4a_footconv with pose_parts=9: shared weights (decoder,
    encoders, fusion) are copied; the resized/missing pieces (query 23->9,
    group_emb 3->9, out_body/left/right -> out_parts) are freshly initialized
    — init_from_checkpoint's name+shape matching.
 4. Gradients flow to the 9 part heads.

Run (touch_gait env):
  python z_note/smoke_v32_parts.py [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.models import AnySoleModelV2
from anysole.models.pose_head import PoseHead
from anysole.train import init_from_checkpoint
from anysole.types import PART_JOINTS, POSE_DIM, TW


def check_shapes(device: torch.device) -> None:
    batch = 2
    for n_parts in (3, 9):
        head = PoseHead(None, tw=TW, head_mode="regress", repr="6d",
                        n_parts=n_parts).to(device)
        F = torch.randn(batch, 2 * TW, 256, device=device)
        out = head(F=F)
        assert out.shape == (batch, TW, POSE_DIM), out.shape
        loss = out.square().mean()
        loss.backward()
        for p in head.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
    # n_parts=9 is regress/6d only.
    for kwargs in ({"head_mode": "diffusion", "n_parts": 9},
                   {"head_mode": "regress", "repr": "pos", "n_parts": 9}):
        try:
            PoseHead(None, **kwargs)
            raise AssertionError("n_parts=9 should reject %r" % kwargs)
        except ValueError:
            pass
    print("1. n_parts 3/9: forward (B,tw,138), grads finite; invalid combos rejected")


def check_scatter(device: torch.device) -> None:
    head = PoseHead(None, tw=TW, head_mode="regress", repr="6d",
                    n_parts=9).to(device)
    for p, lin in enumerate(head.out_parts):
        lin.weight.data.zero_()
        lin.bias.data.fill_(float(p + 1))
    batch = 2
    h = torch.randn(batch, TW, 9, 256, device=device)
    out = head._unembed(h)  # (B, tw, 138)
    part_of = {}
    for p, joints in enumerate(PART_JOINTS):
        for j in joints:
            part_of[j] = p + 1
    for j in range(23):
        for d in range(6):
            col = out[..., 6 * j + d]
            assert torch.allclose(col, torch.full_like(col, float(part_of[j]))), \
                "joint %d dim %d got wrong part marker" % (j, d)
    print("2. scatter: every joint's 6D output lands at its own part marker "
          "(r_leg/r_foot swap covered)")


def check_warm_start(device: torch.device) -> None:
    ckpt_path = GAIT / "results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = AnySoleModelV2(t_encoder="foot_conv", pose_parts=9).to(device)
    init_from_checkpoint(model, ckpt_path)
    sd = ck["model"]
    # Shared: decoder layers + encoders + fusion copied exactly.
    for key in ("pose_head.decoder.layers.0.self_attn.in_proj_weight",
                "encoders.v_enc.proj.weight",
                "encoders.t_enc.conv.0.weight",
                "fusion.encoder.layers.0.self_attn.in_proj_weight"):
        cur = model.state_dict()[key]
        assert torch.equal(cur.cpu(), sd[key]), "%s not copied" % key
    # Resized/missing: fresh init (not equal to any ckpt key).
    assert "pose_head.query" not in sd or \
        model.state_dict()["pose_head.query"].shape != sd["pose_head.query"].shape
    assert model.state_dict()["pose_head.query"].shape == (1, TW, 9, 256)
    assert model.state_dict()["pose_head.group_emb.weight"].shape == (9, 256)
    assert "pose_head.out_body.weight" not in model.state_dict()
    assert "pose_head.out_parts.0.weight" in model.state_dict()
    assert "pose_head.part_place_idx" in model.state_dict()
    print("3. warm-start F4a->9parts: shared weights copied, query/group_emb/"
          "out_parts fresh, part_place_idx registered")


def check_model(device: torch.device) -> None:
    model = AnySoleModelV2(t_encoder="foot_conv", pose_parts=9).to(device)
    batch = 2
    out = model(
        torch.randn(batch, TW, 2051, device=device),
        torch.randn(batch, TW, 96, device=device),
        torch.randn(batch, TW, 12, device=device),
        torch.zeros(batch, dtype=torch.long, device=device),
    )
    assert out["x0_hat"].shape == (batch, TW, POSE_DIM)
    assert out["F"].shape == (batch, 2 * TW, 256)
    out["x0_hat"].square().mean().backward()
    assert model.pose_head.out_parts[7].weight.grad is not None
    assert model.pose_head.out_parts[7].weight.grad.abs().sum() > 0
    print("4. AnySoleModelV2(pose_parts=9): end-to-end forward + part-head grads")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V3-2 9-part pose head smoke test.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    check_shapes(device)
    check_scatter(device)
    check_warm_start(device)
    check_model(device)
    print("smoke_v32_parts: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
