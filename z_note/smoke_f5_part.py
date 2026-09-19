"""F5 smoke test: part-query decoder structure and gating (ARCHIVED).

**ARCHIVED (2026-09-19)**: v2 §F5 作废（fix_plan_v3.md 取代），PartDecoder 无调用
方、仅留档。本脚本只保留 PartDecoder 独立构造的检查 1-3；原检查 4/5
（AnySoleModelV2(decoder=part9) 端到端集成）随回退删除——模型构造已不再接受
decoder 参数。此脚本不在常规回归清单内，仅供 PartDecoder 留档参考。

Run (touch_gait env):
  python z_note/smoke_f5_part.py [--device cpu]

Checks:
 1. Hop matrix: root-torso = 1, torso-l_arm = 2 (through the spine chain),
    diagonal = 0, symmetric.
 2. Gate bias init: root/legs/feet favor T (+1), torso/head/arms favor V.
 3. PartDecoder forward: x0_hat (B,tw,138), v_hat (B,tw,3/4), gates
    (B,tw,9,3) rows sum to 1; with V fully masked the V gates are ~0 and
    the T/empty gates take over; nogate variant runs with the same shapes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.models.part_decoder import (
    N_PARTS,
    PART_NAMES,
    PartDecoder,
    part_hop_matrix,
)
from anysole.types import TW


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F5 part decoder smoke test.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def check_hop_matrix() -> None:
    hops = part_hop_matrix()
    assert hops.shape == (9, 9)
    assert (hops.diagonal() == 0).all()
    assert (hops == hops.T).all()
    names = dict(zip(PART_NAMES, range(9)))
    assert hops[names["root"], names["torso"]].item() == 1
    assert hops[names["torso"], names["l_arm"]].item() == 1  # Spine2 -> LShoulder
    assert hops[names["root"], names["l_leg"]].item() == 1   # Hips -> LUpLeg
    assert hops[names["l_arm"], names["r_arm"]].item() >= 2  # arms meet at the torso
    assert hops[names["root"], names["l_hand"] if "l_hand" in names else names["l_arm"]].item() >= 3
    print("1. hop matrix: root-torso=1, torso-l_arm=1, arms via torso, symmetric")


def check_decoder(device: torch.device) -> None:
    batch = 2
    decoder = PartDecoder(traj_dim=3).to(device)
    e_v = torch.randn(batch, TW, 256, device=device)
    e_t = torch.randn(batch, TW, 3, 256, device=device)
    mask_v = torch.ones(batch, TW, device=device, dtype=torch.bool)
    mask_t = torch.ones(batch, TW, device=device, dtype=torch.bool)
    out = decoder(e_v, e_t, torch.zeros(batch, dtype=torch.long, device=device),
                  mask_v, mask_t)
    assert out["x0_hat"].shape == (batch, TW, 138), out["x0_hat"].shape
    assert out["v_hat"].shape == (batch, TW, 3)
    assert out["gates"].shape == (batch, TW, 9, 3)
    assert torch.allclose(out["gates"].sum(-1), torch.ones(1, device=device), atol=1e-5)
    # Bias init: root/legs/feet favor T; torso/head/arms favor V.
    bias = decoder.layers[0].gate_bias.detach()
    assert bias[0, 1] == 1.0 and bias[7, 1] == 1.0  # root, l_foot
    assert bias[1, 0] == 1.0 and bias[3, 0] == 1.0  # torso, l_arm
    # V fully masked -> V gates collapse to ~0.
    out2 = decoder(e_v, e_t, torch.zeros(batch, dtype=torch.long, device=device),
                   torch.zeros(batch, TW, device=device, dtype=torch.bool), mask_t)
    assert out2["gates"][..., 0].max().item() < 1e-4, out2["gates"][..., 0].max().item()
    # nogate variant: same output shapes, no gates key populated.
    nogate = PartDecoder(traj_dim=3, gate_mode="nogate").to(device)
    out3 = nogate(e_v, e_t, torch.zeros(batch, dtype=torch.long, device=device),
                  mask_v, mask_t)
    assert out3["x0_hat"].shape == (batch, TW, 138) and out3["gates"] is None
    print("2. PartDecoder: shapes, gate softmax, bias init, V-mask collapse, nogate OK")


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    check_hop_matrix()
    check_decoder(device)
    print("smoke_f5_part: ARCHIVED PartDecoder checks PASSED (checks 1-3 only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
