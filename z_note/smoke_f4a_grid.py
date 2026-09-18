"""F4a smoke test: FootConvEncoder on the EXISTING 108-dim tactile channel
(fix_plan_v2.md §F4a items 3-6; items 1-2 — native-rate resampling and
per-session normalization — are deliberately NOT applied, the tactile input
data pipeline is untouched).

Run (touch_gait env):
  python z_note/smoke_f4a_grid.py [--device cpu]

Checks:
 1. Input split: T_raw 96 -> per-foot 4×12 grids (right = cells 48:96);
    in-encoder features (contact-area ratio, dF/dt) finite and in range;
    grid orientation follows pressure.py (cop_y heel=1 -> toe=0 via
    cop_from_grid on synthetic loads).
 2. FootConvEncoder: forward (B, tw, 108) -> (B, tw, d).
 3. Right-foot mirror symmetry: a right foot that is the mirrored left foot
    yields identical conv tokens up to the foot-embedding difference.
 4. ModalEncoders + AnySoleModelV2 with t_encoder="foot_conv": shapes OK,
    null row OK, conv grads flow, T-recon target untouched (L_Trec reads the
    same 96-dim T_raw).
 5. t_encoder="linear" (default) keeps the old raw108 path byte-identical
    (same state-dict layout as the F0b baseline).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.data.pressure import cop_from_grid
from anysole.models import AnySoleModelV2
from anysole.models.encoders import ModalEncoders
from anysole.models.embeddings import SharedEmbeddings
from anysole.models.foot_encoder import FootConvEncoder, GRID_COLS, GRID_ROWS
from anysole.types import CONFIG_V, CONFIG_VT, T_PHYS_DIM, T_RAW_DIM, TW


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F4a FootConvEncoder smoke test.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def check_grid_orientation() -> None:
    """Synthetic loads -> CoP direction (pressure.py 口径: cop_y 1=heel -> 0=toe)."""
    frames = np.zeros((3, GRID_ROWS * GRID_COLS), dtype=np.float32)
    frames[0, 0::GRID_COLS] = 1.0            # col 0 = heel -> cop_y 1
    frames[1, (GRID_COLS - 1)::GRID_COLS] = 1.0  # col 11 = toe -> cop_y 0
    frames[2, (GRID_COLS // 2)::GRID_COLS] = 1.0  # mid
    cop = cop_from_grid(frames)
    assert cop[0, 1] > 0.9 and cop[1, 1] < 0.1 and 0.4 < cop[2, 1] < 0.6, cop[:, 1]
    print("1. grid orientation OK (cop_y heel=1 -> toe=0)")


def check_input_split(device: torch.device) -> None:
    emb = SharedEmbeddings(tw=TW)
    enc = FootConvEncoder(emb).to(device)
    batch = 2
    t_raw = torch.rand(batch, TW, T_RAW_DIM, device=device)
    t_phys = torch.rand(batch, TW, T_PHYS_DIM, device=device)
    grids, feats = enc._per_foot_inputs(torch.cat([t_raw, t_phys], dim=-1))
    assert grids.shape == (batch, TW, 2, GRID_ROWS, GRID_COLS), grids.shape
    assert feats.shape == (batch, TW, 2, 8), feats.shape  # T_phys 6 + area + dF/dt
    assert torch.isfinite(feats).all()
    assert (feats[..., -2] >= 0).all() and (feats[..., -2] <= 1).all()  # contact area
    print("2. input split: grids (B,tw,2,4,12) + feats (B,tw,2,8); area/dFdt finite OK")


def check_encoder(device: torch.device) -> None:
    emb = SharedEmbeddings(tw=TW)
    enc = FootConvEncoder(emb).to(device)
    batch = 2
    x = torch.rand(batch, TW, T_RAW_DIM + T_PHYS_DIM, device=device)
    out = enc(x)
    assert out.shape == (batch, TW, emb.dim), out.shape
    # Mirror symmetry: right grid = mirrored left grid, equal features ->
    # conv tokens differ only by the foot embedding.
    left = torch.rand(batch, TW, GRID_ROWS, GRID_COLS, device=device)
    right = torch.flip(left, dims=[-2])
    t_raw = torch.cat([left.reshape(batch, TW, 48), right.reshape(batch, TW, 48)], dim=-1)
    t_phys = torch.rand(batch, TW, T_PHYS_DIM, device=device)
    t_phys[:, :, 6:] = t_phys[:, :, :6]  # equal per-foot features
    tokens = enc._per_foot_tokens(torch.cat([t_raw, t_phys], dim=-1))
    foot_emb = enc.foot_emb.weight.to(tokens.dtype)
    diff = (tokens[:, :, 0] - tokens[:, :, 1]).abs().mean()
    expected = (foot_emb[0] - foot_emb[1]).abs().mean()
    assert diff.item() < expected.item() + 1e-4, (diff.item(), expected.item())
    print("3. FootConvEncoder: (B,tw,108)->(B,tw,256); right-mirror symmetry OK (diff %.4f)" % diff.item())


def check_integration(device: torch.device) -> None:
    emb = SharedEmbeddings(tw=TW)
    modal = ModalEncoders(emb, tactile_input="raw108", t_encoder="foot_conv").to(device)
    batch = 2
    v = torch.randn(batch, TW, 2051, device=device)
    t = torch.rand(batch, TW, T_RAW_DIM + T_PHYS_DIM, device=device)
    cid = torch.tensor([CONFIG_VT, CONFIG_V], device=device)
    v_tok, t_tok = modal(v, t, cid)
    assert t_tok.shape == (batch, TW, emb.dim), t_tok.shape
    assert torch.allclose(t_tok[1], emb.null_t.expand(1, TW, emb.dim).to(device)[0], atol=1e-5)
    model = AnySoleModelV2(t_encoder="foot_conv").to(device)
    tr = torch.randn(batch, TW, 96, device=device)
    tp = torch.randn(batch, TW, 12, device=device)
    out = model(v, tr, tp, cid)
    assert out["x0_hat"].shape == (batch, TW, 138), out["x0_hat"].shape
    out["x0_hat"].sum().backward()
    assert model.encoders.t_enc.conv[0].weight.grad is not None
    assert model.encoders.t_enc.conv[0].weight.grad.abs().sum() > 0
    # T-recon target unchanged: aux head still reads the 96-dim T_raw.
    assert model.aux_heads.t_rec.weight.shape == (96, 256)
    print("4. ModalEncoders + AnySoleModelV2 (t_encoder=foot_conv): shapes, null row, grads OK")


def check_default_path(device: torch.device) -> None:
    """t_encoder='linear' must keep the F0b baseline layout byte-identical."""
    model = AnySoleModelV2().to(device)
    enc = model.encoders.t_enc
    assert isinstance(enc, torch.nn.Module) and not isinstance(enc, FootConvEncoder)
    # The baseline ckpt state dict must load strictly (layout unchanged).
    ckpt = torch.load(GAIT / "results/AnySole/F0b_seed2/checkpoints/ckpt_last.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    print("5. t_encoder='linear' default: F0b_seed2 ckpt strict-loads (baseline layout unchanged)")


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    check_grid_orientation()
    check_input_split(device)
    check_encoder(device)
    check_integration(device)
    check_default_path(device)
    print("smoke_f4a_grid: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
