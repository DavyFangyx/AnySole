"""F1 smoke test: hmr cache assembly, VEncHMR, and the hmr_gvhmr model path.

Synthetic hmr cache (no GVHMR checkpoints needed) — checks the AnySole-side
integration only.  The real GVHMR extraction + zero-training probe need the
downloaded checkpoints (see anysole/data/extract_hmr.py header).

Run (touch_gait env):
  python z_note/smoke_f1_hmr.py [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.data.dataset import assemble_v_hmr
from anysole.models import AnySoleModelV2
from anysole.models.encoders import VEncHMR
from anysole.models.embeddings import SharedEmbeddings
from anysole.types import (
    CONFIG_V,
    CONFIG_VT,
    TW,
    V_HMR_DIM,
    V_HMR_IMG_DIM,
    V_HMR_KP_DIM,
    V_HMR_MISC_DIM,
    V_HMR_ROT_DIM,
)


def synthetic_cache(n: int) -> dict:
    return {
        "body_pose": np.random.randn(n, 63).astype(np.float32),
        "global_orient": np.random.randn(n, 3).astype(np.float32),
        "betas": np.random.randn(n, 10).astype(np.float32),
        "kp2d": np.random.rand(n, 17, 3).astype(np.float32),
        "f_imgseq": np.random.randn(n, 1024).astype(np.float32),
        "bbx_xys": np.random.rand(n, 3).astype(np.float32) * 1000.0,
        "q_v": np.random.rand(n, 2).astype(np.float32),
        "img_w": 1920,
        "img_h": 1080,
    }


def check_assemble() -> None:
    n = 37
    v_hmr = assemble_v_hmr(synthetic_cache(n), n)
    assert v_hmr.shape == (n, V_HMR_DIM), v_hmr.shape
    total = V_HMR_ROT_DIM + V_HMR_KP_DIM + V_HMR_IMG_DIM + V_HMR_MISC_DIM
    assert total == V_HMR_DIM, total
    # misc block: bbox normalized by image dims.
    bbx, img_w, img_h = 1000.0 * np.random.rand(3), 1920, 1080
    cache = synthetic_cache(4)
    cache["bbx_xys"] = np.tile(bbx[None], (4, 1)).astype(np.float32)
    out = assemble_v_hmr(cache, 4)
    misc = out[:, -V_HMR_MISC_DIM:]
    diag = np.hypot(img_w, img_h)
    assert np.allclose(misc[:, 0], bbx[0] / img_w)
    assert np.allclose(misc[:, 1], bbx[1] / img_h)
    assert np.allclose(misc[:, 2], bbx[2] / diag)
    print("1. assemble_v_hmr: (%d, %d), block split + bbox normalization OK" % v_hmr.shape)


def check_encoder(device: torch.device) -> None:
    emb = SharedEmbeddings(tw=TW)
    enc = VEncHMR(emb).to(device)
    x = torch.randn(2, TW, V_HMR_DIM, device=device)
    out = enc(x)
    assert out.shape == (2, TW, emb.dim), out.shape
    print("2. VEncHMR: (B,tw,1156) -> (B,tw,256) OK")


def check_model(device: torch.device) -> None:
    model = AnySoleModelV2(v_input="hmr_gvhmr").to(device)
    batch = 2
    v_feat = torch.randn(batch, TW, 2051, device=device)
    t_raw = torch.randn(batch, TW, 96, device=device)
    t_phys = torch.randn(batch, TW, 12, device=device)
    v_hmr = torch.randn(batch, TW, V_HMR_DIM, device=device)
    cid = torch.tensor([CONFIG_VT, CONFIG_V], device=device)
    out = model(v_feat, t_raw, t_phys, cid, V_hmr=v_hmr)
    assert out["x0_hat"].shape == (batch, TW, 138), out["x0_hat"].shape
    out["x0_hat"].sum().backward()
    assert model.encoders.v_enc.rot_proj.weight.grad is not None
    # Missing V_hmr must raise.
    try:
        model(v_feat, t_raw, t_phys, cid)
        raise AssertionError("hmr model accepted missing V_hmr")
    except ValueError:
        pass
    print("3. AnySoleModelV2(v_input=hmr_gvhmr): forward + grads + missing-V_hmr guard OK")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="F1 hmr integration smoke test.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    check_assemble()
    check_encoder(device)
    check_model(device)
    print("smoke_f1_hmr: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
