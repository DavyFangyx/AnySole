#!/usr/bin/env python3
"""Diagnostic: rebuild MotionPRO clean features with AnySole bbox_cam3 (crop
too) for a subset of sessions, so we can test whether the v1 YOLOX bbox is
what drags the four-fold PA down."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/fangyuxuan/projects/gait")
for p in (ROOT, ROOT / "Baselines/MotionPRO"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from AnysoleWorkspace.tool.adapters.MotionPRO.build_e5_variants import (  # noqa: E402
    ANYSOLE_BBOX, BASE, OUT, V1, encode_frames, link_inputs,
    load_encoder, session_dirs,
)

TREE = OUT / "adapter_v4_anysolebbox/cam3"


def main() -> int:
    sessions = sys.argv[1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_encoder(device)
    for sid in sessions:
        base_dir, v1_dir, rel = session_dirs(sid)
        out_dir = TREE / rel
        if (out_dir / "feature_hrnet.pth").is_file():
            print(f"skip {sid}")
            continue
        bbox = np.load(ANYSOLE_BBOX / f"{sid}.npy")
        import os
        color_dir = base_dir / "color"
        n = len(bbox)
        paths = [color_dir / f"{i:06d}.jpg" for i in range(n)]
        feat = encode_frames(device, model, base_dir, bbox, paths, batch=16)
        link_inputs(base_dir, out_dir, feat, False)
        print(f"[anysole] {sid} {tuple(feat.shape)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
