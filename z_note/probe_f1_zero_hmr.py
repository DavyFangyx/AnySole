"""F1 zero-training HMR baseline (fix_plan_v2.md §F1 判据): the per-frame
visual readout ceiling of GVHMR on this dataset, BEFORE any training.

Reads the hmr caches written by ``anysole/data/extract_hmr.py`` (joints_hmr =
22 SMPL-X body joints from EnDecoder.fk_v2 in gravity-aligned world space)
and compares them against the BVH-23 GT keypoints with the 15-pair semantic
matching table of fix_plan_v2.md §F1.  Per-frame Procrustes alignment
(PA-MPJPE 口径, same _pa_align as the eval protocol); also raw MPJPE after
a per-sequence world translation alignment of the pelvis.

Usage (touch_gait env, after extract_hmr.py has run):
  python z_note/probe_f1_zero_hmr.py --split val [--model gvhmr] [--device cuda]

The SMPL-X 22-body-joint order (SMPL-X convention):
  0 pelvis, 1 l_hip, 2 r_hip, 3 spine1, 4 l_knee, 5 r_knee, 6 spine2,
  7 l_ankle, 8 r_ankle, 9 spine3, 10 l_foot, 11 r_foot, 12 neck,
  13 l_collar, 14 r_collar, 15 head, 16 l_shoulder, 17 r_shoulder,
  18 l_elbow, 19 r_elbow, 20 l_wrist, 21 r_wrist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, load_split_ids  # noqa: E402
from anysole.types import HMR_CACHE_ROOT, JOINT_NAMES, SEQ_ROOT, SPLIT_CSV  # noqa: E402

# BVH joint index -> SMPL-X body joint index (15 pairs; Spine*/Neck/ToeBase/
# Hands excluded, per fix_plan_v2.md §F1).
MATCH_PAIRS: List[Tuple[int, int]] = [
    (0, 0),    # Hips        <-> pelvis
    (15, 1),   # LUpLeg      <-> l_hip
    (19, 2),   # RUpLeg      <-> r_hip
    (16, 4),   # LLeg        <-> l_knee
    (20, 5),   # RLeg        <-> r_knee
    (17, 7),   # LFoot       <-> l_ankle
    (21, 8),   # RFoot       <-> r_ankle
    (7, 16),   # LShoulder   <-> l_shoulder
    (11, 17),  # RShoulder   <-> r_shoulder
    (8, 18),   # LArm        <-> l_elbow
    (12, 19),  # RArm        <-> r_elbow
    (9, 20),   # LForeArm    <-> l_wrist
    (13, 21),  # RForeArm    <-> r_wrist
    (6, 15),   # Head        <-> head
    (5, 12),   # Neck        <-> neck (bonus pair, both are upper-spine proxies)
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 zero-training HMR baseline probe.")
    parser.add_argument("--split", choices=("val", "test", "train"), default="val")
    parser.add_argument("--model", type=str, default="gvhmr", help="hmr_cache subdir.")
    parser.add_argument("--cam-id", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def load_cache(model: str, session_id: str) -> dict:
    path = HMR_CACHE_ROOT / model / "cam3" / ("%s.pt" % session_id)
    if not path.is_file():
        raise FileNotFoundError(
            "hmr cache missing for %s: %s — run `python -m anysole.data.extract_hmr` first."
            % (session_id, path)
        )
    return torch.load(path, map_location="cpu")


def rigid_align(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Per-sample 3D Procrustes (scale-free): R*, t* = argmin ||R src + t - dst||."""
    src = src - src.mean(dim=1, keepdim=True)
    dst_c = dst - dst.mean(dim=1, keepdim=True)
    H = torch.einsum("bji,bjk->bik", src, dst_c)
    u, _, vt = torch.linalg.svd(H)
    R = torch.einsum("bij,bkj->bik", vt.transpose(1, 2), u.transpose(1, 2))
    t = dst.mean(dim=1) - torch.einsum("bij,bj->bi", R, src.mean(dim=1))
    return torch.einsum("bij,bpj->bpi", R, src) + t[:, None, :]


def main(argv=None) -> int:
    args = parse_args(argv)
    sessions = load_split_ids(SPLIT_CSV, args.split)
    dataset = AnySoleDataset(
        mode="eval",
        seq_root=SEQ_ROOT,
        split_csv=SPLIT_CSV,
        window_length=20,
        session_ids=sessions,
        contact_method="joint_and",
    )
    # kp_gt per session: (T, 23, 3) world.
    kp_gt = {session["session_id"]: torch.from_numpy(session["kp_gt"].astype(np.float32))
             for session in dataset.sessions}

    mpjpe_all, pa_all = [], []
    per_pair = {}
    for session_id in sessions:
        cache = load_cache(args.model, session_id)
        joints = torch.from_numpy(cache["joints_hmr"].astype(np.float32))  # (T,22,3)
        gt = kp_gt[session_id]
        n = min(joints.shape[0], gt.shape[0])
        src = joints[:n][:, [smpl for _, smpl in MATCH_PAIRS]]
        dst = gt[:n][:, [bvh for bvh, _ in MATCH_PAIRS]]
        # World translation alignment at the pelvis (joint 0), per sequence.
        src_root = src - src[:, :1]
        dst_root = dst - dst[:, :1]
        mpjpe_all.append(torch.linalg.vector_norm(src_root - dst_root, dim=-1))
        pa_all.append(torch.linalg.vector_norm(rigid_align(src_root, dst_root) - dst_root, dim=-1))
        for i, (bvh_idx, smpl_idx) in enumerate(MATCH_PAIRS):
            per_pair.setdefault(JOINT_NAMES[bvh_idx], []).append(
                torch.linalg.vector_norm(src_root[:, i] - dst_root[:, i], dim=-1)
            )

    mpjpe = torch.cat(mpjpe_all)
    pa = torch.cat(pa_all)
    result = {
        "checkpoint": "zero-training GVHMR (%s)" % args.model,
        "split": args.split,
        "n_pairs": len(MATCH_PAIRS),
        "MPJPE": float(mpjpe.mean()) * 1000.0,
        "PA-MPJPE": float(pa.mean()) * 1000.0,
        "per_joint_MPJPE": {k: float(torch.cat(v).mean()) * 1000.0 for k, v in sorted(per_pair.items())},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    out = args.out or REPO / "metrics" / "hmr_zero_val.json"
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print("-> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
