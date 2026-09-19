"""Small SMPL-NPZ adapter used by the results_display renderers."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anysole.data.smpl_io import load_smpl


def smpl_joints(path: Path, query_t: np.ndarray, paired_bvh: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if paired_bvh is None:
        candidate = Path(path).parent / "motion_smpl24_blender_world_m.bvh"
        paired_bvh = candidate if candidate.is_file() else None
    data = load_smpl(path, query_t=query_t, paired_bvh=paired_bvh)
    d6 = data["pose_6d"].reshape(-1, 23, 6)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    local = np.stack((b1, b2, b3), axis=-1)
    parents = np.asarray(data["parents"], dtype=np.int64)
    pos = np.zeros((len(d6), 23, 3), dtype=np.float64)
    grot = np.zeros_like(local, dtype=np.float64)
    for j, p in enumerate(parents):
        if p < 0:
            grot[:, j] = local[:, j]
            pos[:, j] = data["trans_m"]
        else:
            grot[:, j] = np.matmul(grot[:, p], local[:, j])
            pos[:, j] = pos[:, p] + np.matmul(grot[:, p], data["offsets_m"][j][:, None]).squeeze(-1)
    return pos, parents


def load_smpl_gt(seq_dir: Path, n_frames: int, fps: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    from anysole.data.dataset import resolve_bvh_path
    from anysole.data.smpl_io import resolve_smpl_path
    from anysole.types import SMPL_ROOTS
    t_grid = float(meta["visual_start_s"]) + np.arange(n_frames, dtype=np.float64) / float(fps)
    t_mocap = t_grid - float(meta["offset_s"])
    bvh = resolve_bvh_path(meta)
    try:
        p = resolve_smpl_path(meta, SMPL_ROOTS)
    except FileNotFoundError:
        # Converted-failure sessions can still be visualized through legacy BVH.
        from bvh_aligner_pose import parse_bvh_aligner, interp_joints, joints_to_meters
        parsed = parse_bvh_aligner(bvh, trim_leading_seconds=0.0)
        return joints_to_meters(interp_joints(parsed["joints"], parsed["frame_time"], t_mocap)), parsed["parents"]
    smpl_bvh = p.parent / "motion_smpl24_blender_world_m.bvh"
    return smpl_joints(p, t_mocap, smpl_bvh if smpl_bvh.is_file() else bvh)
