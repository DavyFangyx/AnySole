#!/usr/bin/env python3
"""Export PoseTransOpt V2M session archives (frame-id based).

Reads the PoseTransOpt work output (opt_result.pth + frame ids) and writes
the standard eval_motion archive:

    results/baselines/VP-MoCap/predictions/eval_motion/<session>.npz
    ├── joint_xyz_world   float32[n,24,3]
    ├── joint_names       <U24
    ├── valid_mask        bool[n]            optimized frames & shared-valid
    ├── frame_indices     int64[n]
    ├── vertices_world / poses / provenance
    └── ...

Frames are placed by their shared frame ids; the retired positional
``start=2`` reconstruction (which broke on fake gaps) is gone.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
RESULTS = REPO_ROOT / "results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

WORK_ROOT = WORKSPACE / "work/VP-MoCap/v1/pose_optimization"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
SMPL_NEUTRAL = WORKSPACE / "assets/third_party/smpl/SMPL_NEUTRAL.pkl"
OUTPUT_ROOT = RESULTS / "baselines/VP-MoCap/predictions/eval_motion"
SMPL_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)


def read_manifest() -> dict[str, dict]:
    rows = {}
    for line in (WORKSPACE / "protocol/manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def split_sessions(name: str) -> list[str]:
    cols = ("train", "val", "test") if name == "all" else (name,)
    result = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in cols)
    return sorted(set(x for x in result if x))


def session_parts(row: dict) -> tuple[str, str, str]:
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2], row["session_id"]


def shared_valid(row: dict) -> np.ndarray:
    date, subject, sid = session_parts(row)
    frames = np.load(FACTS_ROOT / date / subject / sid / "frames.npz")
    valid = np.asarray(frames["valid"], dtype=np.uint8).astype(bool)
    fake = np.asarray(frames["fake"], dtype=np.uint8).astype(bool)
    return valid & ~fake


def smpl_yup_to_display(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)


def export_session(row: dict, force: bool) -> str:
    sid = row["session_id"]
    date, subject, _ = session_parts(row)
    n = int(row["n_frames"])
    output = OUTPUT_ROOT / f"{sid}.npz"
    if output.is_file() and not force:
        return f"skip existing {output}"
    native_path = WORK_ROOT / date / subject / sid / "opt_result.pth"
    if not native_path.is_file():
        raise FileNotFoundError(f"PoseTransOpt output missing: {native_path}")

    import torch
    import smplx

    data = torch.load(native_path, map_location="cpu")
    pose = data["pose"].detach().cpu() if torch.is_tensor(data["pose"]) else torch.as_tensor(data["pose"])
    beta = data.get("beta", data.get("betas"))
    beta = beta.detach().cpu() if torch.is_tensor(beta) else torch.as_tensor(beta)
    trans = data["trans"].detach().cpu() if torch.is_tensor(data["trans"]) else torch.as_tensor(data["trans"])
    # the native result keeps float64 rotations from the Savitzky-Golay
    # post-filter; SMPL forward requires float32
    pose = pose.float()
    beta = beta.float()
    trans = trans.float()
    frame_ids = np.asarray(data["frame_ids"], dtype=np.int64).reshape(-1)
    if pose.ndim != 4 or tuple(pose.shape[1:]) != (24, 3, 3):
        raise ValueError(f"{native_path}: expected pose (T,24,3,3), got {tuple(pose.shape)}")
    if len(frame_ids) != len(pose):
        raise ValueError(f"{native_path}: frame_ids {len(frame_ids)} != pose rows {len(pose)}")
    if beta.ndim == 1:
        beta = beta.reshape(1, -1).repeat(len(pose), 1)

    model = smplx.create(str(SMPL_NEUTRAL), "smpl", gender="neutral",
                         batch_size=len(pose), num_betas=10)
    with torch.no_grad():
        result = model(
            betas=beta[:, :10],
            body_pose=pose[:, 1:],
            global_orient=pose[:, [0]],
            transl=trans,
            pose2rot=False,
        )
    native = result.joints[:, :24].detach().cpu().numpy()
    native_vertices = result.vertices.detach().cpu().numpy()
    from scipy.spatial.transform import Rotation
    poses_aa = Rotation.from_matrix(pose.numpy().reshape(-1, 3, 3)).as_rotvec().reshape(len(pose), 72)

    full = np.zeros((n, 24, 3), dtype=np.float32)
    full_vertices = np.zeros((n, 6890, 3), dtype=np.float32)
    full_poses = np.zeros((n, 72), dtype=np.float32)
    available = np.zeros(n, dtype=bool)
    order = np.searchsorted(frame_ids, np.arange(n, dtype=np.int64))
    for target in range(n):
        if order[target] < len(frame_ids) and frame_ids[order[target]] == target:
            k = order[target]
            full[target] = smpl_yup_to_display(native[k])
            full_vertices[target] = smpl_yup_to_display(native_vertices[k])
            full_poses[target] = poses_aa[k]
            available[target] = True

    valid = shared_valid(row) & available
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        joint_xyz_world=full,
        joint_names=np.asarray(SMPL_NAMES),
        valid_mask=valid,
        frame_indices=np.arange(n, dtype=np.int64),
        motion_protocol=np.asarray("smpl24"),
        model_units=np.asarray("m"),
        coordinate_system=np.asarray("world_z_up"),
        joint_coordinate_system=np.asarray("world_z_up"),
        session_id=np.asarray(sid),
        source_native_output=np.asarray(str(native_path)),
        provenance_source_type=np.asarray("method_optimization"),
        target_fps=np.asarray(40.0, dtype=np.float32),
        vertices_world=full_vertices,
        poses=full_poses,
    )
    return f"wrote {output} frames={int(valid.sum())}/{n}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = read_manifest()
    sessions = [s for s in args.sessions.split(",") if s] if args.sessions else split_sessions(args.split)
    for sid in sessions:
        print(export_session(rows[sid], args.force))


if __name__ == "__main__":
    main()
