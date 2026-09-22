"""Protocol sanity probe for the AnySole SMPL-24 migration.

This is deliberately independent of training.  It catches the failure mode
where an SMPL file is merely wrapped in the old 23-joint representation:
shapes, parent tree, pelvis translation, FK foot heights, rotation round-trip,
and mean/identity/random pose baselines are printed for representative files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import glob
import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from anysole.data.smpl_io import load_smpl, smpl_archive_metadata
from anysole.utils.geometry import fk_pose6d_np, rot6d_to_rotmat_np, rotmat_to_6d_np
from anysole.types import JOINT_NAMES, JOINT_PARENTS, N_JOINTS, POSE_DIM, SMPL_ROOTS


# The first SMPL migration accidentally attached both collars to neck (12)
# instead of spine3 (9).  Keep that tree only inside this diagnostic so the
# numerical effect of the fixed protocol remains visible and regression-tested.
LEGACY_WRONG_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)


def _find(session: str) -> Path:
    for root in SMPL_ROOTS:
        hit = sorted(Path(root).glob(f"**/{session}/motion_neutral_smpl.npz"))
        if hit:
            return hit[0]
    raise FileNotFoundError(session)


def probe(path: Path) -> None:
    raw = dict(np.load(path, allow_pickle=True))
    motion = load_smpl(path)
    pose = motion["pose_6d"]
    if pose.shape[1] != POSE_DIM or POSE_DIM != 24 * 6:
        raise AssertionError(f"native pose shape is {pose.shape}, expected (*,{24*6})")
    if tuple(motion["parents"].tolist()) != tuple(JOINT_PARENTS):
        raise AssertionError("SMPL parent tree mismatch")
    mats = rot6d_to_rotmat_np(pose.reshape(-1, N_JOINTS, 6))
    roundtrip = rotmat_to_6d_np(mats).reshape(pose.shape)
    rot_err = np.abs(roundtrip - pose).max()
    kp = fk_pose6d_np(pose, motion["trans_m"], motion["offsets_m"], motion["parents"])
    # Independent standard-SMPL rigid-chain check.  This is intentionally not
    # built from AnySole's FK helper: it catches a wrong interpretation of the
    # archive's model-origin ``trans`` (in particular rotating the pelvis rest
    # offset by the root orientation).
    aa = np.asarray(raw["poses"], dtype=np.float64).reshape(-1, N_JOINTS, 3)
    rest = np.asarray(motion["joint_rest_m"], dtype=np.float64)
    standard = np.empty_like(kp, dtype=np.float64)
    for t in range(len(aa)):
        local = SciRotation.from_rotvec(aa[t]).as_matrix()
        global_r = np.empty_like(local)
        for j, parent in enumerate(JOINT_PARENTS):
            if parent < 0:
                global_r[j] = local[j]
                standard[t, j] = np.asarray(raw["trans"], dtype=np.float64)[t] + rest[j]
            else:
                global_r[j] = global_r[parent] @ local[j]
                standard[t, j] = standard[t, parent] + global_r[parent] @ (rest[j] - rest[parent])
    smpl_fk_err = float(np.max(np.abs(standard - kp)))
    if smpl_fk_err > 1e-5:
        raise AssertionError(f"standard SMPL FK mismatch {smpl_fk_err:g} m")
    wrong_offsets = np.zeros_like(rest)
    for joint, parent in enumerate(LEGACY_WRONG_PARENTS):
        if parent >= 0:
            wrong_offsets[joint] = rest[joint] - rest[parent]
    wrong_kp = fk_pose6d_np(
        pose, motion["trans_m"], wrong_offsets,
        np.asarray(LEGACY_WRONG_PARENTS, dtype=np.int64),
    )
    wrong_tree_mpjpe = float(np.linalg.norm(wrong_kp - kp, axis=-1).mean() * 1000.0)
    wrong_tree_arm = float(np.linalg.norm(wrong_kp[:, 13:] - kp[:, 13:], axis=-1).mean() * 1000.0)
    # Identity rotations with the same pelvis trajectory are a useful lower
    # bound; a shuffled/random pose is intentionally much worse.
    identity = np.zeros_like(pose)
    identity[:, 0::6] = 1.0
    identity[:, 4::6] = 1.0
    kp_identity = fk_pose6d_np(identity, motion["trans_m"], motion["offsets_m"], motion["parents"])
    mean_pose = np.broadcast_to(np.mean(mats, axis=0), mats.shape)
    mean_pose6d = rotmat_to_6d_np(mean_pose).reshape(pose.shape)
    kp_mean = fk_pose6d_np(mean_pose6d, motion["trans_m"], motion["offsets_m"], motion["parents"])
    identity_err = np.linalg.norm(kp_identity - kp, axis=-1).mean() * 1000.0
    mean_err = np.linalg.norm(kp_mean - kp, axis=-1).mean() * 1000.0
    print(f"{path.parent.name}: pose={pose.shape} offsets={motion['offsets_m'].shape} "
          f"root_y={motion['trans_m'][:,1].mean():.3f}m "
          f"foot_y=[{kp[:,(10,11),1].min():.3f},{kp[:,(10,11),1].max():.3f}]m "
          f"rot6d_roundtrip={rot_err:.2e} smpl_fk={smpl_fk_err:.2e}m "
          f"wrong_parent={wrong_tree_mpjpe:.1f}mm(all)/{wrong_tree_arm:.1f}mm(j13+) "
          f"identity={identity_err:.1f}mm mean={mean_err:.1f}mm")


def scan_archives() -> None:
    paths = sorted({
        path
        for root in SMPL_ROOTS
        for path in Path(root).glob("**/motion_neutral_smpl.npz")
    })
    if not paths:
        raise FileNotFoundError(f"no SMPL archives under {SMPL_ROOTS}")
    model_types, genders, units, coordinates, rates, source_hashes = set(), set(), set(), set(), set(), set()
    pose_shapes, body_shapes = set(), set()
    terminal_max = root_diff = body_diff = 0.0
    nonfinite = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as raw:
            scalar = lambda key, default="": str(np.asarray(raw[key]).reshape(-1)[0]) if key in raw else default
            model_types.add(scalar("surface_model_type"))
            genders.add(scalar("gender"))
            units.add(scalar("model_units"))
            coordinates.add(scalar("coordinate_system"))
            rates.add(float(np.asarray(raw.get("mocap_frame_rate", 120.0)).reshape(-1)[0]))
            source_hashes.add(scalar("surface_model_sha256"))
            poses = np.asarray(raw["poses"], dtype=np.float64)
            body = np.asarray(raw["pose_body"], dtype=np.float64)
            root = np.asarray(raw["root_orient"], dtype=np.float64)
            trans = np.asarray(raw["trans"], dtype=np.float64)
            pose_shapes.add(poses.shape[1:])
            body_shapes.add(body.shape[1:])
            nonfinite += int(not (
                np.isfinite(poses).all() and np.isfinite(body).all()
                and np.isfinite(root).all() and np.isfinite(trans).all()
            ))
            terminal_max = max(terminal_max, float(np.abs(poses[:, -6:]).max(initial=0.0)))
            root_diff = max(root_diff, float(np.abs(poses[:, :3] - root).max(initial=0.0)))
            body_diff = max(body_diff, float(np.abs(poses[:, 3:66] - body).max(initial=0.0)))
    local_hash = str(np.asarray(smpl_archive_metadata()["surface_model_sha256"]).item())
    print(
        "archive scan: files=%d model_type=%s gender=%s units=%s fps=%s "
        "pose_tail=%s pose_body=%s nonfinite_files=%d" % (
            len(paths), sorted(model_types), sorted(genders), sorted(units), sorted(rates),
            sorted(pose_shapes), sorted(body_shapes), nonfinite,
        )
    )
    print(f"coordinate_system={sorted(coordinates)}")
    print(
        f"redundant fields: root_max_diff={root_diff:.2e} "
        f"body_max_diff={body_diff:.2e} terminal_hand_rotation_max={terminal_max:.2e}"
    )
    print(f"source_model_sha256={sorted(source_hashes)}")
    print(f"local_model_sha256={local_hash}")
    if source_hashes != {local_hash}:
        print(
            "PROVENANCE: UNVERIFIED_NUMERICAL_EQUIVALENCE — source/local model file "
            "hashes differ; this may be serialization-only, but array equality has not been proven."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="*", default=["S12011", "S7063", "S7013", "S5011"])
    parser.add_argument("--skip-archive-scan", action="store_true")
    args = parser.parse_args()
    print(f"protocol: N_JOINTS={N_JOINTS}, POSE_DIM={POSE_DIM}, parents={JOINT_PARENTS}")
    for session in args.sessions:
        try:
            probe(_find(session))
        except FileNotFoundError:
            print(f"{session}: NOT_FOUND")
    if not args.skip_archive_scan:
        scan_archives()


if __name__ == "__main__":
    main()
