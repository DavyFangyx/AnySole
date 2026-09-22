"""Independent checks for the native SMPL-24 prediction archive contract.

The loader uses pelvis-root translations internally, while a standard SMPL
archive stores model-origin ``trans``.  This probe writes a short real clip,
reloads it through the public readers, and checks both representations plus
axis-angle/6D/FK consistency.  It intentionally also inspects the raw NPZ so
an encode/decode pair cannot hide a duplicated pelvis offset.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for entry in (REPO_ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from anysole.data.smpl_io import (  # noqa: E402
    load_smpl,
    pelvis_to_smpl_trans,
    smpl24_pose6d_to_poses,
    smpl_archive_metadata,
)
from anysole.utils.geometry import fk_pose6d_np, rot6d_to_rotmat_np  # noqa: E402
from anysole.types import (  # noqa: E402
    JOINT_PROTOCOL_CHECKSUM,
    MOTION_PROTOCOL,
    N_JOINTS,
    SMPL_ROOTS,
)
from utils.motion_io import detect_motion_format  # noqa: E402


def _first_source() -> Path:
    for root in SMPL_ROOTS:
        paths = sorted(Path(root).glob("**/motion_neutral_smpl.npz"))
        if paths:
            return paths[0]
    raise FileNotFoundError("no source SMPL archive found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()

    source = args.source or _first_source()
    motion = load_smpl(source)
    n = min(max(int(args.frames), 1), len(motion["pose_6d"]))
    pose = motion["pose_6d"][:n]
    pelvis = motion["trans_m"][:n]
    betas = motion["betas"]
    poses_aa = smpl24_pose6d_to_poses(pose)
    model_trans = pelvis_to_smpl_trans(pose, pelvis, betas)

    with tempfile.TemporaryDirectory(prefix="anysole-smpl-export-") as tmp:
        out = Path(tmp) / "prediction.npz"
        np.savez_compressed(
            out,
            poses=poses_aa,
            trans=model_trans,
            betas=betas,
            root_orient=poses_aa[:, :3],
            pose_body=poses_aa[:, 3:],
            mocap_frame_rate=np.asarray(motion["fps"], dtype=np.float32),
            source_frame_times_s=np.arange(n, dtype=np.float32) / float(motion["fps"]),
            **smpl_archive_metadata(),
        )
        if detect_motion_format(out) != "smpl":
            raise AssertionError("automatic motion-format detection did not select SMPL")
        with np.load(out, allow_pickle=True) as raw:
            if raw["poses"].shape != (n, N_JOINTS * 3):
                raise AssertionError(f"raw poses shape is {raw['poses'].shape}")
            if raw["pose_body"].shape != (n, (N_JOINTS - 1) * 3):
                raise AssertionError(f"raw pose_body shape is {raw['pose_body'].shape}")
            np.testing.assert_allclose(raw["trans"], model_trans, atol=1e-7)
            np.testing.assert_allclose(raw["betas"], betas, atol=0.0)
            if str(raw["motion_protocol"]) != MOTION_PROTOCOL:
                raise AssertionError(f"motion_protocol metadata is not {MOTION_PROTOCOL}")
            if str(raw["joint_protocol_checksum"]) != JOINT_PROTOCOL_CHECKSUM:
                raise AssertionError("joint_protocol_checksum metadata does not match names/parents")

        restored = load_smpl(out)
        np.testing.assert_allclose(restored["model_trans_m"], model_trans, atol=1e-6)
        np.testing.assert_allclose(restored["trans_m"], pelvis, atol=1e-6)
        np.testing.assert_allclose(restored["betas"], betas, atol=0.0)
        r0 = rot6d_to_rotmat_np(pose.reshape(n, N_JOINTS, 6))
        r1 = rot6d_to_rotmat_np(restored["pose_6d"].reshape(n, N_JOINTS, 6))
        rot_max = float(np.max(np.abs(r0 - r1)))
        joints0 = fk_pose6d_np(pose, pelvis, motion["offsets_m"], motion["parents"])
        joints1 = fk_pose6d_np(
            restored["pose_6d"], restored["trans_m"],
            restored["offsets_m"], restored["parents"],
        )
        fk_max = float(np.max(np.abs(joints0 - joints1)))
        if rot_max > 2e-6 or fk_max > 2e-6:
            raise AssertionError(f"export round-trip drift: rot={rot_max:g}, FK={fk_max:g}m")

        # Tensor width alone cannot identify the joint language.  An early
        # migration artifact may be 24/72-D but use a different parent tree.
        # Generated AnySole files which declare SMPL-24 must therefore carry
        # the exact names/parents checksum.  Raw MoSh inputs predate these
        # AnySole fields and remain valid without them.
        missing_checksum = Path(tmp) / "generated_without_tree_checksum.npz"
        np.savez_compressed(
            missing_checksum,
            poses=poses_aa,
            trans=model_trans,
            betas=betas,
            motion_protocol=np.asarray(MOTION_PROTOCOL),
        )
        try:
            load_smpl(missing_checksum)
        except ValueError as exc:
            if "joint checksum=missing" not in str(exc):
                raise AssertionError(f"unexpected missing-checksum error: {exc}") from exc
        else:
            raise AssertionError("generated SMPL archive without tree checksum was accepted")

        with np.load(source, allow_pickle=True) as raw_source:
            if "motion_protocol" not in raw_source.files:
                load_smpl(source)

    print(f"source={source}")
    print(f"frames={n} poses={poses_aa.shape} pose_body={(n, (N_JOINTS - 1) * 3)}")
    print(f"model_trans/pelvis/rest-offset contract: PASS (max FK drift={fk_max:.3e} m)")
    print(f"axis-angle/6D rotation contract: PASS (max matrix drift={rot_max:.3e})")
    print("format detection + metadata + betas + protocol rejection: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
