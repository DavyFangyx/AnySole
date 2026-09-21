"""Independent SMPL/BVH coordinate-axis audit on semantically matched joints.

AnySole does not train on BVH.  The old capture BVHs are nevertheless useful
as an independent witness for the display transform: this probe resamples both
files on exactly the manifest time grid, aligns only explicitly named common
joints, removes root translation, and scores every signed axis permutation.
It therefore catches an X/Z swap or sign/handedness error that an SMPL-only
encode/decode round-trip cannot expose.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from evaluate_compare import bvh_joints, protocol_gt, select_common_joints
from utils.motion_io import LEGACY_BVH_NAMES, load_motion, smpl_yup_to_display


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "AnysoleWorkspace/manifests/session_manifest.jsonl"


def signed_permutations() -> list[tuple[str, np.ndarray]]:
    out = []
    axes = "xyz"
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for output_axis, input_axis in enumerate(perm):
                matrix[output_axis, input_axis] = signs[output_axis]
            label = "(" + ",".join(
                ("+" if signs[i] > 0 else "-") + axes[perm[i]] for i in range(3)
            ) + ")"
            out.append((label, matrix))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=0, help="0 audits every eligible session")
    parser.add_argument("--frame-stride", type=int, default=5)
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if str(row.get("eligible_anysole", "0")) == "1"]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("manifest contains no eligible AnySole sessions")

    candidates = signed_permutations()
    sums = np.zeros(len(candidates), dtype=np.float64)
    counts = np.zeros(len(candidates), dtype=np.int64)
    display_sum = 0.0
    display_count = 0
    reader_sum = 0.0
    reader_count = 0
    sessions = 0
    for row in rows:
        smpl_display, smpl_names = protocol_gt(row, "smpl24")
        bvh = bvh_joints(Path(row["bvh_path"]), row)
        smpl = select_common_joints(smpl_display, smpl_names, "smpl24")
        bvh = select_common_joints(bvh, tuple(LEGACY_BVH_NAMES), "bvh23")
        n = min(len(smpl), len(bvh))
        stride = max(int(args.frame_stride), 1)
        smpl = smpl[:n:stride] - smpl[:n:stride, :1]
        bvh = bvh[:n:stride] - bvh[:n:stride, :1]

        # protocol_gt already applied the current Y-up -> display transform.
        # Undo it so all candidate matrices start from native SMPL coordinates.
        native = np.stack((smpl[..., 0], smpl[..., 2], -smpl[..., 1]), axis=-1)
        for i, (_, matrix) in enumerate(candidates):
            transformed = np.einsum("ij,tkj->tki", matrix, native)
            error = np.linalg.norm(transformed - bvh, axis=-1)
            sums[i] += float(error.sum())
            counts[i] += int(error.size)

        # The renderer intentionally uses Z-up. Applying the same proper
        # rotation to both native protocols must preserve their discrepancy.
        smpl_zup = smpl_yup_to_display(native)
        bvh_zup = smpl_yup_to_display(bvh)
        display_error = np.linalg.norm(smpl_zup - bvh_zup, axis=-1)
        display_sum += float(display_error.sum())
        display_count += int(display_error.size)

        # Exercise the actual auto-detect display reader as a separate path.
        query_t = (
            float(row["visual_start_s"])
            + np.arange(int(row["n_frames"]), dtype=np.float64) / float(row["target_fps"])
            - float(row["offset_s"])
        )
        bvh_reader = load_motion(Path(row["bvh_path"]), query_t=query_t)
        bvh_reader = select_common_joints(
            bvh_reader["joints"], tuple(bvh_reader["names"]), "bvh23"
        )[:n:stride]
        bvh_reader = bvh_reader - bvh_reader[:, :1]
        reader_error = np.linalg.norm(smpl_zup - bvh_reader, axis=-1)
        reader_sum += float(reader_error.sum())
        reader_count += int(reader_error.size)
        sessions += 1

    mean_mm = sums / np.maximum(counts, 1) * 1000.0
    order = np.argsort(mean_mm)
    native_identity_label = "(+x,+y,+z)"
    native_identity_i = next(
        i for i, (label, _) in enumerate(candidates) if label == native_identity_label
    )
    native_identity_rank = int(np.where(order == native_identity_i)[0][0]) + 1
    native_identity_matrix = candidates[native_identity_i][1]
    print(
        f"sessions={sessions} stride={max(int(args.frame_stride), 1)} "
        f"common_joints=19 candidates={len(candidates)}"
    )
    print(
        f"native_identity={native_identity_label} det={np.linalg.det(native_identity_matrix):+.0f} "
        f"root_relative_MPJPE={mean_mm[native_identity_i]:.3f}mm "
        f"rank={native_identity_rank}/{len(candidates)}"
    )
    print(
        "display=(+x,-z,+y) det=+1 "
        f"paired_transform_MPJPE={display_sum / max(display_count, 1) * 1000.0:.3f}mm "
        f"auto_reader_MPJPE={reader_sum / max(reader_count, 1) * 1000.0:.3f}mm"
    )
    print("best signed permutations:")
    for index in order[:8]:
        label, matrix = candidates[index]
        print(
            f"  {label:12s} det={np.linalg.det(matrix):+.0f} "
            f"MPJPE={mean_mm[index]:.3f}mm"
        )
    if native_identity_rank != 1:
        raise AssertionError(
            f"native SMPL/BVH axes do not agree: identity ranks {native_identity_rank}; "
            f"best is {candidates[order[0]][0]}"
        )
    display_mm = display_sum / max(display_count, 1) * 1000.0
    if abs(display_mm - mean_mm[native_identity_i]) > 1e-6:
        raise AssertionError("common display rotation changed root-relative distance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
