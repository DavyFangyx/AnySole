"""Shared chain: package unit -> canonical 40 Hz frames -> mocap y-up world.

Geometry follows the validated anysole_aligned chain in
``AnysoleWorkspace/tool/eval_rgb_smpl.py`` (camera -> checkerboard world for
both prediction and GT), then inverts the date calibration's
``mocap_raw_to_checkerboard_world`` transform so archives are expressed in the
project's mocap y-up world (the GT SMPL frame), and rebuilds joints with the
project protocol FK (``fk_pose6d_np`` + GT offsets/parents, drift point p).

Time axis: package units carry physical seconds-of-day timestamps; the project
canonical grid is ``shared/facts/.../frames.npz visual_time_s`` (seconds from
recording start).  An epoch offset K is estimated by scan; native-rate SMPL
parameters are resampled onto the canonical grid with SO(3) slerp (rotations)
and linear interpolation (translations/betas/scales).  Target frames beyond the
source span are nearest-neighbour filled within 0.04 s and counted; anything
worse marks the frame invalid.  A session whose alignment does not reach
MAD < 0.02 s or bracket coverage >= 0.97 fails loudly (decision archive 11).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from anysole.data.rgb_smpl import alignment as _alignment  # noqa: E402
from anysole.data.rgb_smpl import coordinates as _coords  # noqa: E402
from anysole.data.rgb_smpl import io as _io  # noqa: E402
from anysole.utils.geometry import fk_pose6d_np, rotmat_to_6d_np  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

WORKSPACE = ROOT / "AnysoleWorkspace"
FACTS = WORKSPACE / "shared" / "facts" / "sessions"
SMPL_MODEL = WORKSPACE / "assets/third_party" / "smpl" / "SMPL_NEUTRAL.pkl"
CALIBRATION_ALT = Path("/data/lizhe/projects/Tactile/4_Dataset/config/calibration")

MAD_LIMIT_S = 0.02
COVERAGE_LIMIT = 0.97
TAIL_TOLERANCE_S = 0.04
# Linear residual ramp over the mapped span: real clock drift stays under
# ~50 ms (S13011 max residual 33 ms); a frame-rate mismatch (30 fps CFR)
# produces seconds-level ramps.  The gate sits in the empty middle.
EPOCH_RAMP_LIMIT_S = 0.1


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def session_parts(row: dict) -> tuple[str, str]:
    """(date, subject) from the manifest ``video_path`` URI."""
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2]


def load_calibration(date: str) -> tuple[dict, str, str]:
    """Project calibration JSON first, absolute upstream path as fallback."""
    candidates = [WORKSPACE / "protocol/calibration" / f"{date}.json",
                  CALIBRATION_ALT / f"{date}.json"]
    for path in candidates:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if "cameras" in value and "mocap_raw_to_checkerboard_world" in value:
                return value, str(path), sha256(path)
    raise FileNotFoundError(
        f"calibration for {date} not found under {candidates[0]} or {candidates[1]}")


def load_unit(package_root: str | Path, baseline: str, date: str, session: str, cam: int,
              manifest_override: str | Path | None = None) -> dict:
    """Read the package unit manifest entry for one method-camera unit.

    ``manifest_override`` names the shared per-unit manifest.jsonl (the
    package keeps one master copy under SAM3DB_baseline/.run/) so exports can
    run while other baseline record archives are still restoring.
    """
    manifest = Path(manifest_override) if manifest_override else (
        Path(package_root) / "reference_results" / "3_Result" / f"{baseline}_baseline" / ".run" / "manifest.jsonl")
    unit_id = f"{date}/method_1_offset/{session}/camera_{cam}"
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value.get("unit_id") == unit_id:
                if value.get("complete") not in (True, "true", 1):
                    raise ValueError(f"package unit incomplete: {unit_id}")
                return value
    raise FileNotFoundError(f"package unit not in manifest: {unit_id} ({manifest})")


def unit_object(entry: dict):
    """Strict Unit built by the rgb_smpl reader (reuses its validation)."""
    return _io._unit_from_json(entry)


def canonical_grid(row: dict, cam: int) -> np.ndarray:
    """Canonical 40 Hz visual timestamps for one session-camera."""
    date, subject = session_parts(row)
    frames = np.load(FACTS / f"cam{cam}" / date / subject / row["session_id"] / "frames.npz",
                     allow_pickle=False)
    times = np.asarray(frames["visual_time_s"], dtype=np.float64)
    n = int(float(row["n_frames"]))
    if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{row['session_id']}: canonical visual_time_s does not match manifest")
    return times


def align_unit_to_canonical(unit, visual_times: np.ndarray) -> tuple[float, dict]:
    """Pin unit frames onto the canonical grid via the unit's frame numbers.

    Timestamps alone cannot decide frame parity (a one-frame slip is absorbed
    into the epoch constant), so the mapping uses the package frame numbers
    directly: the package names frames 1-based (``frame1_...``) while the
    canonical ``frame_id`` is 0-based, i.e. unit frame ``min+k`` maps to
    canonical frame ``k``.  The epoch is the median stamp-grid residual over
    that fixed mapping; a linear ramp in the residuals (frame-rate mismatch)
    or MAD/coverage beyond the gates fails loudly (decision archive 11, risk 7).
    """
    times = _alignment.unwrap_seconds_of_day(unit.physical_times)
    visual = np.asarray(visual_times, dtype=np.float64)
    origin = int(unit.frame_numbers.min())
    indices = unit.frame_numbers - origin
    if indices.size != times.size or np.any(np.diff(indices) != 1) \
            or indices[0] < 0 or indices[-1] >= visual.size:
        raise ValueError(f"{unit.unit_id}: frame numbers do not map contiguously "
                         f"onto the canonical grid (origin {origin}, "
                         f"range {indices[0]}..{indices[-1]} of {visual.size})")
    residual = times - visual[indices]
    offset = float(np.median(residual))
    mad = float(np.median(np.abs(residual - offset)))
    ramp = float(np.abs(np.polyfit(indices, residual, 1)[0]) * (indices[-1] - indices[0]))
    brackets = _alignment.bracket_times(times - offset, visual)
    coverage = float(brackets.valid.mean())
    stats = {"epoch_offset_s": offset, "frame_origin": origin,
             "mapped_frames": int(indices.size), "mad_s": mad,
             "residual_ramp_s": ramp, "bracket_coverage": coverage,
             "max_residual_s": float(np.max(np.abs(residual - offset)))}
    if mad > MAD_LIMIT_S or coverage < COVERAGE_LIMIT or ramp > EPOCH_RAMP_LIMIT_S:
        raise ValueError(f"visual epoch alignment failed gates: {stats}")
    return offset, stats


def resample_on_canonical(spec: dict[str, np.ndarray], unit_times: np.ndarray,
                          visual_times: np.ndarray, epoch: float) -> tuple[dict[str, np.ndarray],
                                                                           np.ndarray, dict]:
    """Resample native-rate fields onto the canonical grid.

    ``spec`` maps field name -> native array; a trailing axis of size 3 is
    treated as SO(3) (slerp via ``interpolate_axis_angle``), everything else
    linear.  Target frames beyond the source span get a nearest-neighbour fill
    within ``TAIL_TOLERANCE_S`` and are counted; the rest stay invalid.
    """
    source = _alignment.unwrap_seconds_of_day(unit_times) - epoch
    target = np.asarray(visual_times, dtype=np.float64)
    brackets = _alignment.bracket_times(source, target)
    valid = brackets.valid.copy()
    count = {name: np.asarray(value, dtype=np.float64) for name, value in spec.items()}
    if not count:
        raise ValueError("resample spec is empty")
    frames = next(iter(count.values())).shape[0]
    if any(value.shape[0] != frames for value in count.values()):
        raise ValueError("resample spec fields must share the native frame count")
    # Tail fill: targets beyond the last source sample (and before the first),
    # nearest neighbour within tolerance.
    tail_count = 0
    for index in np.flatnonzero(~valid):
        t = target[index]
        if t >= source[-1] and t - source[-1] <= TAIL_TOLERANCE_S:
            valid[index] = True
            tail_count += 1
        elif t <= source[0] and source[0] - t <= TAIL_TOLERANCE_S:
            valid[index] = True
            tail_count += 1
    out: dict[str, np.ndarray] = {}
    for name, values in count.items():
        result = np.zeros((target.size,) + values.shape[1:], dtype=np.float64)
        interior = brackets.valid
        if interior.any():
            subset = _alignment.bracket_times(source, target[interior])
            if not subset.valid.all():
                raise ValueError(f"{name}: interior bracket set invalid")
            if values.shape[-1] == 3:
                result[interior] = _alignment.interpolate_axis_angle(values, subset)
            else:
                result[interior] = _alignment.interpolate_linear(values, subset)
        # Tail fill uses the raw native samples (nearest neighbour).
        for index in np.flatnonzero(valid & ~brackets.valid):
            nearest = np.searchsorted(source, target[index])
            nearest = int(np.clip(nearest, 0, source.size - 1))
            result[index] = values[nearest]
        out[name] = result
    stats = {"tail_nn_filled": tail_count, "invalid_target_frames": int((~valid).sum())}
    return out, valid, stats


def checkerboard_to_gt_smpl(points_m: np.ndarray, calibration: dict) -> np.ndarray:
    """Inverse of ``rgb_smpl.coordinates.gt_smpl_to_checkerboard`` (m -> m)."""
    points = np.asarray(points_m, dtype=np.float64)
    transform = calibration["mocap_raw_to_checkerboard_world"]
    rotation = np.asarray(transform["R"], dtype=np.float64)
    translation_cm = np.asarray(transform["t"], dtype=np.float64)
    return np.einsum("ji,...j->...i", rotation, points * 100.0 - translation_cm) / 100.0


def checkerboard_to_gt_root_rotation(rotation: np.ndarray, calibration: dict) -> np.ndarray:
    """Inverse of ``gt_root_rotation_to_checkerboard`` for the global orientation."""
    fixed = np.asarray(calibration["mocap_raw_to_checkerboard_world"]["R"], dtype=np.float64)
    return np.einsum("ji,...jk->...ik", fixed, np.asarray(rotation, dtype=np.float64))


def camera_chain(points_m: np.ndarray, camera: dict, calibration: dict) -> np.ndarray:
    """Native camera-space points -> mocap y-up world (eval chain + inverse)."""
    checkerboard = _coords.camera_to_checkerboard(points_m, camera)
    return checkerboard_to_gt_smpl(checkerboard, calibration)


def protocol_assembly(rotations: np.ndarray, transl: np.ndarray,
                      gt_motion: dict) -> tuple[np.ndarray, np.ndarray]:
    """Protocol FK joints (mocap y-up world) from world rotations + root translation.

    ``rotations`` are (T,24,3,3) with channel 0 already in mocap y-up world;
    body channels are basis-independent body-local rotations (same convention
    as GT ``poses``).  Joints are rebuilt with the GT chain's offsets/parents
    so they match ``protocol_gt`` exactly (drift point p).
    """
    pose_6d = rotmat_to_6d_np(rotations).reshape(rotations.shape[0], -1)
    joints = fk_pose6d_np(pose_6d, transl, gt_motion["offsets_m"], gt_motion["parents"])
    poses = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().reshape(
        rotations.shape[0], 72)
    return joints, poses


def smpl_forward(poses: np.ndarray, betas: np.ndarray,
                 trans: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Chunked CPU neutral-SMPL forward shared by the three exporters."""
    from anysole.data.rgb_smpl.model import SMPLForward

    model = SMPLForward(SMPL_MODEL, device="cpu", batch_size=256)
    return model.forward(poses, betas, trans)


def export_meta(unit, result_path: Path, calibration: dict, calib_path: str, calib_sha: str,
                epoch_stats: dict, resample_stats: dict, replay: float) -> dict:
    """Provenance block shared by the three exporters (decision archive 11)."""
    return {
        "surface_source": "model_prediction",
        "shape_source": "model_prediction",
        "public_surface_metrics": "true",
        "coordinate_transform": "camera_to_checkerboard_to_mocap_world",
        "coordinate_transform_source": calib_path,
        "calibration_sha256": calib_sha,
        "package_unit_id": unit.unit_id,
        "package_result_sha256": sha256(result_path),
        "time_resample": f"epoch_s={epoch_stats['epoch_offset_s']:.6f};"
                         f"mad_s={epoch_stats['mad_s']:.6f};"
                         f"coverage={epoch_stats['bracket_coverage']:.6f};"
                         f"tail_nn={resample_stats['tail_nn_filled']};"
                         f"invalid={resample_stats['invalid_target_frames']}",
        "geometry_replay_rms_mm": float(replay),
    }
