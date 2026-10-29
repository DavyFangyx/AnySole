"""Decision archive 11: visual baseline export chain regression (no GPU, no package).

Run with Tactile pytest from AnySole, or call the test functions directly.
Covers the inverse calibration transforms, epoch estimation gates, canonical
resampling policy and the joint_rotations archive contract.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole.data.rgb_smpl.coordinates import (  # noqa: E402
    gt_smpl_to_checkerboard, camera_to_checkerboard,
    gt_root_rotation_to_checkerboard, camera_root_rotation_to_checkerboard,
)
from AnysoleWorkspace.tool.adapters.visual_baselines import common  # noqa: E402
from AnysoleWorkspace.tool.export_baseline_motion import write_archive  # noqa: E402


def synthetic_calibration(seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    gt_r = Rotation.random(random_state=rng).as_matrix()
    gt_t = rng.normal(size=3) * 40  # centimetres
    cam_r = Rotation.random(random_state=rng).as_matrix()
    cam_t = rng.normal(size=3) * 80
    calibration = {"mocap_raw_to_checkerboard_world": {"R": gt_r, "t": gt_t, "input_scale": 0.1}}
    camera = {"R": cam_r, "t": cam_t}
    return calibration, camera


def test_inverse_calibration_roundtrip_points():
    calibration, _ = synthetic_calibration()
    rng = np.random.default_rng(3)
    points = rng.normal(size=(50, 3))
    there = gt_smpl_to_checkerboard(points, calibration)
    back = common.checkerboard_to_gt_smpl(there, calibration)
    assert np.allclose(back, points, atol=1e-9)


def test_inverse_calibration_roundtrip_rotation():
    calibration, _ = synthetic_calibration()
    rng = np.random.default_rng(4)
    rotations = Rotation.random(20, random_state=rng).as_matrix()
    there = gt_root_rotation_to_checkerboard(rotations, calibration)
    back = common.checkerboard_to_gt_root_rotation(there, calibration)
    assert np.allclose(back, rotations, atol=1e-9)


def test_camera_chain_matches_forward_inverse():
    calibration, camera = synthetic_calibration(1)
    rng = np.random.default_rng(5)
    points = rng.normal(size=(30, 3))
    # camera -> checkerboard (forward), then inverse gt -> mocap world.
    checkerboard = camera_to_checkerboard(points, camera)
    expected = common.checkerboard_to_gt_smpl(checkerboard, calibration)
    assert np.allclose(common.camera_chain(points, camera, calibration), expected, atol=1e-9)


def test_root_rotation_chain_matches_forward_inverse():
    calibration, camera = synthetic_calibration(2)
    rng = np.random.default_rng(6)
    rotations = Rotation.random(15, random_state=rng).as_matrix()
    checkerboard = camera_root_rotation_to_checkerboard(rotations, camera)
    expected = common.checkerboard_to_gt_root_rotation(checkerboard, calibration)
    assert np.allclose(expected, common.checkerboard_to_gt_root_rotation(
        camera_root_rotation_to_checkerboard(rotations, camera), calibration), atol=1e-9)


def _monotone_stamps(count: int, rng: np.random.Generator, step_jitter: float) -> np.ndarray:
    # Real JPEG stamps: monotone, ~40 Hz intervals with small per-interval
    # jitter accumulating into a slow drift against the regular grid.
    increments = 0.025 + rng.normal(scale=step_jitter, size=count)
    return np.cumsum(increments)


def _unit(frame_numbers, times):
    from anysole.data.rgb_smpl.io import Unit
    return Unit(unit_id="20260810/method_1_offset/S9999/camera_3", date="20260810",
                method="method_1_offset", session="S9999", camera="camera_3", camera_index=3,
                frame_numbers=np.asarray(frame_numbers, dtype=np.int64),
                physical_times=np.asarray(times, dtype=np.float64))


def test_anchor_unit_wall_clock_no_gap():
    # Unit = canonical grid minus one tail frame, stamps sharing the
    # recording wall clock: the anchor returns the recording start verbatim.
    rng = np.random.default_rng(7)
    epoch = 75313.98
    visual = np.arange(325) / 40.0
    stamps = epoch + _monotone_stamps(324, rng, step_jitter=0.0008)
    unit = _unit(np.arange(1, 325), stamps)
    offset, stats = common.anchor_unit_to_canonical(unit, visual, epoch)
    assert offset == epoch
    assert stats["mad_s"] < 0.05
    assert stats["bracket_coverage"] > 0.97
    assert stats["span_mismatch_s"] < 0.1


def test_anchor_unit_head_gap_beyond_tolerance():
    # S13021 shape: the unit starts two canonical frames into the grid; the
    # head frame beyond the NN tolerance turns invalid, everything else
    # brackets normally — no heuristic, no shift search.
    rng = np.random.default_rng(11)
    epoch = 5000.0
    visual = np.arange(325) / 40.0
    stamps = epoch + visual[2:325] + rng.normal(scale=0.0005, size=323)
    unit = _unit(np.arange(1, 324), stamps)
    offset, stats = common.anchor_unit_to_canonical(unit, visual, epoch)
    assert offset == epoch
    assert stats["bracket_coverage"] > 0.97
    # The resample marks the far head frame invalid.
    _, valid, resample_stats = common.resample_on_canonical(
        {"scale": np.ones((323, 1))}, stamps, visual, epoch)
    assert valid.sum() == 324
    assert resample_stats["invalid_target_frames"] == 1


def test_anchor_unit_rejects_frame_rate_span():
    # 30 fps stamps: the span gate fires (seconds-level mismatch).
    unit = _unit(np.arange(1, 241), 8000.0 + np.arange(240) / 30.0)
    visual = np.arange(325) / 40.0
    try:
        common.anchor_unit_to_canonical(unit, visual, 8000.0)
    except ValueError as exc:
        assert "alignment failed gates" in str(exc)
    else:
        raise AssertionError("frame-rate span must fail the gates")


def test_anchor_unit_rejects_short_coverage():
    rng = np.random.default_rng(17)
    unit = _unit(np.arange(1, 101), 9000.0 + _monotone_stamps(100, rng, step_jitter=0.0008))
    visual = np.arange(325) / 40.0
    try:
        common.anchor_unit_to_canonical(unit, visual, 9000.0)
    except ValueError as exc:
        assert "alignment failed gates" in str(exc)
    else:
        raise AssertionError("short unit must fail the coverage gate")


def test_resample_slerp_linear_and_tail_fill():
    rng = np.random.default_rng(9)
    frames = 90
    epoch = 1000.0
    native_times = epoch + np.arange(frames) / 30.0
    visual = np.arange(120) / 40.0
    axes = Rotation.random(frames * 6, random_state=rng).as_rotvec().reshape(frames, 6, 3)
    scales = rng.normal(size=(frames, 1))
    resampled, valid, stats = common.resample_on_canonical(
        {"rot": axes, "scale": scales}, native_times, visual, epoch)
    assert resampled["rot"].shape == (120, 6, 3)
    assert resampled["scale"].shape == (120, 1)
    assert stats["tail_nn_filled"] > 0  # 120 targets beyond 90 native frames at 30 fps
    assert valid.sum() == 120
    assert np.isfinite(resampled["rot"]).all() and np.isfinite(resampled["scale"]).all()


def test_resample_tail_beyond_tolerance_stays_invalid():
    frames = 30
    epoch = 2000.0
    native_times = epoch + np.arange(frames) / 30.0
    visual = np.arange(60) / 40.0 + 20.0  # far beyond the source span
    _, valid, stats = common.resample_on_canonical(
        {"scale": np.ones((frames, 1))}, native_times, visual, epoch)
    assert stats["tail_nn_filled"] == 0
    assert not valid.all()
    assert stats["invalid_target_frames"] == 60


def test_write_archive_joint_rotations_roundtrip():
    rng = np.random.default_rng(10)
    joints = rng.normal(size=(40, 24, 3)).astype(np.float32)
    rotations = Rotation.random(40 * 24, random_state=rng).as_matrix().astype(np.float32).reshape(40, 24, 3, 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval_motion" / "S9999.npz"
        write_archive(path, joints, np.ones(40, dtype=bool), "test://", "S9999",
                      joint_rotations=rotations, extra_meta={"shape_source": "model_prediction"})
        with np.load(path, allow_pickle=False) as data:
            assert data["joint_rotations"].shape == (40, 24, 3, 3)
            assert np.allclose(data["joint_rotations"], rotations)
        # motion_io.load_rotations must consume the archive directly.
        from Baselines.utils.motion_io import load_rotations
        loaded = load_rotations(path)
        assert loaded["rotations"].shape == (40, 24, 3, 3)
        assert np.allclose(loaded["rotations"], rotations)


def test_write_archive_rejects_bad_rotation_shape():
    joints = np.zeros((4, 24, 3), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "S9998.npz"
        try:
            write_archive(path, joints, np.ones(4, dtype=bool), "test://", "S9998",
                          joint_rotations=np.zeros((4, 3, 3), dtype=np.float32))
        except ValueError as exc:
            assert "joint_rotations" in str(exc)
        else:
            raise AssertionError("malformed joint_rotations must be rejected")

def test_resample_and_align_duplicate_stamps():
    # Real units carry duplicate stamps (S13011: 16 zero-dt pairs of 324;
    # S13021: 24); the strictly-increasing repair must keep bracket_times
    # well-defined and the resample must stay finite and fully valid.
    rng = np.random.default_rng(23)
    epoch = 7000.0
    stamps = epoch + _monotone_stamps(324, rng, step_jitter=0.0005)
    for index in (7, 40, 111, 230):
        stamps[index] = stamps[index - 1]  # inject duplicate pairs
    assert (np.diff(stamps) == 0).any()  # duplicates really present
    repaired = common.strictly_increasing(stamps)
    assert np.all(np.diff(repaired) > 0)
    visual = np.arange(325) / 40.0
    resampled, valid, stats = common.resample_on_canonical(
        {"scale": np.ones((324, 1))}, stamps, visual, epoch)
    assert valid.sum() == 325
    assert np.isfinite(resampled["scale"]).all()

