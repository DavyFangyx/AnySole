"""Protocol regression checks; run with Tactile pytest from AnySole."""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole.utils.vision_smpl_metrics import evaluate_motion, evaluate_formula_control
from AnysoleWorkspace.tool.eval_rgb_smpl import aggregate, helpers


def motion():
    rng = np.random.default_rng(1)
    joints = np.broadcast_to(rng.normal(size=(1, 24, 3)), (110, 24, 3)).copy()
    return {"joints": joints, "vertices": np.zeros((110, 6890, 3)),
            "rotations": np.broadcast_to(np.eye(3), (110, 24, 3, 3)).copy(),
            "poses": np.zeros((110, 72))}, np.arange(110) / 40


def test_identity_and_removed_metric():
    target, times = motion()
    values = evaluate_motion(target, target, times)
    for metric in ("mpjpe_mm", "pa_mpjpe_mm", "mpjae_deg", "pve_mm", "accel_error_m_s2",
                   "jitter_pred_m_s3", "root_ate_mm", "w_mpjpe100_mm", "foot_sliding_mm"):
        assert abs(values[metric]) < 1e-8
    assert values["pve_t_mm"] is None


def test_native_rotation_includes_root():
    target, times = motion()
    predicted = {**target, "rotations": target["rotations"].copy()}
    predicted["rotations"][:, 0] = Rotation.from_euler("y", 24, degrees=True).as_matrix()
    assert np.isclose(evaluate_motion(predicted, target, times)["mpjae_deg"], 1)
    control = evaluate_formula_control(predicted, target, times, np.ones(times.size, bool))
    assert control["mpjae_deg"] == 0


def test_world_acceleration_is_retained():
    target, times = motion()
    predicted = {**target, "joints": target["joints"].copy()}
    predicted["joints"][:, :, 0] += times[:, None] ** 2
    native = evaluate_motion(predicted, target, times)
    control = evaluate_formula_control(predicted, target, times, np.ones(times.size, bool))
    assert np.isclose(native["accel_error_m_s2"], 2, atol=1e-8)
    assert abs(control["accel_error_m_s2"]) < 1e-8


def test_public_foot_sliding_uses_joints():
    target, times = motion()
    predicted = {**target, "joints": target["joints"].copy()}
    predicted["joints"][:, (7, 8, 10, 11), 0] += times[:, None]
    native = evaluate_motion(predicted, target, times)
    control = evaluate_formula_control(predicted, target, times, np.ones(times.size, bool))
    assert np.isclose(native["foot_sliding_mm"], 25)
    assert control["foot_sliding_mm"] == 0


def test_aggregation_is_session_macro():
    from anysole.utils.vision_smpl_metrics import REQUESTED_METRICS
    rows = []
    for session, value, count in (("S1", 1, 1), ("S2", 3, 3)):
        for _ in range(count):
            rows.append({"baseline": "WHAM", "date": "20260804", "session": session,
                         "method": "method_1_offset", "camera": "camera_1", "frame_count": 100,
                         **{key: None if key == "pve_t_mm" else value for key in REQUESTED_METRICS}})
    sessions, summaries = aggregate(rows)
    assert len(sessions) == 2
    assert summaries[0]["mpjpe_mm"] == 2


def test_helpers_do_not_import_historical_metrics():
    helpers(Path(__file__).resolve().parents[2])
    assert "smpl_evaluation.metrics" not in sys.modules
    assert "smpl_evaluation.runner" not in sys.modules
    assert "_anysole_tactile_helpers.metrics" not in sys.modules


def test_smpl_forward_matches_anysole_native_fk():
    from anysole.data.smpl_io import _smpl_model
    from anysole.types import JOINT_PARENTS
    from anysole.utils.geometry import fk_local_np
    workspace = Path(__file__).resolve().parents[2]
    model_path = workspace / "models/smpl/SMPL_NEUTRAL.pkl"
    _, _, _, model_module = helpers(workspace)
    forward = model_module.SMPLForward(model_path, device="cpu", batch_size=2)
    rng = np.random.default_rng(11)
    poses = rng.normal(scale=.25, size=(3, 72)).astype(np.float32)
    betas = rng.normal(scale=.3, size=10).astype(np.float32)
    trans = rng.normal(scale=.2, size=(3, 3)).astype(np.float32)
    _, actual = forward.forward(poses, betas, trans)
    regressor, template, shapedirs = _smpl_model(str(model_path))
    rest = regressor @ (template + np.einsum("vkc,c->vk", shapedirs[:, :, :10], betas))
    offsets = np.zeros((24, 3))
    for j, parent in enumerate(JOINT_PARENTS):
        if parent >= 0:
            offsets[j] = rest[j] - rest[parent]
    rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(3, 24, 3, 3)
    native, _ = fk_local_np(rotations, offsets, np.array(JOINT_PARENTS))
    native += (trans + rest[0])[:, None]
    assert np.max(np.abs(actual - native)) < 1e-6
