"""Protocol regression checks; run with Tactile pytest from AnySole."""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole.utils.vision_smpl_metrics import evaluate_motion
from AnysoleWorkspace.tool.eval_rgb_smpl import aggregate, helpers


def motion():
    rng = np.random.default_rng(1)
    joints = np.broadcast_to(rng.normal(size=(1, 24, 3)), (110, 24, 3)).copy()
    return {"joints": joints, "vertices": np.zeros((110, 6890, 3)),
            "rotations": np.broadcast_to(np.eye(3), (110, 24, 3, 3)).copy(),
            "poses": np.zeros((110, 72)),
            "tpose_joints": joints.copy(), "tpose_vertices": np.zeros((110, 6890, 3))}, np.arange(110) / 40


def test_identity_and_restored_shape_metric():
    target, times = motion()
    values = evaluate_motion(target, target, times)
    for metric in ("mpjpe_mm", "pa_mpjpe_mm", "mpjae_deg", "pve_mm", "accel_error_m_s2",
                   "pve_t_mm", "jitter_pred_m_s3", "root_ate_mm", "w_mpjpe100_mm", "foot_sliding_mm"):
        assert abs(values[metric]) < 1e-8


def test_body21_rotation_excludes_root_and_hands():
    target, times = motion()
    predicted = {**target, "poses": target["poses"].copy()}
    predicted["poses"][:, :3] = Rotation.from_euler("y", 24, degrees=True).as_rotvec()
    predicted["poses"][:, 66:] = 0.5
    assert evaluate_motion(predicted, target, times)["mpjae_deg"] == 0
    predicted["poses"][:, 3:6] = Rotation.from_euler("y", 21, degrees=True).as_rotvec()
    assert np.isclose(evaluate_motion(predicted, target, times)["mpjae_deg"], 1)


def test_root_acceleration_and_jerk_are_removed():
    target, times = motion()
    predicted = {**target, "joints": target["joints"].copy()}
    predicted["joints"][:, :, 0] += times[:, None] ** 3
    values = evaluate_motion(predicted, target, times)
    assert abs(values["accel_error_m_s2"]) < 1e-8
    assert abs(values["jitter_pred_m_s3"]) < 1e-7


def test_public_foot_sliding_uses_joints():
    target, times = motion()
    predicted = {**target, "joints": target["joints"].copy()}
    predicted["joints"][:, (7, 8, 10, 11), 0] += times[:, None]
    assert np.isclose(evaluate_motion(predicted, target, times)["foot_sliding_mm"], 25)


def test_pve_t_uses_predicted_shape_and_is_translation_invariant():
    target, times = motion()
    predicted = {**target, "tpose_vertices": target["tpose_vertices"].copy(),
                 "tpose_joints": target["tpose_joints"].copy()}
    predicted["tpose_vertices"][:, 0, 0] += 0.1
    expected = 100 / 6890
    assert np.isclose(evaluate_motion(predicted, target, times)["pve_t_mm"], expected)
    predicted["tpose_vertices"] += 3
    predicted["tpose_joints"] += 3
    assert np.isclose(evaluate_motion(predicted, target, times)["pve_t_mm"], expected)


def test_aggregation_is_session_macro():
    from anysole.utils.vision_smpl_metrics import REQUESTED_METRICS
    rows = []
    for session, value, count in (("S1", 1, 1), ("S2", 3, 3)):
        for _ in range(count):
            rows.append({"baseline": "WHAM", "date": "20260804", "session": session,
                         "method": "method_1_offset", "camera": "camera_1", "frame_count": 100,
                         **{key: value for key in REQUESTED_METRICS}})
    sessions, summaries = aggregate(rows)
    assert len(sessions) == 2
    assert summaries[0]["mpjpe_mm"] == 2


def test_helpers_do_not_import_historical_metrics():
    modules = helpers(Path(__file__).resolve().parents[2])
    assert all("AnySole/anysole/data/rgb_smpl" in module.__file__ for module in modules)
    assert "smpl_evaluation.metrics" not in sys.modules
    assert "smpl_evaluation.runner" not in sys.modules
    assert "_anysole_tactile_helpers.metrics" not in sys.modules


def test_original_consumer_uses_body21_and_relative_temporal(monkeypatch):
    import torch
    from anysole.utils import eval_protocol as protocol
    count = 10
    times = np.arange(count) / 40
    base = torch.tensor(np.random.default_rng(3).normal(size=(count, 24, 3)), dtype=torch.float64)
    base[:] = base[0].clone()
    target_rot = np.broadcast_to(np.eye(3), (count, 24, 3, 3)).copy()
    pred_rot = target_rot.copy()
    for joint, angle in ((0, 30), (1, 21), (22, 60), (23, 60)):
        pred_rot[:, joint] = Rotation.from_euler("y", angle, degrees=True).as_matrix()
    pose6d = lambda matrices: torch.tensor(
        np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1).reshape(count, 144),
        dtype=torch.float64)
    pred_trans = torch.zeros((count, 3), dtype=torch.float64)
    pred_trans[:, 0] = torch.tensor(times ** 3)
    # Isolate the consumer's trajectory/rotation selection from surface/FK generation.
    monkeypatch.setattr(protocol, "fk_pose6d", lambda pose, trans, *args: base[None] + trans[:, :, None])
    monkeypatch.setattr(protocol, "pelvis_to_smpl_trans", lambda pose, trans, beta: trans)
    monkeypatch.setattr(protocol, "smpl_vertices_from_archive_params",
                        lambda poses, trans, beta: np.zeros((count, 6890, 3)))
    seq = {"pred_pose": pose6d(pred_rot), "gt_pose": pose6d(target_rot),
           "pred_trans": pred_trans, "gt_trans": torch.zeros_like(pred_trans),
           "kp_gt": base, "offsets": torch.zeros((24, 3)),
           "parents": torch.zeros(24, dtype=torch.long), "betas": torch.zeros(10),
           "floor_y": torch.tensor(0.), "contact_gt": torch.zeros((count, 2))}
    accum = protocol._Accum()
    protocol._session_metrics(seq, 40, 2, 0, accum, dict(tp=0, fp=0, fn=0, tn=0))
    values = accum.means()
    assert np.isclose(values["mpjae_deg"], 1)
    assert abs(values["accel_error_m_s2"]) < 1e-8
    assert abs(values["jitter_pred_m_s3"]) < 1e-7


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
