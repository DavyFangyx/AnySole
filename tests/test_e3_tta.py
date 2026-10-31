"""Scientific mechanism tests for the GT-free G1 TTA/evaluation boundary."""
from dataclasses import replace
from unittest import TestCase, main, mock

import numpy as np
import torch

from anysole.tta import (E3Observations, TrainTemplate, TTAConfig, adapt_multiview,
                        CalibratedReferenceObservations, triangulate_calibrated_reference,
                        fit_calibrated_reference)
from anysole.types import CONFIG_V, FPS, JOINT_PARENTS
from anysole.utils.geometry import rot6d_to_rotmat
from anysole.utils.evaluation_contract import (
    e3_complete_window_starts, e3_evaluation_contract, e3_evaluation_manifest_schema,
)


class TinyMotion(torch.nn.Module):
    """A nonlinear pose/velocity decoder with identifiable token gradients."""
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))
        self.register_buffer("identity", torch.tensor([1., 0, 0, 0, 1, 0]))
        self.contact_calls = 0

    def decode_z(self, z):
        windows, time = z.shape[:2]
        pose = self.identity.repeat(windows, time, 24) + .03 * torch.tanh(z[:, :, 1]).repeat(1, 1, 24) * self.scale
        velocity = z[:, :, 0, :3] * self.scale
        return {"x0_hat": pose, "v_hat": velocity, "trans_hat": velocity.cumsum(1) / FPS}

    def forward(self, visual, raw, phys, config_id):
        self.contact_calls += 1
        z = visual.new_zeros(visual.shape[0], visual.shape[1], 10, 6)
        z[..., 0, :3] = visual[..., :3]
        z[..., 1, :3] = visual[..., :3] * .5
        z[..., 0, 1] += (config_id != CONFIG_V)[:, None] * raw[..., 0] * .01
        return {**self.decode_z(z), "z": z,
                "contact_logits": raw[..., :2] + phys[..., :2] * .1}


def fixture(views=2, *, actual_t=True, empty_points=False):
    ids = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])
    v = torch.zeros(views, 2, 4, 3)
    v[..., 0] = .2
    v[..., 1] = ids * .03
    for view in range(views):
        v[view, ..., 2] = .1 * view
    joints = torch.tensor([] if empty_points else [0, 1, 2, 4, 5, 10, 11, 15, 20, 21], dtype=torch.long)
    points = len(joints)
    k = torch.tensor([[500., 0, 320], [0, 500, 240], [0, 0, 1]]).repeat(views, 1, 1)
    observations = E3Observations(v, ids, torch.zeros(views, 2, 4, points, 2),
                                  torch.ones(views, 2, 4, points), k, joints,
                                  T_raw=torch.ones(2, 4, 96) if actual_t else None,
                                  T_phys=torch.ones(2, 4, 12) if actual_t else None,
                                  tactile_source="synthetic_actual_pressure" if actual_t else None)
    offsets = torch.arange(72).reshape(24, 3).float() / 1000
    offsets[:, 1] = -.08
    offsets[0] = 0
    template = TrainTemplate(offsets, torch.tensor(JOINT_PARENTS), "a" * 64,
                             support_offsets=torch.tensor([[0, -.04, 0], [0, -.04, 0.]]))
    if not empty_points:
        initial = adapt_multiview(TinyMotion(), observations, template, TTAConfig(steps=0, path="0step"))
        xyz = initial["joints"][ids.flatten()][:, joints] + torch.tensor([.2, -.1, 4.])
        projection = torch.einsum("ij,fkj->fki", k[0], xyz)
        pixels = (projection[..., :2] / projection[..., 2:]).reshape(2, 4, points, 2)
        observations = replace(observations, detected_2d=pixels[None].repeat(views, 1, 1, 1, 1))
    return observations, template


class E3TTATests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_step_exact_initial_and_no_network_mutation(self):
        obs, template = fixture()
        model = TinyMotion().train()
        model.scale.grad = torch.tensor(9.)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with mock.patch("torch.optim.Adam", side_effect=AssertionError("no optimizer at zero step")):
            out = adapt_multiview(model, obs, template, TTAConfig(steps=0, path="0step"))
        self.assertTrue(torch.equal(out["pose"], out["initial_pose"]))
        self.assertTrue(torch.equal(out["trans"], out["initial_trans"]))
        self.assertEqual(out["optimizer_steps"], 0)
        self.assertTrue(model.training)
        self.assertTrue(model.scale.requires_grad)
        self.assertEqual(model.scale.grad.item(), 9.)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_C0_and_disabled_contact_are_exact_same_loop(self):
        obs, template = fixture()
        c0 = adapt_multiview(TinyMotion(), obs, template, TTAConfig(steps=4))
        off = adapt_multiview(TinyMotion(), obs, template, TTAConfig(steps=4, path="C_original", use_contact=False))
        self.assertTrue(torch.equal(c0["pose"], off["pose"]))
        self.assertTrue(torch.equal(c0["trans"], off["trans"]))
        self.assertEqual(c0["trace"], off["trace"])

    def test_gradients_camera_rotations_positive_depth_and_frozen_network(self):
        obs, template = fixture()
        model = TinyMotion()
        before = model.scale.detach().clone()
        out = adapt_multiview(model, obs, template, TTAConfig(steps=5, use_contact=True, path="C_original"))
        self.assertTrue(torch.equal(model.scale, before))
        self.assertIsNone(model.scale.grad)
        self.assertFalse(out["contact_weights"].requires_grad)
        self.assertEqual(model.contact_calls, obs.V_feat.shape[0] + 1)
        for name in ("z", "virtual_z", "camera_rotation", "camera_xy", "camera_depth", "ground"):
            self.assertTrue(all(np.isfinite(row["gradient_norms"][name]) for row in out["trace"]))
            self.assertGreater(max(row["gradient_norms"][name] for row in out["trace"]), 0, name)
        r = out["camera_rotation"]
        torch.testing.assert_close(r.transpose(-1, -2) @ r, torch.eye(3).repeat(len(r), 1, 1))
        torch.testing.assert_close(torch.det(r), torch.ones(len(r)))
        self.assertGreaterEqual(out["minimum_depth"].item(), .05 - 1e-6)
        self.assertNotEqual(out["ground"].item(), out["initial_ground"].item())
        body = rot6d_to_rotmat(out["pose"].reshape(-1, 24, 6))
        torch.testing.assert_close(torch.det(body), torch.ones(body.shape[:-2]))

    def test_contact_physical_evidence_not_multiplied_by_views(self):
        values = []
        for views in (1, 2, 4):
            obs, template = fixture(views=1)
            obs = replace(obs, V_feat=obs.V_feat.repeat(views, 1, 1, 1),
                          detected_2d=obs.detected_2d.repeat(views, 1, 1, 1, 1),
                          confidence=obs.confidence.repeat(views, 1, 1, 1),
                          intrinsics=obs.intrinsics.repeat(views, 1, 1))
            out = adapt_multiview(TinyMotion(), obs, template, TTAConfig(steps=1, use_contact=True, path="C_original"))
            values.append(out["trace"][0]["Lcontact_raw_sum"])
            self.assertAlmostEqual(out["trace"][0]["Lcontact"], values[-1] / 7)
        np.testing.assert_allclose(values, [values[0]] * 3, rtol=1e-6, atol=1e-9)

    def test_UP_is_direct_and_never_constructs_optimizer(self):
        obs, template = fixture()
        with mock.patch("torch.optim.Adam", side_effect=AssertionError("UP optimizer constructed")), mock.patch(
                "anysole.tta._initialize_cameras", side_effect=AssertionError("UP camera fitting invoked")):
            direct = adapt_multiview(TinyMotion(), obs, template, TTAConfig(recipe="UP_G1", path="direct", steps=0))
            self.assertEqual(direct["optimizer_steps"], 0)
            with self.assertRaisesRegex(ValueError, "UP is direct only"):
                adapt_multiview(TinyMotion(), obs, template, TTAConfig(recipe="UP_G1", steps=1))

    def test_gt_free_V_only_and_no_contact_no_detections_are_finite(self):
        obs, template = fixture(actual_t=False, empty_points=True)
        out = adapt_multiview(TinyMotion(), obs, template, TTAConfig(steps=2))
        self.assertTrue(torch.isfinite(out["joints"]).all())
        self.assertIsNone(out["contact_weights"])
        self.assertEqual(out["trace"][0]["L2D"], 0.)
        with self.assertRaisesRegex(ValueError, "actual T"):
            adapt_multiview(TinyMotion(), obs, template, TTAConfig(steps=1, use_contact=True, path="C_original"))
        with self.assertRaises(TypeError):
            E3Observations(**{**vars(obs), "pose_gt": torch.zeros(2, 4, 144)})
        full, template = fixture()
        malformed = full.detected_2d.clone()
        malformed[..., 0, :] = float("inf")
        finite = adapt_multiview(TinyMotion(), replace(full, detected_2d=malformed), template, TTAConfig(steps=1))
        self.assertTrue(torch.isfinite(finite["joints"]).all())

    def test_source_degradation_and_shape_provenance_fail_closed(self):
        obs, template = fixture()
        for invalid in (replace(obs, detection_condition="occ010"),
                        replace(obs, detector_source="mocap_projection")):
            with self.assertRaises(ValueError):
                adapt_multiview(TinyMotion(), invalid, template, TTAConfig(steps=0, path="0step"))
        with self.assertRaisesRegex(ValueError, "train-only"):
            adapt_multiview(TinyMotion(), obs, replace(template, split="test"), TTAConfig(steps=0, path="0step"))

    def test_continuous_integration_overlap_tail_gap_and_unique_scoring(self):
        obs, template = fixture(views=1)
        out = adapt_multiview(TinyMotion(), obs, template, TTAConfig(steps=0, path="0step"))
        self.assertEqual(out["frame_ids"].tolist(), list(range(7)))
        torch.testing.assert_close(out["trans"][:, 0], torch.arange(1, 8) * .2 / FPS)
        self.assertEqual(e3_complete_window_starts(7, 4), [0, 3])
        self.assertEqual(e3_complete_window_starts(3, 4), [])
        self.assertEqual(e3_evaluation_contract(out, template_sha256=template.sha256)["scored_frames"], 7)
        gap_obs = replace(obs, frame_ids=torch.tensor([[0, 1, 2, 3], [7, 8, 9, 10]]))
        gap = adapt_multiview(TinyMotion(), gap_obs, template, TTAConfig(steps=0, path="0step"))
        torch.testing.assert_close(gap["trans"][[0, 4], 0], torch.tensor([.2, .2]) / FPS)
        self.assertEqual(gap["segment_ids"].tolist(), [0] * 4 + [1] * 4)
        large_time = obs.frame_ids.double() / FPS + 1700000000.
        timed = adapt_multiview(TinyMotion(), replace(obs, frame_times=large_time), template, TTAConfig(steps=0, path="0step"))
        torch.testing.assert_close(timed["trans"], out["trans"], atol=1e-7, rtol=1e-5)

    def test_scoring_one_rigid_transform_and_never_fits_scale_or_windows(self):
        from anysole.utils.eval_protocol import score_e3_prediction
        obs, template = fixture()
        out = adapt_multiview(TinyMotion(), obs, template, TTAConfig(steps=0, path="0step"))
        p = out["joints"].numpy()
        r = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
        target = p @ r.T + np.array([10., 2., 3.])
        score = score_e3_prediction(out, target, np.arange(7))
        self.assertLess(score["metrics"]["GlobalMPJPE_segment_SE3"], .001)
        self.assertGreater(score["metrics"]["GlobalMPJPE_unaligned"], 1000)
        scaled = score_e3_prediction(out, target * 2, np.arange(7))
        self.assertGreater(scaled["metrics"]["GlobalMPJPE_segment_SE3"], 1)
        warped = target.copy()
        warped[4:] += [1., 0, 0]
        self.assertGreater(score_e3_prediction(out, warped, np.arange(7))["metrics"]["GlobalMPJPE_segment_SE3"], 10)
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            score_e3_prediction(out, target, np.array([0, 1, 2, 3, 3, 5, 6]))

    def test_manifest_all_three_conditions_fifteen_subsets_three_seeds_and_UP_guard(self):
        panel = e3_evaluation_manifest_schema()
        self.assertEqual(len(panel["camera_subsets"]), 15)
        for recipe in ("U0", "UP_G1", "UC_G1"):
            rows = [row for row in panel["rows"] if row["recipe"] == recipe]
            self.assertEqual(len({(r["visual_condition"], tuple(r["camera_subset"]), r["seed"]) for r in rows}), 135)
        self.assertTrue(all(row["path"] == "direct" and row["optimizer_steps"] == 0
                            for row in panel["rows"] if row["recipe"] == "UP_G1"))

    def test_real_V7_decode_interface_and_frozen_TTA(self):
        from anysole.models.common.model_v7 import AnySoleModelV7
        torch.manual_seed(42)
        model = AnySoleModelV7(d=16, tw=4, nhead=4, dropout=0., pose_layers=1,
                              reliability_route=True, task_routes=True, t_only_adapters=True).eval()
        obs, template = fixture(views=1, empty_points=True)
        obs = replace(obs, V_feat=torch.randn(1, 2, 4, 2051))
        with torch.no_grad():
            initial = model(obs.V_feat[0], obs.T_raw, obs.T_phys, torch.zeros(2, dtype=torch.long))
            decoded = model.decode_z(initial["z"])
        self.assertTrue(torch.equal(initial["x0_hat"], decoded["x0_hat"]))
        self.assertTrue(torch.equal(initial["v_hat"], decoded["v_hat"]))
        state = {key: value.clone() for key, value in model.state_dict().items()}
        out = adapt_multiview(model, obs, template, TTAConfig(steps=2, initialization="VT", path="C_original", use_contact=True))
        self.assertTrue(torch.isfinite(out["joints"]).all())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)

    def test_calibrated_reference_DLT_and_fixed_template_FK_fit(self):
        obs, template = fixture(views=1)
        initial = adapt_multiview(TinyMotion(), obs, template, TTAConfig(steps=0, path="0step"))
        xyz = initial["joints"]
        rotations = torch.eye(3).repeat(2, 1, 1)
        translations = torch.tensor([[-.5, 0, 4.], [.5, 0, 4.]])
        camera = xyz[None, :, obs.joint_indices] + translations[:, None, None]
        k = obs.intrinsics.repeat(2, 1, 1)
        uvw = torch.einsum("vij,vfkj->vfki", k, camera)
        reference = CalibratedReferenceObservations(uvw[..., :2] / uvw[..., 2:],
            torch.ones(2, len(xyz), len(obs.joint_indices)), k, rotations, translations,
            obs.joint_indices, initial["frame_ids"], "synthetic_calibration_reference_only")
        triangulated = triangulate_calibrated_reference(reference)
        self.assertTrue(triangulated["valid"].all())
        torch.testing.assert_close(triangulated["points"], xyz[:, obs.joint_indices], atol=3e-6, rtol=1e-4)
        fitted = fit_calibrated_reference(reference, template, steps=40)
        self.assertLess(fitted["trace"][-1]["triangulation_fit"], fitted["trace"][0]["triangulation_fit"])
        self.assertEqual(fitted["coverage"]["supported_frames"], len(xyz))
        joints = fitted["joints"]
        for j, parent in enumerate(JOINT_PARENTS):
            if parent >= 0:
                torch.testing.assert_close((joints[:, j] - joints[:, parent]).norm(dim=-1),
                                           template.offsets[j].norm().expand(len(joints)), atol=1e-6, rtol=1e-4)
        with self.assertRaisesRegex(ValueError, "two real views"):
            triangulate_calibrated_reference(replace(reference, detected_2d=reference.detected_2d[:1]))
        with self.assertRaisesRegex(ValueError, "no triangulable"):
            fit_calibrated_reference(replace(reference, confidence=torch.zeros_like(reference.confidence)), template)


if __name__ == "__main__":
    main()
