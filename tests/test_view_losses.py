"""Numeric, gradient, source-continuity and accumulation loss contracts."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole.data.rgb_smpl.coordinates import (gt_smpl_to_checkerboard, checkerboard_to_camera,
    camera_to_checkerboard, gt_root_rotation_to_checkerboard, camera_root_rotation_to_checkerboard)
from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.models.v3_3b import STRUCTURE
from anysole.types import CONFIG_V, JOINT_PARENTS
from anysole.utils.geometry import fk_pose6d, rotmat_to_6d
from anysole.utils.losses import (compute_visual_loss_terms, compute_pair_loss_terms,
    compute_visual_loss_counts, compute_pair_loss_counts, loss_term_weights, compute_losses)


@pytest.fixture(autouse=True, scope="module")
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config(**updates):
    return {"lambda_sigma": 0, "lambda_assign": 0, "lambda_con": 0,
            "lambda_vrec": 0, "soft_parts": True, "traj_deltas": [2, 4], **updates}


def batch_fixture(cameras=(2,), tw=5, dtype=torch.float64):
    bsz, cmax = len(cameras), max(cameras)
    valid = torch.zeros(bsz, cmax, tw, dtype=torch.bool)
    for g, count in enumerate(cameras):
        valid[g, :count] = True
    times = torch.arange(tw, dtype=torch.float64)[None].expand(bsz, -1).clone() / 40
    pose = torch.tensor([1, 0, 0, 0, 1, 0], dtype=dtype).repeat(bsz, tw, 24)
    root = torch.zeros(bsz, tw, 3, dtype=dtype)
    root[..., 0] = times * 2 + torch.arange(bsz, dtype=dtype)[:, None]
    velocity = torch.zeros_like(root)
    velocity[:, 1:, 0] = 2
    offsets = torch.arange(72, dtype=dtype).reshape(1, 24, 3).expand(bsz, -1, -1).clone() / 200
    offsets[:, 0] = 0
    parents = torch.tensor(JOINT_PARENTS)[None].expand(bsz, -1).clone()
    pressure = torch.arange(bsz * tw * 96, dtype=dtype).reshape(bsz, tw, 96) / 10000
    generator = torch.Generator().manual_seed(41)
    return {"visual": {"V_feat": torch.randn(bsz, cmax, tw, 2051, generator=generator, dtype=dtype),
                        "view_valid": valid, "view_edges": valid[..., 1:] & valid[..., :-1],
                        "target_time_s": times},
            "labels": {"pose6d": pose, "pressure": pressure, "velocity": velocity, "root": root,
                       "offsets": offsets, "parents": parents, "betas": torch.zeros(bsz, 10, dtype=dtype),
                       "kp_gt": fk_pose6d(pose, root, offsets, parents)},
            "masks": {"motion_valid": torch.ones(bsz, tw, dtype=torch.bool),
                      "pressure_valid": torch.ones(bsz, tw, 96, dtype=torch.bool),
                      "velocity_valid": torch.cat([torch.zeros(bsz, 1, dtype=torch.bool),
                                                   torch.ones(bsz, tw-1, dtype=torch.bool)], dim=1),
                      "motion_edges": torch.ones(bsz, tw-1, dtype=torch.bool),
                      "pressure_edges": torch.ones(bsz, tw-1, 96, dtype=torch.bool)},
            "meta": {"camera_id": torch.stack([torch.cat([torch.arange(1, c+1), torch.full((cmax-c,), -1)])
                                                 for c in cameras]),
                     "sample_key": [f"recording_{g}:0:{tw}" for g in range(bsz)],
                     "target_frame_id": torch.arange(tw)[None].expand(bsz, -1).clone()}}


def view_index(batch):
    return batch["visual"]["view_valid"].any(dim=-1).nonzero()


def predictions(batch, index):
    g = index[:, 0]
    return {"x0_hat": batch["labels"]["pose6d"][g].clone().requires_grad_(),
            "pressure_hat": batch["labels"]["pressure"][g].clone().requires_grad_(),
            "v_hat": batch["labels"]["velocity"][g].clone().requires_grad_(),
            "trans_hat": torch.full_like(batch["labels"]["velocity"][g], float("nan")),
            "sigma_frozen": False}


def mean(term):
    return term["raw_sum"] / term["valid_count"] if term["valid_count"] else term["raw_sum"]


def test_supervised_raw_sums_counts_and_per_delta_weights_are_hand_computable():
    batch = batch_fixture()
    index = view_index(batch)
    out = predictions(batch, index)
    out["x0_hat"] = out["x0_hat"] + 2
    out["pressure_hat"] = out["pressure_hat"] + 3
    out["v_hat"] = out["v_hat"] + 4
    terms = compute_visual_loss_terms(out, batch, index, config())
    assert terms["pose"]["valid_count"] == 2 * 5 * 144
    assert mean(terms["pose"]) == 4
    assert terms["pressure"]["valid_count"] == 2 * 5 * 96
    assert mean(terms["pressure"]) == 9
    assert terms["traj_vel"]["valid_count"] == 2 * 4 * 3
    assert mean(terms["traj_vel"]) == 16
    assert terms["traj_delta_2"]["valid_count"] == 2 * 3 * 3
    torch.testing.assert_close(mean(terms["traj_delta_2"]), torch.tensor((4 * .05)**2, dtype=torch.float64))
    assert terms["traj_delta_4"]["valid_count"] == 2 * 1 * 3
    weights = loss_term_weights(config(), 5, lambda_mv=.2)
    assert weights["traj_delta_2"] == .25 and weights["traj_delta_4"] == .125
    assert weights["pose"] == 3 and weights["pressure"] == .1
    assert weights["local23_pair"] == .2 and weights["pressure_pair"] == .2


@pytest.mark.parametrize("c,pairs", ((1, 0), (2, 1), (3, 3), (4, 6)))
def test_pair_counts_c1_to_c4_and_identical_outputs_zero(c, pairs):
    batch = batch_fixture((c,))
    index = view_index(batch)
    terms = compute_pair_loss_terms(predictions(batch, index), batch, index, config())
    counts = compute_pair_loss_counts(batch, index, config())
    assert counts == {"local23_pair": pairs * 5 * 23, "pressure_pair": pairs * 5 * 96}
    for name, term in terms.items():
        assert term["valid_count"] == counts[name]
        assert term["raw_sum"] == 0


def test_pair_chordal_90_degrees_is_four_per_local_joint_root_excluded_and_both_ends_train():
    batch = batch_fixture()
    index = view_index(batch)
    out = predictions(batch, index)
    rotation = torch.tensor(Rotation.from_euler("z", 90, degrees=True).as_matrix(), dtype=torch.float64)
    raw = out["x0_hat"].detach().clone().reshape(2, 5, 24, 6)
    raw[1] = rotmat_to_6d(rotation)
    out["x0_hat"] = raw.reshape(2, 5, 144).requires_grad_()
    pressure = torch.zeros(2, 5, 96, dtype=torch.float64)
    pressure[1, :, :48] = 1
    out["pressure_hat"] = pressure.requires_grad_()
    terms = compute_pair_loss_terms(out, batch, index, config())
    torch.testing.assert_close(mean(terms["local23_pair"]), torch.tensor(4., dtype=torch.float64))
    torch.testing.assert_close(mean(terms["pressure_pair"]), torch.tensor(.5, dtype=torch.float64))
    (mean(terms["local23_pair"]) + mean(terms["pressure_pair"])).backward()
    for key in ("x0_hat", "pressure_hat"):
        assert out[key].grad[0].abs().sum() > 0 and out[key].grad[1].abs().sum() > 0
    assert out["x0_hat"].grad[:, :, :6].abs().sum() == 0


def test_pair_masks_are_per_frame_cell_and_invariant_to_view_order():
    batch = batch_fixture((3,))
    batch["visual"]["view_valid"][0, 2, 2] = False
    batch["masks"]["pressure_valid"][0, 1, :48] = False
    batch["masks"]["motion_valid"][0, 3] = False
    index = view_index(batch)
    out = predictions(batch, index)
    out["pressure_hat"] = out["pressure_hat"] + torch.arange(3)[:, None, None]
    before = compute_pair_loss_terms(out, batch, index, config())
    order = torch.tensor([2, 0, 1])
    reordered = {key: value[order] if torch.is_tensor(value) else value for key, value in out.items()}
    after = compute_pair_loss_terms(reordered, batch, index[order], config())
    for name in before:
        assert before[name]["valid_count"] == after[name]["valid_count"]
        torch.testing.assert_close(before[name]["raw_sum"], after[name]["raw_sum"])
    assert before["local23_pair"]["valid_count"] == (3*5 - 2 - 3) * 23
    assert before["pressure_pair"]["valid_count"] == 3*5*96 - 2*96 - 3*48


def test_masked_nan_labels_predictions_are_selected_out_before_arithmetic():
    batch = batch_fixture()
    batch["masks"]["motion_valid"][:, 2] = False
    batch["masks"]["velocity_valid"][:, 2:4] = False
    batch["masks"]["motion_edges"][:, 1:3] = False
    batch["masks"]["pressure_valid"][:, 2] = False
    for name in ("pose6d", "root", "kp_gt", "pressure", "velocity"):
        batch["labels"][name][:, 2] = float("nan")
    index = view_index(batch)
    out = predictions(batch, index)
    terms = compute_visual_loss_terms(out, batch, index, config())
    terms.update(compute_pair_loss_terms(out, batch, index, config()))
    sum(term["raw_sum"] for term in terms.values()).backward()
    assert all(torch.isfinite(term["raw_sum"]) for term in terms.values())
    assert all(torch.isfinite(out[key].grad).all() for key in ("x0_hat", "v_hat", "pressure_hat"))
    assert out["x0_hat"].grad[:, 2].abs().sum() == 0


@pytest.mark.parametrize("name", ("pose6d", "root", "kp_gt", "pressure", "velocity", "offsets"))
def test_true_label_mask_with_nonfinite_values_is_an_explicit_error(name):
    batch = batch_fixture()
    batch["labels"][name].reshape(-1)[min(400, batch["labels"][name].numel()-1)] = float("nan")
    # velocity at index0 has no predecessor, so force a valid ending edge.
    if name == "velocity":
        batch["labels"][name][0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="valid"):
        compute_visual_loss_terms(predictions(batch, view_index(batch)), batch, view_index(batch), config())


def test_no_valid_views_or_pairs_return_finite_connected_zero():
    batch = batch_fixture()
    batch["visual"]["view_valid"][:] = False
    for value in batch["masks"].values():
        value[:] = False
    for name in ("pose6d", "root", "kp_gt", "pressure", "velocity", "offsets"):
        batch["labels"][name][:] = float("nan")
    index = view_index(batch)
    out = predictions(batch, index)
    terms = compute_visual_loss_terms(out, batch, index, config())
    terms.update(compute_pair_loss_terms(out, batch, index, config()))
    assert all(term["valid_count"] == 0 and term["raw_sum"] == 0 for term in terms.values())
    sum(term["raw_sum"] for term in terms.values()).backward()
    assert out["x0_hat"].grad is not None and out["pressure_hat"].grad is not None


def test_true_dt_source_reuse_gap_and_long_span_denominators():
    batch = batch_fixture((1,), tw=8)
    times = torch.tensor([[0, .02, .04, .065, .09, .13, .15, .175]], dtype=torch.float64)
    batch["visual"]["target_time_s"] = times
    batch["visual"]["view_edges"][0, 0, 2] = False
    index = view_index(batch)
    out = predictions(batch, index)
    out["v_hat"] = out["v_hat"] + 1
    terms = compute_visual_loss_terms(out, batch, index, config())
    counts = compute_visual_loss_counts(batch, index, config())
    assert counts["traj_delta_2"] == 4 * 3 and counts["traj_delta_4"] == 1 * 3
    for delta in (2, 4):
        spans = [float(times[0, end] - times[0, end-delta]) for end in range(delta, 8)
                 if not end-delta <= 2 < end]
        expected = torch.tensor(sum(x*x for x in spans) * 3, dtype=torch.float64)
        torch.testing.assert_close(terms[f"traj_delta_{delta}"]["raw_sum"], expected)
    assert all(terms[key]["valid_count"] == value for key, value in counts.items())
    out["v_hat"] = out["v_hat"].clone()
    out["v_hat"][:, 0] += 1e6
    after = compute_visual_loss_terms(out, batch, index, config())
    for key in ("traj_vel", "traj_delta_2", "traj_delta_4"):
        torch.testing.assert_close(terms[key]["raw_sum"], after[key]["raw_sum"])


def test_production_t20_all_four_delta_scales_and_no_reweighting_after_gap():
    batch = batch_fixture(tw=20)
    index = view_index(batch)
    out = predictions(batch, index)
    out["v_hat"] = out["v_hat"] + 1
    cfg = config(traj_deltas=[2, 4, 8, 19])
    full = compute_visual_loss_terms(out, batch, index, cfg)
    weights = loss_term_weights(cfg, 20)
    for delta in (2, 4, 8, 19):
        key = f"traj_delta_{delta}"
        assert full[key]["valid_count"] == 2 * (20-delta) * 3
        torch.testing.assert_close(mean(full[key]), torch.tensor((delta/40)**2, dtype=torch.float64))
        assert weights[key] == 1 / (4 * delta)
    batch["visual"]["view_edges"][:, :, 9] = False
    gapped = compute_visual_loss_terms(out, batch, index, cfg)
    assert gapped["traj_delta_19"]["valid_count"] == 0
    assert gapped["traj_delta_19"]["raw_sum"] == 0
    assert loss_term_weights(cfg, 20) == weights


def test_shuffle_routes_actual_camera_ids_and_does_not_change_gt_or_rng():
    batch = batch_fixture((3, 2))
    batch["meta"]["camera_id"][:] = torch.tensor([[1, 2, 4], [2, 4, -1]])
    index = view_index(batch)
    out = predictions(batch, index)
    out["pressure_hat"] = torch.arange(5, dtype=torch.float64)[:, None, None].expand(-1, 5, 96).clone().requires_grad_()
    original = {key: value.clone() for key, value in batch["labels"].items()}
    rng = torch.get_rng_state().clone()
    normal = compute_pair_loss_terms(out, batch, index, config())
    shuffled = compute_pair_loss_terms(out, batch, index, config(), torch.tensor([1, 0]))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, original[key]) for key, value in batch["labels"].items())
    assert normal["pressure_pair"]["raw_sum"] == 7 * 5 * 96
    assert shuffled["pressure_pair"]["raw_sum"] == 35 * 5 * 96
    counts = compute_pair_loss_counts(batch, index, config(), torch.tensor([1, 0]))
    assert counts["pressure_pair"] == 4 * 5 * 96
    shuffled["pressure_pair"]["raw_sum"].backward()
    assert (out["pressure_hat"].grad.abs().sum(dim=(1, 2)) > 0).all()


@pytest.mark.parametrize("mapping", ([0, 1], [1, 1], [1.0, 0.0], [1, 0, 2]))
def test_invalid_shuffle_mapping_is_rejected(mapping):
    batch = batch_fixture((2, 2))
    with pytest.raises(ValueError, match="derangement"):
        compute_pair_loss_counts(batch, view_index(batch), config(), mapping)


def test_shuffle_duplicate_sample_keys_or_single_group_rejected_and_missing_camera_count_zero():
    batch = batch_fixture((2, 2))
    batch["meta"]["sample_key"][1] = batch["meta"]["sample_key"][0]
    with pytest.raises(ValueError, match="sample_key"):
        compute_pair_loss_counts(batch, view_index(batch), config(), [1, 0])
    single = batch_fixture()
    with pytest.raises(ValueError, match="derangement"):
        compute_pair_loss_counts(single, view_index(single), config(), [0])
    batch = batch_fixture((2, 1))
    assert compute_pair_loss_counts(batch, view_index(batch), config(), [1, 0])["pressure_pair"] == 0


class FixtureStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(3, 8)
        self.pose = nn.Linear(8, 144)
        self.pressure = nn.Linear(8, 96)
        self.velocity = nn.Linear(8, 3)
        self.sigma = nn.Linear(8, 18)
        self.logits = nn.Parameter(torch.randn(9, 24) * .1)

    def forward(self, batch, index):
        g, c = index.unbind(1)
        h = self.trunk(batch["visual"]["V_feat"][g, c, :, :3]).tanh()
        identity = h.new_tensor([1, 0, 0, 0, 1, 0]).repeat(24)
        pose = identity + .05 * self.pose(h)
        sigma = self.sigma(h).reshape(*h.shape[:2], 9, 2)
        gate = torch.cat([sigma.detach(), torch.zeros_like(sigma[..., :1])], dim=-1).softmax(-1)
        return {"x0_hat": pose, "pressure_hat": self.pressure(h), "v_hat": self.velocity(h),
                "gate_sigma": sigma, "gate_g": gate,
                "part_logits": self.logits, "part_assignment": self.logits.detach().softmax(0),
                "part_hypotheses": ((pose.detach() - .2) / .7).reshape(1, *pose.shape[:2], 24, 6).expand(9, -1, -1, -1, -1),
                "pose_mean": h.new_full((144,), .2), "pose_std": h.new_full((144,), .7),
                "sigma_frozen": False}


def subset(batch, rows):
    return {section: {key: value[rows] if torch.is_tensor(value) else [value[i] for i in rows]
                      for key, value in fields.items()} for section, fields in batch.items()}


@pytest.mark.parametrize("shuffle", (False, True))
def test_variable_c_masked_accumulation_matches_merged_loss_and_all_parameter_gradients(shuffle):
    torch.manual_seed(193)
    batch = batch_fixture((4, 2, 3, 1))
    batch["visual"]["view_valid"][0, 2, 1] = False
    batch["visual"]["view_edges"][0, 1, 2] = False
    batch["masks"]["motion_valid"][2, 3] = False
    batch["masks"]["pressure_valid"][1, 2, :48] = False
    cfg = config(lambda_assign=.05, lambda_sigma=.01)
    model = FixtureStudent().double()
    accumulated = copy.deepcopy(model)
    index = view_index(batch)
    mapping = [1, 0, 3, 2] if shuffle else None
    full_outputs = model(batch, index)
    full = compute_visual_loss_terms(full_outputs, batch, index, cfg)
    full.update(compute_pair_loss_terms(full_outputs, batch, index, cfg, mapping))
    weights = loss_term_weights(cfg, 5, .2)
    full_loss = sum(weights[key] * mean(term) for key, term in full.items())
    full_loss.backward()
    micros = [subset(batch, [0, 1]), subset(batch, [2, 3])]
    counts, micro_counts = {}, []
    for i, micro in enumerate(micros):
        idx = view_index(micro)
        item = compute_visual_loss_counts(micro, idx, cfg, include_assignment=i == 0)
        item.update(compute_pair_loss_counts(micro, idx, cfg, [1, 0] if shuffle else None))
        micro_counts.append(item)
        for key, count in item.items():
            counts[key] = counts.get(key, 0) + count
    totals = {key: 0. for key in full}
    for i, micro in enumerate(micros):
        idx = view_index(micro)
        outputs = accumulated(micro, idx)
        terms = compute_visual_loss_terms(outputs, micro, idx, cfg, include_assignment=i == 0)
        terms.update(compute_pair_loss_terms(outputs, micro, idx, cfg, [1, 0] if shuffle else None))
        assert all(terms[key]["valid_count"] == value for key, value in micro_counts[i].items())
        if i:
            assert terms["assignment"]["valid_count"] == 0 and terms["assignment"]["raw_sum"] == 0
        loss = sum(weights[key] * term["raw_sum"] / counts[key] if counts[key] else term["raw_sum"]
                   for key, term in terms.items())
        loss.backward()
        for key, term in terms.items():
            totals[key] += float(term["raw_sum"].detach())
    for key, term in full.items():
        assert counts[key] == term["valid_count"]
        assert totals[key] == pytest.approx(float(term["raw_sum"].detach()), rel=1e-10, abs=1e-10)
    for (name, parameter), (_, other) in zip(model.named_parameters(), accumulated.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-9, atol=1e-10, msg=name)


def test_lambda_zero_preserves_supervised_loss_gradient_and_rng():
    torch.manual_seed(22)
    batch = batch_fixture()
    index = view_index(batch)
    a = FixtureStudent().double()
    b = copy.deepcopy(a)
    cfg = config(lambda_sigma=.01, lambda_assign=.05)
    rng = torch.get_rng_state().clone()
    sup = compute_visual_loss_terms(a(batch, index), batch, index, cfg)
    weights = loss_term_weights(cfg, 5, 0)
    loss_a = sum(weights[key] * mean(value) for key, value in sup.items())
    out = b(batch, index)
    both = compute_visual_loss_terms(out, batch, index, cfg)
    both.update(compute_pair_loss_terms(out, batch, index, cfg))
    loss_b = sum(weights[key] * mean(value) for key, value in both.items())
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    loss_a.backward(); loss_b.backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_real_model_sigma_visual_path_denormalized_pose_and_original_regularizers_match():
    batch = batch_fixture(dtype=torch.float32)
    index = view_index(batch)
    model = AnySoleModelV2(d=32, tw=5, pose_layers=1, dropout=0, visual_only=True,
                           pressure_head="time_pe", **STRUCTURE).eval()
    model.pose_head.set_stats(torch.linspace(.1, .5, 144), torch.linspace(.2, .7, 144))
    g, c = index.unbind(1)
    out = model.forward_visual(batch["visual"]["V_feat"][g, c], batch["visual"]["view_valid"][g, c],
                                batch["visual"]["target_time_s"][g])
    cfg = config(lambda_sigma=.01, lambda_assign=.05)
    terms = compute_visual_loss_terms(out, batch, index, cfg)
    old_batch = {"pose_gt": batch["labels"]["pose6d"][g], "T_raw": batch["labels"]["pressure"][g],
                 "vel_gt": batch["labels"]["velocity"][g], "trans_gt": batch["labels"]["root"][g],
                 "kp_gt": batch["labels"]["kp_gt"][g], "offsets": batch["labels"]["offsets"][g],
                 "parents": batch["labels"]["parents"][g], "V_feat": batch["visual"]["V_feat"][g, c]}
    old = compute_losses(out, old_batch, torch.full((len(index),), CONFIG_V), cfg)
    for name, legacy in (("pose", "L_pose"), ("pressure", "L_Trec"), ("kp", "L_kp"),
                         ("sigma", "L_sigma"), ("assignment", "L_assign")):
        torch.testing.assert_close(mean(terms[name]), old[legacy], rtol=2e-6, atol=1e-6)
    mean(terms["sigma"]).backward(retain_graph=True)
    assert model.encoders.v_enc.proj.weight.grad.abs().sum() > 0
    mean(terms["assignment"]).backward()
    assert model.pose_head.part_logits.grad.abs().sum() > 0
    assert compute_visual_loss_counts(batch, index, cfg, sigma_frozen=True)["sigma"] == 0


def test_fk_is_train_only_pose_supervision_and_ignores_predicted_root_trajectory():
    batch = batch_fixture()
    index = view_index(batch)
    out = predictions(batch, index)
    out["x0_hat"] = (out["x0_hat"].detach() + .1).requires_grad_()
    term = compute_visual_loss_terms(out, batch, index, config())["kp"]
    term["raw_sum"].backward()
    assert out["x0_hat"].grad.abs().sum() > 0
    assert out["v_hat"].grad is None
    before = term["raw_sum"].detach()
    out["v_hat"] = out["v_hat"] + 10000
    torch.testing.assert_close(compute_visual_loss_terms(out, batch, index, config())["kp"]["raw_sum"], before)


def test_cpu_bf16_raw_losses_are_fp32_finite_and_sigma_freeze_counts_match():
    batch = batch_fixture(dtype=torch.float32)
    index = view_index(batch)
    cfg = config(lambda_sigma=.01, lambda_assign=.05)
    student = FixtureStudent()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = student(batch, index)
        terms = compute_visual_loss_terms(out, batch, index, cfg)
        terms.update(compute_pair_loss_terms(out, batch, index, cfg))
        total = sum(mean(value) for value in terms.values())
    assert all(value["raw_sum"].dtype == torch.float32 and torch.isfinite(value["raw_sum"]) for value in terms.values())
    total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
    out = student(batch, index)
    out["sigma_frozen"] = True
    terms = compute_visual_loss_terms(out, batch, index, cfg, include_assignment=False)
    counts = compute_visual_loss_counts(batch, index, cfg, include_assignment=False, sigma_frozen=True)
    assert terms["sigma"]["valid_count"] == counts["sigma"] == 0
    assert terms["sigma"]["raw_sum"] == 0 and terms["assignment"]["raw_sum"] == 0


def test_calibrated_point_composition_vector_direction_and_rotation_inverse():
    rm = Rotation.from_euler("xyz", [17, -31, 9], degrees=True).as_matrix()
    rv = Rotation.from_euler("xyz", [-11, 8, 43], degrees=True).as_matrix()
    tm, tv = np.array([30., -70., 20.]), np.array([-50., 90., 170.])
    calibration = {"mocap_raw_to_checkerboard_world": {"R": rm.tolist(), "t": tm.tolist()}}
    camera = {"R": rv.tolist(), "t": tv.tolist()}
    points = np.array([[.2, -.3, 1.4], [-1, 2, .5]])
    board = gt_smpl_to_checkerboard(points, calibration)
    projected = checkerboard_to_camera(board, camera)
    expected = points @ (rv @ rm).T + (rv @ tm + tv) / 100
    np.testing.assert_allclose(projected, expected, atol=1e-12)
    np.testing.assert_allclose(camera_to_checkerboard(projected, camera), board, atol=1e-12)
    vector = np.array([.1, -.2, .3])
    moved = checkerboard_to_camera(gt_smpl_to_checkerboard(points + vector, calibration), camera)
    np.testing.assert_allclose(moved - projected, np.broadcast_to((rv @ rm) @ vector, points.shape), atol=1e-12)
    root = Rotation.from_euler("y", 33, degrees=True).as_matrix()
    world_root = gt_root_rotation_to_checkerboard(root, calibration)
    np.testing.assert_allclose(camera_root_rotation_to_checkerboard(rv @ world_root, camera), world_root, atol=1e-12)
    np.testing.assert_allclose(world_root.T @ world_root, np.eye(3), atol=1e-12)
    assert np.linalg.det(world_root) == pytest.approx(1)
