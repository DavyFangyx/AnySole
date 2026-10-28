"""B1 sum/count, camera-frame and empty-target regression checks."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole.data.dataset import grouped_collate
from anysole.train import grouped_view_index
from anysole.utils.b1_training import camera_supervision, teacher_supervision, kd_counts, kd_terms, predictions_to_world
from anysole.utils.b1_training import validation_diagnostics
from anysole.utils.geometry import rot6d_to_rotmat
from test_view_training import ToyGroups


def batch():
    items = [ToyGroups((3, 3))[i] for i in (0, 1)]
    r = torch.tensor([[0., -1, 0], [1., 0, 0], [0., 0, 1]])
    for item in items:
        item["visual"].update(camera_R_cw=r.expand(3, 3, 3).clone(), reference_R_cw=r.clone(),
                               camera_geometry=torch.zeros(3, 16))
        item["meta"]["camera_id"] = torch.tensor([1, 2, 4])
    return grouped_collate(items)


def config(**changes):
    value = dict(pressure_contact_threshold=[1., 1.], pressure_log_total_scale=[.2, .3],
                 kd_pose_local23=.2, kd_pressure=.02, kd_total=0., kd_allocation=0., kd_contact=0.)
    value.update(changes)
    return value


def outputs(n, *, grad=False):
    pose = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(n, 20, 24).requires_grad_(grad)
    pressure = torch.full((n, 20, 96), .1, requires_grad=grad)
    return dict(x0_hat=pose, pressure_hat=pressure, pressure_total=pressure.reshape(n, 20, 2, 48).sum(-1),
                pressure_allocation=pressure.reshape(n, 20, 2, 48).softmax(-1),
                pressure_contact_logits=torch.ones(n, 20, 2, requires_grad=grad), v_hat=torch.ones(n, 20, 3))


def test_camera_rotation_local23_origin_and_inverse_without_gt_input():
    source = batch()
    source["labels"]["root"] += torch.tensor([10., 20, 30])
    source["masks"]["motion_valid"][:, :2] = False
    source["labels"]["root"][:, :2] = float("nan")
    source["labels"]["pose6d"][:, :2] = float("nan")
    index = grouped_view_index(source)
    adapted, flat = camera_supervision(source, index)
    torch.testing.assert_close(adapted["labels"]["pose6d"][:, 2:, 6:], source["labels"]["pose6d"][index[:, 0], 2:, 6:])
    assert torch.isfinite(adapted["labels"]["pose6d"][:, 2:]).all()
    torch.testing.assert_close(adapted["labels"]["root"][:, 2], torch.zeros(6, 3))
    torch.testing.assert_close(adapted["labels"]["velocity"][:, 2], torch.tensor([0., 2, 0]).expand(6, 3))
    out = {"x0_hat": adapted["labels"]["pose6d"], "v_hat": adapted["labels"]["velocity"]}
    restored = predictions_to_world(out, source["visual"]["camera_R_cw"][index[:, 0], index[:, 1]])
    torch.testing.assert_close(restored["x0_hat"][:, 2:], source["labels"]["pose6d"][index[:, 0], 2:])
    torch.testing.assert_close(restored["v_hat"][:, 2:], source["labels"]["velocity"][index[:, 0], 2:])
    assert torch.equal(flat[:, 1], torch.zeros(6, dtype=torch.long))


def test_teacher_edges_require_continuous_source_not_only_or_frame_mask():
    source = batch()
    source["visual"]["view_edges"][:, :, 5] = False
    adapted, _ = teacher_supervision(source, source["visual"]["view_valid"])
    assert adapted["visual"]["view_valid"].all()
    assert not adapted["visual"]["view_edges"][:, :, 5].any()


def test_kd_local23_ignores_root_and_teacher_gradients():
    source, cfg = batch(), config()
    index = grouped_view_index(source)
    student, teacher = outputs(6, grad=True), outputs(2, grad=True)
    teacher["x0_hat"].data[..., :6] = torch.tensor([0., 1, 0, -1, 0, 0])
    teacher["pressure_hat"].data.add_(.2)
    terms = kd_terms(student, teacher, source, index, cfg)
    assert terms["kd_pose_local23"]["raw_sum"].item() == 0.
    assert terms["kd_pose_local23"]["valid_count"] == 6*20*23
    assert terms["kd_pressure"]["valid_count"] == 6*20*96
    objective = sum(v["raw_sum"]/max(v["valid_count"], 1) for v in terms.values())
    objective.backward()
    assert student["pressure_hat"].grad.abs().sum() > 0
    assert teacher["pressure_hat"].grad is None and teacher["x0_hat"].grad is None


def test_empty_teacher_eligibility_retains_gt_support_and_finite_zero_kd():
    source, cfg = batch(), config(kd_pressure=0., kd_total=.02, kd_allocation=.01, kd_contact=.005)
    source["visual"]["view_valid"][:, 1:] = False
    index = grouped_view_index(source)
    terms = kd_terms(outputs(2, grad=True), outputs(2), source, index, cfg)
    assert source["masks"]["motion_valid"].all()
    assert all(v["valid_count"] == 0 and v["raw_sum"].item() == 0. for v in terms.values())
    sum(v["raw_sum"] for v in terms.values()).backward()


def test_kd_denominator_whole_update_and_loaded_foot_allocation_mask():
    source = batch()
    source["masks"]["pressure_valid"][1, :, :48] = False
    index = grouped_view_index(source)
    cfg = config(kd_pressure=0., kd_total=.02, kd_allocation=.01, kd_contact=.005)
    counts = kd_counts(source, index, cfg)
    assert counts["kd_total"] == counts["kd_allocation"] == counts["kd_contact"] == 180
    left = {name: {k: v[:1] if torch.is_tensor(v) else v for k, v in section.items()} for name, section in source.items()}
    right = {name: {k: v[1:] if torch.is_tensor(v) else v for k, v in section.items()} for name, section in source.items()}
    split = [kd_counts(part, grouped_view_index(part), cfg) for part in (left, right)]
    assert counts == {key: split[0][key]+split[1][key] for key in counts}


def test_validation_diagnostics_all_metrics_use_identical_two_view_support():
    import copy
    import numpy as np
    truth = np.tile(np.array([1., 0, 0, 0, 1, 0]), (3, 24))
    item = dict(valid=np.ones(3, bool), teacher_eligible=np.array([True, True, False]),
        motion_valid=np.ones(3, bool), pressure_valid=np.ones((3, 96), bool), frame_id=np.arange(3),
        gt_pose6d=truth, rotation=np.tile(np.eye(3), (3, 24, 1, 1)),
        gt_pressure=np.full((3, 96), .2), pressure=np.full((3, 96), .1))
    series = {("test", 1): item, ("test", 2): copy.deepcopy(item)}
    cfg = dict(pressure_active_threshold=[.05]*96, pressure_highload_threshold=[.15]*96,
               pressure_contact_threshold=[1., 1.])
    result = validation_diagnostics(series, cfg)
    assert result["val/gate/pose_local23"] == 0.
    assert result["val/gate/pose_local23_count"] == 4*23
    assert result["val/gate/pressure_rmse_count"] == 4*96
    assert result["val/gate/total_rmse_count"] == 4*2
    for key in ("pressure_rmse", "active_rmse", "highload_rmse", "total_rmse", "peak_mae"):
        assert result["val/gate/"+key] == pytest.approx(.1)
    assert result["val/gate/contact_f1"] == 1.
    changed = copy.deepcopy(series)
    changed[("test", 1)]["pressure"][:] = .3
    assert validation_diagnostics(changed, cfg)["val/gate_coverage_hash"] == result["val/gate_coverage_hash"]


def test_resume_keeps_selected_validation_when_no_later_checkpoint_wins(tmp_path):
    import json
    from anysole import train
    from test_view_training import config as lifecycle_config, student
    value = lifecycle_config(tmp_path)
    def fixed_validation(*_):
        return {"val/mpjpe": 1., "val/pressure_rmse": 2., "val/motion_frames": 80,
                "val/pressure_cells": 80*96, "val/coverage_hash": "fixed_toy_domain"}, None
    directory = tmp_path/"run"
    def run(**kw):
        return train.train_visual_run(value, directory, dataset=ToyGroups(), validation_dataset=ToyGroups(),
            model=student(), validation_fn=fixed_validation, **kw)
    interrupted = run(stop_after_updates=2)
    assert interrupted["selection"]["best_step"] == 1
    expected = interrupted["best_validation"]
    finished = run(resume=directory/"last.pt")
    assert finished["selection"]["best_step"] == 1
    assert finished["best_validation"] == expected
    assert json.loads((directory/"run.json").read_text())["best_validation"] == expected
