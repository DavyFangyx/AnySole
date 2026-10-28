"""B1 runtime protocol tests use synthetic arrays only, never heldout data."""
from pathlib import Path
import copy
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.tools import b1_evaluate as runtime
from anysole.data.dataset import grouped_collate
from anysole.utils.eval_protocol import aggregate_visual_windows
from test_view_training import ToyGroups


def batch(*, teacher=False, length=20):
    item = ToyGroups((3 if teacher else 1,))[0]
    c = 3 if teacher else 1
    item["visual"].update(camera_R_cw=torch.eye(3).repeat(c, 1, 1),
                          reference_R_cw=torch.eye(3), camera_geometry=torch.zeros(c, 16))
    item["meta"].update(camera_id=torch.tensor([1, 2, 4] if teacher else [3]),
                         source_frame_id=torch.arange(20).repeat(c, 1),
                         source_time_s=torch.arange(20, dtype=torch.float64).repeat(c, 1) / 40,
                         reuse=torch.zeros(c, 20, dtype=torch.bool), time_residual_ms=torch.zeros(c, 20))
    if length != 20:
        for section in ("visual", "labels", "masks", "meta"):
            for name, value in item[section].items():
                if not torch.is_tensor(value):
                    continue
                if name in ("V_feat", "view_valid", "source_frame_id", "source_time_s", "reuse", "time_residual_ms"):
                    item[section][name] = value[:, :length]
                elif name == "view_edges":
                    item[section][name] = value[:, :max(length - 1, 0)]
                elif name in ("pose6d", "pressure", "root", "velocity", "kp_gt", "motion_valid", "pressure_valid", "velocity_valid", "target_time_s", "target_frame_id"):
                    item[section][name] = value[:length]
                elif name in ("motion_edges", "pressure_edges"):
                    item[section][name] = value[:max(length - 1, 0)]
    return grouped_collate([item])


class Oracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward_visual(self, features, valid, times):
        self.seen.append((features.clone(), valid.clone(), times.clone()))
        n, t = valid.shape
        return {"x0_hat": features.new_tensor([1., 0, 0, 0, 1, 0]).repeat(n, t, 24),
                "pressure_hat": features.new_full((n, t, 96), 1.2),
                "v_hat": features.new_tensor([1., 0, 0]).repeat(n, t, 1)}

    def forward_multiview(self, features, valid, times, geometry):
        self.seen.append((features.clone(), valid.clone(), times.clone(), geometry.clone()))
        union = valid.any(1)
        n, t = union.shape
        return {"x0_hat": features.new_tensor([1., 0, 0, 0, 1, 0]).repeat(n, t, 24),
                "pressure_hat": features.new_full((n, t, 96), 1.2),
                "v_hat": features.new_tensor([1., 0, 0]).repeat(n, t, 1),
                "frame_valid": union, "selected_view_valid": valid.clone()}


def test_student_forward_is_invariant_to_gt_and_does_not_clip_pressure():
    original = batch()
    changed = copy.deepcopy(original)
    for value in changed["labels"].values():
        if value.is_floating_point():
            value.fill_(float("nan"))
    model = Oracle()
    left = runtime.collect_batch(model, original, "cpu")[0]
    right = runtime.collect_batch(model, changed, "cpu")[0]
    assert left["camera_id"] == 3
    for name in ("rotation", "pressure", "velocity", "valid"):
        np.testing.assert_array_equal(left[name], right[name])
    assert left["pressure"].min() == pytest.approx(1.2)
    assert all(len(call) == 3 for call in model.seen)


def test_teacher_one_output_uses_actual_union_and_continuous_input_edges():
    source = batch(teacher=True)
    source["visual"]["view_valid"][0, 0, 4] = False
    source["visual"]["view_valid"][0, :, 8] = False
    source["visual"]["view_edges"][0, :, 5] = False
    model = Oracle()
    result = runtime.collect_batch(model, source, "cpu", teacher=True)
    assert len(result) == 1 and result[0]["camera_id"] == 1
    item = result[0]
    assert item["valid"][4] and not item["valid"][8]
    assert not item["view_edges"][5] and not item["view_edges"][7]
    assert item["available_allowed_view_count"][4] == 2
    assert not item["teacher_eligible"][8]
    assert len(model.seen[0]) == 4


def test_teacher_rejects_cam3_before_forward_and_student_rejects_other_views():
    source = batch(teacher=True)
    source["meta"]["camera_id"][0, 1] = 3
    model = Oracle()
    with pytest.raises(ValueError, match="1/2/4"):
        runtime.collect_batch(model, source, "cpu", teacher=True)
    with pytest.raises(ValueError, match="one camera"):
        runtime.collect_batch(model, batch(teacher=True), "cpu")
    assert not model.seen


def test_native_short_tail_keeps_padding_out_of_prediction_extent():
    source = batch(length=7)
    window = runtime.collect_batch(Oracle(), source, "cpu")[0]
    item = aggregate_visual_windows([window])[("toy_0", 3)]
    assert len(item["frame_id"]) == 7
    assert item["valid"].all()
    np.testing.assert_allclose(item["trajectory"][:, 0], np.arange(7) / 40)


def test_overlap_so3_and_velocity_integrate_once_and_reset_at_gap():
    window = runtime.collect_batch(Oracle(), batch(), "cpu")[0]
    duplicate = copy.deepcopy(window)
    window["valid"][9] = duplicate["valid"][9] = False
    duplicate["velocity"] *= 3
    item = aggregate_visual_windows([window, duplicate])[("toy_0", 3)]
    np.testing.assert_allclose(np.linalg.det(item["rotation"][item["valid"]]), 1)
    np.testing.assert_allclose(item["trajectory"][8], [.4, 0, 0])
    np.testing.assert_allclose(item["trajectory"][10], [0, 0, 0])
    np.testing.assert_allclose(item["trajectory"][19], [.45, 0, 0])
    assert item["segment_id"][9] == -1


def test_frozen_plan_checkpoint_and_execution_receipt_are_bound(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture")
    plan = {"frozen": True, "test_access_authorized": True,
            "checkpoint_roles": {"best": "primary: val selector", "last": "diagnostic only"},
            "runs": [{"run_id": "B0-s1", "experiment": "B0", "seed": 1,
                      "checkpoints": {"best": str(checkpoint)},
                      "checkpoint_hashes": {"best": runtime.sha256_file(checkpoint)}}]}
    plan_path = tmp_path / "plan.json"
    runtime.write_json(plan_path, plan)
    _, entry = runtime.load_frozen_entry(plan_path, "B0-s1", "best")
    assert entry["entry_id"] == "B0-s1/best"
    receipt = {"test_inference_authorized": True, "evaluation_plan_sha256": runtime.sha256_file(plan_path),
               "evaluator_sha256": runtime.sha256_file(Path(runtime.__file__)), "allowed_devices": ["cpu"]}
    receipt_path = tmp_path / "receipt.json"
    runtime.write_json(receipt_path, receipt)
    runtime.verify_execution_authorization(receipt_path, plan_path, "cpu", "cpu")
    with pytest.raises(ValueError, match="allocation"):
        runtime.verify_execution_authorization(receipt_path, plan_path, "cuda:0", "cpu")
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="fingerprint"):
        runtime.load_frozen_entry(plan_path, "B0-s1", "best")


def test_workspace_output_and_no_overwrite(tmp_path):
    assert runtime.inside_workspace(Path(runtime.__file__)) == Path(runtime.__file__).resolve()
    with pytest.raises(ValueError, match="workspace"):
        runtime.inside_workspace(Path("/tmp/unapproved"))
    path = tmp_path / "record.json"
    runtime.write_json(path, {"ok": True})
    with pytest.raises(FileExistsError):
        runtime.write_json(path, {"ok": False})


def test_prediction_and_rescoring_archives_have_separate_truth_boundaries(tmp_path, monkeypatch):
    from anysole.data import smpl_io
    monkeypatch.setattr(smpl_io, "_rest_joints", lambda *a, **kw: np.zeros((24, 3)))
    monkeypatch.setattr(smpl_io, "smpl24_pose6d_to_poses", lambda pose: np.zeros((len(pose), 72), np.float32))
    series = aggregate_visual_windows(runtime.collect_batch(Oracle(), batch(), "cpu"))
    series[("toy_0", 3)]["coordinate_frame"] = "world"
    predictions = runtime.export_series(series, tmp_path / "predictions", {"smpl_model_path": "unused"}, {"checkpoint_sha256": "frozen"})
    scoring = runtime.export_scoring_inputs(series, tmp_path / "scoring", {"checkpoint_sha256": "frozen"})
    with np.load(predictions[0]["path"], allow_pickle=False) as archive:
        assert not any(name.startswith("gt_") for name in archive.files)
        assert "motion_valid" not in archive.files and "pressure_valid" not in archive.files
        assert np.max(archive["pressure"]) > 1
        np.testing.assert_array_equal(archive["betas"], np.zeros(10))
        assert json.loads(str(archive["metadata_json"]))["absolute_world_origin"] is None
    with np.load(scoring[0]["path"], allow_pickle=False) as archive:
        assert "gt_pose6d" in archive.files and "motion_edges" in archive.files
        assert "evaluator-only" in json.loads(str(archive["metadata_json"]))["purpose"]
    with pytest.raises(FileExistsError):
        runtime.export_series(series, tmp_path / "predictions", {"smpl_model_path": "unused"}, {})


def test_empty_student_rgb_retains_label_denominator_and_no_forward():
    source = batch()
    source["visual"]["view_valid"].zero_()
    model = Oracle()
    window = runtime.collect_batch(model, source, "cpu")[0]
    item = aggregate_visual_windows([window])[("toy_0", 3)]
    assert len(item["frame_id"]) == 20 and not item["valid"].any()
    assert item["motion_valid"].sum() == 20 and item["pressure_valid"].sum() == 20 * 96
    assert not model.seen


def test_recording_collector_templates_preserve_gapped_native_ids_and_restore_rng():
    class NativeFixture:
        def __init__(self):
            n = 23
            self.ids = np.r_[np.arange(10), np.arange(11, 24)]
            self.times = np.arange(n, dtype=np.float64) / 40
            self.windows = [(0, 0, 20, (3,)), (0, 3, 23, (3,))]
            self.valid = np.ones(n, bool)
            self.valid[7] = False
            view = {"camera_id": 3, "source_frame_id": self.ids, "source_time_s": self.times,
                    "reuse": np.zeros(n, bool), "time_residual_ms": np.zeros(n),
                    "view_edges": np.diff(self.ids) == 1}
            self.records = [{"recording_id": "toy_0", "target_frame_id": self.ids, "target_time_s": self.times,
                             "views": [view], "motion": {"valid": np.ones(n, bool)},
                             "pressure": {"foot_valid": np.ones((n, 2), bool)},
                             "motion_edges": np.diff(self.ids) == 1,
                             "pressure_foot_edges": np.repeat((np.diff(self.ids) == 1)[:, None], 2, 1),
                             "available_allowed_view_count": np.full(n, 3)}]
            self.labels = {key: array.numpy() for key, array in ToyGroups((1,))[0]["labels"].items()}
            for key in ("pose6d", "pressure", "root", "velocity", "kp_gt"):
                source = self.labels[key]
                self.labels[key] = np.concatenate((source, source[-3:]), 0)

        def __len__(self):
            return len(self.windows)

        def load_labels(self, _ri):
            return self.labels

        def __getitem__(self, i):
            _, start, end, _ = self.windows[i]
            item = ToyGroups((1,))[0]
            item["visual"].update(camera_R_cw=torch.eye(3)[None], reference_R_cw=torch.eye(3),
                                  camera_geometry=torch.zeros(1, 16),
                                  view_valid=torch.from_numpy(self.valid[start:end][None]),
                                  view_edges=torch.from_numpy((self.valid[:-1] & self.valid[1:] & (np.diff(self.ids) == 1))[start:end-1][None]),
                                  target_time_s=torch.from_numpy(self.times[start:end]))
            item["labels"] = {key: torch.from_numpy(value[start:end] if key in
                              ("pose6d", "pressure", "root", "velocity", "kp_gt") else value)
                              for key, value in self.labels.items()}
            item["masks"]["motion_edges"] = torch.from_numpy((np.diff(self.ids) == 1)[start:end-1])
            item["masks"]["pressure_edges"] = item["masks"]["motion_edges"][:, None].repeat(1, 96)
            item["meta"].update(camera_id=torch.tensor([3]), target_frame_id=torch.from_numpy(self.ids[start:end]),
                                source_frame_id=torch.from_numpy(self.ids[start:end][None]),
                                source_time_s=torch.from_numpy(self.times[start:end][None]),
                                reuse=torch.zeros(1, 20, dtype=torch.bool), time_residual_ms=torch.zeros(1, 20))
            return item
    dataset = NativeFixture()
    original = dataset.windows
    model = Oracle().train()
    torch.manual_seed(17)
    state = torch.random.get_rng_state()
    item = runtime.collect_recording(model, dataset, "toy_0", torch.device("cpu"), batch_size=2)[("toy_0", 3)]
    assert dataset.windows is original and model.training
    assert torch.equal(torch.random.get_rng_state(), state)
    np.testing.assert_array_equal(item["frame_id"], dataset.ids)
    assert len(item["valid"]) == 23 and not item["valid"][7]
    assert item["overlap_count"][10] == 2 and not item["edges"][9]
    assert np.all(item["available_allowed_view_count"] == 3)
