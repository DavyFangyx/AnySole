"""B1 single-view equivalence, true set fusion, masks and initialization."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.models.common.pressure_a2 import QueryPressureHead
from anysole.models.multiview_teacher import AnySoleMultiviewTeacher
from anysole.models.v3_3b import STRUCTURE
from anysole.models.visual import build, effective_config
from anysole.utils.a2_config import A3_VERSION


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_model(teacher=False, *, head="query_factorized"):
    torch.manual_seed(91)
    cls = AnySoleMultiviewTeacher if teacher else AnySoleModelV2
    model = cls(d=32, tw=5, dropout=0, nhead=8, pose_layers=1,
                visual_only=True, pressure_head="time_pe" if head == "query_factorized" else head,
                **STRUCTURE)
    if head == "query_factorized":
        model.pressure_readout = QueryPressureHead(kind="query_factorized", dim=32, nhead=8,
            hidden=16, dropout=0, cell_mean=[.2] * 96, contact_rate=[.3, .4],
            memory="fusion_tactile_slice")
        model.aux_heads.t_rec = torch.nn.Identity()
        model.pressure_head = head
    return model.eval()


def multiview_inputs(g=2, c=3):
    generator = torch.Generator().manual_seed(332)
    features = torch.randn(g, c, 5, 2051, generator=generator)
    valid = torch.ones(g, c, 5, dtype=torch.bool)
    times = torch.arange(5, dtype=torch.float64)[None].expand(g, -1) / 40
    geometry = torch.randn(g, c, 16, generator=generator)
    return features, valid, times, geometry


def assert_predictions_equal(first, second, *, atol=1e-6, rtol=1e-5):
    for key in ("x0_hat", "v_hat", "trans_hat", "pressure_hat", "F"):
        torch.testing.assert_close(first[key], second[key], atol=atol, rtol=rtol)


@pytest.mark.parametrize("head", ["legacy", "time_pe", "query_factorized"])
@pytest.mark.parametrize("partial", [False, True])
def test_visual_encode_decode_matches_original_computation_exactly(head, partial):
    model = small_model(head=head)
    features, valid, times, _ = multiview_inputs(g=1, c=2)
    features, valid, times = features[0], valid[0], times.expand(2, -1)
    if partial:
        valid[:, 2] = False
        features[:, 2] = float("nan")
    # Replay the pre-extraction path directly through the unchanged modules.
    from anysole.models.common.embeddings import relative_time_pe
    pe = relative_time_pe(times, model.d, 40., features.dtype) if model.pressure_time_pe_enabled else None
    v_tok, t_tok = model.encoders.forward_visual(features, valid, pe)
    memory_mask = torch.cat([~valid, ~valid], dim=1)
    padding = memory_mask if bool(memory_mask.any()) else None
    memory = model.fusion(v_tok, t_tok, memory_padding_mask=padding)
    memory = torch.where(memory_mask[..., None], torch.zeros_like(memory), memory)
    expected_pose = model.pose_head(F=memory, frame_valid=valid, memory_padding_mask=padding)
    expected_velocity, expected_translation = model.traj_head(memory, frame_valid=valid, timestamps=times)
    expected_pressure = (model.pressure_readout(memory, valid, times)["pressure_hat"]
                         if hasattr(model, "pressure_readout") else model.aux_heads(memory)[0])
    encoded = model.encode_visual(features, valid, times)
    decoded = model.decode_memory(encoded["F"], valid, times, encoded["v_tok"], encoded["t_tok"])
    actual = model.forward_visual(features, valid, times)
    for key, expected in (("F", memory), ("x0_hat", expected_pose), ("v_hat", expected_velocity),
                          ("trans_hat", expected_translation), ("pressure_hat", expected_pressure)):
        torch.testing.assert_close(actual[key], expected, atol=0, rtol=0)
        torch.testing.assert_close(decoded[key], expected, atol=0, rtol=0)


def test_teacher_is_invariant_to_synchronized_view_permutation():
    model = small_model(teacher=True)
    features, valid, times, geometry = multiview_inputs()
    valid[0, 1, (1, 4)] = False
    first = model.forward_multiview(features, valid, times, geometry)
    permutation = torch.tensor([2, 0, 1])
    second = model.forward_multiview(features[:, permutation], valid[:, permutation], times, geometry[:, permutation])
    assert_predictions_equal(first, second, atol=2e-6)
    assert torch.equal(first["frame_valid"], second["frame_valid"])
    assert torch.equal(first["teacher_valid"], second["teacher_valid"])
    assert torch.equal(first["selected_views"][:, permutation], second["selected_views"])


def test_teacher_uses_all_views_geometry_and_one_decoder_without_tactile():
    model = small_model(teacher=True)
    features, valid, times, geometry = multiview_inputs(g=1)
    features.requires_grad_()
    geometry.requires_grad_()
    calls = []
    def forbidden(*_args):
        raise AssertionError("real tactile encoder must not execute")
    handles = [model.encoders.t_enc.register_forward_pre_hook(forbidden),
               model.pose_head.register_forward_pre_hook(lambda *_args: calls.append(True))]
    try:
        result = model.forward_multiview(features, valid, times, geometry)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [True]
    result["x0_hat"].square().mean().backward()
    assert (features.grad.abs().sum((2, 3)) > 0).all()
    assert (geometry.grad.abs().sum(-1) > 0).all()
    changed = features.detach().clone()
    changed[:, 2] = torch.randn_like(changed[:, 2]) * 8
    other = model.forward_multiview(changed, valid, times, geometry.detach())
    assert not torch.allclose(result["x0_hat"], other["x0_hat"])
    shared = {id(p) for p in model.embeddings.parameters()}
    assert all(p.grad is None for p in model.encoders.t_enc.parameters() if id(p) not in shared)
    with pytest.raises(TypeError):
        model.forward_multiview(features, valid, times, geometry, labels={})
    assert tuple(inspect.signature(model.forward_visual).parameters) == ("V_feat", "frame_valid", "timestamps")


def test_missing_views_times_and_groups_are_zero_finite_and_have_no_gradients():
    model = small_model(teacher=True)
    features, valid, times, geometry = multiview_inputs(g=3)
    valid[0, 2] = False
    valid[0, :, 2] = False
    valid[1] = False
    valid[2, 1:, 3] = False
    clean = model.forward_multiview(features, valid, times, geometry)
    features[~valid] = float("nan")
    geometry[~valid.any(-1)] = float("inf")
    features.requires_grad_()
    geometry.requires_grad_()
    out = model.forward_multiview(features, valid, times, geometry)
    assert_predictions_equal(clean, out, atol=0, rtol=0)
    assert torch.equal(out["frame_valid"], valid.any(1))
    assert torch.equal(out["teacher_valid"], valid.sum(1) >= 2)
    for key in ("x0_hat", "v_hat", "trans_hat", "pressure_hat", "vfeat_hat", "gate_sigma",
                "pressure_contact_logits", "pressure_total", "pressure_allocation"):
        assert torch.isfinite(out[key]).all(), key
        assert torch.count_nonzero(out[key][~out["frame_valid"]]) == 0, key
    assert torch.count_nonzero(out["F"][1]) == 0
    out["x0_hat"].square().mean().backward()
    assert torch.count_nonzero(features.grad[~valid]) == 0
    assert torch.count_nonzero(geometry.grad[~valid.any(-1)]) == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_all_empty_teacher_batch_never_calls_attention_or_decoders():
    model = small_model(teacher=True)
    features, valid, times, geometry = multiview_inputs()
    valid.zero_()
    features.fill_(float("nan"))
    geometry.fill_(float("nan"))
    def forbidden(*_args):
        raise AssertionError("empty input must not be evaluated as an observation")
    modules = [model.fusion, model.pose_head, model.traj_head, model.multiview_fusion.refinement]
    modules += [stream.attention for stream in model.multiview_fusion.streams]
    handles = [module.register_forward_pre_hook(forbidden) for module in modules]
    try:
        out = model.forward_multiview(features, valid, times, geometry)
    finally:
        for handle in handles:
            handle.remove()
    assert not out["frame_valid"].any()
    assert not out["teacher_valid"].any()
    assert out["memory_padding_mask"].all()
    for key in ("x0_hat", "v_hat", "trans_hat", "pressure_hat", "F", "gate_sigma"):
        assert torch.isfinite(out[key]).all()
        assert torch.count_nonzero(out[key]) == 0


def test_teacher_cpu_autocast_missing_frames_keep_outputs_and_gradients_finite():
    teacher = small_model(teacher=True).train()
    features, valid, times, geometry = multiview_inputs()
    valid[0, :, 2] = False
    valid[1, 2] = False
    features[~valid] = float("nan")
    features.requires_grad_()
    geometry.requires_grad_()
    selected, _ = teacher.select_views(valid)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = teacher.forward_multiview(features, valid, times, geometry, selected_view_valid=selected)
        loss = out["x0_hat"].square().mean() + out["pressure_hat"].square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(features.grad[~selected]) == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in teacher.parameters())


def test_empty_padded_view_cannot_change_result_or_dilute_attention():
    model = small_model(teacher=True)
    features, valid, times, geometry = multiview_inputs(c=2)
    original = model.forward_multiview(features, valid, times, geometry)
    padded = model.forward_multiview(
        torch.cat([features, torch.full_like(features[:, :1], float("nan"))], 1),
        torch.cat([valid, torch.zeros_like(valid[:, :1])], 1), times,
        torch.cat([geometry, torch.full_like(geometry[:, :1], float("nan"))], 1))
    assert_predictions_equal(original, padded, atol=2e-6)
    assert torch.equal(original["view_counts"], padded["view_counts"])


def test_dropout_precedes_encoder_and_preselected_masks_do_not_draw_again():
    model = small_model(teacher=True).train()
    model.three_view_probability = 0.
    features, valid, times, geometry = multiview_inputs(g=2)
    selected, selected_views = model.select_views(valid)
    assert (selected_views.sum(1) == 2).all()
    features[~selected] = float("nan")
    seen = []
    handle = model.encoders.v_enc.register_forward_pre_hook(lambda _module, args: seen.append(args[0].shape[0]))
    rng = torch.get_rng_state().clone()
    try:
        out = model.forward_multiview(features, valid, times, geometry, selected_view_valid=selected)
    finally:
        handle.remove()
    assert seen == [4]
    # dropout=0 for modules, so only subset selection could consume RNG.
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(out["selected_views"], selected_views)
    assert torch.equal(out["selected_view_valid"], selected)
    assert torch.equal(out["rgb_window_counts"], torch.tensor([2, 2]))
    bad = selected.clone()
    bad[0, selected_views[0].nonzero()[0], 1] = False
    with pytest.raises(ValueError, match="whole views"):
        model.forward_multiview(features, valid, times, geometry, selected_view_valid=bad)


def test_default_dropout_probabilities_and_eval_use_all_available_views():
    model = small_model(teacher=True).train()
    valid = torch.ones(10000, 3, 5, dtype=torch.bool)
    torch.manual_seed(233)
    selected, views = model.select_views(valid)
    three = (views.sum(1) == 3).float().mean().item()
    assert .68 < three < .72
    assert (views.sum(1) >= 2).all()
    dropped = (~views).sum(0).float()
    assert ((dropped / dropped.sum()) - 1 / 3).abs().max() < .025
    valid[:2, 2] = False
    model.eval()
    selected, views = model.select_views(valid)
    assert torch.equal(selected, valid)
    assert torch.equal(views, valid.any(-1))


def test_baseline_copy_is_complete_and_parameters_remain_independent():
    baseline = small_model()
    teacher = small_model(teacher=True)
    new_before = {k: value.clone() for k, value in teacher.state_dict().items()
                  if k.startswith(teacher.NEW_STATE_PREFIX)}
    receipt = teacher.initialize_from_baseline(baseline.state_dict())
    assert set(receipt["copied_keys"]) == set(baseline.state_dict())
    assert receipt["new_keys"]
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, teacher.state_dict()[key], atol=0, rtol=0)
        assert value.data_ptr() != teacher.state_dict()[key].data_ptr()
    for key, value in new_before.items():
        torch.testing.assert_close(value, teacher.state_dict()[key], atol=0, rtol=0)
    state = baseline.state_dict()
    incomplete = dict(state)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="missing"):
        teacher.initialize_from_baseline(incomplete)
    with pytest.raises(ValueError, match="extra"):
        teacher.initialize_from_baseline({**state, "unrelated.weight": torch.zeros(1)})
    wrong = dict(state)
    wrong["encoders.v_enc.proj.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape"):
        teacher.initialize_from_baseline(wrong)
    teacher.initialize_from_baseline(baseline)
    with torch.no_grad():
        teacher.encoders.v_enc.proj.weight.add_(1)
    assert not torch.equal(teacher.encoders.v_enc.proj.weight, baseline.encoders.v_enc.proj.weight)


def b1_config(role):
    return {"b1_version": "anysole.b1.v1", "b1_role": role, "a2_version": A3_VERSION,
            "pressure_head": "query_factorized", "query_memory": "fusion_tactile_slice",
            "a2_head_seed": 1, "pressure_cell_mean": [.2] * 96,
            "pressure_contact_rate": [.3, .4], "dropout": 0}


def test_factory_teacher_student_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(5)
    baseline = build(b1_config("baseline"))
    student = build(b1_config("student")).eval()
    student.load_state_dict(baseline.state_dict(), strict=True)
    teacher = build(b1_config("teacher")).eval()
    assert isinstance(teacher, AnySoleMultiviewTeacher)
    teacher.initialize_from_baseline(baseline)
    assert student.tw == 20 and student.pressure_readout.memory == "fusion_tactile_slice"
    assert teacher.structure_hash != baseline.structure_hash
    config = teacher.effective_config
    assert effective_config(config)["structure_hash"] == teacher.structure_hash
    path = tmp_path / "teacher.pt"
    torch.save({"model": teacher.state_dict(), "config": config}, path)
    restored_checkpoint = torch.load(path, weights_only=True)
    restored = build(restored_checkpoint["config"]).eval()
    restored.load_state_dict(restored_checkpoint["model"], strict=True)
    features = torch.randn(1, 2, 20, 2051)
    valid = torch.ones(1, 2, 20, dtype=torch.bool)
    times = torch.arange(20, dtype=torch.float64)[None] / 40
    geometry = torch.randn(1, 2, 16)
    expected = teacher.forward_multiview(features, valid, times, geometry)
    actual = restored.forward_multiview(features, valid, times, geometry)
    assert_predictions_equal(expected, actual, atol=0, rtol=0)
    student_out = student.forward_visual(features[:, 0], valid[:, 0], times)
    baseline.eval()
    assert_predictions_equal(student_out, baseline.forward_visual(features[:, 0], valid[:, 0], times), atol=0, rtol=0)


@pytest.mark.parametrize("change", [
    {"b1_version": "future"}, {"b1_role": "joint"}, {"query_memory": "fusion_all"},
    {"teacher_reference_camera": 2}, {"teacher_cross_view_layers": 2},
    {"teacher_view_dropout": {"three_view_probability": .5, "min_views": 2}},
])
def test_b1_factory_rejects_incompatible_architecture(change):
    with pytest.raises(ValueError):
        build({**b1_config("teacher"), **change})


def test_teacher_bad_input_contracts():
    teacher = small_model(teacher=True)
    features, valid, times, geometry = multiview_inputs()
    with pytest.raises(ValueError, match="geometry"):
        teacher.forward_multiview(features, valid, times, geometry[..., :15])
    with pytest.raises(ValueError, match="non-decreasing"):
        teacher.forward_multiview(features, valid, times.flip(1), geometry)
    features[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="valid visual"):
        teacher.forward_multiview(features, valid, times, geometry)
