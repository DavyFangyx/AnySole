"""RGB student input, mask, temporal and historical-model regression checks."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole.models.common.model import AnySoleModel
from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.models.common.traj_head import TrajHead
from anysole.models.visual import build, effective_config
from anysole.models.v3_3b import STRUCTURE
from anysole.registry import get_builder
from anysole.types import CONFIG_V, JOINT_PARENTS, PART_JOINTS
from anysole.utils.geometry import rot6d_to_rotmat
from anysole.utils.losses import compute_losses


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_model(head="time_pe", visual_only=True):
    torch.manual_seed(184)
    return AnySoleModelV2(d=32, tw=5, dropout=0, pose_layers=1,
                          visual_only=visual_only, pressure_head=head, **STRUCTURE).eval()


def inputs(n=1):
    generator = torch.Generator().manual_seed(72)
    v = torch.randn(n, 5, 2051, generator=generator)
    valid = torch.ones(n, 5, dtype=torch.bool)
    times = torch.arange(5, dtype=torch.float64).expand(n, -1) / 40
    return v, valid, times


def assert_outputs_close(a, b, valid=None, atol=1e-6):
    for key in ("x0_hat", "pressure_hat", "v_hat", "trans_hat"):
        x, y = a[key], b[key]
        if valid is not None:
            x, y = x[valid], y[valid]
        torch.testing.assert_close(x, y, rtol=1e-5, atol=atol)


def test_strict_visual_does_not_read_tactile_or_labels():
    m = small_model()
    args = inputs()
    assert tuple(inspect.signature(m.forward_visual).parameters) == ("V_feat", "frame_valid", "timestamps")
    def forbidden(*_args):
        raise AssertionError("tactile encoder was evaluated")
    handle = m.encoders.t_enc.register_forward_pre_hook(forbidden)
    try:
        out = m.forward_visual(*args)
    finally:
        handle.remove()
    assert out["F"].shape == (1, 10, 32)
    assert out["x0_hat"].shape == (1, 5, 144)
    assert out["pressure_hat"].shape == (1, 5, 96)
    assert out["memory_padding_mask"].shape == (1, 10)
    with pytest.raises(TypeError):
        m.forward_visual(*args, labels={})
    with pytest.raises(ValueError, match="forward_visual"):
        m(args[0], torch.zeros(1, 5, 96), torch.zeros(1, 5, 12), torch.tensor([CONFIG_V]))


def test_masked_nan_and_inf_cannot_pollute_output_or_gradients():
    m = small_model()
    v, valid, times = inputs(2)
    valid[:, (1, 4)] = False
    clean = m.forward_visual(v, valid, times)
    damaged = v.clone()
    damaged[:, 1] = float("nan")
    damaged[:, 4] = float("inf")
    damaged.requires_grad_()
    changed = m.forward_visual(damaged, valid, times)
    assert_outputs_close(clean, changed)
    loss = changed["x0_hat"][valid].square().mean() + changed["pressure_hat"][valid].square().mean()
    loss.backward()
    assert torch.count_nonzero(damaged.grad[~valid]) == 0
    assert damaged.grad[valid].abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    # T's encoder references the shared null/modality embeddings too; those
    # legitimately receive RGB/pose gradients without executing t_enc.
    shared = {id(p) for p in m.embeddings.parameters()}
    assert all(p.grad is None for p in m.encoders.t_enc.parameters() if id(p) not in shared)
    assert torch.equal(changed["memory_padding_mask"], torch.cat([~valid, ~valid], dim=1))


def test_pose_invalid_queries_are_not_keys_for_valid_queries():
    m = small_model()
    v, valid, times = inputs()
    valid[:, (1, 3)] = False
    before = m.forward_visual(v, valid, times)
    with torch.no_grad():
        m.pose_head.query[:, (1, 3)] += torch.randn_like(m.pose_head.query[:, (1, 3)]) * 1000
    after = m.forward_visual(v, valid, times)
    assert_outputs_close(before, after, valid=valid)


def test_attention_receives_memory_source_and_time_major_query_masks():
    m = small_model()
    v, valid, times = inputs()
    valid[:, (0, 3)] = False
    seen = {}
    def record(name):
        def hook(_module, _args, kwargs):
            seen[name] = kwargs
        return hook
    block = m.pose_head.decoder.layers[0]
    modules = {"fusion": m.fusion.encoder, "query": block.self_attn,
               "v_cross": block.multihead_attn, "t_cross": block.multihead_attn_t,
               "traj": m.traj_head.encoder}
    handles = [module.register_forward_pre_hook(record(name), with_kwargs=True)
               for name, module in modules.items()]
    try:
        m.forward_visual(v, valid, times)
    finally:
        for handle in handles:
            handle.remove()
    memory = torch.cat([~valid, ~valid], dim=1)
    assert torch.equal(seen["fusion"]["src_key_padding_mask"], memory)
    assert torch.equal(seen["query"]["key_padding_mask"], (~valid).repeat_interleave(9, dim=1))
    assert torch.equal(seen["v_cross"]["key_padding_mask"], memory | (torch.arange(10)[None] >= 5))
    assert torch.equal(seen["t_cross"]["key_padding_mask"], memory | (torch.arange(10)[None] < 5))
    assert torch.equal(seen["traj"]["src_key_padding_mask"], ~valid)


@pytest.mark.parametrize("valid_position", (0, 2, 4))
def test_one_valid_frame_never_creates_all_mask_attention(valid_position):
    m = small_model()
    v, valid, times = inputs()
    valid[:] = False
    valid[:, valid_position] = True
    v[~valid] = float("nan")
    out = m.forward_visual(v, valid, times)
    for key in ("x0_hat", "pressure_hat", "v_hat", "trans_hat", "gate_sigma"):
        assert torch.isfinite(out[key]).all()
    assert torch.count_nonzero(out["trans_hat"]) == 0


@pytest.mark.parametrize("c", (1, 2, 3, 4))
def test_flat_views_are_independent_for_variable_camera_counts(c):
    m = small_model()
    v, valid, times = inputs(c)
    out = m.forward_visual(v, valid, times)
    for i in range(c):
        solo = m.forward_visual(v[i:i+1], valid[i:i+1], times[i:i+1])
        assert_outputs_close({k: value[i:i+1] for k, value in out.items() if torch.is_tensor(value)
                              and k in ("x0_hat", "pressure_hat", "v_hat", "trans_hat")}, solo)
    if c > 1:
        v[1:] = torch.randn_like(v[1:]) * 100
        changed = m.forward_visual(v, valid, times)
        for key in ("x0_hat", "pressure_hat", "v_hat", "trans_hat"):
            torch.testing.assert_close(out[key][0], changed[key][0], rtol=0, atol=0)


def test_all_invalid_empty_and_bad_inputs_are_rejected():
    m = small_model()
    v, valid, times = inputs(2)
    valid[1] = False
    with pytest.raises(ValueError, match="all-invalid"):
        m.forward_visual(v, valid, times)
    with pytest.raises(ValueError, match="empty"):
        m.forward_visual(v[:0], valid[:0], times[:0])
    valid[:] = True
    v[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="valid visual"):
        m.forward_visual(v, valid, times)
    v[0, 0, 0] = 0
    with pytest.raises(ValueError, match="bool"):
        m.forward_visual(v, valid.float(), times)
    with pytest.raises(ValueError, match="non-decreasing"):
        m.forward_visual(v, valid, times.flip(1))


def test_native_duplicate_timestamps_preserve_all_rgb_observations():
    m = small_model()
    v, valid, times = inputs()
    times = times.clone()
    times[:, 2] = times[:, 1]
    out = m.forward_visual(v, valid, times)
    for key in ("x0_hat", "pressure_hat", "v_hat", "trans_hat"):
        assert out[key].shape[:2] == valid.shape
        assert torch.isfinite(out[key]).all()
    assert torch.equal(out["frame_valid"], valid)


def test_relative_time_pe_ignores_absolute_clock_and_preserves_gaps():
    m = small_model()
    v, valid, times = inputs()
    out = m.forward_visual(v, valid, times)
    shifted = m.forward_visual(v, valid, times + 65000.0)
    assert_outputs_close(out, shifted, atol=2e-5)
    gap_times = torch.tensor([[0, .025, .050, .150, .175]], dtype=torch.float64)
    gap = m.forward_visual(v, valid, gap_times)
    # The internal T-token PE retains target positions across the gap.
    assert not torch.allclose(out["t_tok"][:, 3], gap["t_tok"][:, 3])
    torch.testing.assert_close(out["v_tok"], gap["v_tok"], rtol=0, atol=0)


def test_legacy_visual_matches_old_full_valid_vonly_and_stays_constant():
    m = small_model("legacy", visual_only=False)
    v, valid, times = inputs(2)
    out = m.forward_visual(v, valid, times)
    old = m(v, torch.randn(2, 5, 96), torch.randn(2, 5, 12), torch.full((2,), CONFIG_V))
    for key in ("x0_hat", "v_hat", "pressure_hat", "F", "v_tok", "t_tok"):
        torch.testing.assert_close(out[key], old[key], rtol=0, atol=0)
    torch.testing.assert_close(out["pressure_hat"], out["pressure_hat"][:, :1].expand(-1, 5, -1), rtol=0, atol=1e-6)
    assert torch.count_nonzero(out["trans_hat"][:, 0]) == 0
    assert not torch.allclose(old["trans_hat"][:, 0], out["trans_hat"][:, 0])


def test_time_pe_is_deterministic_visual_dependent_and_keeps_foot_channel_order():
    m = small_model()
    v, valid, times = inputs()
    v.requires_grad_()
    a = m.forward_visual(v, valid, times)
    b = m.forward_visual(v, valid, times)
    assert_outputs_close(a, b, atol=0)
    assert (a["pressure_hat"] - a["pressure_hat"][:, :1]).abs().max() > 1e-5
    changed = m.forward_visual(v + torch.randn_like(v), valid, times)
    assert not torch.allclose(a["pressure_hat"], changed["pressure_hat"])
    a["pressure_hat"].square().mean().backward()
    assert v.grad.abs().sum() > 0
    with torch.no_grad():
        m.aux_heads.t_rec.bias[:48] += 1
    left_changed = m.forward_visual(v, valid, times)["pressure_hat"]
    torch.testing.assert_close(left_changed[..., :48] - a["pressure_hat"][..., :48],
                               torch.ones_like(left_changed[..., :48]), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(left_changed[..., 48:], a["pressure_hat"][..., 48:], rtol=0, atol=0)


def test_pose_is_denormalized_once_and_rotation_conversion_is_legal():
    m = small_model()
    args = inputs()
    normalized = m.forward_visual(*args)["x0_hat"]
    mean = torch.linspace(.1, .9, 144)
    std = torch.linspace(.2, .8, 144)
    m.pose_head.set_stats(mean, std)
    raw = m.forward_visual(*args)["x0_hat"]
    torch.testing.assert_close(raw, normalized * std + mean, rtol=0, atol=0)
    rotations = rot6d_to_rotmat(raw.reshape(1, 5, 24, 6))
    identity = torch.eye(3).expand_as(rotations)
    torch.testing.assert_close(rotations.transpose(-2, -1) @ rotations, identity, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(1, 5, 24), rtol=1e-4, atol=1e-5)


def test_regularizer_targets_are_detached_but_sigma_and_assignment_train():
    m = small_model()
    v, valid, times = inputs()
    v.requires_grad_()
    out = m.forward_visual(v, valid, times)
    assert out["gate_sigma"].requires_grad and out["part_logits"].requires_grad
    assert not out["gate_g"].requires_grad
    assert not out["part_assignment"].requires_grad
    assert not out["part_hypotheses"].requires_grad
    batch = {"pose_gt": torch.randn(1, 5, 144), "trans_gt": torch.zeros(1, 5, 3),
             "vel_gt": torch.zeros(1, 5, 3), "T_raw": torch.zeros(1, 5, 96), "V_feat": v,
             "offsets": torch.zeros(1, 24, 3), "parents": torch.tensor(JOINT_PARENTS)[None],
             "kp_gt": torch.zeros(1, 5, 24, 3)}
    terms = compute_losses(out, batch, torch.tensor([CONFIG_V]),
                           {"soft_parts": True, "lambda_sigma": .01, "lambda_con": 0,
                            "traj_deltas": [2, 4]})
    terms["L_sigma"].backward(retain_graph=True)
    assert v.grad.abs().sum() > 0
    terms["L_assign"].backward()
    assert m.pose_head.part_logits.grad.abs().sum() > 0
    m.pose_head.set_sigma_frozen(True)
    frozen = m.forward_visual(v, valid, times)
    assert frozen["sigma_frozen"] is True
    assert torch.count_nonzero(frozen["gate_sigma"]) == 0


def test_actual_dt_trajectory_has_zero_origin_and_restarts_after_missing_or_gap():
    head = TrajHead(dim=8, tw=5, nhead=2, dropout=0).eval()
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.bias.fill_(1)
    memory = torch.zeros(1, 10, 8)
    valid = torch.tensor([[True, True, False, True, True]])
    times = torch.tensor([[0, .02, .06, .20, .225]], dtype=torch.float64)
    velocity, position = head(memory, valid, times)
    torch.testing.assert_close(position[0, :, 0], torch.tensor([0, .02, 0, 0, .025]))
    assert torch.count_nonzero(velocity[~valid]) == 0
    valid[:] = True
    _, position = head(memory, valid, torch.tensor([[0, .025, .05, .15, .175]], dtype=torch.float64))
    torch.testing.assert_close(position[0, :, 0], torch.tensor([0, .025, .05, 0, .025]))


def test_cpu_autocast_keeps_masked_forward_and_gradients_finite():
    m = small_model()
    v, valid, times = inputs()
    valid[:, 2] = False
    v[:, 2] = float("nan")
    v.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m.forward_visual(v, valid, times)
        loss = out["x0_hat"][valid].square().mean() + out["pressure_hat"][valid].square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert torch.count_nonzero(v.grad[~valid]) == 0


@pytest.mark.parametrize("name", ("F0b", "F4a", "F2", "F2p4", "V3_2", "V3_3A", "V3_3B",
                                  "V3_3C", "V3_4a", "V3_4b", "V3_4c", "V4A", "V4B"))
def test_historical_regression_families_keep_all_modes(name):
    config = {"d_model": 32, "tw": 5, "pose_layers": 1, "dropout": 0}
    m = get_builder(name).build(config, part_joints=PART_JOINTS if name == "V4B" else None).eval()
    v, _, _ = inputs()
    for cid in (0, 1, 2):
        out = m(v, torch.zeros(1, 5, 96), torch.zeros(1, 5, 12), torch.tensor([cid]))
        assert out["x0_hat"].shape == (1, 5, 144)
        assert all(torch.isfinite(out[key]).all() for key in ("x0_hat", "v_hat", "trans_hat", "pressure_hat"))


def test_historical_v1_diffusion_positional_signature_still_works():
    m = AnySoleModel(d=32, tw=5, pose_layers=1, dropout=0).eval()
    v, _, _ = inputs()
    out = m(v, torch.zeros(1, 5, 96), torch.zeros(1, 5, 12), torch.zeros(1, 5, 144),
            torch.zeros(1, dtype=torch.long), torch.tensor([CONFIG_V]))
    assert out["x0_hat"].shape == (1, 5, 144)
    assert torch.isfinite(out["x0_hat"]).all()


def test_registered_checkpoint_builder_roundtrip_and_structure_validation(tmp_path):
    cfg = effective_config({"pressure_head": "time_pe", "dropout": 0})
    model = get_builder("visual").build(cfg).eval()
    legacy = build({"pressure_head": "legacy", "dropout": 0})
    assert model.pressure_time_pe_enabled is True and model.effective_config["pressure_time_pe_enabled"] is True
    assert legacy.pressure_time_pe_enabled is False and legacy.effective_config["pressure_time_pe_enabled"] is False
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in legacy.parameters())
    assert model.structure_hash != legacy.structure_hash
    assert effective_config({**cfg, "run_dir": "/different", "seed": 9})["structure_hash"] == model.structure_hash
    path = tmp_path / "student.pt"
    torch.save({"model": model.state_dict(), "config": model.effective_config}, path)
    saved = torch.load(path, weights_only=True)
    restored = build(saved["config"]).eval()
    restored.load_state_dict(saved["model"], strict=True)
    v = torch.randn(1, 20, 2051)
    valid = torch.ones(1, 20, dtype=torch.bool)
    times = torch.arange(20, dtype=torch.float64)[None] / 40
    assert_outputs_close(model.forward_visual(v, valid, times), restored.forward_visual(v, valid, times), atol=0)
    for invalid in ({}, {**cfg, "pressure_head": "query"}, {**cfg, "pressure_head": "legacy"},
                    {**cfg, "pressure_time_pe": "after_fusion"}, {**cfg, "tw": 40},
                    {**cfg, "visual_structure_version": "unknown"}, {**cfg, "nhead": 4},
                    {**cfg, "modal": "anysolev1"}, {**cfg, "pose_repr": "pos"},
                    {"pressure_head": "time_pe", "pressure_time_pe_enabled": False}):
        with pytest.raises(ValueError):
            build(invalid)
