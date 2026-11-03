"""V7_Cv5AR2 冒烟（2026-11-01，V7_AR相对运动与航向状态方案 §7）：

(a) 零初始化契约：同 seed 构建 Cv5/AR2，prev_state=zeros + causal 下
    body 138 维逐位一致、v_hat 逐位一致、根 = deyaw(Cv5 根)、dyaw=0、
    vis_gain≈sigmoid(-4)；
(b) 通路有效：扰动 ar2 专属参数后输出变化、输入梯度非零；
(c) 因果性：只改未来反馈帧，当前帧输出逐位不变；
(d) dataset 列：v7_ar2 的 dyaw_gt/v_head_gt/psi_base/prev_state 形状与
    帧 0 口径（首窗全零、连续窗真实前驱）。
"""

import copy
import unittest

import torch

from anysole.models.v7_c2ar2 import build as build_ar2
from anysole.models.v7_c2 import build as build_cv5
from anysole.utils.geometry import deyaw_rotmat, rot6d_to_rotmat, yaw_rotmat


def _cfg(ar2=False):
    return {
        "d_model": 16, "tw": 4, "pose_layers": 1, "dropout": 0.0,
        "v7_task_routes": True, "v7_reliability_route": True,
        "v7_ar2": ar2, "model_name": "V7_Cv5AR2" if ar2 else "V7_Cv5",
    }


def _batch(device="cpu"):
    return {
        "V_feat": torch.randn(2, 4, 2051, device=device),
        "T_raw": torch.rand(2, 4, 96, device=device),
        "T_phys": torch.rand(2, 4, 12, device=device),
        "T_s2m": torch.rand(2, 4, 50, device=device),
    }


def _fwd(model, b, cid, prev_state=None, causal=True):
    return model(b["V_feat"], b["T_raw"], b["T_phys"], cid,
                 T_s2m=b["T_s2m"], prev_state=prev_state, causal=causal)


class TestAR2Model(unittest.TestCase):
    def test_zero_init_contract(self):
        # eval 模式（编码器 dropout 关闭，前向确定）；causal=False——
        # 掩码是行为项不是 init 项，fb 零初始化下 ar2 前向应逐位 = Cv5。
        b = _batch()
        cid = torch.zeros(2, dtype=torch.long)
        torch.manual_seed(7)
        cv5 = build_cv5(_cfg(ar2=False)).eval()
        torch.manual_seed(7)
        ar2 = build_ar2(_cfg(ar2=True)).eval()
        with torch.inference_mode():
            o_c = _fwd(cv5, b, cid, causal=False)
            o_a = _fwd(ar2, b, cid,
                       prev_state=torch.zeros(2, 4, 148), causal=False)
        # body 138 维 + v_hat 逐位一致（同一批共享权重）
        self.assertAlmostEqual(
            float((o_a["x0_hat"][..., 6:] - o_c["x0_hat"][..., 6:]).abs().max()),
            0.0, places=6)
        self.assertAlmostEqual(
            float((o_a["v_hat"] - o_c["v_hat"]).abs().max()), 0.0, places=6)
        # 根 = deyaw(Cv5 根)
        r_c = rot6d_to_rotmat(o_c["x0_hat"][..., :6])
        r_a = rot6d_to_rotmat(o_a["x0_hat"][..., :6])
        self.assertLess(float((r_a - deyaw_rotmat(r_c)).abs().max()), 1e-5)
        # dyaw = 0、vis_gain ≈ sigmoid(-4)
        self.assertAlmostEqual(float(o_a["dyaw_hat"].abs().max()), 0.0, places=6)
        self.assertLess(abs(float(o_a["vis_gain"].mean()) - 0.018), 1e-3)

    def test_pathway_and_grad(self):
        b = _batch()
        cid = torch.zeros(2, dtype=torch.long)
        torch.manual_seed(7)
        m = build_ar2(_cfg(ar2=True))
        ps0 = torch.zeros(2, 4, 148)
        with torch.inference_mode():
            o0 = _fwd(m, b, cid, prev_state=ps0, causal=True)
        # 扰动 ar2 专属参数 → 输出改变
        m2 = copy.deepcopy(m)
        with torch.no_grad():
            for p in m2.pose_head.ar2_fb.parameters():
                p.add_(0.5)
        with torch.inference_mode():
            o1 = _fwd(m2, b, cid, prev_state=torch.randn(2, 4, 148), causal=True)
        self.assertGreater(float((o1["x0_hat"] - o0["x0_hat"]).abs().max()), 0.0)
        # dyaw 监督梯度可达首层（末层零初始化不死梯度）
        from anysole.utils.losses import compute_losses
        batch = dict(b)
        batch.update({
            "pose_gt": torch.randn(2, 4, 144), "vel_gt": torch.randn(2, 4, 3),
            "v_head_gt": torch.randn(2, 4, 3), "dyaw_gt": torch.randn(2, 4),
            "psi_gt": torch.randn(2, 4), "psi_base": torch.randn(2),
            "contact_gt": torch.zeros(2, 4, 2),
            "trans_anchor": torch.randn(2, 3),
            "offsets": torch.randn(24, 3), "parents": torch.zeros(24, dtype=torch.long),
        })
        cfg = _cfg(ar2=True)
        cfg.update({"v7": True, "lambda_pose": 3.0, "lambda_traj": 1.0,
                    "lambda_dyaw": 1.0, "lambda_cumyaw": 1.0,
                    "lambda_dyaw_td": 0.5, "lambda_pose_vel": 0.5,
                    "lambda_visyaw": 0.5, "lambda_yawcorr": 0.5,
                    "traj_deltas": [2], "traj_delta_weight_power": 1.0})
        out = _fwd(m, b, cid, prev_state=ps0, causal=True)
        losses = compute_losses(out, batch, cid, cfg)
        losses["loss"].backward()
        # 末层零初始化（slow-start）：第一步末层 W 必须收到梯度（头可训练
        # 的必要条件——首层梯度经 W2=0 截断，第一步后解冻）。
        self.assertGreater(
            float(m.pose_head.dyaw_head[2].weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(m.vis_yaw_head[2].weight.grad.abs().sum()), 0.0)
        opt = torch.optim.SGD(m.parameters(), lr=1e-2)
        opt.step(); opt.zero_grad()
        out = _fwd(m, b, cid, prev_state=ps0, causal=True)
        losses = compute_losses(out, batch, cid, cfg)
        losses["loss"].backward()
        self.assertGreater(
            float(m.pose_head.dyaw_head[0].weight.grad.abs().sum()), 0.0)

    def test_causality(self):
        # eval 模式（消除编码器 dropout 的随机性）+ 扰动 ar2_fb（零初始化
        # 下 fb≡0，不改参数则反馈通道不可见——扰动后 fb 非零，改未来帧
        # 必须只影响未来输出）。
        b = _batch()
        cid = torch.full((2,), 2)  # T 行（无视觉校正干扰）
        torch.manual_seed(7)
        m = build_ar2(_cfg(ar2=True)).eval()
        with torch.no_grad():
            for p in m.pose_head.ar2_fb.parameters():
                p.add_(0.5)
        ps_a = torch.zeros(2, 4, 148)
        ps_b = ps_a.clone()
        ps_b[:, 2:] = torch.randn(2, 2, 148)  # 只改未来反馈帧（2、3）
        with torch.inference_mode():
            o_a = _fwd(m, b, cid, prev_state=ps_a, causal=True)
            o_b = _fwd(m, b, cid, prev_state=ps_b, causal=True)
        self.assertAlmostEqual(
            float((o_b["x0_hat"][:, :2] - o_a["x0_hat"][:, :2]).abs().max()),
            0.0, places=6)
        self.assertGreater(
            float((o_b["x0_hat"][:, 2:] - o_a["x0_hat"][:, 2:]).abs().max()), 0.0)


class TestAR2Dataset(unittest.TestCase):
    def test_dataset_columns(self):
        from pathlib import Path
        from AnysoleWorkspace.tool.workspace import resolve_uri
        from anysole.data.dataset import AnySoleDataset
        SMPL_ROOTS = [Path("/data/lizhe/projects/Tactile/Mocap/0804"),
                      Path("/data/lizhe/projects/Tactile/Mocap/0807"),
                      Path("/data/lizhe/projects/Tactile/Mocap/0808"),
                      Path("/data/lizhe/projects/Tactile/Mocap/0810")]
        ds = AnySoleDataset(
            mode="train", seq_root=resolve_uri("shared://facts/sessions/cam3"),
            split_csv=resolve_uri("protocol://splits/default/splits.csv"),
            cache_root=resolve_uri(
                "model-input://AnySole/adapter_v1/visual/hrnet/cam3"),
            window_length=20, session_ids=["S14072"],
            contact_method="f6_soft_time_accel_state_smpl24_v1",
            stride=20, tactile_input="raw108", cam_id=3, smpl_roots=SMPL_ROOTS,
            smpl_support_policy="exclude", v7_ar2=True)
        i0, i1 = ds[0], ds[1]
        self.assertEqual(tuple(i0["prev_state"].shape), (20, 148))
        self.assertEqual(tuple(i0["dyaw_gt"].shape), (20,))
        self.assertEqual(tuple(i0["v_head_gt"].shape), (20, 3))
        self.assertEqual(tuple(i0["psi_gt"].shape), (20,))
        # 首窗帧 0 全零（会话起点 reset 口径）
        self.assertTrue(bool((i0["prev_state"][0] == 0).all()))
        # 连续窗帧 0 = 上一窗末帧 GT 状态
        st = lambda dy, po, vh: torch.cat([dy, po, vh])
        self.assertTrue(bool((i1["prev_state"][0] == st(
            i0["dyaw_gt"][19:20], i0["pose_gt"][19], i0["v_head_gt"][19])).all()))
        # psi_base = 窗前一帧航向
        self.assertEqual(float(i1["psi_base"]), float(i0["psi_gt"][19]))


class TestAR2ObservationContext(unittest.TestCase):
    def test_mixed_history_matches_individual_forward_and_has_gradients(self):
        torch.manual_seed(31)
        cfg = {**_cfg(ar2=True), "obs_context": 12}
        m = build_ar2(cfg).eval()
        b = {key: torch.cat([value, value[:1]]) for key, value in _batch().items()}
        history = [{key: torch.rand(length, b[key].shape[-1], requires_grad=True)
                    for key in ("V_feat", "T_raw", "T_phys")}
                   for length in (8, 0, 8)]
        cid = torch.tensor([0, 2, 2])
        prev = torch.randn(3, 4, 148)
        out = m(b["V_feat"], b["T_raw"], b["T_phys"], cid,
                prev_state=prev, causal=True, obs_history=history)
        for key, width in (("x0_hat", 144), ("v_hat", 3), ("dyaw_hat", 1),
                           ("contact_logits", 2), ("vis_yaw", 1)):
            self.assertEqual(tuple(out[key].shape), (3, 4, width))
        for i in range(3):
            single = m(b["V_feat"][i:i + 1], b["T_raw"][i:i + 1],
                       b["T_phys"][i:i + 1], cid[i:i + 1],
                       prev_state=prev[i:i + 1], causal=True,
                       obs_history=[history[i]])
            torch.testing.assert_close(out["x0_hat"][i:i + 1], single["x0_hat"])
        out["x0_hat"][2, :, 6:].square().mean().backward()
        self.assertGreater(float(history[2]["T_raw"].grad.abs().sum()), 0.)
        bad = [{**history[0], "T_phys": history[0]["T_phys"][:-1]}, *history[1:]]
        with self.assertRaisesRegex(ValueError, "different lengths"):
            m(b["V_feat"], b["T_raw"], b["T_phys"], cid,
              prev_state=prev, causal=True, obs_history=bad)

    def test_context_keeps_target_positions_and_parameter_shapes(self):
        from anysole.models.v7_c2ar2m import build as build_ar2m
        torch.manual_seed(37)
        short = build_ar2({**_cfg(ar2=True), "obs_context": 4}).eval()
        torch.manual_seed(37)
        wide = build_ar2({**_cfg(ar2=True), "obs_context": 12}).eval()
        torch.testing.assert_close(short.embeddings.time_pe, wide.embeddings.time_pe[-4:])
        for key, value in short.named_parameters():
            torch.testing.assert_close(value, dict(wide.named_parameters())[key])
        m = build_ar2m({**_cfg(ar2=True), "obs_context": 12,
                       "v7_ar2_fb": False}).eval()
        self.assertFalse(any(".ar_fb." in key or ".ar2_fb." in key
                             for key, _ in m.named_parameters()))
        b = _batch()
        history = [{key: torch.rand(8, b[key].shape[-1])
                    for key in ("V_feat", "T_raw", "T_phys")} for _ in range(2)]
        with torch.inference_mode():
            outputs = [m(b["V_feat"], b["T_raw"], b["T_phys"], torch.tensor([2, 2]),
                         prev_state=state, causal=True, obs_history=history)
                       for state in (torch.zeros(2, 4, 148), torch.randn(2, 4, 148))]
        torch.testing.assert_close(outputs[0]["x0_hat"], outputs[1]["x0_hat"])
        self.assertEqual(tuple(m.pose_head.causal_mask.shape), (40, 40))

    def test_real_context_keeps_targets_coverage_and_training(self):
        from anysole.data.dataset import AnySoleDataset, collate_windows
        from anysole.train import backward_v7_enum
        short = AnySoleDataset(mode="train", window_length=20, stride=20,
                              session_ids=["S14072"], v7_ar2=True,
                              smpl_support_policy="exclude")
        wide = AnySoleDataset(mode="train", window_length=20, stride=20,
                             session_ids=["S14072"], v7_ar2=True, obs_context=80,
                             smpl_support_policy="exclude")
        self.assertEqual(short.valid_windows, wide.valid_windows)
        for i in range(len(wide)):
            a, b = short[i], wide[i]
            for key in ("V_feat", "T_raw", "pose_gt", "dyaw_gt", "v_head_gt"):
                torch.testing.assert_close(a[key], b[key])
            fi, left, _ = wide.valid_windows[i]
            past = b["obs_history"]
            self.assertEqual(set(past), {"V_feat", "T_raw", "T_phys"})
            start = wide.observation_starts[i]
            for key in past:
                torch.testing.assert_close(past[key], torch.from_numpy(
                    wide.sessions[fi][key][start:left]).float())
        batch = collate_windows([wide[0], wide[3]])
        self.assertEqual([h["T_raw"].shape[0] for h in batch["obs_history"]], [0, 60])
        cfg = {**_cfg(ar2=True), "tw": 20, "obs_context": 80,
               "v7": True, "config_probs": [0.5, 0.25, 0.25]}
        m = build_ar2(cfg).eval()
        _, losses = backward_v7_enum(m, batch, cfg, ar_ss_p=1.0)
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        self.assertGreater(float(m.pose_head.dyaw_head[2].weight.grad.abs().sum()), 0.)

    def test_context_checkpoint_and_matrix_identity(self):
        import contextlib
        import io
        from anysole.eval import _load_model
        from anysole.train import parse_args
        from anysole.registry import get_builder
        from anysole.types import JOINT_PROTOCOL_CHECKSUM, MOTION_PROTOCOL, variant_from_config
        from configs.z_gen.model_registry import MODELS
        from configs.tools.common import TRAIN_ARG_FLAG_MAP, effective_variant
        help_text = io.StringIO()
        with contextlib.redirect_stdout(help_text), self.assertRaises(SystemExit) as stopped:
            parse_args(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--obs-context", help_text.getvalue())
        cfg = {**_cfg(ar2=True), **get_builder("V7_Cv5AR2").STRUCTURE,
               "obs_context": 12, "modal": "anysolev2",
               "motion_protocol": MOTION_PROTOCOL,
               "joint_protocol_checksum": JOINT_PROTOCOL_CHECKSUM}
        m = build_ar2(cfg)
        restored = _load_model({"config": cfg, "model": m.state_dict()}, cfg,
                               torch.device("cpu"))
        self.assertEqual(restored.obs_context, 12)
        self.assertEqual(restored.tw, 4)
        names = ["V7_Cv5AR2M_s1", "V7_Cv5AR2M_tw40_s1", "V7_Cv5AR2M_tw80_s1",
                 "V7_Cv5AR2_ctx20_s1", "V7_Cv5AR2_tw40_s1", "V7_Cv5AR2_tw80_s1"]
        paths = []
        for name in names:
            spec = MODELS[name]
            args = spec["train_args"]
            self.assertEqual((args["tw"], args["stride"], spec["batch_size"],
                              args["lambda_cumyaw"]), (20, 20, 64, 1.0))
            expected = variant_from_config({"tw": 20, "obs_context": args["obs_context"]})
            if args["obs_context"] > 20:
                self.assertTrue(expected.endswith("oc%d" % args["obs_context"]))
            paths.append((spec["name"], effective_variant(spec)))
        self.assertEqual(len(set(paths)), 6)
        self.assertEqual(TRAIN_ARG_FLAG_MAP["obs_context"], "--obs-context")

    def test_standalone_infer_rejects_unsupported_stateful_models(self):
        from anysole.infer import _require_supported_inference
        _require_supported_inference({"tw": 20, "model_name": "V7_Cv5"})
        for config in ({"tw": 20, "v7_ar2": True},
                       {"tw": 20, "v7_ar2": True, "obs_context": 80},
                       {"tw": 20, "obs_context": 40}):
            with self.subTest(config=config), self.assertRaisesRegex(
                    NotImplementedError, "anysole.eval"):
                _require_supported_inference(config)

    def test_context_stops_at_invalid_window(self):
        from unittest.mock import patch
        from anysole.data.dataset import AnySoleDataset, load_smpl

        def with_gap(*args, **kwargs):
            motion = load_smpl(*args, **kwargs)
            motion["query_valid"] = motion["query_valid"].copy()
            motion["query_valid"][45] = False
            return motion

        with patch("anysole.data.dataset.load_smpl", side_effect=with_gap):
            ds = AnySoleDataset(mode="train", window_length=20, stride=20,
                                session_ids=["S14072"], v7_ar2=True, obs_context=80,
                                smpl_support_policy="exclude")
        starts = [left for _, left, _ in ds.valid_windows]
        self.assertNotIn(40, starts)
        self.assertEqual(ds[starts.index(60)]["obs_history"]["T_raw"].shape[0], 0)
        self.assertEqual(ds[starts.index(80)]["obs_history"]["T_raw"].shape[0], 20)

    def test_session_generation_receives_real_history_for_every_window(self):
        from unittest.mock import patch
        from anysole.data.dataset import AnySoleDataset
        from anysole.utils.eval_protocol import ar2_session_chunks
        ds = AnySoleDataset(mode="eval", window_length=20, stride=20,
                            session_ids=["S14072"], v7_ar2=True, obs_context=80,
                            smpl_support_policy="exclude")
        m = build_ar2({**_cfg(ar2=True), "tw": 20, "obs_context": 80}).eval()
        forward = m.forward
        lengths = []

        def record(*args, **kwargs):
            if kwargs.get("obs_history") is not None:
                lengths.append(kwargs["obs_history"][0]["T_raw"].shape[0])
            return forward(*args, **kwargs)

        with torch.inference_mode(), patch.object(m, "forward", side_effect=record):
            chunks, _ = ar2_session_chunks(ds, m, "S14072", list(range(4)),
                                          2, torch.device("cpu"), 20, {})
        self.assertEqual(lengths, [n for n in (0, 20, 40, 60) for _ in range(20)])
        self.assertEqual([c["start"] for c in chunks], [0, 20, 40, 60])
        self.assertTrue(all(c["pred_pose"].shape == (20, 144) for c in chunks))


if __name__ == "__main__":
    unittest.main()


class TestAR2Regression(unittest.TestCase):
    """V7_Cv5AR2R 纯回归格冒烟（2026-11-02）：双向无反馈无因果。"""

    def test_regression_build_no_feedback_bidirectional(self):
        from anysole.models.v7_c2ar2r import build as build_ar2r
        m = build_ar2r({**_cfg(ar2=True), "v7_ar2_fb": False, "tw": 4}).eval()
        self.assertTrue(m.ar2)
        self.assertFalse(m.ar2_fb)
        # ar2_fb 属性存布尔开关；反馈 Linear 只在 ar2_fb=True 时构建
        # （pose_head.ar2_fb 此时为 False 而非 nn.Module）。
        self.assertFalse(m.pose_head.ar2_fb)
        self.assertNotIsInstance(m.pose_head.ar2_fb, torch.nn.Module)
        b = _batch()
        cid = torch.zeros(2, dtype=torch.long)
        with torch.inference_mode():
            out = _fwd(m, b, cid, prev_state=None, causal=False)
            out_mask = _fwd(m, b, cid, prev_state=None, causal=True)
        self.assertEqual(out["dyaw_hat"].shape, (2, 4, 1))
        self.assertEqual(out["x0_hat"].shape, (2, 4, 144))
        self.assertEqual(out["v_hat"].shape, (2, 4, 3))
        self.assertEqual(out["vis_yaw"].shape, (2, 4, 1))
        self.assertEqual(out["vis_gain"].shape, (2, 4, 1))
        # 无反馈支路：prev_state 不消费；因果掩码仍作用于 query 自注意力
        # （帧间协调通道），故双向/因果输出不同——回归格用双向（causal
        # =False），掩码存在但不启用。
        self.assertFalse(torch.allclose(out["x0_hat"], out_mask["x0_hat"]))

    def test_regression_session_chunks_single_parallel_forward(self):
        from unittest.mock import patch
        from anysole.data.dataset import AnySoleDataset
        from anysole.utils.eval_protocol import ar2_session_chunks
        from anysole.models.v7_c2ar2r import build as build_ar2r
        ds = AnySoleDataset(mode="eval", window_length=20, stride=20,
                            session_ids=["S14072"], v7_ar2=True, obs_context=80,
                            smpl_support_policy="exclude")
        m = build_ar2r({**_cfg(ar2=True), "v7_ar2_fb": False, "tw": 20,
                         "obs_context": 80}).eval()
        forward = m.forward
        calls = []

        def record(*args, **kwargs):
            calls.append(kwargs)
            return forward(*args, **kwargs)

        with torch.inference_mode(), patch.object(m, "forward", side_effect=record):
            chunks, _ = ar2_session_chunks(ds, m, "S14072", list(range(4)),
                                           2, torch.device("cpu"), 20, {},
                                           regression=True)
        # 每窗一次并行前向（串行格为 20 次/窗）；obs_history 分组内部
        # 会递归调用 forward（obs_history 非 None = 顶层调用）。
        outer = [c for c in calls if c.get("obs_history") is not None]
        self.assertEqual(len(outer), 4)
        self.assertTrue(all(c["prev_state"] is None and not c["causal"]
                            for c in outer))
        self.assertEqual([c["start"] for c in chunks], [0, 20, 40, 60])
        self.assertTrue(all(c["pred_pose"].shape == (20, 144) for c in chunks))
        self.assertTrue(all(c["pred_trans"].shape == (20, 3) for c in chunks))


# ---------------------------------------------------------------------------
# yaw-fix 损失（2026-11-02，v7_ar2_yaw_fix）：合成 out/batch 直测
# _compute_losses_v7_ar2 的数学行为（不建模型，CPU 快测）。
# ---------------------------------------------------------------------------

import torch.nn as nn  # noqa: E402

from anysole.types import CONFIG_T, CONFIG_V  # noqa: E402
from anysole.utils.losses import (  # noqa: E402
    _ar2_heading_chain, _compute_losses_v7_ar2, _wrap_angle)

TW = 8


def _synth_out(bsz, dyaw, vis_yaw, gain, v_head):
    return {
        "x0_hat": torch.randn(bsz, TW, 144),
        "v_hat": v_head.clone(),
        "trans_hat": torch.cumsum(v_head, dim=1) / 40.0,
        "dyaw_hat": dyaw[..., None],
        "vis_yaw": vis_yaw[..., None],
        "vis_gain": gain[..., None],
        "contact_logits": torch.zeros(bsz, TW, 1),
        "z_root": None,
        "route_gate": None,
    }


def _synth_batch(bsz, dyaw_gt, psi_gt, vel_gt, v_head_gt, base0=None):
    batch = {
        "pose_gt": torch.randn(bsz, TW, 144),
        "dyaw_gt": dyaw_gt,
        "psi_gt": psi_gt,
        "psi_base": psi_gt[:, 0].clone(),
        "vel_gt": vel_gt,
        "v_head_gt": v_head_gt,
        "contact_gt": torch.zeros(bsz, TW, 1),
        "trans_anchor": torch.zeros(bsz, 3),
    }
    if base0 is not None:
        # 148 维布局 = [Δyaw 1 + tilt/body 144 + v_head 3]；帧 0 通道 0 =
        # 真实前驱 Δψ（历史基线起点），其余槽位零。
        prev_state = torch.zeros(bsz, TW, 148)
        prev_state[:, 0, 0] = base0
        batch["prev_state"] = prev_state
    return batch


def _losses(out, batch, config_id, **extra):
    weights = {"v7": True, "v7_ar2_yaw_fix": True, "lambda_air": 0.0,
               "ar2_dyaw_base": "history", "ar2_dyaw_t_w": 0.1}
    weights.update(extra)
    return _compute_losses_v7_ar2(out, batch, config_id, weights)


class TestAR2YawFixLoss(unittest.TestCase):
    def _const_batch(self, bsz, psi, v=1.2):
        # GT 口径：vel_gt = Y(psi_gt) · v_head_gt（数据集分解定义）。
        dyaw_gt = torch.zeros(bsz, TW)
        psi_gt = torch.full((bsz, TW), psi)
        v_head_gt = torch.zeros(bsz, TW, 3)
        v_head_gt[..., 2] = v
        vel_gt = torch.einsum("btij,btj->bti", yaw_rotmat(psi_gt), v_head_gt)
        return dyaw_gt, psi_gt, vel_gt, v_head_gt

    def test_t_rows_heading_weak_not_zero(self):
        # 正面 Δψ 结构（2026-11-02 修订）：T 行弱监督（t_w=0.1）——混合批
        # 中 T 行误差对池化损失的贡献仅 1/10；t_w=1 恢复全监督（池化损失
        # 更大）。单一条件批的均值对 t_w 不变（权在归一化中相消）。
        dyaw_gt, psi_gt, vel_gt, v_head_gt = self._const_batch(2, 0.3)
        dyaw_hat = torch.zeros(2, TW)
        dyaw_hat[0] = 0.02                       # T 行大修正
        dyaw_hat[1] = 0.001                      # V 行小修正
        out = _synth_out(2, dyaw_hat, torch.full((2, TW), 0.3),
                         torch.full((2, TW), 0.5), v_head_gt)
        batch = _synth_batch(2, dyaw_gt, psi_gt, vel_gt, v_head_gt)
        cid = torch.tensor([CONFIG_T, CONFIG_V])
        l_weak = _losses(out, batch, cid)
        l_full = _losses(out, batch, cid, ar2_dyaw_t_w=1.0)
        for key in ("L_dyaw", "L_cumyaw", "L_dyaw_td"):
            self.assertGreater(float(l_weak[key]), 0.0)
            self.assertLess(float(l_weak[key]), float(l_full[key]))

    def test_history_baseline_continuation_zero_hat(self):
        # 基线延续：hat=0、base0=GT Δψ 且 GT 恒定 → Δψ_total 逐位复现 GT，
        # 三个航向损失 ≈ 0（结构保证「不幻觉转向」）；零基线（旧语义）下
        # 同批损失 = mrad²·c² > 0。
        c = 0.02
        dyaw_gt = torch.full((2, TW), c)
        psi_gt = torch.cumsum(dyaw_gt, dim=1)
        vel_gt = torch.zeros(2, TW, 3)
        vel_gt[..., 2] = 1.2
        v_head_gt = vel_gt.clone()
        batch = _synth_batch(2, dyaw_gt, psi_gt, vel_gt, v_head_gt,
                             base0=torch.full((2,), c))
        cid = torch.full((2,), CONFIG_T, dtype=torch.long)
        out = _synth_out(2, torch.zeros(2, TW), torch.zeros(2, TW),
                         torch.zeros(2, TW), v_head_gt)
        l_hist = _losses(out, batch, cid)
        for key in ("L_dyaw", "L_cumyaw", "L_dyaw_td"):
            self.assertLess(float(l_hist[key]), 1e-6)
        l_zero = _losses(out, batch, cid, ar2_dyaw_base="zero")
        self.assertGreater(float(l_zero["L_dyaw"]), 1e-3)

    def test_yawcorr_regime_prefers_large_gain(self):
        # 自由生成口径（ψ 从 0 起）：gain 大 → 校正到位 → 损失小；
        # 旧口径（GT psi_base 起）gain 越小越好——本测试锁定新口径。
        dyaw_gt, psi_gt, vel_gt, v_head_gt = self._const_batch(2, 0.3)
        batch = _synth_batch(2, dyaw_gt, psi_gt, vel_gt, v_head_gt)
        cid = torch.full((2,), CONFIG_V, dtype=torch.long)
        l_lo = _losses(_synth_out(2, torch.zeros(2, TW), torch.full((2, TW), 0.3),
                                  torch.full((2, TW), 0.01), v_head_gt),
                       batch, cid)["L_yawcorr"]
        l_hi = _losses(_synth_out(2, torch.zeros(2, TW), torch.full((2, TW), 0.3),
                                  torch.full((2, TW), 0.9), v_head_gt),
                       batch, cid)["L_yawcorr"]
        self.assertLess(float(l_hi), float(l_lo))

    def test_world_traj_zero_when_assembly_matches_gt(self):
        # 组装与 GT 逐位一致（gain=1、vis_yaw=ψ_gt、v̂=v_gt、Δψ̂=0）→ 世界
        # 轨迹损失 ≈ 0；gain 小（航向没纠回来）→ 世界位移旋转错 → 损失 >0。
        dyaw_gt, psi_gt, vel_gt, v_head_gt = self._const_batch(2, 0.3)
        batch = _synth_batch(2, dyaw_gt, psi_gt, vel_gt, v_head_gt)
        cid = torch.full((2,), CONFIG_V, dtype=torch.long)
        l_ok = _losses(_synth_out(2, torch.zeros(2, TW), torch.full((2, TW), 0.3),
                                  torch.full((2, TW), 1.0), v_head_gt),
                       batch, cid)["L_world_traj"]
        l_bad = _losses(_synth_out(2, torch.zeros(2, TW), torch.full((2, TW), 0.3),
                                   torch.full((2, TW), 0.01), v_head_gt),
                        batch, cid)["L_world_traj"]
        self.assertLess(float(l_ok), 1e-6)
        self.assertGreater(float(l_bad), 1e-3)

    def test_world_traj_translation_invariant(self):
        # 位移口径：平移锚点不影响损失（锚点相消）。
        dyaw_gt, psi_gt, vel_gt, v_head_gt = self._const_batch(2, 0.3)
        batch = _synth_batch(2, dyaw_gt, psi_gt, vel_gt, v_head_gt)
        cid = torch.full((2,), CONFIG_V, dtype=torch.long)
        out = _synth_out(2, torch.zeros(2, TW), torch.full((2, TW), 0.3),
                         torch.full((2, TW), 0.01), v_head_gt)
        l_a = _losses(out, batch, cid)["L_world_traj"]
        batch["trans_anchor"] = torch.tensor([[0.85, 0.0, 0.0]] * 2)
        l_b = _losses(out, batch, cid)["L_world_traj"]
        self.assertAlmostEqual(float(l_a), float(l_b), places=5)

    def test_heading_chain_matches_inference_update(self):
        # _ar2_heading_chain 必须与 eval 侧 ar2_session_chunks 的逐帧更新
        # 同构（d_total = 基线 + Δψ̂；ψ += d_total；EMA；ψ += g·wrap(ema−ψ)）
        # ——手写循环对照（含 base0 基线）。
        torch.manual_seed(3)
        dyaw = torch.randn(2, TW) * 0.02
        vis = torch.randn(2, TW) * 0.5
        gain = torch.rand(2, TW)
        base0 = torch.randn(2) * 0.01
        psi_ref = torch.zeros(2, TW)
        total_ref = torch.zeros(2, TW)
        for b in range(2):
            cur, ema = torch.zeros(()), None
            cur_d = base0[b]
            for t in range(TW):
                d_total = dyaw[b, t] + cur_d
                cur = cur + d_total
                if vis is not None:
                    v_t = vis[b, t]
                    ema = v_t if ema is None else 0.5 * ema + 0.5 * v_t
                    cur = cur + gain[b, t] * _wrap_angle(ema - cur)
                cur_d = d_total
                psi_ref[b, t] = cur
                total_ref[b, t] = d_total
        psi, total = _ar2_heading_chain(dyaw, vis, gain, base0=base0)
        self.assertLess(float((psi - psi_ref).abs().max()), 1e-6)
        self.assertLess(float((total - total_ref).abs().max()), 1e-6)

    def test_yawcorr_grad_flows_to_gain(self):
        # 新口径下 vis_gain 必须有梯度（旧口径梯度把 gain 推向 0 仍存在，
        # 但新口径的梯度应能把 gain 拉大——方向检查：gain 全 0.01 时
        # 增大 gain 应降低总损失）。用合成 out 的 gain 常量 + 参数化重建。
        dyaw_gt, psi_gt, vel_gt, v_head_gt = self._const_batch(2, 0.3)
        batch = _synth_batch(2, dyaw_gt, psi_gt, vel_gt, v_head_gt)
        cid = torch.full((2,), CONFIG_V, dtype=torch.long)
        gain = nn.Parameter(torch.full((2, TW), 0.01))
        out = _synth_out(2, torch.zeros(2, TW), torch.full((2, TW), 0.3),
                         gain, v_head_gt)
        weights = {"v7": True, "v7_ar2_yaw_fix": True, "lambda_air": 0.0}
        total = _compute_losses_v7_ar2(out, batch, cid, weights)["loss"]
        total.backward()
        self.assertIsNotNone(gain.grad)
        self.assertTrue(bool((gain.grad < 0).all()))
