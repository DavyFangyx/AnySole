"""V7_Cv5AR2RL / V7_Cv5AR2RLD 冒烟（2026-11-03，z_note/修改/04_AR训练目标
与自由生成对齐.md 改动 A/B）：

(a) 零改动等价契约：改动全关（rollout 0 / lowdim off）→ R 入口与 AR2
    同 seed 逐位一致（state_dict 键值 bit-equal、前向 bit-equal）；与
    AR2M 的共享参数逐位一致 + fb 零初始化下前向输出逐位一致；
    backward_v7_enum 损失与 AR2 逐位一致；
(b) rollout 反馈来源与时间位置：帧 t 反馈 = 帧 t−1 模型输出组装
    [Δyaw_total + tilt/body + v_head]（与 eval ar2_session_chunks 逐帧
    更新式同构——手写参考循环逐位比对）；扰动 GT 反馈帧 1..L 不影响
    闭环帧输出、扰动帧 0 携带反馈影响帧 0/1 输出（来源断言）；
(c) 前缀解码逐位等价：pose_head max_frames=k 与全窗解码在 ≤k 帧逐位
    一致（闭环 rollout 省算力依据）；
(d) 因果性：扰动未来帧（>L）GT 反馈不影响闭环帧输出；
(e) 改动 B 降维：投影维度、切片布局（Δyaw1 + tilt6 + proj k + v_head3）、
    零初始化契约、评估链不感知（ar2_session_chunks 照常组装 148 维）；
(f) 课程公式与死开关防御。
"""

import copy
import unittest

import torch

from anysole.models.v7_c2ar2 import build as build_ar2
from anysole.models.v7_c2ar2m import build as build_ar2m
from anysole.models.v7_c2ar2rl import build as build_ar2rl
from anysole.train import _ar_rollout_curriculum, backward_v7_enum
from anysole.types import CONFIG_T, CONFIG_V


def _cfg(ar2=True, **extra):
    cfg = {
        "d_model": 16, "tw": 4, "pose_layers": 1, "dropout": 0.0,
        "v7_task_routes": True, "v7_reliability_route": True,
        "v7_ar2": ar2, "model_name": "V7_Cv5AR2RL" if ar2 else "V7_Cv5",
    }
    cfg.update(extra)
    return cfg


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


def _perturb_fb(model, delta=0.5):
    # 零初始化下 fb≡0 反馈通道不可见——扰动后 fb 非零（同 test_v7_ar2
    # 的因果性测试口径）。
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in m2.pose_head.ar2_fb.parameters():
            p.add_(delta)
    return m2


def _loss_batch(b):
    batch = dict(b)
    batch.update({
        "pose_gt": torch.randn(2, 4, 144), "vel_gt": torch.randn(2, 4, 3),
        "v_head_gt": torch.randn(2, 4, 3), "dyaw_gt": torch.randn(2, 4),
        "psi_gt": torch.randn(2, 4), "psi_base": torch.randn(2),
        "contact_gt": torch.zeros(2, 4, 2),
        "trans_anchor": torch.randn(2, 3),
        "offsets": torch.randn(24, 3), "parents": torch.zeros(24, dtype=torch.long),
        "prev_state": torch.randn(2, 4, 148),
    })
    return batch


def _enum_cfg(**extra):
    cfg = _cfg(ar2=True)
    cfg.update({"v7": True, "config_probs": [0.5, 0.25, 0.25],
                "lambda_pose": 3.0, "lambda_traj": 1.0,
                "lambda_dyaw": 1.0, "lambda_cumyaw": 1.0,
                "lambda_dyaw_td": 0.5, "lambda_pose_vel": 0.5,
                "lambda_visyaw": 0.5, "lambda_yawcorr": 0.5,
                "lambda_air": 0.0, "traj_deltas": [2],
                "traj_delta_weight_power": 1.0})
    cfg.update(extra)
    return cfg


class TestContractAllOff(unittest.TestCase):
    def test_all_off_bit_reproduces_ar2(self):
        # 改动全关 → R 入口与 AR2 同 seed 逐位一致（结构 + 前向）。
        b = _batch()
        cid = torch.zeros(2, dtype=torch.long)
        ps = torch.randn(2, 4, 148)
        torch.manual_seed(7)
        ar2 = build_ar2(_cfg(ar2=True)).eval()
        torch.manual_seed(7)
        r = build_ar2rl(_cfg(ar2=True)).eval()
        self.assertEqual(set(ar2.state_dict()), set(r.state_dict()))
        for key, value in ar2.state_dict().items():
            torch.testing.assert_close(r.state_dict()[key], value,
                                       rtol=0.0, atol=0.0)
        with torch.inference_mode():
            o_a = _fwd(ar2, b, cid, prev_state=ps)
            o_r = _fwd(r, b, cid, prev_state=ps)
        torch.testing.assert_close(o_r["x0_hat"], o_a["x0_hat"],
                                   rtol=0.0, atol=0.0)
        torch.testing.assert_close(o_r["v_hat"], o_a["v_hat"],
                                   rtol=0.0, atol=0.0)
        torch.testing.assert_close(o_r["dyaw_hat"], o_a["dyaw_hat"],
                                   rtol=0.0, atol=0.0)

    def test_all_off_matches_ar2m_shared_params_and_forward(self):
        # AR2M 无反馈支路：共享参数逐位一致（dyaw_head 首层除外——AR2 的
        # ar2_fb 构造在 RNG 存/复块内先于 dyaw_head 消耗 RNG，AR2 vs
        # AR2M 的首层 init 差异为既有事实；末层零初始化保证 init 输出
        # 仍逐位一致）；fb 零初始化 → R 前向输出与 AR2M 逐位一致（任何
        # prev_state——AR2M 不消费）。
        b = _batch()
        cid = torch.full((2,), 2)
        torch.manual_seed(7)
        ar2m = build_ar2m(_cfg(ar2=True, v7_ar2_fb=False)).eval()
        torch.manual_seed(7)
        r = build_ar2rl(_cfg(ar2=True)).eval()
        for key, value in ar2m.state_dict().items():
            self.assertIn(key, r.state_dict())
            if key.startswith("pose_head.dyaw_head.0"):
                continue
            torch.testing.assert_close(r.state_dict()[key], value,
                                       rtol=0.0, atol=0.0)
        self.assertEqual(
            set(r.state_dict()) - set(ar2m.state_dict()),
            {"pose_head.ar2_fb.weight", "pose_head.ar2_fb.bias"})
        for ps in (torch.zeros(2, 4, 148), torch.randn(2, 4, 148)):
            with torch.inference_mode():
                o_m = _fwd(ar2m, b, cid, prev_state=ps)
                o_r = _fwd(r, b, cid, prev_state=ps)
            torch.testing.assert_close(o_r["x0_hat"], o_m["x0_hat"],
                                       rtol=0.0, atol=0.0)

    def test_training_loss_bit_equal_with_rollout_off(self):
        # ar_rollout_len=0（默认）→ backward_v7_enum 与 AR2 逐位一致
        # （训练路径零改动等价）。eval 模式关 dropout（位级断言需要
        # 确定性；dropout 本身与契约无关）。
        b = _loss_batch(_batch())
        torch.manual_seed(7)
        ar2 = build_ar2(_cfg(ar2=True)).eval()
        torch.manual_seed(7)
        r = build_ar2rl(_cfg(ar2=True)).eval()
        _, losses_a = backward_v7_enum(ar2, b, _enum_cfg(), ar_rollout_len=0)
        _, losses_r = backward_v7_enum(r, b, _enum_cfg(), ar_rollout_len=0)
        self.assertAlmostEqual(float(losses_a["loss"]), float(losses_r["loss"]),
                               places=6)

    def test_rollout_noop_at_zero_init(self):
        # fb 零初始化 → 输出与 prev_state 无关 → 闭环自反馈 ≡ TF 输出
        # → rollout 开启时损失仍与 TF 逐位一致（init 语义的延伸契约）。
        b = _loss_batch(_batch())
        torch.manual_seed(7)
        r0 = build_ar2rl(_cfg(ar2=True)).eval()
        torch.manual_seed(7)
        r1 = build_ar2rl(_cfg(ar2=True)).eval()
        _, l0 = backward_v7_enum(r0, b, _enum_cfg(), ar_rollout_len=0)
        _, l1 = backward_v7_enum(r1, b, _enum_cfg(), ar_rollout_len=3)
        self.assertAlmostEqual(float(l0["loss"]), float(l1["loss"]), places=6)
        # 扰动 fb 后 rollout 有作用对象：损失变化。
        torch.manual_seed(7)
        rp = _perturb_fb(build_ar2rl(_cfg(ar2=True)).eval())
        _, lp = backward_v7_enum(rp, b, _enum_cfg(), ar_rollout_len=3)
        self.assertGreater(abs(float(lp["loss"]) - float(l1["loss"])), 1e-4)
        self.assertTrue(bool(torch.isfinite(lp["loss"])))


class TestRolloutSemantics(unittest.TestCase):
    def test_manual_reference_closed_loop(self):
        # 手写参考循环（eval ar2_session_chunks 逐帧更新式：d_total(t) =
        # Δψ̂(t) + 基线、反馈 = [d_total, tilt/body, v_head]）与
        # rollout_forward 逐位一致；帧 0 保留 batch 携带、尾帧 = TF。
        torch.manual_seed(11)
        m = _perturb_fb(build_ar2rl(_cfg(ar2=True)).eval())
        b = _batch()
        cid = torch.full((2,), CONFIG_T)
        ps = torch.randn(2, 4, 148)
        with torch.inference_mode():
            out = m.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                                    cid, T_s2m=b["T_s2m"], prev_state=ps,
                                    rollout_len=2, dyaw_base="zero")
            out_tf = _fwd(m, b, cid, prev_state=ps)
            # 手写参考（与 rollout_forward 同源编码，显式更新式）。
            v_present = (cid != CONFIG_T).to(b["V_feat"].dtype)
            t_present = (cid != CONFIG_V).to(b["V_feat"].dtype)
            t_tac = torch.cat([b["T_raw"], b["T_phys"]], dim=-1)
            v_tok, t_tok = m.encoders(b["V_feat"], t_tac, cid)
            m_v = m.v_temporal(v_tok)
            m_t = m.t_temporal(t_tok)
            gate = m._route_gate(v_tok, t_tok)

            def decode(pstate, frames):
                return m.pose_head(m_v, m_t, v_present, t_present,
                                   gate=gate, prev_state=pstate, causal=True,
                                   max_frames=frames)

            pose0, v0, dyaw0, _, _ = decode(ps, 4)
            # 行 1 = 帧 0 状态（闭环链起点，替换 GT 携带的行 1）。
            d_total0 = dyaw0[:, 0, 0].detach()  # dyaw_base="zero"
            prev_filled = torch.cat([
                ps[:, :1],
                torch.cat([d_total0[:, None], pose0[:, 0].detach(),
                           v0[:, 0].detach()], dim=-1)[:, None, :],
                ps[:, 2:]], dim=1)
            roll_pose, roll_v, roll_dyaw = [], [], []
            for t in range(1, 3):
                pose_t, v_t, dyaw_t, _, _ = decode(prev_filled, t + 1)
                x_t, v_t_, d_t = (pose_t[:, t].detach(), v_t[:, t].detach(),
                                  dyaw_t[:, t, 0].detach())
                d_total = d_t  # dyaw_base="zero"（与 eval 同规则）
                fb_t = torch.cat([d_total[:, None], x_t, v_t_],
                                 dim=-1)[:, None, :]
                if t < 2:
                    prev_filled = torch.cat(
                        [prev_filled[:, :t + 1], fb_t, prev_filled[:, t + 2:]],
                        dim=1)
                roll_pose.append(x_t)
                roll_v.append(v_t_)
                roll_dyaw.append(d_t)
        # 帧 0 = TF（batch 携带反馈口径）；尾帧（3）= TF；帧 1..2 = 闭环。
        torch.testing.assert_close(out["x0_hat"][:, :1], out_tf["x0_hat"][:, :1],
                                   rtol=0.0, atol=0.0)
        torch.testing.assert_close(out["x0_hat"][:, 3:], out_tf["x0_hat"][:, 3:],
                                   rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            out["x0_hat"][:, 1:3], torch.stack(roll_pose, dim=1),
            rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            out["v_hat"][:, 1:3], torch.stack(roll_v, dim=1),
            rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            out["dyaw_hat"][:, 1:3, 0], torch.stack(roll_dyaw, dim=1),
            rtol=0.0, atol=0.0)
        # 闭环帧 ≠ TF 帧（fb 已扰动，rollout 确实替换了反馈来源）。
        self.assertGreater(
            float((out["x0_hat"][:, 1:3] - out_tf["x0_hat"][:, 1:3]).abs().max()),
            0.0)

    def test_feedback_source_per_frame(self):
        # 来源断言：帧 1..L 的 GT 反馈被自预测替换（扰动无效）；帧 0
        # batch 携带反馈有效并传导到帧 1；帧 0 反馈扰动不越过窗口。
        torch.manual_seed(13)
        m = _perturb_fb(build_ar2rl(_cfg(ar2=True)).eval())
        b = _batch()
        cid = torch.full((2,), CONFIG_T)
        ps_a = torch.randn(2, 4, 148)
        ps_b = ps_a.clone()
        ps_b[:, 1] = torch.randn(2, 148)   # 只改帧 1 的 GT 反馈
        with torch.inference_mode():
            out_a = m.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                                      cid, T_s2m=b["T_s2m"], prev_state=ps_a,
                                      rollout_len=2)
            out_b = m.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                                      cid, T_s2m=b["T_s2m"], prev_state=ps_b,
                                      rollout_len=2)
        # 帧 1 反馈 = 自预测（帧 0 输出）→ 扰动 GT 帧 1 不影响帧 1 输出；
        # 帧 0 输出同样不受影响（其反馈 = 帧 0 携带，未动）。
        torch.testing.assert_close(out_b["x0_hat"][:, 1], out_a["x0_hat"][:, 1],
                                   rtol=0.0, atol=0.0)
        torch.testing.assert_close(out_b["x0_hat"][:, 0], out_a["x0_hat"][:, 0],
                                   rtol=0.0, atol=0.0)
        # 帧 0 携带反馈：扰动 ps[:, 0] 影响帧 0 输出并经闭环反馈传导到帧 1。
        ps_d = ps_a.clone()
        ps_d[:, 0] = torch.randn(2, 148)
        with torch.inference_mode():
            out_d = m.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                                      cid, T_s2m=b["T_s2m"], prev_state=ps_d,
                                      rollout_len=2)
        self.assertGreater(
            float((out_d["x0_hat"][:, 0] - out_a["x0_hat"][:, 0]).abs().max()),
            0.0)
        self.assertGreater(
            float((out_d["x0_hat"][:, 1] - out_a["x0_hat"][:, 1]).abs().max()),
            0.0)
        # 帧 2 反馈同样自预测（扰动 GT 帧 2 不影响帧 2 输出）。
        ps_c = ps_a.clone()
        ps_c[:, 2] = torch.randn(2, 148)
        with torch.inference_mode():
            out_c = m.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                                      cid, T_s2m=b["T_s2m"], prev_state=ps_c,
                                      rollout_len=2)
        torch.testing.assert_close(out_c["x0_hat"][:, 2], out_a["x0_hat"][:, 2],
                                   rtol=0.0, atol=0.0)

    def test_prefix_decode_matches_full(self):
        # 前缀解码（max_frames=k）与全窗解码在 ≤k 帧数学一致——闭环
        # rollout 每帧一次前缀解码的省算力依据（含门切片路径）。容差
        # 1e-5：前缀/全窗的 softmax 求和项数不同 → 浮点非结合舍入
        # （~1e-6），非语义差异。
        torch.manual_seed(17)
        m = _perturb_fb(build_ar2rl(_cfg(ar2=True)).eval())
        v_present = torch.ones(2, dtype=torch.float32)
        t_present = torch.ones(2, dtype=torch.float32)
        m_v = torch.randn(2, 4, 16)
        m_t = torch.randn(2, 4, 16)
        gate = torch.rand(2, 4, 10, 1)     # Cv5 双门形状（窗口级展开）
        ps = torch.randn(2, 4, 148)
        with torch.inference_mode():
            full = m.pose_head(m_v, m_t, v_present, t_present, gate=gate,
                               prev_state=ps, causal=True)
            for k in (1, 2, 3):
                pre = m.pose_head(m_v, m_t, v_present, t_present, gate=gate,
                                  prev_state=ps, causal=True, max_frames=k)
                self.assertEqual(tuple(pre[0].shape), (2, k, 144))
                torch.testing.assert_close(pre[0], full[0][:, :k],
                                           rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(pre[1], full[1][:, :k],
                                           rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(pre[2], full[2][:, :k],
                                           rtol=1e-5, atol=1e-5)

    def test_causality_future_feedback(self):
        # 因果性：扰动未来帧（>L 的 GT 尾帧反馈）不影响闭环帧输出。
        torch.manual_seed(19)
        m = _perturb_fb(build_ar2rl(_cfg(ar2=True)).eval())
        b = _batch()
        cid = torch.full((2,), CONFIG_T)
        ps_a = torch.randn(2, 4, 148)
        ps_b = ps_a.clone()
        ps_b[:, 3] = torch.randn(2, 148)   # 只改帧 3（TF 尾帧）的反馈
        with torch.inference_mode():
            out_a = m.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                                      cid, T_s2m=b["T_s2m"], prev_state=ps_a,
                                      rollout_len=2)
            out_b = m.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                                      cid, T_s2m=b["T_s2m"], prev_state=ps_b,
                                      rollout_len=2)
        torch.testing.assert_close(out_b["x0_hat"][:, :3], out_a["x0_hat"][:, :3],
                                   rtol=0.0, atol=0.0)
        self.assertGreater(
            float((out_b["x0_hat"][:, 3] - out_a["x0_hat"][:, 3]).abs().max()),
            0.0)


class TestLowdimFeedback(unittest.TestCase):
    def test_dims_and_slice_layout(self):
        # 改动 B：投影维度与反馈切片布局（Δyaw 1 + tilt 6 + proj k +
        # v_head 3；总 26 ≤ 32）；外部接口仍 148 维。
        torch.manual_seed(23)
        m = build_ar2rl(_cfg(ar2=True, v7_ar2_lowdim=True,
                              v7_ar2_fb_dim=16)).eval()
        self.assertEqual(m.pose_head.ar2_fb.in_features, 26)
        self.assertEqual(tuple(m.pose_head.ar2_body_proj.weight.shape), (16, 138))
        b = _batch()
        cid = torch.zeros(2, dtype=torch.long)
        ps = torch.zeros(2, 4, 148)
        for c in range(148):
            ps[..., c] = (c + 1) / 100.0
        captured = []

        def hook(mod, inp):
            captured.append(inp[0].detach().clone())

        handle = m.pose_head.ar2_fb.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                _fwd(m, b, cid, prev_state=ps)
        finally:
            handle.remove()
        fb_in = captured[0]
        self.assertEqual(tuple(fb_in.shape), (2, 4, 26))
        torch.testing.assert_close(fb_in[..., :1], ps[..., :1],
                                   rtol=0.0, atol=0.0)      # Δyaw
        torch.testing.assert_close(fb_in[..., 1:7], ps[..., 1:7],
                                   rtol=0.0, atol=0.0)      # 根倾斜
        torch.testing.assert_close(fb_in[..., -3:], ps[..., 145:148],
                                   rtol=0.0, atol=0.0)      # v_head
        torch.testing.assert_close(
            fb_in[..., 7:23], m.pose_head.ar2_body_proj(ps[..., 7:145]),
            rtol=0.0, atol=0.0)                              # body 投影
        # 反馈通道扰动：只改 body 一个通道 → 输出经投影变化（通路有效；
        # ar2_fb 零初始化 → 需一并扰动，否则 fb≡0 不可见）。
        with torch.no_grad():
            for p in m.pose_head.ar2_body_proj.parameters():
                p.add_(0.5)
            for p in m.pose_head.ar2_fb.parameters():
                p.add_(0.5)
        ps2 = ps.clone()
        ps2[..., 100] += 0.3
        with torch.inference_mode():
            o1 = _fwd(m, b, cid, prev_state=ps)
            o2 = _fwd(m, b, cid, prev_state=ps2)
        self.assertGreater(
            float((o2["x0_hat"] - o1["x0_hat"]).abs().max()), 0.0)

    def test_zero_init_contract_with_lowdim_on(self):
        # 低维投影零初始化 → 改动 B 开启时 init 输出仍与 AR2 逐位一致
        # （init 逐位契约的 B 侧延伸）。
        b = _batch()
        cid = torch.zeros(2, dtype=torch.long)
        ps = torch.randn(2, 4, 148)
        torch.manual_seed(7)
        ar2 = build_ar2(_cfg(ar2=True)).eval()
        torch.manual_seed(7)
        rd = build_ar2rl(_cfg(ar2=True, v7_ar2_lowdim=True,
                              v7_ar2_fb_dim=16)).eval()
        with torch.inference_mode():
            o_a = _fwd(ar2, b, cid, prev_state=ps)
            o_d = _fwd(rd, b, cid, prev_state=ps)
        torch.testing.assert_close(o_d["x0_hat"], o_a["x0_hat"],
                                   rtol=0.0, atol=0.0)
        torch.testing.assert_close(o_d["v_hat"], o_a["v_hat"],
                                   rtol=0.0, atol=0.0)

    def test_rollout_runs_with_lowdim(self):
        # RD 组合（rollout × 低维反馈）前向有限、形状正确。
        torch.manual_seed(29)
        m = _perturb_fb(build_ar2rl(
            _cfg(ar2=True, v7_ar2_lowdim=True, v7_ar2_fb_dim=16)).eval())
        b = _batch()
        cid = torch.full((2,), CONFIG_T)
        ps = torch.randn(2, 4, 148)
        with torch.inference_mode():
            out = m.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                                    cid, T_s2m=b["T_s2m"], prev_state=ps,
                                    rollout_len=3)
        self.assertEqual(tuple(out["x0_hat"].shape), (2, 4, 144))
        self.assertEqual(tuple(out["v_hat"].shape), (2, 4, 3))
        self.assertEqual(tuple(out["dyaw_hat"].shape), (2, 4, 1))
        self.assertTrue(bool(torch.isfinite(out["x0_hat"]).all()))

    def test_eval_chain_unaware_of_reduction(self):
        # 评估链不感知降维：ar2_session_chunks 照常组装 148 维反馈，
        # 模型内部投影——RD 走现役自由生成链（02/03 评估链零改动）。
        from anysole.data.dataset import AnySoleDataset
        from anysole.utils.eval_protocol import ar2_session_chunks
        ds = AnySoleDataset(mode="eval", window_length=20, stride=20,
                            session_ids=["S14072"], v7_ar2=True,
                            smpl_support_policy="exclude")
        m = build_ar2rl({**_cfg(ar2=True), "tw": 20,
                         "v7_ar2_lowdim": True, "v7_ar2_fb_dim": 16}).eval()
        with torch.inference_mode():
            chunks, _ = ar2_session_chunks(ds, m, "S14072", list(range(2)),
                                           CONFIG_T, torch.device("cpu"), 20,
                                           {})
        self.assertEqual([c["start"] for c in chunks], [0, 20])
        self.assertTrue(all(c["pred_pose"].shape == (20, 144) for c in chunks))


class TestGuardsAndCurriculum(unittest.TestCase):
    def test_curriculum_values(self):
        # L：起点 1、斜坡终点 = L_max、其后固定 L_max、上界 tw−1。
        self.assertEqual(_ar_rollout_curriculum(0, 1000, 20, l_max=8,
                                                schedule_frac=0.6), 1)
        self.assertEqual(_ar_rollout_curriculum(300, 1000, 20, l_max=8,
                                                schedule_frac=0.6),
                         1 + round(7 * 0.5))
        self.assertEqual(_ar_rollout_curriculum(600, 1000, 20, l_max=8,
                                                schedule_frac=0.6), 8)
        self.assertEqual(_ar_rollout_curriculum(999, 1000, 20, l_max=8,
                                                schedule_frac=0.6), 8)
        self.assertEqual(_ar_rollout_curriculum(10 ** 6, 1000, 20, l_max=19,
                                                schedule_frac=0.6), 19)
        with self.assertRaises(ValueError):
            _ar_rollout_curriculum(0, 1000, 20, l_max=20, schedule_frac=0.6)

    def test_rollout_guards(self):
        # 死开关防御：AR2M（无反馈支路）上 rollout 必须报错而非静默回退。
        b = _loss_batch(_batch())
        m = build_ar2m(_cfg(ar2=True, v7_ar2_fb=False)).eval()
        with self.assertRaisesRegex(ValueError, "ar2 with feedback"):
            backward_v7_enum(m, b, _enum_cfg(), ar_rollout_len=1)
        ps = torch.randn(2, 4, 148)
        with self.assertRaisesRegex(ValueError, "ar2 with feedback"):
            m.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                              torch.zeros(2, dtype=torch.long),
                              T_s2m=b["T_s2m"], prev_state=ps, rollout_len=1)
        # SS 与 rollout 互斥。
        m2 = build_ar2(_cfg(ar2=True))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            backward_v7_enum(m2, b, _enum_cfg(), ar_ss_p=0.3, ar_rollout_len=1)
        # rollout 长度界。
        with self.assertRaisesRegex(ValueError, "tw-1"):
            m2.rollout_forward(b["V_feat"], b["T_raw"], b["T_phys"],
                               torch.zeros(2, dtype=torch.long),
                               T_s2m=b["T_s2m"], prev_state=ps, rollout_len=4)
        # 低维反馈必须有反馈支路（V7MotionHead 构造守卫）。
        from anysole.models.common.model_v7 import V7MotionHead
        from anysole.models.common.embeddings import SharedEmbeddings
        emb = SharedEmbeddings(dim=16, tw=4)
        with self.assertRaisesRegex(ValueError, "ar2 with feedback"):
            V7MotionHead(emb, dim=16, tw=4, ar2=True, ar2_fb=False,
                         ar2_lowdim=True)
        with self.assertRaisesRegex(ValueError, "1, 22"):
            V7MotionHead(emb, dim=16, tw=4, ar2=True, ar2_fb=True,
                         ar2_lowdim=True, ar2_fb_dim=30)


if __name__ == "__main__":
    unittest.main()
