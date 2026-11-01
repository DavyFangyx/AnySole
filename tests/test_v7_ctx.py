"""V7 续写协议（2026-11-01，V7_续写协议方案 §1-§3）冒烟：

- ctx 零向量 / prev_ctx=None 时逐位复现 Cv5（零初始化契约）；
- ctx 经 ctx_emb 加性注入根速度 query（不改变输出头/解码器层）；
- dataset v7_ctx 列 = 窗首帧 [根 6D + 根速度 3 + 有效标志]，首窗 flag=0；
- 三条件（VT/V/T）均读到 ctx（注入在 query 侧，与模态存在掩码无关）。
"""
import copy
import unittest

import torch

from anysole.models.v7_c2ctx import build as build_ctx
from anysole.models.v7_c2ctx2 import build as build_ctx2
from anysole.models.v7_c2ctx3 import build as build_ctx3
from anysole.models.v7_c2ctx4 import build as build_ctx4
from anysole.models.v7_c2ctx4b import build as build_ctx4b
from anysole.models.v7_c2ctx5 import build as build_ctx5
from anysole.models.v7_c2ar import build as build_ar
from anysole.models.v7_c2 import build as build_cv5
from anysole.utils.geometry import compose_rot6d, rot6d_to_rotmat


def _cfg(**over):
    base = {"model_name": "V7_Cv5Ctx", "modal": "anysolev2", "d_model": 16,
            "tw": 4, "pose_layers": 1, "dropout": 0., "v7_prev_ctx": True,
            "v7_task_routes": True}
    base.update(over)
    return base


def _batch(B, tw, d_v=2051):
    return {
        "V_feat": torch.randn(B, tw, 2051),
        "T_raw": torch.randn(B, tw, 96),
        "T_phys": torch.randn(B, tw, 12),
        "T_s2m": torch.randn(B, tw, 50),
    }


class V7CtxSmokeTests(unittest.TestCase):
    def test_zero_ctx_reproduces_cv5_at_init(self):
        # 同种子构建 Cv5 与 Ctx：prev_ctx=None / 全零时输出逐位一致。
        torch.manual_seed(0)
        cv5 = build_cv5({**_cfg(), "v7_prev_ctx": False})
        torch.manual_seed(0)
        ctx = build_ctx(_cfg())
        cv5.eval(); ctx.eval()
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            out_a = cv5(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"])
            out_b = ctx(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                        prev_ctx=None)
            out_c = ctx(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                        prev_ctx=torch.zeros(2, 10))
        torch.testing.assert_close(out_a["x0_hat"], out_b["x0_hat"])
        torch.testing.assert_close(out_b["x0_hat"], out_c["x0_hat"])

    def test_ctx_gradient_and_output_change(self):
        # 零初始化下输出恒零 → 输入梯度恒零（数学性质）；梯度/通路验证
        # 必须先把 ctx_emb 随机化，再检查 (a) 输出随 ctx 改变 (b) 输入
        # 梯度非零。
        torch.manual_seed(2)
        ctx = build_ctx(_cfg())
        ctx.eval()
        for p_ in ctx.ctx_emb.parameters():
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            a = ctx(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                    prev_ctx=torch.zeros(2, 10))["x0_hat"]
            c = ctx(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                    prev_ctx=torch.ones(2, 10))["x0_hat"]
        self.assertGreater(float((a - c).abs().mean()), 0.0)
        p0 = torch.randn(2, 10, requires_grad=True)
        out = ctx(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                  prev_ctx=p0)
        out["v_hat"].square().mean().backward()
        self.assertIsNotNone(p0.grad)
        self.assertGreater(float(p0.grad.abs().sum()), 0.0)

    def test_three_conditions_all_see_ctx(self):
        # ctx 加性注入根 query（query 侧不乘模态存在掩码）：三条件
        # （VT/V/T）下 ctx 都应影响输出。
        torch.manual_seed(3)
        ctx = build_ctx(_cfg())
        ctx.eval()
        for p_ in ctx.ctx_emb.parameters():
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            for cid in (0, 1, 2):  # VT / V / T
                a = ctx(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), cid), T_s2m=b["T_s2m"],
                        prev_ctx=torch.zeros(2, 10))["x0_hat"]
                c = ctx(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), cid), T_s2m=b["T_s2m"],
                        prev_ctx=torch.ones(2, 10))["x0_hat"]
                self.assertGreater(float((a - c).abs().mean()), 0.0,
                                  f"condition {cid} does not see ctx")


class V7Ctx2SmokeTests(unittest.TestCase):
    """续写 v2（2026-11-01）：Level 1 部位 query 注入 + Level 2 输出残差。"""

    def test_zero_ctx_reproduces_cv5_at_init(self):
        # v2 的两条通路均零初始化 → init 时贡献严格为零、逐位复现 Cv5。
        torch.manual_seed(0)
        cv5 = build_cv5({**_cfg(), "v7_prev_ctx": False})
        torch.manual_seed(0)
        ctx2 = build_ctx2({**_cfg(), "v7_ctx_v2": True})
        cv5.eval(); ctx2.eval()
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            out_a = cv5(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"])
            out_b = ctx2(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                         prev_ctx=torch.zeros(2, 10))
        torch.testing.assert_close(out_a["x0_hat"], out_b["x0_hat"])

    def test_ctx_affects_root6_and_grad(self):
        # 随机化两条 ctx 通路后：根朝向 6D 输出随 ctx 改变（Level 1 部位
        # 注入 + Level 2 残差），且 ctx 输入有非零梯度。
        torch.manual_seed(4)
        ctx2 = build_ctx2({**_cfg(), "v7_ctx_v2": True})
        ctx2.eval()
        for p_ in list(ctx2.ctx_emb.parameters()) + list(ctx2.ctx_pose_emb.parameters()):
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            a = ctx2(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                     prev_ctx=torch.zeros(2, 10))["x0_hat"]
            c = ctx2(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                     prev_ctx=torch.ones(2, 10))["x0_hat"]
        self.assertGreater(float((a[..., :6] - c[..., :6]).abs().mean()), 0.0)
        p0 = torch.randn(2, 10, requires_grad=True)
        out = ctx2(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                   prev_ctx=p0)
        out["x0_hat"].square().mean().backward()
        self.assertGreater(float(p0.grad.abs().sum()), 0.0)

    def test_three_conditions_all_see_ctx(self):
        torch.manual_seed(5)
        ctx2 = build_ctx2({**_cfg(), "v7_ctx_v2": True})
        ctx2.eval()
        for p_ in ctx2.ctx_pose_emb.parameters():
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            for cid in (0, 1, 2):  # VT / V / T
                a = ctx2(b["V_feat"], b["T_raw"], b["T_phys"],
                         config_id=torch.full((2,), cid), T_s2m=b["T_s2m"],
                         prev_ctx=torch.zeros(2, 10))["x0_hat"]
                c = ctx2(b["V_feat"], b["T_raw"], b["T_phys"],
                         config_id=torch.full((2,), cid), T_s2m=b["T_s2m"],
                         prev_ctx=torch.ones(2, 10))["x0_hat"]
                self.assertGreater(float((a - c).abs().mean()), 0.0,
                                  f"condition {cid} does not see ctx")


class V7Ctx3SmokeTests(unittest.TestCase):
    """续写 v3（2026-11-01）：T-only 硬填式 10 帧槽位替换 + V 在场门控。"""

    def test_zero_ctx_reproduces_cv5_at_init(self):
        # flag=0 块不替换（保留学习 query）→ init 逐位 = Cv5。
        torch.manual_seed(0)
        cv5 = build_cv5({**_cfg(), "v7_prev_ctx": False})
        torch.manual_seed(0)
        ctx3 = build_ctx3({**_cfg(), "v7_ctx3": True})
        cv5.eval(); ctx3.eval()
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            out_a = cv5(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"])
            out_b = ctx3(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                         prev_ctx=None)
            out_c = ctx3(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                         prev_ctx=torch.zeros(2, 2, 10))
        torch.testing.assert_close(out_a["x0_hat"], out_b["x0_hat"])
        torch.testing.assert_close(out_b["x0_hat"], out_c["x0_hat"])

    def test_gating_v_present_ignores_ctx(self):
        # 门控 A：V 在场（VT 行）时 ctx 块不影响输出；T-only 时影响。
        torch.manual_seed(6)
        ctx3 = build_ctx3({**_cfg(), "v7_ctx3": True})
        ctx3.eval()
        for p_ in ctx3.ctx_emb.parameters():
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        blk = torch.randn(2, 2, 10)
        blk[..., -1] = 1.0
        with torch.inference_mode():
            a_vt = ctx3(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), 0), T_s2m=b["T_s2m"],
                        prev_ctx=blk)["x0_hat"]
            c_vt = ctx3(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), 0), T_s2m=b["T_s2m"],
                        prev_ctx=torch.zeros(2, 2, 10))["x0_hat"]
            self.assertAlmostEqual(float((a_vt - c_vt).abs().max()), 0.0, places=6,
                                   msg="VT 行不应读 ctx（门控失败）")
            a_t = ctx3(b["V_feat"], b["T_raw"], b["T_phys"],
                       config_id=torch.full((2,), 2), T_s2m=b["T_s2m"],
                       prev_ctx=blk)["x0_hat"]
            c_t = ctx3(b["V_feat"], b["T_raw"], b["T_phys"],
                       config_id=torch.full((2,), 2), T_s2m=b["T_s2m"],
                       prev_ctx=torch.zeros(2, 2, 10))["x0_hat"]
            self.assertGreater(float((a_t - c_t).abs().mean()), 0.0,
                               "T-only 应读 ctx 块")

    def test_ctx_block_grad(self):
        torch.manual_seed(7)
        ctx3 = build_ctx3({**_cfg(), "v7_ctx3": True})
        ctx3.eval()
        for p_ in ctx3.ctx_emb.parameters():
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        p0 = torch.randn(2, 2, 10)
        p0[..., -1] = 1.0
        p0 = p0.requires_grad_()
        out = ctx3(b["V_feat"], b["T_raw"], b["T_phys"],
                   config_id=torch.full((2,), 2), T_s2m=b["T_s2m"],
                   prev_ctx=p0)
        out["x0_hat"].square().mean().backward()
        self.assertGreater(float(p0.grad.abs().sum()), 0.0)

    def test_dataset_ctx3_column(self):
        from anysole.data.dataset import AnySoleDataset
        ds = AnySoleDataset(mode="train", window_length=20,
                            session_ids=["S14072"], v7_ctx=True, v7_ctx3=True)
        first = ds[0]
        self.assertEqual(tuple(first["ctx"].shape), (10, 10))
        import torch as _t
        want = _t.cat([first["pose_gt"][:10, :6], first["vel_gt"][:10, :3],
                       _t.full((10, 1), 1.0 if int(first["frame_start"]) != 0 else 0.0)], dim=-1)
        self.assertAlmostEqual(float((first["ctx"] - want).abs().max()), 0.0, places=6)


class V7Ctx4SmokeTests(unittest.TestCase):
    """续写 v4（2026-11-01）：锚点系残差分解（v4-a 混合空间版）。"""

    def test_zero_ctx_reproduces_cv5_at_init(self):
        torch.manual_seed(0)
        cv5 = build_cv5({**_cfg(), "v7_prev_ctx": False})
        torch.manual_seed(0)
        ctx4 = build_ctx4({**_cfg(), "v7_ctx4": True})
        cv5.eval(); ctx4.eval()
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            out_a = cv5(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"])
            out_b = ctx4(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                         prev_ctx=None)
            out_c = ctx4(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                         prev_ctx=torch.zeros(2, 2, 10))
        torch.testing.assert_close(out_a["x0_hat"], out_b["x0_hat"])
        torch.testing.assert_close(out_b["x0_hat"], out_c["x0_hat"])

    def test_gating_v_present_ignores_ctx(self):
        torch.manual_seed(8)
        ctx4 = build_ctx4({**_cfg(), "v7_ctx4": True})
        ctx4.eval()
        for p_ in ctx4.ctx_emb.parameters():
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        blk = torch.randn(2, 2, 10)
        blk[..., -1] = 1.0
        with torch.inference_mode():
            a_vt = ctx4(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), 0), T_s2m=b["T_s2m"],
                        prev_ctx=blk)["x0_hat"]
            c_vt = ctx4(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), 0), T_s2m=b["T_s2m"],
                        prev_ctx=torch.zeros(2, 2, 10))["x0_hat"]
            self.assertAlmostEqual(float((a_vt - c_vt).abs().max()), 0.0, places=6)
            a_t = ctx4(b["V_feat"], b["T_raw"], b["T_phys"],
                       config_id=torch.full((2,), 2), T_s2m=b["T_s2m"],
                       prev_ctx=blk)["x0_hat"]
            c_t = ctx4(b["V_feat"], b["T_raw"], b["T_phys"],
                       config_id=torch.full((2,), 2), T_s2m=b["T_s2m"],
                       prev_ctx=torch.zeros(2, 2, 10))["x0_hat"]
            self.assertGreater(float((a_t - c_t).abs().mean()), 0.0)

    def test_compose_roundtrip_and_identity(self):
        # R_a ⊗ (R_aᵀ ⊗ x) ≈ x（x 取合法旋转 6D——正交化对合法旋转可逆；
        # 随机 6D 列不单位正交，正交化丢尺度属预期）；残差帧 0 根 = 恒等。
        torch.manual_seed(9)
        from anysole.utils.geometry import rotmat_to_6d
        R_a = rot6d_to_rotmat(torch.randn(2, 6))
        x = rotmat_to_6d(torch.matmul(
            rot6d_to_rotmat(torch.randn(2, 3, 6)),
            rot6d_to_rotmat(torch.randn(2, 3, 6)).transpose(-1, -2)))
        res = compose_rot6d(R_a.transpose(-1, -2), x)
        back = compose_rot6d(R_a, res)
        self.assertLess(float((back - x).abs().max()), 1e-5)
        d6 = rotmat_to_6d(rot6d_to_rotmat(torch.randn(2, 2, 6)))
        R0 = rot6d_to_rotmat(d6[:, 0])
        res0 = compose_rot6d(R0.transpose(-1, -2), d6[:, 0])
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertLess(float((res0 - identity).abs().max()), 1e-5)

    def test_train_residual_batch(self):
        from anysole.train import _ctx4_residual_batch
        b = {
            "pose_gt": torch.randn(2, 4, 144),
            "vel_gt": torch.randn(2, 4, 3),
            "trans_gt": torch.randn(2, 4, 3),
        }
        rb = _ctx4_residual_batch(b)
        # 帧 0 根残差 = 恒等（变换后直接检查，不再二次组合）；
        # 速度往返 = 原值。
        R_a = rot6d_to_rotmat(b["pose_gt"][:, 0, :6])
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertLess(float((rb["pose_gt"][:, 0, :6] - identity).abs().max()), 1e-5)
        vel_back = torch.einsum("bij,btj->bti", R_a, rb["vel_gt"])
        self.assertLess(float((vel_back - b["vel_gt"]).abs().max()), 1e-5)
        # trans 由残差 vel 重积分
        want = torch.cumsum(rb["vel_gt"], dim=1) / 40.0
        self.assertLess(float((rb["trans_gt"] - want).abs().max()), 1e-6)

    def test_dataset_ctx4_column(self):
        from anysole.data.dataset import AnySoleDataset
        ds = AnySoleDataset(mode="train", window_length=20,
                            session_ids=["S14072"], v7_ctx=True, v7_ctx3=True)
        first = ds[0]
        self.assertEqual(tuple(first["ctx"].shape), (10, 10))


class V7Ctx5SmokeTests(unittest.TestCase):
    """续写 v5（2026-11-01）：块自回归——全姿态硬填（nf × 148）。"""

    def test_zero_ctx_reproduces_cv5_at_init(self):
        torch.manual_seed(0)
        cv5 = build_cv5({**_cfg(), "v7_prev_ctx": False})
        torch.manual_seed(0)
        ctx5 = build_ctx5({**_cfg(), "v7_ctx5": True})
        cv5.eval(); ctx5.eval()
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            out_a = cv5(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"])
            out_b = ctx5(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                         prev_ctx=None)
            out_c = ctx5(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                         prev_ctx=torch.zeros(2, 2, 148))
        torch.testing.assert_close(out_a["x0_hat"], out_b["x0_hat"])
        torch.testing.assert_close(out_b["x0_hat"], out_c["x0_hat"])

    def test_gating_v_present_ignores_ctx(self):
        torch.manual_seed(10)
        ctx5 = build_ctx5({**_cfg(), "v7_ctx5": True})
        ctx5.eval()
        for mod in [ctx5.pose_head.ctx_root_emb] + list(ctx5.pose_head.ctx_part_embs):
            for p_ in mod.parameters():
                p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        blk = torch.randn(2, 2, 148)
        blk[..., -1] = 1.0
        with torch.inference_mode():
            a_vt = ctx5(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), 0), T_s2m=b["T_s2m"],
                        prev_ctx=blk)["x0_hat"]
            c_vt = ctx5(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), 0), T_s2m=b["T_s2m"],
                        prev_ctx=torch.zeros(2, 2, 148))["x0_hat"]
            self.assertAlmostEqual(float((a_vt - c_vt).abs().max()), 0.0, places=6)
            a_t = ctx5(b["V_feat"], b["T_raw"], b["T_phys"],
                       config_id=torch.full((2,), 2), T_s2m=b["T_s2m"],
                       prev_ctx=blk)["x0_hat"]
            c_t = ctx5(b["V_feat"], b["T_raw"], b["T_phys"],
                       config_id=torch.full((2,), 2), T_s2m=b["T_s2m"],
                       prev_ctx=torch.zeros(2, 2, 148))["x0_hat"]
            self.assertGreater(float((a_t - c_t).abs().mean()), 0.0)

    def test_ctx_block_grad(self):
        torch.manual_seed(11)
        ctx5 = build_ctx5({**_cfg(), "v7_ctx5": True})
        ctx5.eval()
        for mod in [ctx5.pose_head.ctx_root_emb] + list(ctx5.pose_head.ctx_part_embs):
            for p_ in mod.parameters():
                p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        p0 = torch.randn(2, 2, 148)
        p0[..., -1] = 1.0
        p0 = p0.requires_grad_()
        out = ctx5(b["V_feat"], b["T_raw"], b["T_phys"],
                   config_id=torch.full((2,), 2), T_s2m=b["T_s2m"],
                   prev_ctx=p0)
        out["x0_hat"].square().mean().backward()
        self.assertGreater(float(p0.grad.abs().sum()), 0.0)

    def test_dataset_ctx5_column(self):
        from anysole.data.dataset import AnySoleDataset
        ds = AnySoleDataset(mode="train", window_length=20,
                            session_ids=["S14072"], v7_ctx=True,
                            v7_ctx3=True, v7_ctx5=True)
        first = ds[0]
        self.assertEqual(tuple(first["ctx"].shape), (10, 148))
        import torch as _t
        want = _t.cat([first["pose_gt"][:10], first["vel_gt"][:10, :3],
                       _t.full((10, 1), 1.0 if int(first["frame_start"]) != 0 else 0.0)], dim=-1)
        self.assertAlmostEqual(float((first["ctx"] - want).abs().max()), 0.0, places=6)


class V7ArSmokeTests(unittest.TestCase):
    """帧级双模式 AR（2026-11-01）：V/VT 回归 + T 因果自回归，共享网络。"""

    def test_zero_feedback_reproduces_cv5_at_init(self):
        # 反馈零初始化 → 全零 prev_state 时逐位 = Cv5。
        torch.manual_seed(0)
        cv5 = build_cv5({**_cfg(), "v7_prev_ctx": False})
        torch.manual_seed(0)
        ar = build_ar({**_cfg(), "v7_ar": True})
        cv5.eval(); ar.eval()
        b = _batch(2, 4, 8)
        with torch.inference_mode():
            out_a = cv5(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"])
            out_b = ar(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                       prev_state=None)
            out_c = ar(b["V_feat"], b["T_raw"], b["T_phys"], T_s2m=b["T_s2m"],
                       prev_state=torch.zeros(2, 4, 147))
        torch.testing.assert_close(out_a["x0_hat"], out_b["x0_hat"])
        torch.testing.assert_close(out_b["x0_hat"], out_c["x0_hat"])

    def test_all_conditions_feedback_and_causality(self):
        # 全条件因果 + 反馈（2026-11-01 用户裁定）：VT/T 都吃反馈；
        # 因果性：帧 0 输出与后续帧反馈无关（只改帧 1 反馈，帧 0 逐位不变）。
        torch.manual_seed(12)
        ar = build_ar({**_cfg(), "v7_ar": True})
        ar.eval()
        for p_ in ar.pose_head.ar_fb.parameters():
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        fb = torch.randn(2, 4, 147)
        with torch.inference_mode():
            for cid in (0, 2):  # VT / T
                a = ar(b["V_feat"], b["T_raw"], b["T_phys"],
                       config_id=torch.full((2,), cid), T_s2m=b["T_s2m"],
                       prev_state=fb, causal=True)["x0_hat"]
                c = ar(b["V_feat"], b["T_raw"], b["T_phys"],
                       config_id=torch.full((2,), cid), T_s2m=b["T_s2m"],
                       prev_state=torch.zeros(2, 4, 147), causal=True)["x0_hat"]
                self.assertGreater(float((a - c).abs().mean()), 0.0,
                                   f"condition {cid} should see feedback")
            fb2 = torch.zeros(2, 4, 147)
            fb2[:, 1] = fb[:, 1]
            for cid in (0, 2):
                d0 = ar(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), cid), T_s2m=b["T_s2m"],
                        prev_state=fb2, causal=True)["x0_hat"][:, 0]
                c0 = ar(b["V_feat"], b["T_raw"], b["T_phys"],
                        config_id=torch.full((2,), cid), T_s2m=b["T_s2m"],
                        prev_state=torch.zeros(2, 4, 147), causal=True)["x0_hat"][:, 0]
                self.assertAlmostEqual(float((d0 - c0).abs().max()), 0.0, places=6,
                                       msg=f"condition {cid} causality broken")

    def test_feedback_grad(self):
        torch.manual_seed(13)
        ar = build_ar({**_cfg(), "v7_ar": True})
        ar.eval()
        for p_ in ar.pose_head.ar_fb.parameters():
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        p0 = torch.randn(2, 4, 147, requires_grad=True)
        out = ar(b["V_feat"], b["T_raw"], b["T_phys"],
                 config_id=torch.full((2,), 2), T_s2m=b["T_s2m"],
                 prev_state=p0, causal=True)
        out["x0_hat"].square().mean().backward()
        self.assertGreater(float(p0.grad.abs().sum()), 0.0)

    def test_dataset_prev_state(self):
        from anysole.data.dataset import AnySoleDataset
        ds = AnySoleDataset(mode="train", window_length=20,
                            session_ids=["S14072"], v7_ar=True)
        first = ds[0]
        self.assertEqual(tuple(first["prev_state"].shape), (20, 147))
        import torch as _t
        want = _t.cat([
            _t.cat([_t.zeros(1, 144), first["pose_gt"][:-1]], dim=0),
            _t.cat([_t.zeros(1, 3), first["vel_gt"][:-1]], dim=0)], dim=-1)
        self.assertAlmostEqual(float((first["prev_state"] - want).abs().max()), 0.0, places=6)


class V7CtxDatasetTests(unittest.TestCase):
    def test_dataset_ctx_column(self):
        import numpy as np
        from anysole.data.dataset import AnySoleDataset
        ds = AnySoleDataset(
            mode="train",
            window_length=20,
            session_ids=["S14072"],
            v7_ctx=True,
        )
        self.assertTrue(len(ds) > 0)
        starts = [int(ds[i]["frame_start"]) for i in range(min(len(ds), 4))]
        self.assertEqual(starts, sorted(starts))
        first = ds[0]
        self.assertEqual(tuple(first["ctx"].shape), (10,))
        # ctx = 窗首帧 [根 6D + 根速度 3 + flag]，与窗口自身 GT 首帧一致
        import torch as _t
        want = _t.cat([first["pose_gt"][0, :6], first["vel_gt"][0, :3],
                       _t.tensor([1.0 if starts[0] != 0 else 0.0])])
        self.assertAlmostEqual(float((first["ctx"] - want).abs().max()), 0.0, places=6)
        second = ds[1]
        self.assertAlmostEqual(float(second["ctx"][-1]), 1.0)


if __name__ == "__main__":
    unittest.main()


class V7Ctx4BSmokeTests(unittest.TestCase):
    """v4-b（统一残差空间 + V/VT 锚估计头）冒烟（2026-11-01）。"""

    def _build(self, **over):
        cfg = {**_cfg(), "v7_ctx3": True, "v7_ctx4b": True}
        cfg.update(over)
        return build_ctx4b(cfg)

    def test_zero_ctx_reproduces_cv5_at_init(self):
        # 零 ctx / 无 ctx 时逐位复现 Cv5（锚估计头零初始化不影响前向）。
        torch.manual_seed(30)
        cv5 = build_cv5({**_cfg(), "v7_prev_ctx": False})
        torch.manual_seed(30)
        m = self._build()
        cv5.eval(); m.eval()
        b = _batch(2, 4, 8)
        for cid in (0, 2):
            cid_t = torch.full((2,), cid)
            a = m(b["V_feat"], b["T_raw"], b["T_phys"], cid_t, T_s2m=b["T_s2m"],
                  prev_ctx=torch.zeros(2, 2, 10))["x0_hat"]
            c = cv5(b["V_feat"], b["T_raw"], b["T_phys"], cid_t, T_s2m=b["T_s2m"])["x0_hat"]
            self.assertAlmostEqual(float((a - c).abs().max()), 0.0, places=6)

    def test_anchor_identity_at_init(self):
        torch.manual_seed(31)
        m = self._build()
        b = _batch(2, 4, 8)
        out = m(b["V_feat"], b["T_raw"], b["T_phys"],
                torch.zeros(2, dtype=torch.long), T_s2m=b["T_s2m"])
        eye = torch.eye(3).expand(2, 3, 3)
        self.assertAlmostEqual(float((out["anchor_rot"] - eye).abs().max()),
                               0.0, places=6)

    def test_anchor_head_grad_and_response(self):
        torch.manual_seed(32)
        m = self._build()
        for p_ in m.anchor_off.parameters():
            p_.data.normal_(0, 0.1)
        b = _batch(2, 4, 8)
        out = m(b["V_feat"], b["T_raw"], b["T_phys"],
                torch.zeros(2, dtype=torch.long), T_s2m=b["T_s2m"])
        out["anchor_rot"].sum().backward()
        self.assertGreater(float(m.anchor_off[0].weight.grad.abs().sum()), 0.0)

    def test_l_anchor_masked_for_t(self):
        from anysole.utils.losses import compute_losses
        torch.manual_seed(33)
        m = self._build()
        b = _batch(2, 4, 8)
        b.update({"pose_gt": torch.randn(2, 4, 144),
                  "vel_gt": torch.randn(2, 4, 3),
                  "contact_gt": torch.rand(2, 4, 2)})
        w = {"v7": True, "lambda_pose": 3.0, "lambda_traj": 1.0,
             "lambda_anchor": 1.0, "v7_contact_root": False}
        for cid, expect_pos in ((0, True), (2, False)):
            out = m(b["V_feat"], b["T_raw"], b["T_phys"],
                    torch.full((2,), cid, dtype=torch.long), T_s2m=b["T_s2m"])
            loss = compute_losses(out, b, torch.full((2,), cid, dtype=torch.long), w)
            if expect_pos:
                self.assertGreater(float(loss["L_anchor"]), 0.0)
            else:
                self.assertEqual(float(loss["L_anchor"]), 0.0)

    def test_guards(self):
        from anysole.models.common.model_v7 import AnySoleModelV7
        with self.assertRaises(ValueError):
            AnySoleModelV7(d=16, tw=4, ctx4b=True, ctx4=True, ctx3=True,
                           prev_ctx=True)
        with self.assertRaises(ValueError):
            AnySoleModelV7(d=16, tw=4, ctx4b=True, prev_ctx=True)


class V7ArMaskSmokeTests(unittest.TestCase):
    """ARM 对照格（因果掩码、无反馈）冒烟（2026-11-01）。"""

    def _build(self):
        from anysole.models.v7_c2arm import build as build_arm
        return build_arm({**_cfg(), "v7_prev_ctx": False, "v7_ar": True,
                          "v7_ar_fb": False})

    def test_feedback_ignored_and_mask_active(self):
        torch.manual_seed(34)
        m = self._build()
        m.eval()
        b = _batch(2, 4, 8)
        cid = torch.full((2,), 2)
        with torch.inference_mode():
            a = m(b["V_feat"], b["T_raw"], b["T_phys"], cid, T_s2m=b["T_s2m"],
                  prev_state=torch.randn(2, 4, 147), causal=True)["x0_hat"]
            c = m(b["V_feat"], b["T_raw"], b["T_phys"], cid, T_s2m=b["T_s2m"],
                  prev_state=torch.zeros(2, 4, 147), causal=True)["x0_hat"]
            d = m(b["V_feat"], b["T_raw"], b["T_phys"], cid, T_s2m=b["T_s2m"],
                  prev_state=torch.zeros(2, 4, 147), causal=False)["x0_hat"]
        self.assertAlmostEqual(float((a - c).abs().max()), 0.0, places=6)
        self.assertGreater(float((d - c).abs().max()), 0.0)
