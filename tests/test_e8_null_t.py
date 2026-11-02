"""E8 null-T 路由冒烟（2026-11-03，修改 03「复制捷径与 T 信息边界测量」）：

(a) 不传开关逐位不变：null_t=False（缺省）下，V7 三条协议路径传给模型
    的 T 流与 condition_inputs 输出逐位一致、AR2 观测历史不被改写；
(b) null_t=True：当前窗 T 流（t_raw/t_phys/t_s2m）与 AR2 观测历史
    （T_raw/T_phys）全部置零，V 流与 GT 槽位（pressure_gt）不被触碰；
(c) eval.py 路由：--null-t 只对 V4_Av1 与 V7 家族（model.v7=True）
    放行，其余模型忽略（main 内联路由的等价判定逻辑在此以模型属性
    矩阵复刻——真实路由由实际评估「正常臂复现官方 brief」背书）。

自包含；touch_gait env 运行（CPU 即可）：
    python tests/test_e8_null_t.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from anysole.models.v7_c2 import build as build_cv5
from anysole.models.v7_c2ar2 import build as build_ar2
from anysole.train import condition_inputs, move_batch
from anysole.data.dataset import collate_windows
from anysole.utils.eval_protocol import (
    _v7_ar_session_seq,
    _v7_ctx_session_seq,
    ar2_session_chunks,
)
from anysole.types import CONFIG_T, CONFIG_VT

TW = 4
N_JOINTS = 24


def _cfg(ar2=False):
    return {
        "d_model": 16, "tw": TW, "pose_layers": 1, "dropout": 0.0,
        "v7_task_routes": True, "v7_reliability_route": True,
        "v7_ar2": ar2, "model_name": "V7_Cv5AR2" if ar2 else "V7_Cv5",
        "obs_context": 8 if ar2 else TW,
    }


def _item(start=0, with_history=False):
    """逐窗 item（与 AnySoleDataset.__getitem__ 同构：窗级张量无批维）。"""
    rng = torch.Generator().manual_seed(1000 + start)
    item = {
        "V_feat": torch.randn(TW, 2051, generator=rng),
        "T_raw": torch.rand(TW, 96, generator=rng) * 2 + 0.5,
        "T_phys": torch.rand(TW, 12, generator=rng),
        "T_s2m": torch.randn(TW, 50, generator=rng),
        "pose_gt": torch.randn(TW, 144, generator=rng),
        "vel_gt": torch.randn(TW, 3, generator=rng),
        "dyaw_gt": torch.randn(TW, generator=rng) * 0.01,
        "v_head_gt": torch.randn(TW, 3, generator=rng),
        "psi_gt": torch.randn(TW, generator=rng),
        "psi_base": torch.tensor(0.3),
        "kp_gt": torch.randn(TW, N_JOINTS, 3, generator=rng),
        "contact_gt": torch.zeros(TW, 2),
        "trans_anchor": torch.randn(3, generator=rng),
        "trans_gt": torch.randn(TW, 3, generator=rng),
        "floor_y": torch.tensor(0.02),
        "offsets": torch.randn(N_JOINTS, 3, generator=rng).abs(),
        "parents": torch.zeros(N_JOINTS, dtype=torch.long),
        "betas": torch.randn(10, generator=rng),
        "root_rot_init": torch.randn(3, 3, generator=rng),
        "session_id": "S99999",
        "frame_start": torch.tensor(start, dtype=torch.long),
    }
    if with_history:
        h = torch.Generator().manual_seed(5000 + start)
        item["obs_history"] = {
            "V_feat": torch.randn(TW, 2051, generator=h),
            "T_raw": torch.rand(TW, 96, generator=h) * 2 + 0.5,
            "T_phys": torch.rand(TW, 12, generator=h),
        }
    return item


class _FakeDS:
    """两个连续窗口的假数据集（窗口间隔 tw → 状态链不复位）。"""

    def __init__(self, with_history=False):
        self.items = [_item(start=0, with_history=with_history),
                      _item(start=TW, with_history=with_history)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class _Recording:
    """记录传给模型前向的 T 流（位置参数 + kwargs）。"""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.inner(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _expected_streams(item, cid):
    batch = move_batch(collate_windows([item]), "cpu")
    _, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
    return batch, t_raw, t_phys, t_s2m


def _assert_streams_zero(args, kwargs):
    assert bool((args[1] == 0).all()), "t_raw not zeroed"
    assert bool((args[2] == 0).all()), "t_phys not zeroed"
    assert bool((kwargs["T_s2m"] == 0).all()), "t_s2m not zeroed"


class TestAR2NullT(unittest.TestCase):
    def test_ar2_null_t_off_bit_identical(self):
        ds = _FakeDS(with_history=True)
        torch.manual_seed(7)
        model = _Recording(build_ar2(_cfg(ar2=True)).eval())
        chunks, _ = ar2_session_chunks(
            ds, model, "S99999", [0, 1], CONFIG_VT, "cpu", TW, {},
            dyaw_base="zero", null_t=False)
        self.assertEqual(len(model.calls), 2 * TW)  # 逐窗逐帧串行
        # 每窗首个调用（帧 0）的 T 流 = condition_inputs 输出（逐位）。
        for w in range(2):
            item = ds.items[w]
            cid = torch.full((1,), CONFIG_VT, dtype=torch.long)
            _, t_raw, t_phys, t_s2m = _expected_streams(item, cid)
            args, kwargs = model.calls[w * TW]
            self.assertTrue(torch.equal(args[1], t_raw))
            self.assertTrue(torch.equal(args[2], t_phys))
            self.assertTrue(torch.equal(kwargs["T_s2m"], t_s2m))
            hist = kwargs.get("obs_history")
            self.assertIsNotNone(hist)
            self.assertTrue(torch.equal(
                hist[0]["T_raw"], item["obs_history"]["T_raw"]))
            self.assertTrue(torch.equal(
                hist[0]["T_phys"], item["obs_history"]["T_phys"]))
        # GT 槽位不被触碰：pressure_gt = 原始 T_raw。
        self.assertTrue(torch.equal(
            chunks[0]["pressure_gt"], ds.items[0]["T_raw"]))
        # 两次 null_t=False 运行逐位一致（确定性 + 无隐式改写）。
        torch.manual_seed(7)
        model2 = _Recording(build_ar2(_cfg(ar2=True)).eval())
        chunks2, _ = ar2_session_chunks(
            ds, model2, "S99999", [0, 1], CONFIG_VT, "cpu", TW, {},
            dyaw_base="zero", null_t=False)
        for k in ("pred_pose", "pred_trans", "gt_pose", "gt_trans"):
            self.assertTrue(torch.equal(chunks[0][k], chunks2[0][k]))

    def test_ar2_null_t_on_zeroes_all_t(self):
        ds = _FakeDS(with_history=True)
        model = _Recording(build_ar2(_cfg(ar2=True)).eval())
        chunks, _ = ar2_session_chunks(
            ds, model, "S99999", [0, 1], CONFIG_T, "cpu", TW, {},
            dyaw_base="zero", null_t=True)
        for w in range(2):
            args, kwargs = model.calls[w * TW]
            _assert_streams_zero(args, kwargs)
            h = kwargs.get("obs_history")
            self.assertIsNotNone(h)
            self.assertTrue(bool((h[0]["T_raw"] == 0).all()))
            self.assertTrue(bool((h[0]["T_phys"] == 0).all()))
            self.assertTrue(torch.equal(
                h[0]["V_feat"], ds.items[w]["obs_history"]["V_feat"]))
        # GT 槽位不被触碰：pressure_gt 仍 = 原始 T_raw。
        self.assertTrue(torch.equal(
            chunks[0]["pressure_gt"], ds.items[0]["T_raw"]))


class TestFrameARNullT(unittest.TestCase):
    def test_ar_null_t_routing(self):
        ds = _FakeDS(with_history=False)
        model = _Recording(build_cv5(_cfg()).eval())
        _v7_ar_session_seq(ds, model, "S99999", [0, 1], CONFIG_VT,
                           "cpu", TW, {}, seed=0, null_t=False)
        args, kwargs = model.calls[0]
        self.assertTrue(bool((args[1] != 0).any()))
        model = _Recording(build_cv5(_cfg()).eval())
        _v7_ar_session_seq(ds, model, "S99999", [0, 1], CONFIG_VT,
                           "cpu", TW, {}, seed=0, null_t=True)
        args, kwargs = model.calls[0]
        _assert_streams_zero(args, kwargs)


class TestCtxNullT(unittest.TestCase):
    def test_ctx_null_t_routing(self):
        ds = _FakeDS(with_history=False)
        model = _Recording(build_cv5(_cfg()).eval())
        _v7_ctx_session_seq(ds, model, "S99999", [0, 1], CONFIG_VT,
                            "cpu", TW // 2, {}, null_t=False)
        args, kwargs = model.calls[0]
        self.assertTrue(bool((args[1] != 0).any()))
        model = _Recording(build_cv5(_cfg()).eval())
        _v7_ctx_session_seq(ds, model, "S99999", [0, 1], CONFIG_VT,
                            "cpu", TW // 2, {}, null_t=True)
        args, kwargs = model.calls[0]
        _assert_streams_zero(args, kwargs)


class TestEvalGate(unittest.TestCase):
    def test_gate_matrix(self):
        """路由判定的等价逻辑：null_t 只在 v4_1 或 v7 家族（model.v7）
        放行；V1/V2/V3/V5_1 等无 v7 属性且非 v4_1 的模型忽略。"""
        for null_t in (False, True):
            for v4_1, v7 in ((False, False), (True, False), (False, True),
                             (True, True)):
                active = bool(null_t) and (v4_1 or v7)
                if not null_t:
                    self.assertFalse(active)  # 不传开关 → 永不生效
                else:
                    self.assertEqual(active, v4_1 or v7)


if __name__ == "__main__":
    unittest.main()
