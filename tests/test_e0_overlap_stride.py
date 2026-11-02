"""E0 重叠拼接推理冒烟（2026-11-03，修改 02「接缝归因与重叠拼接推理」）：

(a) 三角权重：对称、窗缘 1/tw > 0、窗中心最大、从缘到心单调；
(b) tile 锚点网格：重叠窗继承所属 tile 的锚，stride=tw 时与现协议
    per-window 锚（left−1）逐位相同；
(c) assemble_overlap_seq stride=tw 逐位等价：单覆盖帧直接拷贝 →
    输出与现协议 reshape 拼接 torch.equal（等价性硬门的基础断言）；
(d) 重叠混合数值：线性通道 = 三角权重加权平均；pred_pose 6D 通道 =
    加权平均后 6D→rotmat→6D 投影（列正交、与朴素线性平均不同）；
(e) blend_overlap_windows_np（导出侧）：单覆盖逐位拷贝、多覆盖加权
    平均 + 6D 投影。
"""
import unittest

import numpy as np
import torch

from anysole.utils.eval_protocol import (
    _tile_anchor_index,
    _triangle_window_weights,
    assemble_overlap_seq,
    blend_overlap_windows_np,
    prune_overlap_windows,
)
from anysole.utils.geometry import rot6d_to_rotmat


def _rand_6d(rng, n, joints=24):
    """随机但规范的 6D（列正交化后取前两列）→ 旋转通道来自 SO(3)。"""
    a = rng.normal(size=(n, joints, 3))
    b = rng.normal(size=(n, joints, 3))
    b1 = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)
    b2 = b - (b1 * b).sum(-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    return np.concatenate([b1, b2], axis=-1).reshape(n, joints * 6)


class TestTriangleWeights(unittest.TestCase):
    def test_shape(self):
        tw = 20
        w = _triangle_window_weights(tw)
        self.assertEqual(len(w), tw)
        # 对称
        for k in range(tw):
            self.assertAlmostEqual(w[k], w[tw - 1 - k], places=12)
        # 窗缘 = 1/tw > 0（会话首尾单覆盖帧权重合法）
        self.assertAlmostEqual(w[0], 1.0 / tw, places=12)
        self.assertAlmostEqual(w[-1], 1.0 / tw, places=12)
        # 窗中心最大（tw 偶数 → 中心两帧并列最大）
        peak = max(w)
        self.assertAlmostEqual(peak, w[tw // 2 - 1], places=12)
        self.assertAlmostEqual(peak, w[tw // 2], places=12)
        # 从缘到心单调增
        for k in range(1, tw // 2):
            self.assertGreater(w[k], w[k - 1])


class TestTileAnchor(unittest.TestCase):
    def test_grid(self):
        tw = 20
        # 非重叠网格：left = w*tw → 锚 = w*tw − 1（首窗 → 0）
        self.assertEqual(_tile_anchor_index(0, tw), 0)
        self.assertEqual(_tile_anchor_index(20, tw), 19)
        self.assertEqual(_tile_anchor_index(40, tw), 39)
        # 重叠窗继承所属 tile 的锚
        self.assertEqual(_tile_anchor_index(10, tw), 0)      # tile 0
        self.assertEqual(_tile_anchor_index(19, tw), 0)
        self.assertEqual(_tile_anchor_index(30, tw), 19)     # tile 1
        self.assertEqual(_tile_anchor_index(58, tw), 39)     # tile 2


def _parts(tw, stride, seed=0):
    """合成两个重叠窗（start = 0 与 stride），各键按帧位置确定性生成。"""
    rng = np.random.RandomState(seed)
    n_frames = tw + stride
    gt = torch.from_numpy(_rand_6d(rng, n_frames)).float().reshape(n_frames, 24, 6)
    parts = []
    for start in range(0, stride + 1, stride):
        d = {
            "pred_pose": (gt[start:start + tw].reshape(tw, 144) * 2.0).clone(),
            "pred_trans": torch.arange(tw, dtype=torch.float32)[:, None].repeat(1, 3) + start,
            "gt_pose": gt[start:start + tw].reshape(tw, 144).clone(),
            "gt_trans": torch.zeros(tw, 3),
            "kp_gt": torch.zeros(tw, 24, 3),
            "contact_gt": torch.zeros(tw, 2),
            "contact_pred": torch.zeros(tw, 2, dtype=torch.bool),
            "pressure_gt": torch.zeros(tw, 96),
            "pressure_pred": None,
            "pos_corr": None,
        }
        parts.append((start, d))
    return parts, gt


class TestPruneOverlapWindows(unittest.TestCase):
    def test_prune_invalid_tile(self):
        """tile 首窗无效时同 tile 平移窗剔除（GT 泄漏硬约束：锚点集与
        评分帧集 = 现协议非重叠网格）；stride=tw 恒等。"""
        tw = 20
        # 会话 1：tile 0 首窗 [0,20) 无效、平移窗 [5,25) 有效 → 剔除；
        # tile 1 首窗 [20,40) 有效 → 平移窗 [25,45) 保留。
        windows = [(0, 5, 25), (0, 20, 40), (0, 25, 45)]
        out = prune_overlap_windows(windows, tw)
        self.assertEqual(out, [(0, 20, 40), (0, 25, 45)])
        # 覆盖 tile 数 = 有效首窗数 = 现协议锚点数
        self.assertEqual(len({(fi, (l // tw) * tw) for fi, l, _r in out}), 1)
        # stride=tw：恒等
        self.assertEqual(prune_overlap_windows(windows, tw=20) is not windows,
                         True)
        self.assertEqual(prune_overlap_windows(
            [(0, 0, 20), (0, 20, 40)], tw=20), [(0, 0, 20), (0, 20, 40)])

    def test_prune_preserves_coverage(self):
        """剔除后覆盖帧集 == 有效 tile 窗的并集（= 现协议评分帧集）。"""
        tw = 20
        windows = [(0, 0, 20), (0, 5, 25), (0, 15, 35), (0, 20, 40)]
        out = prune_overlap_windows(windows, tw)
        cov = {f for _fi, l, r in out for f in range(l, r)}
        base = {f for _fi, l, r in [(0, 0, 20), (0, 20, 40)]
                for f in range(l, r)}
        self.assertEqual(cov, base)


class TestAssembleOverlapSeq(unittest.TestCase):
    def test_stride_eq_tw_bitwise(self):
        """等价性硬门的基础：stride=tw 单覆盖 → 输出与 reshape 拼接逐位
        一致（torch.equal）。"""
        tw, stride = 20, 20
        parts, _ = _parts(tw, stride)
        seq, frames = assemble_overlap_seq(parts, tw)
        concat = {key: torch.cat([p[1][key] for p in parts], dim=0)
                  for key in ("pred_pose", "pred_trans", "gt_pose",
                              "gt_trans", "kp_gt", "contact_gt",
                              "contact_pred", "pressure_gt")}
        for key in concat:
            self.assertTrue(torch.equal(seq[key], concat[key]),
                            "key %s 未逐位一致" % key)
        self.assertEqual(frames, list(range(tw * 2)))
        self.assertIsNone(seq["pressure_pred"])
        self.assertIsNone(seq["pos_corr"])

    def test_overlap_linear_and_pose_projection(self):
        tw, stride = 20, 10
        parts, _ = _parts(tw, stride)
        seq, frames = assemble_overlap_seq(parts, tw)
        self.assertEqual(frames, list(range(tw + stride)))
        w = torch.tensor(_triangle_window_weights(tw), dtype=torch.float64)
        # 帧 f 的覆盖窗：(parts 索引, 窗内 k)：窗 0 覆盖 [0, tw)，窗 1
        # 覆盖 [stride, stride+tw)
        for f in frames:
            cov = []
            if f < tw:
                cov.append((0, f))
            if stride <= f < stride + tw:
                cov.append((1, f - stride))
            if len(cov) == 1:
                continue
            kws = torch.tensor([w[k] for _, k in cov], dtype=torch.float64)
            # 线性通道：三角权重加权平均
            vals = torch.stack([parts[i][1]["pred_trans"][k]
                                for i, k in cov]).double()
            expect = (vals * kws.view(-1, 1)).sum(0) / kws.sum()
            self.assertTrue(torch.allclose(seq["pred_trans"][f].double(),
                                           expect, atol=1e-10))
            # 6D 通道：加权平均 → 6D→rotmat→6D 投影（列正交）
            vals6 = torch.stack([parts[i][1]["pred_pose"][k]
                                 for i, k in cov]).double()
            mean6 = (vals6 * kws.view(-1, 1)).sum(0) / kws.sum()
            R = rot6d_to_rotmat(seq["pred_pose"][f].reshape(24, 6))
            for j in range(24):
                eye = R[j].T @ R[j]
                self.assertTrue(torch.allclose(eye, torch.eye(3),
                                               atol=1e-5))
            # 投影 ≠ 朴素线性平均（对非平均的随机 6D 必然不同）
            self.assertFalse(torch.equal(
                seq["pred_pose"][f], mean6.float().reshape(144)),
                "6D 通道应经 SO(3) 投影而非朴素线性平均")
        # GT 键取首覆盖窗
        f = stride
        self.assertTrue(torch.equal(seq["gt_pose"][f],
                                    parts[0][1]["gt_pose"][f]))


class TestBlendOverlapWindowsNp(unittest.TestCase):
    def test_single_coverage_bitwise(self):
        tw, stride = 20, 20
        rng = np.random.RandomState(0)
        pose_a = _rand_6d(rng, tw).astype(np.float32)
        pose_b = _rand_6d(rng, tw).astype(np.float32)
        ta = rng.normal(size=(tw, 3)).astype(np.float32)
        tb = rng.normal(size=(tw, 3)).astype(np.float32)
        windows = [(0, pose_a, ta, ta), (tw, pose_b, tb, tb)]
        pose, trans, frames = blend_overlap_windows_np(windows, tw)
        self.assertEqual(list(frames), list(range(2 * tw)))
        self.assertTrue(np.array_equal(pose[:tw], pose_a))
        self.assertTrue(np.array_equal(pose[tw:], pose_b))
        self.assertTrue(np.array_equal(trans[:tw], ta))
        self.assertTrue(np.array_equal(trans[tw:], tb))

    def test_overlap_weighted(self):
        tw, stride = 20, 10
        rng = np.random.RandomState(1)
        w_a = _rand_6d(rng, tw).astype(np.float32)
        w_b = _rand_6d(rng, tw).astype(np.float32)
        ta = np.ones((tw, 3), np.float32)
        tb = np.full((tw, 3), 3.0, np.float32)
        windows = [(0, w_a, ta, ta), (stride, w_b, tb, tb)]
        pose, trans, frames = blend_overlap_windows_np(windows, tw)
        self.assertEqual(list(frames), list(range(tw + stride)))
        ws = _triangle_window_weights(tw)
        for f in range(stride, tw):
            w0, w1 = ws[f], ws[f - stride]
            s = w0 + w1
            expect = (w0 * 1.0 + w1 * 3.0) / s
            self.assertAlmostEqual(float(trans[f, 0]), expect, places=5)
            self.assertAlmostEqual(float(trans[f, 1]), expect, places=5)


if __name__ == "__main__":
    unittest.main()
