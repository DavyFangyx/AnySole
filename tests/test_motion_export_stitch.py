"""_pack_motion_export 窗口拼接语义（2026-10-31 修复交叉淡化）。

历史 4 帧交叉淡化把不同时间的 6D 向量互相混合（上一窗末 4 帧 × 下一窗
前 4 帧），T2M 两窗航向差大时经重正交化放大为 180° 翻转。修复后：
背靠背/有缺口窗口直接拼接（零混合）；仅真正时间重叠（continuation）
时对同时间帧对做淡化；pos_mode 保持无淡化。
"""
import unittest

import numpy as np

from anysole.eval import _pack_motion_export

N_JOINTS = 24
POSE_DIM = 6 * N_JOINTS
IDENTITY6 = np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], N_JOINTS)
# 绕 Y 轴 180°（R = diag(-1, 1, -1)）的 6D 表示：前两列 (-1,0,0),(0,1,0)
YAW1806 = np.tile([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0], N_JOINTS)


def _win(start, pose, length=None):
    length = len(pose) if length is None else length
    trans = np.zeros((length, 3), dtype=np.float32)
    return (start, pose[:length].astype(np.float32), trans, trans.copy())


def _session(n_frames):
    parents = np.zeros(N_JOINTS, dtype=np.int64)
    parents[0] = -1
    return {
        "V_feat": np.zeros((n_frames, 8), dtype=np.float32),
        "fake_mask": np.zeros(n_frames, dtype=np.uint8),
        "offsets": np.zeros((N_JOINTS, 3), dtype=np.float32),
        "parents": parents,
        "trans_global": np.zeros((n_frames, 3), dtype=np.float32),
    }


class MotionExportStitchTests(unittest.TestCase):
    def test_back_to_back_windows_are_not_blended(self):
        # 两窗航向差 180°：修复前淡化会把 16-19/0-3 帧写成跨窗混合；
        # 修复后应为原始拼接，逐位相等。
        a = np.tile(IDENTITY6, (20, 1))
        b = np.tile(YAW1806, (20, 1))
        packed = _pack_motion_export(_session(40),
                                     [_win(0, a), _win(20, b)], False)
        np.testing.assert_array_equal(packed["pose"][:20], a)
        np.testing.assert_array_equal(packed["pose"][20:], b)

    def test_gapped_windows_are_not_blended(self):
        a = np.tile(IDENTITY6, (20, 1))
        b = np.tile(YAW1806, (15, 1))
        packed = _pack_motion_export(_session(40),
                                     [_win(0, a), _win(25, b)], False)
        np.testing.assert_array_equal(packed["pose"][:20], a)
        np.testing.assert_array_equal(packed["pose"][25:], b)

    def test_overlap_blends_same_time_frames_only(self):
        # 重叠 10 帧（continuation 口径）：淡化只发生在同时间帧对
        # （prev a=16..19 ↔ current b=6..9），其余帧保持原值。
        a = np.tile(IDENTITY6, (20, 1)) * 1.0
        b = np.tile(YAW1806, (20, 1)) * 2.0
        packed = _pack_motion_export(_session(40),
                                     [_win(0, a), _win(10, b)], False)
        pose = packed["pose"]
        # 非重叠区：0-9 = prev 原值，10-15 = current 后写覆盖（原值），
        # 20-29 = current 原值（30-39 无覆盖，保持函数初始 identity）
        np.testing.assert_array_equal(pose[:10], a[:10])
        np.testing.assert_array_equal(pose[10:16], b[0:6])
        np.testing.assert_array_equal(pose[20:30], b[10:20])
        # 重叠区淡化侧（current 帧 6..9，j=0..3, alpha=0.2..0.8）：
        # pose[16+j] = (1-alpha)·b[6+j] + alpha·a[16+j]
        for j in range(4):
            alpha = (j + 1) / 5.0
            want = (1 - alpha) * b[6 + j] + alpha * a[16 + j]
            np.testing.assert_allclose(pose[16 + j], want, rtol=0, atol=1e-6)

    def test_pos_mode_never_blends(self):
        # pos_mode 无淡化（保持 6D 形状以便 fk 通路）；重叠窗后写覆盖，
        # 但重叠帧不得出现混合值。
        a = np.zeros((20, POSE_DIM), dtype=np.float32)
        b = np.ones((20, POSE_DIM), dtype=np.float32)
        packed = _pack_motion_export(_session(40),
                                     [_win(0, a), _win(10, b)], True)
        np.testing.assert_array_equal(packed["pose"][:10], a[:10])
        np.testing.assert_array_equal(packed["pose"][10:30], b[:20])


if __name__ == "__main__":
    unittest.main()
