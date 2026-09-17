"""E6.8 --no-imu 检验（AnySole）：确认 IMU 通道被真删，无 IMU 数据进入模型。

检查项：
1. 数据层：38 维 T_s2m 构建 == 50 维构建删掉 IMU 列 [16:22]+[41:47]（逐元素一致）；
   no_imu 路径从不调用 synthesize_imu（monkeypatch 抛异常仍能构建）。
2. 数据集层：AnySoleDataset(no_imu=True) 的 item T_s2m 为 (TW, 38)。
3. 模型层：TactileEncoder(no_imu=True) 只有 6 个编码组（无 IMU 组、无 imu 参数）；
   AnySoleModel(no_imu=True) 前向可跑、state_dict 无 imu 键、50 维输入被拒绝。
4. 回归：默认参数下 50 维路径（8 组）与 raw108 路径行为不变。

用法：/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probe_e6_noimu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gait repo root

import numpy as np
import torch

from anysole.data.dataset import AnySoleDataset
from anysole.data.tactile_s2m import S2M_IMU_COLUMNS, build_t_s2m, drop_imu_columns, synthesize_imu
from anysole.models.tactile_encoder import TactileEncoder
from anysole.models import AnySoleModel
from anysole.types import CONFIG_VT, T_S2M_DIM, T_S2M_NOIMU_DIM, TW, V_FEAT_DIM

SESSION = "S7013"


def check_data_layer() -> None:
    ds50 = AnySoleDataset(mode="train", session_ids=[SESSION], no_imu=False)
    ds38 = AnySoleDataset(mode="train", session_ids=[SESSION], no_imu=True)
    s50 = ds50.sessions[0]
    s38 = ds38.sessions[0]
    t50 = s50["T_s2m"]
    t38 = s38["T_s2m"]
    assert t50.shape == (t50.shape[0], T_S2M_DIM), t50.shape
    assert t38.shape == (t38.shape[0], T_S2M_NOIMU_DIM), t38.shape
    # 38 维构建必须等于 50 维构建删掉 IMU 列（逐元素一致）。
    np.testing.assert_array_equal(t38, drop_imu_columns(t50))
    assert tuple(sorted(S2M_IMU_COLUMNS)) == tuple(range(16, 22)) + tuple(range(41, 47))
    # 数据集中非 IMU 列也逐列一致（pressure16/force/cop 分块对拍）。
    keep = [i for i in range(T_S2M_DIM) if i not in S2M_IMU_COLUMNS]
    np.testing.assert_array_equal(t38, t50[:, keep])
    print("[1] 数据层 OK: 38 维 == 50 维删 IMU 列; session=%s T=%d" % (SESSION, t50.shape[0]))


def check_noimu_never_synthesizes() -> None:
    s = AnySoleDataset(mode="train", session_ids=[SESSION], no_imu=False).sessions[0]

    def boom(*args, **kwargs):
        raise AssertionError("synthesize_imu must not be called on the no_imu path")

    orig = synthesize_imu
    import anysole.data.tactile_s2m as mod

    mod.synthesize_imu = boom
    try:
        out = build_t_s2m(
            s["pose_gt"], s["trans_global"], s["offsets"], s["parents"], s["T_raw"], no_imu=True
        )
    finally:
        mod.synthesize_imu = orig
    assert out.shape[1] == T_S2M_NOIMU_DIM
    print("[2] no_imu 路径从不调用 synthesize_imu OK (dim=%d)" % out.shape[1])


def check_dataset_item() -> None:
    ds = AnySoleDataset(mode="train", session_ids=[SESSION], no_imu=True)
    item = ds[0]
    assert item["T_s2m"].shape == (TW, T_S2M_NOIMU_DIM), item["T_s2m"].shape
    print("[3] 数据集 item T_s2m 形状 OK: %s" % (item["T_s2m"].shape,))


def check_encoder_model() -> None:
    # no-imu: 6 组（l_heel/l_toes/l_others/r_heel/r_toes/r_others），无 imu 组。
    te = TactileEncoder(no_imu=True)
    assert len(te.group_mlps) == 6, len(te.group_mlps)
    in_dims = [int(mlp[0].in_features) for mlp in te.group_mlps]
    assert in_dims == [8, 8, 3, 8, 8, 3], in_dims
    assert not any("imu" in name for name in te.state_dict()), \
        [n for n in te.state_dict() if "imu" in n]
    te(torch.randn(2, TW, T_S2M_NOIMU_DIM))
    try:
        te(torch.randn(2, TW, T_S2M_DIM))
        raise AssertionError("TactileEncoder(no_imu=True) accepted 50-dim input")
    except ValueError:
        pass

    # 回归：默认仍是 8 组、50 维。
    te50 = TactileEncoder()
    assert len(te50.group_mlps) == 8
    te50(torch.randn(2, TW, T_S2M_DIM))
    print("[4] 编码器 OK: no_imu=6 组无 imu 参数、50 维被拒; 默认 8 组回归通过")


def forward(model: AnySoleModel, t_s2m_dim: int, t_s2m=None):
    b = 2
    v = torch.randn(b, TW, V_FEAT_DIM)
    tr = torch.randn(b, TW, 96)
    tp = torch.randn(b, TW, 12)
    xt = torch.randn(b, TW, model.pose_head.pose_dim)
    tau = torch.zeros(b, dtype=torch.long)
    cid = torch.full((b,), CONFIG_VT, dtype=torch.long)
    if t_s2m is None and t_s2m_dim is not None:
        t_s2m = torch.randn(b, TW, t_s2m_dim)
    out = model(v, tr, tp, xt, tau, cid, ["S7013"] * b, T_s2m=t_s2m)
    return out["x0_hat"]


def check_full_model() -> None:
    for direct in (False, True):
        model = AnySoleModel(
            tactile_input="s2m50", tactile_direct=direct, no_imu=True,
        )
        bad = [n for n in model.state_dict() if "imu" in n]
        assert not bad, bad
        x0 = forward(model, T_S2M_NOIMU_DIM)
        assert x0.shape == (2, TW, model.pose_head.pose_dim)
    # 50 维输入必须被 no-imu 模型拒绝（直连路径在 TactileEncoder 里报错）。
    model = AnySoleModel(tactile_input="s2m50", tactile_direct=True, no_imu=True)
    try:
        forward(model, T_S2M_DIM)
        raise AssertionError("no_imu model accepted 50-dim T_s2m")
    except ValueError:
        pass
    # 守卫：raw108 下 no_imu 无意义。
    for kwargs in (dict(tactile_input="raw108", no_imu=True),
                   dict(tactile_input="s2m50", tactile_direct=True, no_imu=False)):
        pass
    try:
        AnySoleModel(tactile_input="raw108", no_imu=True)
        raise AssertionError("no_imu with raw108 must raise")
    except ValueError:
        pass
    # 回归：默认 raw108 与 s2m50（带 IMU）仍可前向。
    forward(AnySoleModel(), None)
    forward(AnySoleModel(tactile_input="s2m50", tactile_direct=True), T_S2M_DIM)
    print("[5] 模型 OK: no_imu 前向通过、state_dict 无 imu 键、50 维被拒、守卫与默认回归通过")


if __name__ == "__main__":
    check_data_layer()
    check_noimu_never_synthesizes()
    check_dataset_item()
    check_encoder_model()
    check_full_model()
    print("probe_e6_noimu: ALL OK")
