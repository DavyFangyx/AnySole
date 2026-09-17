"""Step2Motion --no-imu 检验：确认 IMU 通道被真删（导出 38 维、模型无 IMU 编码）。

检查项：
1. 数据层：--no-imu 导出的 38 维 insole == 50 维导出删掉 IMU 列 [16:22]+[41:47]
   （在世界系转换、首帧丢弃之后逐元素一致）；
   no_imu 路径从不调用 synthesize_imu（monkeypatch 抛异常仍能导出）。
2. 数据集状态：has_imu=False、acc/gyro 索引字段为 None、insole 维数 38；
   MotionDataset.load 可加载、__getitem__ 返回 (1, 38)；data_augmentation
   对 no-IMU 数据不改动 c；Normalizer 拟合出 38 维统计量。
3. 模型层：ControlTransformer(imu_available=False) 无任何 imu 参数、6 条条件
   流、跨注意力 6 头；PriorTransformer+Control 前向可跑；model_from_config
   读 config_gait_noimu 时不构建 translation 模型。
4. 回归：默认参数下 50 维 8 流路径、config_gait 的 translation 构建不变。

用法：/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probe_s2m_noimu.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

GAIT_ROOT = Path(__file__).resolve().parents[1]
SRC = GAIT_ROOT / "Baselines" / "Step2Motion" / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import process_gait as pg  # noqa: E402

IMU_COLUMNS = tuple(range(16, 22)) + tuple(range(41, 47))
KEEP = [i for i in range(50) if i not in IMU_COLUMNS]


def get_session_dir() -> Path:
    dirs = pg.list_sequence_dirs(pg.SEQ_ROOT)
    for sid in ("S13013", "S7013"):
        if sid in dirs:
            return dirs[sid]
    return next(iter(dirs.values()))


def check_data_layer(seq_dir: Path) -> None:
    clips_full = pg.convert_session(seq_dir, pg.MIN_FRAMES, no_imu=False)
    clips_noimu = pg.convert_session(seq_dir, pg.MIN_FRAMES, no_imu=True)
    # 状态级（世界系转换 + 首帧丢弃之后）对比。
    state_full = pg.build_motion_dataset_state(clips_full, no_imu=False)
    state_noimu = pg.build_motion_dataset_state(clips_noimu, no_imu=True)
    full = torch.cat(state_full.insole, dim=0).numpy()
    noimu = torch.cat(state_noimu.insole, dim=0).numpy()
    assert full.shape[1] == 50, full.shape
    assert noimu.shape[1] == 38, noimu.shape
    np.testing.assert_array_equal(noimu, full[:, KEEP])
    # 状态标记正确。
    assert state_noimu.has_imu is False
    assert state_noimu.l_acceleration_idx is None
    assert state_noimu.r_acceleration_idx is None
    assert state_noimu.l_angular_velocity_idx is None
    assert state_noimu.r_angular_velocity_idx is None
    assert state_noimu.quat_lIMU is None and state_noimu.quat_rIMU is None
    assert state_noimu.left_acceleration_local is None
    assert state_noimu.l_total_force_idx == (16, 17)
    assert state_noimu.l_center_of_pressure_idx == (17, 19)
    assert state_noimu.r_pressure_idx == (19, 35)
    assert state_full.has_imu is True
    print("[1] 数据层 OK: 38 维 == 50 维删 IMU 列 (%d 帧); 状态标记正确" % full.shape[0])
    return clips_full, clips_noimu


def check_noimu_never_synthesizes(seq_dir: Path) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("synthesize_imu must not be called on the no_imu path")

    orig = pg.synthesize_imu
    pg.synthesize_imu = boom
    try:
        clips = pg.convert_session(seq_dir, pg.MIN_FRAMES, no_imu=True)
    finally:
        pg.synthesize_imu = orig
    assert clips and clips[0]["insole"].shape[1] == 38
    print("[2] no_imu 路径从不调用 synthesize_imu OK")


def check_dataset_and_augmentation(clips_noimu) -> None:
    from dataset import MotionDataset
    from utils import data_augmentation

    state = pg.build_motion_dataset_state(clips_noimu, no_imu=True)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "gait_noimu_train.pt"
        torch.save({"_kind": "MotionDatasetState", "state": state.__dict__}, out_path)
        db = MotionDataset.load(str(out_path), torch.device("cpu"))
    c = db[0][1]
    assert c.shape == (1, 38), c.shape
    # data_augmentation 对 no-IMU 数据不改动条件通道。
    x = db[0][0].clone()
    x2, c2 = data_augmentation(x, c.clone(), db)
    assert torch.equal(c, c2), "no-IMU insole must be unchanged by yaw augmentation"
    assert not torch.equal(x, x2), "pose should still rotate"
    # Normalizer 拟合 38 维统计量。
    from normalizer import Normalizer

    norm = Normalizer(db)
    assert norm.insoles_mean.shape == (38,)
    print("[3] 数据集 OK: item (1,38); 增强不改 c; normalizer 38 维")


def check_model() -> None:
    from model import ControlTransformer, PriorTransformer

    kw = dict(input_T=100, output_dim=69, hidden_dim=256, n_hidden=2, n_heads=4,
              d_ff=512, dropout=0.1, diffusion_T=200)
    ct = ControlTransformer(input_T=100, c_dim=38, output_dim=69, hidden_dim=256,
                            n_hidden=2, n_heads=4, d_ff=512, dropout=0.1,
                            imu_available=False)
    bad = [n for n, _ in ct.named_parameters() if "imu" in n]
    assert not bad, bad
    assert ct.condition_emb_limu is None and ct.condition_emb_rimu is None
    for layer in ct.decoder_layers:
        assert layer.cross_attn.num_heads == 6
    # 6 条条件流，前向可跑。
    c = torch.randn(1, 100, 38)
    t_emb = torch.zeros(1, 1, 256)
    streams = ct.forward_condition_emb(c, None, t_emb, torch.ones(1))
    assert len(streams) == 6, len(streams)
    # 完整 PriorTransformer + Control 前向。
    prior = PriorTransformer(**kw)
    x = torch.randn(1, 100, 66)
    distances = torch.rand(22)
    pred, _ = prior(x, distances, torch.zeros(1, dtype=torch.long),
                    control=ct, c=c, c_mask=torch.ones(1))
    assert pred.shape == (1, 100, 66), pred.shape

    # 回归：默认（带 IMU）仍是 8 流、8 头、含 limu/rimu。
    ct50 = ControlTransformer(input_T=100, c_dim=50, output_dim=69, hidden_dim=256,
                              n_hidden=2, n_heads=4, d_ff=512, dropout=0.1)
    streams50 = ct50.forward_condition_emb(torch.randn(1, 100, 50), None,
                                           torch.zeros(1, 1, 256), torch.ones(1))
    assert len(streams50) == 8, len(streams50)
    assert ct50.condition_emb_limu is not None
    for layer in ct50.decoder_layers:
        assert layer.cross_attn.num_heads == 8
    print("[4] 模型 OK: no-imu 无 imu 参数/6 流/6 头, 前向通过; 默认 8 流/8 头回归通过")


def check_model_from_config() -> None:
    from backward_diffusion import model_from_config
    from config import load_config

    cfg = load_config(str(GAIT_ROOT / "Baselines/Step2Motion/configs/config_gait_noimu.json"))
    assert cfg["input_dim"] == 38 and cfg["imu_available"] is False
    assert cfg["train_translation"] is False
    model, trans = model_from_config(cfg, is_prior=False)
    assert model.imu_available is False
    assert trans is None, "--no-imu config must not build the IMU-conditioned translation model"
    # 回归：gait 配置仍构建 translation 模型。
    cfg_full = load_config(str(GAIT_ROOT / "Baselines/Step2Motion/configs/config_gait.json"))
    model_full, trans_full = model_from_config(cfg_full, is_prior=False)
    assert model_full.imu_available is True
    assert trans_full is not None
    print("[5] config OK: gait_noimu -> 38 维/无 IMU 编码/无 translation; gait -> 原行为不变")


if __name__ == "__main__":
    seq_dir = get_session_dir()
    print("session: %s" % seq_dir.name)
    clips_full, clips_noimu = check_data_layer(seq_dir)
    check_noimu_never_synthesizes(seq_dir)
    check_dataset_and_augmentation(clips_noimu)
    check_model()
    check_model_from_config()
    print("probe_s2m_noimu: ALL OK")
