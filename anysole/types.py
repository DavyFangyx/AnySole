"""Frozen V1 tensor shapes, joint layout, and path defaults."""

import os
from pathlib import Path


GAIT_ROOT = Path("/data/fangyuxuan/projects/gait")
ANYSOLE_ROOT = Path(__file__).resolve().parents[1]
MOTIONPRO_ROOT = GAIT_ROOT / "Baselines" / "MotionPRO"
WORKSPACE_ROOT = GAIT_ROOT / "AnysoleWorkspace"
SEQ_ROOT = WORKSPACE_ROOT / "derived" / "MotionPRO" / "sequences" / "cam3"
SPLIT_CSV = WORKSPACE_ROOT / "splits" / "default" / "splits.csv"
FAKE_MARKED_ROOT = (
    WORKSPACE_ROOT
    / "sources"
    / "PressureWasher"
    / "outputs"
    / "fake_marked"
    / "reconstruction_20260817_161459_fake_marked"
)
HRNET_CACHE_ROOT = WORKSPACE_ROOT / "derived" / "AnySole" / "hrnet_cache" / "cam3"
# F1: GVHMR per-session caches (<model>/cam3/<session>.pt), written by
# anysole/data/extract_hmr.py.
HMR_CACHE_ROOT = WORKSPACE_ROOT / "derived" / "AnySole" / "hmr_cache"
CLIFF_CKPT = (
    WORKSPACE_ROOT
    / "dependencies"
    / "MotionPRO"
    / "cliff_ckpt"
    / "hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"
)


def anysole_model_dir(modal: str, contact_method: str) -> Path:
    """Centralized results dir for one model variant: AnySole/<modal>_<contact_method>.

    The dir name is self-describing (``anysolev1`` / ``anysolev1_insole_drift``
    modal plus the training contact-label scheme), so eval/infer only need the
    two identifiers to locate checkpoints, predictions, and metrics.
    """
    results_root = Path(os.environ.get("ANYSOLE_RESULTS", str(GAIT_ROOT / "results")))
    return results_root / "AnySole" / ("%s_%s" % (modal, contact_method))
HRNET_YAML = (
    MOTIONPRO_ROOT
    / "lib"
    / "model"
    / "backbone"
    / "hrnet"
    / "models"
    / "cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
)

TW = 20
FPS = 40.0
D_MODEL = 256
N_JOINTS = 23
POSE_DIM = N_JOINTS * 6  # 138
# E6.1: root-local 3D positions of the 22 non-root joints (meters, session
# frame-0 orientation). The position representation removes the FK error
# amplification of the 6D space (verified per-joint profile: hip 60mm ->
# hand 398mm in 6D vs no amplification in positions).
POSE_POS_DIM = 22 * 3  # 66
# F2a: heading-frame trajectory target: [psi_dot (rad/s), v_hx, v_hz
# (heading-frame horizontal velocity, m/s), h (world root height, m)].
TRAJ_F2_DIM = 4
T_RAW_DIM = 96
T_PHYS_DIM = 12
# E6.6a: Step2Motion-口径触觉通道（每脚 25 维 = 16 压力池化[heel8 toes8] +
# 合成 IMU acc3/gyro3 + 总力1 + CoP2）。IMU 由 GT BVH 脚部运动学合成
# （anysole/data/tactile_s2m.py），与 Step2Motion 的 gait 导出同口径。
T_S2M_DIM = 50
# --no-imu：真删 IMU 通道后的布局（每脚 19 维 = 16 压力池化 + 总力1 + CoP2）。
# 数据侧不合成、模型侧不编码 IMU；编码器按 38 维分组（无 IMU 组）。
T_S2M_NOIMU_DIM = 38
V_FEAT_DIM = 2051  # HRNet 2048 + normalized CLIFF bbox_info [cx, cy, b]
# F1: GVHMR visual channel (v_input="hmr_gvhmr").  Per frame:
#   body_pose aa 63 (SMPL-X 21 body joints) + global_orient 3 + betas 10
#   + kp2d COCO-17 51 (full-img px + conf) + HMR2 f_imgseq 1024
#   + bbox_info 3 (normalized cx, cy, size) + q_V 2 (kp mean conf, trunc ratio)
V_HMR_DIM = 1156
V_HMR_ROT_DIM = 76   # body_pose 63 + global_orient 3 + betas 10
V_HMR_KP_DIM = 51    # COCO-17 (x, y, conf)
V_HMR_IMG_DIM = 1024 # HMR2 ViT backbone features
V_HMR_MISC_DIM = 5   # bbox_info 3 + q_V 2
FUSE_LEN = 40  # V(20) + combined T(20)
N_CONTACT = 2

JOINT_NAMES = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
)

# BVH parent indices matching Skeleton3 hierarchy.
JOINT_PARENTS = (
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    4,
    7,
    8,
    9,
    4,
    11,
    12,
    13,
    0,
    15,
    16,
    17,
    0,
    19,
    20,
    21,
)

LEFT_LEG_JOINTS = (15, 16, 17, 18)
RIGHT_LEG_JOINTS = (19, 20, 21, 22)
BODY_JOINTS = tuple(i for i in range(N_JOINTS) if i not in LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS)

LEFT_LEG_SLICE = slice(LEFT_LEG_JOINTS[0] * 6, (LEFT_LEG_JOINTS[-1] + 1) * 6)  # 90:114
RIGHT_LEG_SLICE = slice(RIGHT_LEG_JOINTS[0] * 6, (RIGHT_LEG_JOINTS[-1] + 1) * 6)  # 114:138
BODY_SLICE = slice(0, LEFT_LEG_JOINTS[0] * 6)  # 0:90

LEFT_FOOT_JOINT = JOINT_NAMES.index("LeftFoot")
LEFT_TOE_JOINT = JOINT_NAMES.index("LeftToeBase")
RIGHT_FOOT_JOINT = JOINT_NAMES.index("RightFoot")
RIGHT_TOE_JOINT = JOINT_NAMES.index("RightToeBase")
FOOT_JOINTS = (LEFT_FOOT_JOINT, LEFT_TOE_JOINT, RIGHT_FOOT_JOINT, RIGHT_TOE_JOINT)

CONFIG_VT = 0
CONFIG_V = 1
CONFIG_T = 2
CONFIG_NAMES = ("VT", "V", "T")
# User-facing names for inference/evaluation exports. CONFIG_NAMES is kept for
# existing internal logs and checkpoints that use the short names.
CONFIG_MODE_NAMES = ("VT2M", "V2M", "T2M")
CONFIG_PROBS = (0.50, 0.25, 0.25)

DIFFUSION_TRAIN_STEPS = 1000
DIFFUSION_SAMPLE_STEPS = 50

PRESSURE_CLIP = 1023.0
CONTACT_SUM_THRESH = 100.0

IMG_NORM_MEAN = (0.485, 0.456, 0.406)
IMG_NORM_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 256

# Batch field names and shapes. Values are (B, ...) with B omitted.
BATCH_SHAPES = {
    "V_feat": (TW, V_FEAT_DIM),
    "V_hmr": (TW, V_HMR_DIM),  # F1 optional: only v_input="hmr_*" batches carry it
    "T_raw": (TW, T_RAW_DIM),
    "T_phys": (TW, T_PHYS_DIM),
    "pose_gt": (TW, POSE_DIM),
    "pose_gt_pos": (TW, POSE_POS_DIM),
    "trans_gt": (TW, 3),
    "vel_gt": (TW, 3),
    "traj_gt_f2": (TW, TRAJ_F2_DIM),  # F2a optional: only f2_repr batches carry it
    "trans_anchor": (3,),
    "psi_anchor": (),  # F2a optional scalar
    "root_rot_init": (3, 3),
    "kp_gt": (TW, N_JOINTS, 3),
    "contact_gt": (TW, N_CONTACT),
    "offsets": (N_JOINTS, 3),
    "parents": (N_JOINTS,),
    "config_id": (),
    "session_id": (),
}


def assert_batch_shapes(batch, batch_size=None, tw=TW):
    """Raise if a training batch is missing a frozen field or has the wrong shape."""
    import torch

    for name, expected in BATCH_SHAPES.items():
        if name in ("session_id",):
            continue
        if name in ("V_hmr", "traj_gt_f2", "psi_anchor") and name not in batch:
            # F1/F2a: these channels only exist for their respective modes;
            # every other batch legitimately omits them.
            continue
        if name not in batch:
            raise KeyError("batch missing field %s" % name)
        value = batch[name]
        if not torch.is_tensor(value):
            raise TypeError("%s must be a tensor, got %s" % (name, type(value)))
        if batch_size is None:
            batch_size = value.shape[0]
        # Temporal fields use the runtime window length (E6.3: tw may differ
        # from the frozen TW=20).
        expected = tuple(int(tw) if dim == TW else dim for dim in expected)
        got = tuple(value.shape[1:])
        if got != expected:
            raise ValueError("%s shape %s != %s" % (name, (value.shape[0],) + got, (batch_size,) + expected))
        if value.shape[0] != batch_size:
            raise ValueError("%s batch %d != %d" % (name, value.shape[0], batch_size))
    return batch_size
