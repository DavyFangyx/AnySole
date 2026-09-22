"""AnySole tensor shapes and the canonical SMPL-24 motion protocol.

AnySole consumes and predicts the standard SMPL-24 kinematic tree.  BVH is a
Step2Motion/legacy interchange format and is deliberately *not* represented by
these constants.  Keeping the protocol here prevents an input adapter from
silently changing the model's joint language again.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


GAIT_ROOT = Path("/data/fangyuxuan/projects/gait")
ANYSOLE_ROOT = Path(__file__).resolve().parents[1]
# Default training config lives with the package, not at the repo root.
# (ANYSOLE_ROOT above is the repo root for historical reasons; anchor this
# one to the package directory itself.)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "v1.yaml"
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
# SMPL archives are kept outside the repository.  Override with
# ``ANYSOLE_SMPL_ROOTS`` (path-separator separated) when relocating them.
_smpl_env = os.environ.get("ANYSOLE_SMPL_ROOTS", "")
SMPL_ROOTS = tuple(Path(p).expanduser() for p in _smpl_env.split(os.pathsep) if p) or (
    Path("/data/lizhe/projects/Tactile/Mocap/0804"),
    Path("/data/lizhe/projects/Tactile/Mocap/0807"),
    Path("/data/lizhe/projects/Tactile/Mocap/0808"),
    Path("/data/lizhe/projects/Tactile/Mocap/0810"),
)
CLIFF_CKPT = (
    WORKSPACE_ROOT
    / "dependencies"
    / "MotionPRO"
    / "cliff_ckpt"
    / "hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"
)
SMPL_MODEL_PATH = WORKSPACE_ROOT / "dependencies" / "smpl" / "SMPL_NEUTRAL.pkl"


def anysole_model_dir(modal: str, contact_method: str, variant: Optional[str] = None) -> Path:
    """Centralized results dir for one model variant: AnySole/<modal>_<contact_method>.

    The dir name is self-describing (``anysolev1`` / ``anysolev1_insole_drift``
    modal plus the training contact-label scheme), so eval/infer only need the
    two identifiers to locate checkpoints, predictions, and metrics.

    ``variant`` (2026-09-22 stacked-field naming) is the hyperparameter
    combination under the model dir: ``tw40`` / ``tw40_st20`` /
    ``t0003_tw40_lr3e4_lp5`` — fields stack in fixed order (tw, st, lr, lp,
    lt, lk) and defaults are omitted, so the variant name itself is the
    hyperparameter record.  Model + contact + variant == the ckpt address
    (base/variant/checkpoints/ckpt_last.pt), which is the single locator
    used by eval / visualization / probes.
    """
    results_root = Path(os.environ.get("ANYSOLE_RESULTS", str(GAIT_ROOT / "results")))
    base = results_root / "AnySole" / ("%s_%s" % (modal, contact_method))
    return base / variant if variant else base
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
N_JOINTS = 24
POSE_DIM = N_JOINTS * 6
# E6.1: root-local 3D positions of the 23 non-root joints (meters, session
# frame-0 orientation). The position representation removes the FK error
# amplification of the 6D space (verified per-joint profile: hip 60mm ->
# hand 398mm in 6D vs no amplification in positions).
POSE_POS_DIM = (N_JOINTS - 1) * 3
# F2a: heading-frame trajectory target: [psi_dot (rad/s), v_hx, v_hz
# (heading-frame horizontal velocity, m/s), h (world root height, m)].
TRAJ_F2_DIM = 4
T_RAW_DIM = 96
T_PHYS_DIM = 12
# E6.6a: Step2Motion-口径触觉通道（每脚 25 维 = 16 压力池化[heel8 toes8] +
# 合成 IMU acc3/gyro3 + 总力1 + CoP2）。IMU 沿用 Step2Motion 的原始
# BVH-23/ToeBase 运动学链生成；它不是 AnySole 的 SMPL-24 motion target。
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
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)

# Standard SMPL-24 kinematic tree (the order used by poses/root_orient in the
# mocap NPZ files).  This is the only parent tree used by AnySole geometry.
JOINT_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)
MOTION_PROTOCOL = "smpl24"
JOINT_PROTOCOL_CHECKSUM = hashlib.sha256(json.dumps(
    {"names": list(JOINT_NAMES), "parents": list(JOINT_PARENTS)},
    ensure_ascii=False,
    separators=(",", ":"),
).encode()).hexdigest()

LEFT_LEG_JOINTS = (1, 4, 7, 10)
RIGHT_LEG_JOINTS = (2, 5, 8, 11)
BODY_JOINTS = tuple(i for i in range(N_JOINTS) if i not in LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS)

# Semantic metric groups and angle triples.  Keep these next to the canonical
# SMPL names/tree: evaluation code must not carry unexplained integer triples
# copied from the legacy Skeleton3/BVH-23 ordering.
LOWER_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS
UPPER_JOINTS = tuple(i for i in range(N_JOINTS) if i not in LOWER_JOINTS)
ANKLE_FOOT_JOINTS = (
    JOINT_NAMES.index("left_ankle"), JOINT_NAMES.index("right_ankle"),
    JOINT_NAMES.index("left_foot"), JOINT_NAMES.index("right_foot"),
)
HAND_JOINTS = (
    JOINT_NAMES.index("left_hand"), JOINT_NAMES.index("right_hand"),
)
ELBOW_ANGLE_TRIPLES = (
    tuple(JOINT_NAMES.index(name) for name in ("left_shoulder", "left_elbow", "left_wrist")),
    tuple(JOINT_NAMES.index(name) for name in ("right_shoulder", "right_elbow", "right_wrist")),
)
KNEE_ANGLE_TRIPLES = (
    tuple(JOINT_NAMES.index(name) for name in ("left_hip", "left_knee", "left_ankle")),
    tuple(JOINT_NAMES.index(name) for name in ("right_hip", "right_knee", "right_ankle")),
)

# V3-2: 9-part grouping of SMPL-24 joints.  The part order is the decoder
# query order, not the numerical SMPL order.
# Non-root joints are parent-local rotations, so part grouping is representation-
# safe.  Single source of truth — eval_protocol.py and the archived
# part_decoder.py import from here.
PART_NAMES = (
    "root", "torso", "headneck", "l_arm", "r_arm",
    "l_leg", "r_leg", "l_foot", "r_foot",
)
PART_JOINTS = (
    (0,), (3, 6, 9), (12, 15), (13, 16, 18, 20, 22),
    (14, 17, 19, 21, 23), (1, 4), (2, 5), (7, 10), (8, 11),
)
N_PARTS = len(PART_NAMES)

LEFT_LEG_SLICE = tuple(j for j in LEFT_LEG_JOINTS)
RIGHT_LEG_SLICE = tuple(j for j in RIGHT_LEG_JOINTS)
BODY_SLICE = tuple(j for j in BODY_JOINTS)

# SMPL has ankle and foot joints but no separate toe-base joint.  Use the
# terminal foot joints for both the foot-height and tactile-foot direction;
# never invent a 25th joint or silently map back to Skeleton3.
LEFT_FOOT_JOINT = JOINT_NAMES.index("left_foot")
LEFT_TOE_JOINT = LEFT_FOOT_JOINT
RIGHT_FOOT_JOINT = JOINT_NAMES.index("right_foot")
RIGHT_TOE_JOINT = RIGHT_FOOT_JOINT
FOOT_JOINTS = (LEFT_FOOT_JOINT, RIGHT_FOOT_JOINT)

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
    "floor_y": (),  # per-session native-SMPL ground reference (meters)
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
