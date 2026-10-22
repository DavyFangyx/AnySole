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
from typing import Optional, Tuple


GAIT_ROOT = Path("/data/fangyuxuan/projects/gait")
ANYSOLE_ROOT = Path(__file__).resolve().parents[1]
# Default training config lives with the package, not at the repo root.
# (ANYSOLE_ROOT above is the repo root for historical reasons; anchor this
# one to the package directory itself.)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "v1.yaml"
# The grouped visual protocol is opt-in; historical single-view defaults above
# and below remain available to their existing consumers.
VIEW_CONFIG_PATH = DEFAULT_CONFIG_PATH.parent / "view_consistency.yaml"
VIEW_SCHEMA_VERSION = "view_consistency.v1"
VIEW_MANIFEST_SCHEMA = "anysole.grouped.v1"
VIEW_CACHE_SCHEMA = "anysole.visual-cache.v1"
VIEW_STATS_SCHEMA = "anysole.stats.v1"
VISUAL_STRUCTURE_VERSION = "visual_v1"
VIEW_EXPERIMENTS = (
    "base_legacy", "base_time", "base_pool", "view_consist",
    "pose_pair", "press_pair", "pair_shuffle",
)
MOTIONPRO_ROOT = GAIT_ROOT / "Baselines" / "MotionPRO"
WORKSPACE_ROOT = GAIT_ROOT / "AnysoleWorkspace"
# AnySole's default input contract is the public shared-facts tree.
SEQ_ROOT = WORKSPACE_ROOT / "shared" / "facts" / "sessions" / "cam3"
SPLIT_CSV = WORKSPACE_ROOT / "protocol" / "splits" / "default" / "splits.csv"
FAKE_MARKED_ROOT = (
    WORKSPACE_ROOT
    / "work"
    / "data_pipeline"
    / "pressure_washer"
    / "final_fake_marked"
)
HRNET_CACHE_ROOT = WORKSPACE_ROOT / "model_inputs" / "AnySole" / "adapter_v1" / "visual" / "hrnet" / "cam3"
# F1: GVHMR per-session caches (<model>/cam3/<session>.pt), written by
# anysole/data/extract_hmr.py.
HMR_CACHE_ROOT = WORKSPACE_ROOT / "model_inputs" / "AnySole" / "adapter_v1" / "visual" / "hmr"
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
    / "assets"
    / "third_party"
    / "MotionPRO"
    / "cliff_ckpt"
    / "hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"
)
SMPL_MODEL_PATH = WORKSPACE_ROOT / "assets" / "third_party" / "smpl" / "SMPL_NEUTRAL.pkl"


# Variant dir naming (2026-09-24 用户裁定：统一目录层级 + 完整超参记录).
# Every run — including an all-default one — lands in
# <model>_<contact>/<variant fields>/; the variant name IS the complete
# hyperparameter record of the run.  Fields are stacked in fixed order and
# EVERY field present in the effective config is written (no default
# omission) — a run that varies only λ_assign still gets its own dir.
VARIANT_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("tw", "tw%d"),
    ("stride", "st%d"),
    ("lr", "lr%.4g"),
    ("lambda_pose", "lp%.4g"),
    ("lambda_traj", "lt%.4g"),
    ("lambda_kp", "lk%.4g"),
    ("lambda_assign", "la%.4g"),
    ("lambda_assign_ent", "lae%.4g"),
    ("lambda_assign_conc", "lac%.4g"),
    ("lambda_assign_dead", "lad%.4g"),
    ("assign_dead_beta", "adb%.4g"),
    ("assign_temp_init", "ati%.4g"),
    ("assign_temp_final", "atf%.4g"),
    ("assign_anneal_frac", "aaf%.4g"),
    ("assign_lock_frac", "alf%.4g"),
    ("assign_lr_mult", "alm%.4g"),
    ("lambda_sigma", "ls%.4g"),
    ("sigma_freeze_frac", "sf%.4g"),
    ("lr_warmup_frac", "wu%.4g"),
    ("grad_clip", "gc%.4g"),
    ("loss_cap", "lcp%.4g"),
    ("epochs", "ep%d"),
    ("batch_size", "bs%d"),
    ("seed", "sd%d"),
    ("part_joints", "pk%d"),
)


def variant_from_config(config: dict) -> str:
    """Full stacked variant name from an effective config (fixed order,
    write-if-present; stride=None follows tw; part_joints writes its K)."""
    parts: list[str] = []
    for key, fmt in VARIANT_FIELDS:
        if key not in config:
            continue
        value = config[key]
        if key == "stride":
            value = config["tw"] if value is None else int(value)
        elif isinstance(value, (list, tuple)):
            value = len(value)
        parts.append(fmt % (int(value) if "d" in fmt else float(value)))
    return "_".join(parts) if parts else "default"


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


def camera_config_path(cam: int) -> Path:
    """Per-camera training config (四相机扩展 D2, 2026-10-12).

    ``anysole/configs/v1_cam{N}.yaml`` = a copy of v1.yaml with
    cam_id/seq_root/cache_root overridden for camera N; cam3 URIs equal the
    frozen defaults.
    """
    return DEFAULT_CONFIG_PATH.parent / f"v1_cam{cam}.yaml"


def camera_roots(cam: int) -> dict:
    """Canonical per-camera input roots (四相机扩展 D2); cam3 == frozen defaults."""
    return {
        "cam_id": cam,
        "seq_root": f"shared://facts/sessions/cam{cam}",
        "cache_root": f"model-input://AnySole/adapter_v1/visual/hrnet/cam{cam}",
    }


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


def assert_visual_batch_shapes(batch, *, tw=TW):
    """Validate the grouped visual/labels/masks/meta boundary (True=valid).

    This is a data-wrapper check, never a student input API. In particular,
    labels and metadata must not be forwarded to ``forward_visual``.
    """
    import torch

    for section in ("visual", "labels", "masks", "meta"):
        if not isinstance(batch.get(section), dict):
            raise TypeError(f"grouped batch.{section} must be a dict")
    visual, labels, masks, meta = (batch[k] for k in ("visual", "labels", "masks", "meta"))
    features = visual.get("V_feat")
    if not torch.is_tensor(features) or features.ndim != 4:
        raise ValueError("visual.V_feat must be [B,C,T,2051]")
    b, c, t, width = features.shape
    if not (1 <= c <= 4) or t != tw or width != V_FEAT_DIM:
        raise ValueError(f"visual.V_feat invalid shape {tuple(features.shape)}")

    def check(section, name, shape, dtype=None):
        value = section.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
            raise ValueError(f"grouped field {name} must have shape {tuple(shape)}")
        if dtype is not None and value.dtype != dtype:
            raise TypeError(f"grouped field {name} must have dtype {dtype}")
        return value

    check(visual, "V_feat", (b, c, t, V_FEAT_DIM), torch.float32)
    view_valid = check(visual, "view_valid", (b, c, t), torch.bool)
    times = check(visual, "target_time_s", (b, t), torch.float64)
    if not torch.isfinite(times).all() or (t > 1 and not (times[:, 1:] > times[:, :-1]).all()):
        raise ValueError("target_time_s must be finite and strictly increasing")
    check(visual, "view_edges", (b, c, t - 1), torch.bool)
    check(labels, "pose6d", (b, t, POSE_DIM), torch.float32)
    check(labels, "pressure", (b, t, T_RAW_DIM), torch.float32)
    for name in ("velocity", "root"):
        check(labels, name, (b, t, 3), torch.float32)
    for name, shape in (("motion_valid", (b, t)),
                        ("pressure_valid", (b, t, T_RAW_DIM)),
                        ("velocity_valid", (b, t)),
                        ("motion_edges", (b, t - 1)),
                        ("pressure_edges", (b, t - 1, T_RAW_DIM))):
        check(masks, name, shape, torch.bool)
    target_ids = check(meta, "target_frame_id", (b, t))
    if target_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("meta.target_frame_id must contain integer frame IDs")
    for name in ("source_frame_id", "source_feature_row"):
        ids = check(meta, name, (b, c, t))
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"meta.{name} must contain integer IDs")
    check(meta, "source_time_s", (b, c, t), torch.float64)
    for name in ("reuse", "padding"):
        check(meta, name, (b, c, t), torch.bool)
    check(meta, "time_residual_ms", (b, c, t))
    # Missing input may be NaN; only valid samples promise finite features.
    if not torch.isfinite(features[view_valid]).all():
        raise ValueError("valid visual features contain non-finite values")
    for name, valid in (("pose6d", masks["motion_valid"]),
                        ("pressure", masks["pressure_valid"])):
        if not torch.isfinite(labels[name][valid]).all():
            raise ValueError(f"valid {name} labels contain non-finite values")
    return b, c, t


def validate_view_config(config: dict, *, formal: bool = False) -> None:
    """Reject incompatible grouped settings and unmeasured formal budgets.

    P1 permits unresolved measured values in the template. P2's formal entry
    must call this with ``formal=True`` after the plan has been frozen.
    """
    if config.get("schema_version") != VIEW_SCHEMA_VERSION:
        raise ValueError("unsupported view-consistency config schema")
    if config.get("model_name") != "visual":
        raise ValueError("view-consistency requires model_name=visual")
    if config.get("visual_structure_version") != VISUAL_STRUCTURE_VERSION:
        raise ValueError("unsupported visual structure version")
    from anysole.utils.a2_config import A2_HEADS, is_a2, validate_a2_config
    a2 = is_a2(config)
    validate_a2_config(config, formal=formal)
    if config.get("pressure_head") not in (A2_HEADS if a2 else ("legacy", "time_pe")):
        raise ValueError("unsupported pressure head for the declared schema")
    if config.get("pressure_time_pe") != "before_fusion" or config.get("time_pe_scale") != FPS:
        raise ValueError("pressure PE must use before_fusion and nominal frame phase")
    if config.get("d_model") != D_MODEL or config.get("window_length") != TW or config.get("tw") != TW:
        raise ValueError("view-consistency requires d_model=256 and T=20")
    if config.get("strict_visual") is not True or config.get("config_probs") != [0.0, 1.0, 0.0]:
        raise ValueError("view-consistency must use strict V-only input")
    if config.get("generate_contact_cache") is not False:
        raise ValueError("grouped loading must not generate contact cache")
    if config.get("validation_cameras") != [1, 2, 3, 4]:
        raise ValueError("all grouped experiments require the same four-camera old-val domain")
    if (config.get("crop_height"), config.get("crop_width")) != (256, 192):
        raise ValueError("frozen CLIFF frontend requires a 256x192 RGB crop")
    for key, expected in (("manifest_schema", VIEW_MANIFEST_SCHEMA),
                          ("cache_schema", VIEW_CACHE_SCHEMA), ("stats_schema", VIEW_STATS_SCHEMA)):
        if config.get(key) != expected:
            raise ValueError(f"unsupported {key}")
    if any(config.get(key) is not False for key in ("horizontal_flip", "time_stretch", "mix_occlusion_cache")):
        raise ValueError("unaudited grouped augmentation/cache mixing must remain disabled")
    if (config.get("selection_rule") != "pareto_weighted" or
            config.get("selection_motion_weight") != .5 or config.get("selection_pressure_weight") != .5 or
            config.get("selection_tie_break") != ["motion", "pressure", "earlier_successful_update"]):
        raise ValueError("unsupported shared joint checkpoint selection rule")
    if not formal:
        return
    if config.get("stage") != "formal" or config.get("plan_frozen") is not True:
        raise ValueError("formal training requires a frozen formal plan")
    required = (
        "experiment", "manifest_hash", "split_hash", "stats_hash", "frontend_hash",
        "source_hash", "structure_hash", "plan_hash", "selection_coverage_hash",
        "target_updates", "target_rgb_windows", "effective_batch_rgb_windows",
        "reference_steps_per_epoch", "selection_m_ref", "selection_p_ref",
        "runs_per_gpu", "lambda_mv", "microbatch_rgb_windows",
        "validation_every_updates", "checkpoint_every_updates", "amp",
        "num_workers", "disk_min_free_bytes", "hrnet_checkpoint_hash",
        "cache_index_hash", "stats_file_hash", "detector_checkpoint_hash",
        "smpl_runtime_model_hash", "microbatch_packing_rule",
        "n_cam3_train_windows", "reference_batch_rgb_windows", "reference_epochs",
        "frontend_runtime_hash",
        "resource_version",
    )
    missing = [key for key in required if config.get(key) is None or config.get(key) == ""]
    if missing:
        raise ValueError("formal config has unresolved fields: " + ", ".join(missing))
    for key in ("manifest_hash", "split_hash", "stats_hash", "frontend_hash", "source_hash",
                "structure_hash", "plan_hash", "selection_coverage_hash", "hrnet_checkpoint_hash",
                "cache_index_hash", "stats_file_hash", "detector_checkpoint_hash", "smpl_runtime_model_hash",
                "frontend_runtime_hash"):
        value = config[key]
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"formal config {key} must be a SHA-256 hex digest")
    if config["experiment"] not in VIEW_EXPERIMENTS:
        raise ValueError("unknown formal experiment")
    positive = (
        "target_updates", "target_rgb_windows", "effective_batch_rgb_windows",
        "reference_steps_per_epoch", "selection_m_ref", "selection_p_ref",
        "runs_per_gpu", "microbatch_rgb_windows", "validation_every_updates",
        "checkpoint_every_updates", "disk_min_free_bytes",
        "n_cam3_train_windows", "reference_batch_rgb_windows", "reference_epochs",
    )
    for key in positive:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 < value < float("inf")):
            raise ValueError(f"formal config {key} must be finite and positive")
    for key in ("target_updates", "target_rgb_windows", "effective_batch_rgb_windows",
                "reference_steps_per_epoch", "runs_per_gpu", "microbatch_rgb_windows",
                "validation_every_updates", "checkpoint_every_updates", "num_workers",
                "disk_min_free_bytes"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise ValueError(f"formal config {key} must be an integer")
    for key in ("n_cam3_train_windows", "reference_batch_rgb_windows", "reference_epochs"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise ValueError(f"formal config {key} must be an integer")
    reference_batch = config["reference_batch_rgb_windows"]
    reference_n = config["n_cam3_train_windows"]
    steps = reference_n // reference_batch if config.get("drop_last") else (reference_n+reference_batch-1)//reference_batch
    if (config["reference_epochs"] != 740 or steps != config["reference_steps_per_epoch"] or
            config["target_updates"] != 740*steps*(4 if a2 and config["budget_mode"] == "cam3_exact" else 1) or
            config["effective_batch_rgb_windows"] != reference_batch or
            config["target_rgb_windows"] != config["target_updates"]*reference_batch):
        raise ValueError("formal reference 740-epoch update/RGB budgets must match the shared frozen batch rule")
    for key in ("lambda_mv", "num_workers"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 <= value < float("inf")):
            raise ValueError(f"formal config {key} must be finite and nonnegative")
    if not isinstance(config["amp"], bool):
        raise ValueError("formal amp must be a boolean")
    floor = config.get("selection_reference_floor", 1.e-8)
    if not isinstance(floor, (int, float)) or not 0 < floor < float("inf") or min(
            config["selection_m_ref"], config["selection_p_ref"]) < floor:
        raise ValueError("formal selection references must respect the frozen positive floor")
    if config.get("resource_frozen") is not True:
        raise ValueError("formal resource envelope must be measured and frozen")
    if config.get("packing_rule") != "subset_sum_pending.v1" or config.get("microbatch_packing_rule") != "contiguous_complete_pairs.v1":
        raise ValueError("unsupported formal complete-group packing rule")
    if isinstance(config.get("seed"), bool) or config.get("seed") not in (1, 2, 3):
        raise ValueError("formal seed must be 1, 2, or 3")
    if config["experiment"] in ("pose_pair", "press_pair", "pair_shuffle") and config["seed"] != 1:
        raise ValueError("formal mechanism checks use only seed 1")
    for key in ("max_steps", "limit_sessions", "max_batches", "smoke_steps", "stop_after_updates", "pilot_train_sample_keys"):
        if config.get(key) is not None:
            raise ValueError(f"formal config cannot use truncation {key}")
