"""Train AnySole V1."""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from anysole.data.dataset import (
    AnySoleDataset,
    collate_windows,
    load_split_ids,
    sample_config_ids,
)
from anysole.utils.diffusion import GaussianDiffusion
from anysole.utils.losses import compute_losses
from anysole.models import AnySoleModel, MODEL_NAMES, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_POS, MODEL_ANYSOLEV2
from anysole.types import (
    CONFIG_PROBS,
    CONFIG_T,
    CONFIG_V,
    DEFAULT_CONFIG_PATH,
    FOOT_JOINTS,
    GAIT_ROOT,
    HRNET_CACHE_ROOT,
    JOINT_PROTOCOL_CHECKSUM,
    MOTION_PROTOCOL,
    N_JOINTS,
    PART_NAMES,
    POSE_DIM,
    SEQ_ROOT,
    SMPL_ROOTS,
    SPLIT_CSV,
    T_S2M_DIM,
    WORKSPACE_ROOT,
    anysole_model_dir,
    assert_batch_shapes,
    variant_from_config,
)
from anysole.registry import get_builder, infer_anysole_paths
from anysole.data.workspace_adapter import ADAPTER_VERSION, sha256_file
from AnysoleWorkspace.tool.workspace import resolve_uri
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.utils.geometry import f2_to_world, fk_pose6d
from anysole.utils.losses import soft_contact_from_keypoints
from anysole.types import CONFIG_NAMES


DEFAULT_CONFIG = {
    "d_model": 256,
    "dropout": 0.1,
    "tw": 20,
    "batch_size": 256,
    # Native SMPL-24 reinitializes the pose/traj heads when warm-starting a
    # BVH-23 checkpoint.  A real-batch F4 fixed-batch probe showed 1e-3
    # increases full MPJPE even with 20-step warmup; 1e-4 is the verified
    # stable starting rate for this protocol.
    "lr": 1.0e-4,
    "epochs": 200,
    "num_workers": 4,
    "diffusion_train_steps": 1000,
    "diffusion_sample_steps": 50,
    "pose_layers": 6,
    # ---- E6.x 系列开关（默认值保持既有行为；E3 基础用 noise_scaled=false +
    # lr_schedule=constant，详见 model_fix_note/archived/E6x_series_plan.md）----
    "pose_repr": "6d",          # E6.1: "pos" = 根局部位置表示
    "stride": None,             # E6.3: 训练窗 stride（None = 非重叠）
    "lambda_pose_vel": 0.0,     # E6.2: 帧间平滑损失权重（0 = 关闭）
    "lambda_bone": 0.0,         # E6.2: 骨长刚性损失权重（0 = 关闭，仅 pos 模式）
    "continuation": False,      # E6.4: eval/infer 续写式推理
    "warm_start": False,        # E6.5: 采样起点对齐训练边缘分布
    "noise_scaled": False,      # E3 口径（E4 的噪声缩放被证伪）；默认与 anysole/configs/v1.yaml 对齐
    "lr_schedule": "constant",  # E3 口径（E5 的 cosine 被证伪）
    "lambda_pose": 3.0,
    "lambda_traj": 1.0,
    "lambda_trec": 0.1,
    "lambda_vrec": 0.1,
    "lambda_kp": 1.0,
    "lambda_con": 0.0,          # E3 口径（接触标签默认不参与训练损失）
    "traj_velocity_w": 1.0,
    "traj_delta_w": 1.0,
    "traj_delta_weight_power": 1.0,
    "traj_deltas": [2, 4, 8, 19],
    "config_probs": list(CONFIG_PROBS),
    "seed": 1,
    "cam_id": 3,
    # 路径一律引用 anysole/types.py 的集中常量，避免与配置文件的字面量双重维护。
    "seq_root": str(SEQ_ROOT),
    "split_csv": str(SPLIT_CSV),
    "cache_root": str(HRNET_CACHE_ROOT),
    "use_insole_drift": False,
    "contact_method": "f6_soft",
    "smpl_roots": [str(p) for p in SMPL_ROOTS],
    "modal": MODEL_ANYSOLEV1,
        "template_path": str(WORKSPACE_ROOT / "assets" / "insole_templates.json"),
    "wandb_mode": "disabled",
    "wandb_project": "Anysole",
    "wandb_entity": "davyfangyuxuan-nanjing-university-of-aeronautics-and-ast",
    "wandb_eval_interval": 5,
    "wandb_experiment_tag": "anysole_v1",
}


def load_config(path: Path) -> Dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read %s" % path) from exc
    loaded = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping: %s" % path)
    if "contact_method" not in loaded:
        raise ValueError(
            f"{path}: contact_method is required explicitly; choose an AnySole-private "
            "label method such as 'f6_soft'"
        )
    for key in ("seq_root", "split_csv", "cache_root"):
        value = loaded.get(key)
        if isinstance(value, str) and "://" in value:
            loaded[key] = str(resolve_uri(value))
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return device


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def condition_inputs(batch: dict, config_id: torch.Tensor, t_s2m_dim: int = T_S2M_DIM) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero absent modalities without modifying reconstruction targets in batch.

    Returns (v_feat, t_raw, t_phys, t_s2m); t_s2m is only consumed when the
    model was trained with tactile_input="s2m50" and is zeroed for V-only rows.
    ``t_s2m_dim`` selects the fallback width for hand-built batches that omit
    the E6.6 channel (50 by default, 38 for --no-imu checkpoints).
    """
    drop_v = (config_id == CONFIG_T).view(-1, 1, 1)
    drop_t = (config_id == CONFIG_V).view(-1, 1, 1)
    v_feat = torch.where(drop_v, torch.zeros_like(batch["V_feat"]), batch["V_feat"])
    # F1: zero the GVHMR channel for T-only rows (same rule as V_feat); the
    # batch dict is mutated so model call sites can pick it up directly.
    if batch.get("V_hmr") is not None:
        batch["V_hmr"] = torch.where(drop_v, torch.zeros_like(batch["V_hmr"]), batch["V_hmr"])
    t_raw = torch.where(drop_t, torch.zeros_like(batch["T_raw"]), batch["T_raw"])
    t_phys = torch.where(drop_t, torch.zeros_like(batch["T_phys"]), batch["T_phys"])
    t_s2m = batch.get("T_s2m")
    if t_s2m is None:
        # Manually-built batches (probes) may omit the E6.6 channel.
        t_s2m = t_raw.new_zeros((t_raw.shape[0], t_raw.shape[1], t_s2m_dim))
    else:
        t_s2m = torch.where(drop_t, torch.zeros_like(t_s2m), t_s2m)
    return v_feat, t_raw, t_phys, t_s2m


def fit_pose_stats(loader: DataLoader, device: torch.device, key: str = "pose_gt") -> Tuple[torch.Tensor, torch.Tensor]:
    """One pass over the training loader: per-dim mean/std of the pose target.

    The pose head normalizes its input/output with these fixed stats (MDM/RoHM
    normalize the motion representation before diffusing), so the values are
    fitted once on the training set and frozen for the whole run. ``key``
    selects "pose_gt" (6D) or "pose_gt_pos" (E6.1 positions).
    """
    sums = torch.zeros(POSE_DIM, dtype=torch.float64, device=device)
    sumsq = torch.zeros(POSE_DIM, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            pose = raw_batch[key].to(device=device, dtype=torch.float64, non_blocking=True)
            if sums.shape[0] != pose.shape[-1]:
                sums = torch.zeros(pose.shape[-1], dtype=torch.float64, device=device)
                sumsq = torch.zeros(pose.shape[-1], dtype=torch.float64, device=device)
            flat = pose.reshape(-1, pose.shape[-1])
            sums += flat.sum(dim=0)
            sumsq += flat.square().sum(dim=0)
            count += flat.shape[0]
    if count == 0:
        raise RuntimeError("cannot fit pose stats: training loader is empty")
    mean = sums / count
    var = (sumsq / count - mean.square()).clamp(min=0.0)
    return mean.float(), var.sqrt().float()


def fit_traj_stats(loader: DataLoader, device: torch.device) -> Dict[str, list]:
    """One pass over the training loader: per-dim mean/std of traj_gt_f2 (4).

    F2a: the trajectory loss compares psi_dot / v_h / h in one normalized
    space, so the per-dim scales are balanced by construction.  Saved into
    the checkpoint config (traj_f2_stats) and consumed by losses.py.
    """
    dim = 4  # TRAJ_F2_DIM
    sums = torch.zeros(dim, dtype=torch.float64, device=device)
    sumsq = torch.zeros(dim, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            traj = raw_batch["traj_gt_f2"].to(device=device, dtype=torch.float64, non_blocking=True)
            flat = traj.reshape(-1, traj.shape[-1])
            sums += flat.sum(dim=0)
            sumsq += flat.square().sum(dim=0)
            count += flat.shape[0]
    if count == 0:
        raise RuntimeError("cannot fit traj stats: training loader is empty")
    mean = sums / count
    var = (sumsq / count - mean.square()).clamp(min=0.0)
    std = var.sqrt().clamp(min=1e-2)
    return {"mean": [float(v) for v in mean], "std": [float(v) for v in std]}


def init_from_checkpoint(model, path: Path, drop_prefixes=()) -> None:
    """Warm-start: copy matching parameters from another checkpoint's model.

    A key is copied only when it exists in the target state dict with the
    same shape; everything else (missing keys, shape changes, explicit
    ``drop_prefixes``) keeps the target's fresh initialization.  This makes
    anysolev1 -> anysolev2 warm-start work with no key mapping: the V2 pose
    head's ``query`` replaces the V1 ``proj_*``/``timestep_token`` (skipped
    by the shape/name rule), while encoders/fusion/traj/aux transfer as-is.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("--init-from checkpoint has no 'model' state dict: %s" % path)
    src = checkpoint["model"]
    src_pose_dim = int(getattr(src.get("pose_head.pose_mean"), "shape", (0,))[0])
    src_config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
    src_joints = int(src_config.get("motion_n_joints", src_pose_dim // 6 if src_pose_dim else 0))
    src_protocol = str(src_config.get("motion_protocol", ""))
    src_checksum = str(src_config.get("joint_protocol_checksum", ""))
    protocol_mismatch = (
        src_pose_dim != POSE_DIM
        or src_joints != N_JOINTS
        or src_protocol != MOTION_PROTOCOL
        or src_checksum != JOINT_PROTOCOL_CHECKSUM
    )
    effective_drop = tuple(drop_prefixes)
    if protocol_mismatch:
        # A motion head from another/unknown protocol is not merely a tensor
        # with a possibly different width: joint order, parent tree, body/leg
        # grouping and trajectory target are semantic state.  In particular,
        # early 24/144 SMPL migration checkpoints used a wrong collar parent
        # tree and therefore must not pass this gate on shape alone.
        # Shape-only loading would still copy most decoder and trajectory
        # tensors and silently retain that language.  Transfer only the
        # protocol-independent encoders/fusion/auxiliary heads.
        effective_drop += ("pose_head.", "traj_head.")
        print("=" * 79)
        reason = []
        if src_pose_dim != POSE_DIM or src_joints != N_JOINTS:
            reason.append("shape=%d-joint/%dD" % (src_joints, src_pose_dim))
        if src_protocol != MOTION_PROTOCOL:
            reason.append("motion_protocol=%r" % src_protocol)
        if src_checksum != JOINT_PROTOCOL_CHECKSUM:
            reason.append("joint-tree checksum=%s" % (src_checksum or "missing"))
        print("WARNING: --init-from uses a legacy or unknown motion protocol (%s)." %
              ", ".join(reason))
        print("         Protocol-specific pose query/decoder/output/stats and traj_head.* are reinitialized")
        print("         for SMPL-24/144D; shared embeddings plus encoder/fusion/aux may transfer.")
        print("=" * 79)
    dst = model.state_dict()
    loaded, skipped = [], []
    for key, value in src.items():
        if any(key.startswith(p) for p in effective_drop) or key not in dst or dst[key].shape != value.shape:
            skipped.append(key)
            continue
        dst[key] = value.to(dst[key].dtype)
        loaded.append(key)
    model.load_state_dict(dst)
    pose_skipped = [key for key in skipped if key.startswith("pose_head.")]
    if pose_skipped and not protocol_mismatch:
        print("WARNING: %d pose-head tensors were reinitialized because their name/shape did not match."
              % len(pose_skipped))
    print("init-from %s: %d/%d keys copied%s" % (
        path, len(loaded), len(dst),
        " (skipped: %s)" % ", ".join(skipped) if skipped else ""))


def warn_legacy_output_checkpoint(out_dir: Path) -> None:
    """Warn when a run directory contains a non-current motion checkpoint."""
    candidates = [Path(out_dir) / "ckpt_last.pt", Path(out_dir) / "ckpt_best.pt"]
    legacy = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            old = torch.load(candidate, map_location="cpu", weights_only=False)
            state = old.get("model", {}) if isinstance(old, dict) else {}
            dim = int(getattr(state.get("pose_head.pose_mean"), "shape", (0,))[0])
            cfg = old.get("config", {}) if isinstance(old, dict) else {}
            joints = int(cfg.get("motion_n_joints", dim // 6 if dim else 0))
            protocol = str(cfg.get("motion_protocol", ""))
            checksum = str(cfg.get("joint_protocol_checksum", ""))
            if (dim != POSE_DIM or joints != N_JOINTS
                    or protocol != MOTION_PROTOCOL
                    or checksum != JOINT_PROTOCOL_CHECKSUM):
                legacy.append(
                    "%s (%d joints/%dD, protocol=%r, tree=%s)"
                    % (candidate.name, joints, dim, protocol, checksum[:12] or "missing")
                )
        except Exception as exc:
            print("WARNING: could not inspect existing output checkpoint %s: %s" % (candidate, exc))
    if legacy:
        print("=" * 79)
        print("WARNING: output directory contains legacy checkpoints: %s" % ", ".join(legacy))
        print("         They are not resumable/evaluable under the current SMPL-24 joint tree")
        print("         and will be replaced by this run.")
        print("         Until the first new checkpoint is saved, do not treat this directory as a new result.")
        print("=" * 79)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AnySole V1.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preflight-only", action="store_true", help="Read-only grouped protocol and asset checks; never constructs Dataset.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Grouped run directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume this exact grouped run from its atomic last checkpoint.")
    parser.add_argument("--stop-after-updates", type=int, default=None, help="Pilot-only controlled checkpoint boundary for recovery verification.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training RNG seed. Saved into the checkpoint config for registered experiments.",
    )
    parser.add_argument(
        "--tw",
        type=int,
        default=None,
        help="Window length in frames; overrides the config tw (e.g. --tw 100).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Model dropout rate (0 disables dropout; used by Test9 overfitting and retrain experiments). Saved into the checkpoint config.",
    )
    parser.add_argument(
        "--config-probs",
        type=float,
        nargs=3,
        metavar=("VT", "V", "T"),
        default=None,
        help="Sampling probabilities for VT, V-only, and T-only configs.",
    )
    parser.add_argument(
        "--dropoutVT",
        dest="dropout_vt",
        type=str,
        nargs="+",
        metavar=("DROP_V_PCT", "DROP_T_PCT"),
        default=None,
        help="Modality dropout as percentages: V is dropped DROP_V_PCT%% of steps, T is dropped "
        "DROP_T_PCT%% (config_probs becomes [1-v-t, t, v] in VT/V/T order; overrides --config-probs). "
        "Accepts '--dropoutVT 20 30' or '--dropoutVT 20,30'; '0,0' disables modality dropout.",
    )
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument(
        "--tau-max",
        type=int,
        default=None,
        help="Upper bound for training-step sampling: tau ~ U(0, tau_max) instead of "
        "U(0, diffusion_train_steps). Test11 low-noise band training (e.g. --tau-max 100).",
    )
    parser.add_argument(
        "--tau-fixed",
        type=int,
        default=None,
        help="Fix tau to this value for every training step. Test11 tau0 identity training "
        "(--tau-fixed 0: x_tau = x0 clean input, task degenerates to the identity map).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="E6.3: training window stride (None = non-overlapping, stride == tw; "
        "1 = every-frame alignment, Step2Motion-style).",
    )
    parser.add_argument(
        "--part-json",
        type=Path,
        default=None,
        help="V4B only: load a learned hard partition (JSON, either a list "
        "of joint-index lists or the probe output with a 'partition' key of "
        "joint-name lists) and train the hard pose head with it. The "
        "partition is structural data for the V4B entry; other models reject "
        "it. Saved into the checkpoint config so eval/infer reconstruct it.",
    )
    parser.add_argument(
        "--lambda-assign",
        type=float,
        default=None,
        help="Weight of L_assign (assignment entropy + slot-load balance; "
        "only active for soft-A models, i.e. V3_3A/B / V3_4b/c / V4A). "
        "Default 0.05. V4A clustering passes 1.0 here and scales the three "
        "cluster terms with --lambda-assign-*.",
    )
    parser.add_argument(
        "--assign-cluster",
        action="store_true",
        help="V4A only (its entry fixes 24 slots + soft A): learn the part "
        "grouping from a near-uniform assignment. Replaces the V3-3 L_assign "
        "with annealed entropy + concentration reward + dead-slot tax; A gets "
        "its own optimizer lr (--assign-lr-mult) and is frozen after "
        "--assign-lock-frac of the run. Saved into the checkpoint.",
    )
    parser.add_argument(
        "--lambda-assign-ent", type=float, default=0.05,
        help="V4A: weight of the ANNEALED per-joint assignment entropy (early "
        "softness; anneals to 0 by --assign-anneal-frac).",
    )
    parser.add_argument(
        "--lambda-assign-conc", type=float, default=0.01,
        help="V4A: weight of the concentration reward (N_JOINTS^2 - sum mass^2, "
        ">= 0). Bigger = fewer, larger slots (K emerges from the recon-loss tension).",
    )
    parser.add_argument(
        "--lambda-assign-dead", type=float, default=0.02,
        help="V4A: weight of the dead-slot tax (sum exp(-beta*mass)) that "
        "rounds crumb loads to 0 so the effective K reads cleanly.",
    )
    parser.add_argument(
        "--assign-dead-beta", type=float, default=2.0,
        help="V4A: sharpness of the dead-slot tax exp(-beta*mass).",
    )
    parser.add_argument(
        "--assign-temp-init", type=float, default=1.0,
        help="V4A: assignment softmax temperature at step 0 (soft start).",
    )
    parser.add_argument(
        "--assign-temp-final", type=float, default=0.2,
        help="V4A: assignment softmax temperature at the end of the anneal "
        "(sharp). eval/readout restore this value from the checkpoint.",
    )
    parser.add_argument(
        "--assign-anneal-frac", type=float, default=0.7,
        help="V4A: fraction of total steps over which temperature and entropy "
        "anneal linearly to their final/zero values.",
    )
    parser.add_argument(
        "--assign-lock-frac", type=float, default=0.7,
        help="V4A: fraction of total steps after which part_logits (A) is "
        "frozen so the heads settle against a fixed grouping (M step).",
    )
    parser.add_argument(
        "--assign-lr-mult", type=float, default=5.0,
        help="V4A: optimizer lr multiplier for part_logits (A) relative to the "
        "rest of the model. Only active with --assign-cluster.",
    )
    parser.add_argument(
        "--lambda-sigma",
        type=float,
        default=None,
        help="V3-3: weight of the beta-NLL sigma supervision (plan §3, mixture-"
        "variance form; only active with --gate sigma). Default 0 (off); "
        "recommended 0.01 when enabling sigma calibration.",
    )
    parser.add_argument(
        "--sigma-freeze-frac",
        type=float,
        default=None,
        help="V3-3: fraction of total steps where sigma is frozen at 0 "
        "(gate runs on the tau_p prior bias alone, plan §3 'first 10% sigma "
        "fixed'). Default 0.1 with --gate sigma.",
    )
    parser.add_argument(
        "--lr-warmup-frac",
        type=float,
        default=None,
        help="V3 structural steps (fix_plan_v3 §2.2): linear LR warmup over the "
        "first this-fraction of total steps (e.g. 0.05), multiplying the "
        "epoch cosine LR. 0/absent = no warmup (F0b/F4a behavior).",
    )
    parser.add_argument("--lambda-pose", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_pose.")
    parser.add_argument("--lambda-kp", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_kp.")
    parser.add_argument("--lambda-traj", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_traj.")
    parser.add_argument("--lambda-trec", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_trec.")
    parser.add_argument("--lambda-vrec", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_vrec.")
    parser.add_argument("--lambda-con", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_con.")
    parser.add_argument("--lambda-pose-vel", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_pose_vel (E6.2, pos mode).")
    parser.add_argument("--lambda-bone", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_bone (E6.2, pos mode).")
    parser.add_argument("--device", default="auto", help="Device such as cuda, cuda:0, or cpu.")
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--contact-method",
        default=None,
        help="AnySole-private contact-label scheme for contact_gt. Labels are generated "
        "under model_inputs/AnySole by anysole.data.contact_adapter.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Override checkpoint output directory.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model directory identifier (e.g. F0b / V3_4a). With --out-dir omitted the "
        "output dir is INFERRED: <model-name>_<contact-method>[/<stacked fields>]/checkpoints, "
        "where the stacked fields (tw/st/lr/lp/lt/lk, defaults omitted) are built from the "
        "effective config — model+contact+fields == the ckpt address (2026-09-22 naming). "
        "--out-dir remains an explicit override only.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Explicit variant dir name override (e.g. --variant tw20 puts a "
        "default-hyperparameter run in <model>_<contact>/tw20/ instead of the "
        "bare base dir, keeping the results tree uniform). When omitted, the "
        "variant is auto-derived from the stacked non-default fields.",
    )
    parser.add_argument("--init-from", type=Path, default=None,
                        help="DEBUG OVERRIDE ONLY: warm-start from an existing "
                             "checkpoint.  Independent-models rule (2026-09-24): "
                             "every run starts from random init by default — no "
                             "warm-start is ever inferred.  Only state-dict keys "
                             "whose names AND shapes match the current model are "
                             "copied (everything else keeps its fresh init). "
                             "--init-drop adds explicit prefixes to skip.")
    parser.add_argument("--init-drop", type=str, default="",
                        help="Comma-separated state-dict key prefixes to skip in addition "
                             "to the automatic name/shape matching (e.g. 'pose_head.' to "
                             "re-init the whole pose head while reusing the encoders).")
    parser.add_argument("--wandb_mode", choices=("disabled", "offline", "online"), default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_eval_interval", type=int, default=None,
                        help="Epochs between validation evals (3 condition configs x full val set, "
                             "DDIM sampling). One eval costs ~16 normal epochs of wall-clock, so this "
                             "knob dominates total time: 10 -> 57s stall every 10 epochs (~64%% overhead), "
                             "50 -> ~26%% overhead. Runs regardless of wandb mode.")
    parser.add_argument("--wandb_experiment_tag", default=None, help="Batch tag used by Wandb_Analyzer to group runs.")
    parser.add_argument("--loss-cap", type=float, default=None,
                        help="If set, clip the logged train loss metrics (loss_total, loss_pose, loss_traj, "
                             "loss_con, loss_kp, loss_Trec_Tmissing, loss_Vrec_Vmissing, grad_norm) at this "
                             "value, keeping their original names (no *_cap duplicates). Raw charts get "
                             "y-squashed by the first epochs' spike (loss_total 30 -> 0.1, loss_pose 1.5 -> "
                             "0.01), hiding convergence; e.g. --loss-cap 1.0 pins the y-axis to 0-1.")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Max-norm gradient clipping (torch.nn.utils.clip_grad_norm_). "
                             "None = off. F0 lesson: the original F0b_warm run exploded at "
                             "ep~370 via a FINITE huge gradient (nonfinite guard never fired) "
                             "and never recovered; healthy regression runs stay at "
                             "grad_norm <= 1.3 after warm-up, so 5.0 only bites on explosions.")
    parser.add_argument("--lr", type=float, default=None,
                        help="V3-6 Optuna tuning: override the yaml learning rate per "
                             "trial. The manual's base commands keep the yaml value; "
                             "this flag exists for the tuning loop only.")
    return parser.parse_args(argv)


def _evaluate(model, diffusion, loader, config, device):
    """Return the epoch-level VT/V/T metrics used by the training dashboard."""
    pos_mode = getattr(model.pose_head, "repr", "6d") == "pos"
    regress_mode = str(config.get("modal")) == MODEL_ANYSOLEV2
    f2_mode = bool(config.get("f2_repr", False))
    result = {}
    with torch.inference_mode():
        for config_value, config_name in zip((0, 1, 2), CONFIG_NAMES):
            totals = {"mpjpe": 0.0, "root_ate": 0.0, "contact_f1": 0.0, "foot_slide": 0.0, "n": 0}
            tau0_mpjpe = 0.0
            first_batch = True
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                bsz = batch["pose_gt"].shape[0]
                cid = torch.full((bsz,), config_value, device=device, dtype=torch.long)
                v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
                v_kw = {"V_hmr": batch.get("V_hmr")} if str(config.get("v_input", "hrnet")) == "hmr_gvhmr" else {}
                if regress_mode:
                    # F0b: single forward, no DDIM sampling.
                    out = model(v_feat, t_raw, t_phys, cid, batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                    pred_pose = out["x0_hat"]
                else:
                    warm_prior = None
                    if bool(config.get("warm_start", False)):
                        # E6.5: initialize the chain on the training marginal
                        # (mean pose), not pure noise.
                        warm_prior = model.pose_head.pose_mean.view(1, 1, -1).expand(
                            bsz, int(config["tw"]), -1)
                    pred_pose = diffusion.ddim_sample_loop(
                        model, tau_related_kwargs={"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                        "T_s2m": t_s2m, "config_id": cid, "session_id": batch.get("session_id")},
                        shape=(bsz, int(config["tw"]), model.pose_head.pose_dim),
                        steps=int(config["diffusion_sample_steps"]), eta=0.0, device=device,
                        prior=warm_prior)
                    zero = torch.zeros(bsz, device=device, dtype=torch.long)
                    out = model(v_feat, t_raw, t_phys, pred_pose, zero, cid, batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                anchor = batch["trans_anchor"][:, None, :]
                pred_trans = out["trans_hat"] + anchor
                gt_trans = batch["trans_gt"] + anchor
                n = bsz
                if pos_mode:
                    # E6.1: compare in the metric space directly (23 joints,
                    # root-local; root covered by root_ate below).
                    err = torch.linalg.vector_norm(
                        pred_pose.reshape(bsz, -1, N_JOINTS - 1, 3)
                        - batch["pose_gt_pos"].reshape(bsz, -1, N_JOINTS - 1, 3),
                        dim=-1,
                    )
                    totals["mpjpe"] += float(err.mean().item()) * n * 1000.0
                else:
                    if f2_mode:
                        # F2a: recover the world pose/trans from the tilt pose
                        # + heading-frame trajectory, then the same FK scope.
                        world_pose, world_trans = f2_to_world(
                            pred_pose, out["v_hat"], batch["psi_anchor"],
                            batch["trans_anchor"],
                        )
                        pred_kp = fk_pose6d(world_pose, world_trans,
                                            batch["offsets"], batch["parents"])
                        pred_trans = world_trans
                    else:
                        pred_kp = fk_pose6d(pred_pose, pred_trans, batch["offsets"], batch["parents"])
                    totals["mpjpe"] += float(torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1).mean().item()) * n * 1000.0
                if first_batch and not regress_mode:
                    # tau=0 clean-input reconstruction MPJPE on one batch (no
                    # diffusion sampling): tells apart "cannot fit the training
                    # data" from "cannot denoise", on the same joint scope as
                    # eval.py (all 23 joints).  F0b regress has no clean-input
                    # reconstruction — its direct output IS the prediction, so
                    # tau0 is set to the full MPJPE below.
                    pose_in = batch["pose_gt_pos"] if pos_mode else batch["pose_gt"]
                    zero = torch.zeros(bsz, device=device, dtype=torch.long)
                    out0 = model(v_feat, t_raw, t_phys, pose_in, zero, cid, batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                    if pos_mode:
                        err0 = torch.linalg.vector_norm(
                            out0["x0_hat"].reshape(bsz, -1, N_JOINTS - 1, 3)
                            - batch["pose_gt_pos"].reshape(bsz, -1, N_JOINTS - 1, 3),
                            dim=-1,
                        )
                        tau0_mpjpe = float(err0.mean().item()) * 1000.0
                    else:
                        kp0 = fk_pose6d(out0["x0_hat"], batch["trans_gt"] + anchor, batch["offsets"], batch["parents"])
                        tau0_mpjpe = float(torch.linalg.vector_norm(kp0 - batch["kp_gt"], dim=-1).mean().item()) * 1000.0
                    first_batch = False
                pred_rel = pred_trans - pred_trans[:, :1]
                gt_rel = gt_trans - gt_trans[:, :1]
                totals["root_ate"] += float(torch.linalg.vector_norm(pred_rel - gt_rel, dim=-1).mean().item()) * n
                if not pos_mode:
                    pred_contact = soft_contact_from_keypoints(pred_kp, batch["floor_y"]) > 0.5
                    gt_contact = batch["contact_gt"] > 0.5
                    tp = (pred_contact & gt_contact).sum().item()
                    fp = (pred_contact & ~gt_contact).sum().item()
                    fn = (~pred_contact & gt_contact).sum().item()
                    tn = (~pred_contact & ~gt_contact).sum().item()
                    totals.setdefault("tp", 0); totals.setdefault("fp", 0); totals.setdefault("fn", 0); totals.setdefault("tn", 0)
                    totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn; totals["tn"] += tn
                    foot = pred_kp[:, :, FOOT_JOINTS, :][:, :, :, [0, 2]]
                    # Keypoints are meters; report horizontal contact displacement
                    # in mm/frame, matching eval_protocol.py.  Each foot is masked
                    # by its own contact state.
                    # Each foot is masked by its own contact state. The old
                    # any-foot mask averaged the fast swing foot into the
                    # stationary stance foot and over-reported sliding.
                    speed = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1) * 1000.0
                    slide = speed[pred_contact[:, 1:]]
                    if slide.numel():
                        totals.setdefault("foot_slide_sum", 0.0)
                        totals.setdefault("foot_slide_n", 0)
                        totals["foot_slide_sum"] += float(slide.sum().item())
                        totals["foot_slide_n"] += int(slide.numel())
                totals["n"] += n
            count = max(totals["n"], 1)
            p = totals.get("tp", 0) / max(totals.get("tp", 0) + totals.get("fp", 0), 1)
            r = totals.get("tp", 0) / max(totals.get("tp", 0) + totals.get("fn", 0), 1)
            air_recall = totals.get("tn", 0) / max(totals.get("tn", 0) + totals.get("fp", 0), 1)
            balanced_acc = 0.5 * (r + air_recall)
            if regress_mode:
                # F0b: no denoising chain — the direct regression output is
                # both the final prediction and the "clean-input" value.
                tau0_mpjpe = totals["mpjpe"] / count
            result.update({f"val/mpjpe/{config_name}": totals["mpjpe"] / count,
                           f"val/root_ate/{config_name}": totals["root_ate"] / count,
                           f"val/contact_f1/{config_name}": 2 * p * r / max(p + r, 1e-8),
                           f"val/contact_recall/{config_name}": r,
                           f"val/air_recall/{config_name}": air_recall,
                           f"val/contact_balanced_acc/{config_name}": balanced_acc,
                           f"val/foot_slide/{config_name}": totals.get("foot_slide_sum", 0.0) / max(totals.get("foot_slide_n", 0), 1),
                           f"val/tau0_mpjpe/{config_name}": tau0_mpjpe})
    result["val/gap_mpjpe_dropT"] = result["val/mpjpe/V"] - result["val/mpjpe/VT"]
    result["val/gap_mpjpe_dropV"] = result["val/mpjpe/T"] - result["val/mpjpe/VT"]
    return result


VIEW_CHECKPOINT_SCHEMA = "anysole.visual-training.v1"
_OPERATIONAL_KEYS = {
    "run_id", "run_dir", "results_root", "device", "pid", "pgid", "started_at",
    "resume", "resume_path", "semantic_hash", "run_hash", "plan_hash",
    "wandb_id", "history_step", "attempt_id", "heartbeat_sec", "max_retries",
    "disk_min_free_bytes", "runs_per_gpu", "resource_version", "resource_frozen",
    "log_every_updates", "tensorboard", "plan_frozen",
    "dataset_root", "manifest_path", "split_csv", "feature_cache_root", "stats_path",
    "pressure_metadata_root", "smpl_model_path", "hrnet_module_root",
    "hrnet_checkpoint", "hrnet_config", "detector_checkpoint", "checkpoint_every_updates",
    "control_dir", "wandb_shared_dir", "recovery_note",
    "heartbeat_timeout_sec", "worker_startup_timeout_sec",
}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def visual_run_identity(config: dict) -> str:
    """Content/algorithm identity, independent of relocation, process and W&B."""
    return canonical_hash({key: value for key, value in config.items()
                           if key not in _OPERATIONAL_KEYS and not key.startswith("wandb_")})


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, state: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def process_starttime(pid: int):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError, ValueError):
        return None


class ProcessLease:
    """OS lock plus PID/starttime evidence; never signals an unknown process."""
    def __init__(self, path: Path, identity: str):
        self.path, self.identity, self.handle = Path(path), identity, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handle.seek(0)
            old = json.loads(self.handle.read() or "{}")
            if old.get("status") == "active" and process_starttime(old.get("pid", -1)) == old.get("starttime") and old.get("starttime") is not None:
                raise RuntimeError(f"live owner still recorded in {self.path}")
            self.metadata = {"status": "active", "pid": os.getpid(), "pgid": os.getpgrp(),
                             "starttime": process_starttime(os.getpid()), "identity": self.identity,
                             "owner": "view-runtime", "started_at": time.time()}
            self._write()
            return self
        except Exception:
            self.handle.close()
            self.handle = None
            raise

    def _write(self):
        self.handle.seek(0)
        self.handle.truncate()
        json.dump(self.metadata, self.handle)
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def __exit__(self, *_):
        if self.handle is not None:
            self.metadata.update(status="released", released_at=time.time())
            self._write()
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def capture_training_rng(device: torch.device) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_training_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        if device.type != "cuda":
            raise ValueError("CUDA training cannot resume as a CPU run")
        torch.cuda.set_rng_state_all(state["cuda"])


class GroupBudgetSampler:
    """Deterministic complete-group subset packing; no model RNG or prefetch.

    Pending groups, epoch and permutation cursor are checkpointed. A bounded
    subset-sum pack selects exactly the RGB exposure target without splitting
    a physical group or silently dropping a non-fitting group.
    """
    def __init__(self, weights, keys, seed: int):
        self.weights, self.keys, self.seed = list(weights), list(keys), int(seed)
        if not self.weights or min(self.weights) < 1 or max(self.weights) > 4:
            raise ValueError("sampler needs nonempty groups with 1..4 visual windows")
        self.epoch = self.cursor = 0
        self.pending = []
        self._order = None

    def _next(self):
        if self._order is None:
            self._order = torch.randperm(len(self.weights), generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        if self.cursor == len(self._order):
            self.epoch += 1
            self.cursor = 0
            self._order = torch.randperm(len(self.weights), generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        value = self._order[self.cursor]
        self.cursor += 1
        return value

    def next_update(self, rgb_windows: int) -> list[int]:
        if not isinstance(rgb_windows, int) or rgb_windows < 1:
            raise ValueError("positive integer RGB budget required")
        # Four possible costs; 4*budget guarantees enough of one compatible
        # class for the recommended common multiple 240. 256/all-C3 fails.
        bound = max(4 * rgb_windows, len(self.weights))
        while len(self.pending) < bound:
            reachable = {0: []}
            for position, index in enumerate(self.pending):
                for total, selected in list(reachable.items())[::-1]:
                    updated = total + self.weights[index]
                    if updated <= rgb_windows and updated not in reachable:
                        reachable[updated] = selected + [position]
                if rgb_windows in reachable:
                    chosen = set(reachable[rgb_windows])
                    result = [index for pos, index in enumerate(self.pending) if pos in chosen]
                    self.pending = [index for pos, index in enumerate(self.pending) if pos not in chosen]
                    return result
            self.pending.append(self._next())
        raise ValueError("cannot pack exact RGB exposure; freeze a compatible common batch such as 240")

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch, "cursor": self.cursor,
                "pending": self.pending[:], "keys_hash": canonical_hash(self.keys),
                "weights_hash": canonical_hash(self.weights)}

    def load_state_dict(self, state):
        if state["seed"] != self.seed or state["keys_hash"] != canonical_hash(self.keys) or state["weights_hash"] != canonical_hash(self.weights):
            raise ValueError("sampler sample order/domain changed")
        self.epoch, self.cursor, self.pending = int(state["epoch"]), int(state["cursor"]), list(state["pending"])
        self._order = torch.randperm(len(self.weights), generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()


def grouped_view_index(batch):
    return batch["visual"]["view_valid"].any(dim=-1).nonzero(as_tuple=False)


def visual_forward(model, batch, view_index):
    if len(view_index) == 0:
        raise ValueError("all-invalid views must be filtered before forward")
    g, c = view_index.unbind(1)
    visual = batch["visual"]
    return model.forward_visual(visual["V_feat"][g, c], visual["view_valid"][g, c],
                                visual["target_time_s"][g])


def shuffle_pair_groups(batch, seed: int, update: int, micro: int):
    keys = batch["meta"]["sample_key"]
    n = len(keys)
    if n < 2:
        raise ValueError("pair_shuffle requires at least two distinct microbatch groups")
    salt = int(canonical_hash([seed, update, micro, keys])[:16], 16) % (2**63 - 1)
    generator = torch.Generator().manual_seed(salt)
    neighbors = [torch.randperm(n, generator=generator).tolist() for _ in range(n)]
    right_to_left = {}
    def augment(left, visited):
        for right in neighbors[left]:
            if right == left or keys[right] == keys[left] or right in visited:
                continue
            visited.add(right)
            if right not in right_to_left or augment(right_to_left[right], visited):
                right_to_left[right] = left
                return True
        return False
    if not all(augment(left, set()) for left in range(n)):
        raise ValueError("no derangement exists across distinct physical sample keys")
    result = [0] * n
    for right, left in right_to_left.items():
        result[left] = right
    return torch.tensor(result, device=batch["visual"]["V_feat"].device, dtype=torch.long)


class JointSelector:
    """Finite/common-coverage Pareto candidates and deterministic fixed score."""
    def __init__(self, m_ref=None, p_ref=None, coverage_hash=None, motion_weight=.5, pressure_weight=.5):
        self.m_ref, self.p_ref, self.coverage_hash = m_ref, p_ref, coverage_hash
        if not (0 < motion_weight < 1 and 0 < pressure_weight < 1 and
                math.isclose(motion_weight + pressure_weight, 1., rel_tol=0, abs_tol=1e-12)):
            raise ValueError("selection weights must be positive and sum to one")
        self.motion_weight, self.pressure_weight = motion_weight, pressure_weight
        self.candidates = []
        self.best_step = None

    def consider(self, metrics: dict, step: int):
        m, p = float(metrics["val/mpjpe"]), float(metrics["val/pressure_rmse"])
        if not (math.isfinite(m) and math.isfinite(p)) or min(metrics["val/motion_frames"], metrics["val/pressure_cells"]) <= 0:
            raise ValueError("nonfinite or empty validation coverage")
        coverage = metrics["val/coverage_hash"]
        if self.coverage_hash is None:
            self.coverage_hash = coverage
        if coverage != self.coverage_hash:
            raise ValueError("validation coverage changed")
        if self.m_ref is None or self.p_ref is None:
            return False  # explicit unselected pilot; never invent a scale
        if not (0 < float(self.m_ref) < math.inf and 0 < float(self.p_ref) < math.inf):
            raise ValueError("selection references must be finite and positive")
        candidate = {"step": int(step), "motion": m, "pressure": p,
                     "score": self.motion_weight * m / self.m_ref + self.pressure_weight * p / self.p_ref}
        pool = self.candidates + [candidate]
        self.candidates = [a for a in pool if not any(
            b["motion"] <= a["motion"] and b["pressure"] <= a["pressure"] and
            (b["motion"] < a["motion"] or b["pressure"] < a["pressure"])
            for b in pool)]
        selected = min(self.candidates, key=lambda a: (a["score"], a["motion"], a["pressure"], a["step"]))
        changed = selected["step"] != self.best_step
        self.best_step = selected["step"]
        return changed and self.best_step == step

    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        if state["m_ref"] != self.m_ref or state["p_ref"] != self.p_ref:
            raise ValueError("selection reference changed on resume")
        if (state.get("motion_weight", .5), state.get("pressure_weight", .5)) != (self.motion_weight, self.pressure_weight):
            raise ValueError("selection weights changed on resume")
        self.__dict__.update(copy.deepcopy(state))


def visual_source_hash():
    root = Path(__file__).resolve().parent
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*.py"))}
    files["../configs/tools/runner.py"] = sha256_file(root.parent / "configs/tools/runner.py")
    return canonical_hash(files)


def visual_fingerprints(config: dict):
    from anysole.data.workspace_adapter import load_grouped_manifest, load_grouped_stats
    records, meta = load_grouped_manifest(Path(config["manifest_path"]))
    stats = load_grouped_stats(Path(config["stats_path"]), manifest_sha256=meta["manifest_sha256"])
    index_path = Path(config["feature_cache_root"]) / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {"entries": {}}
    frontend = {key: config[key] for key in ("hrnet_checkpoint_hash", "detector_checkpoint_hash",
                                             "detector_version", "preprocessing_version", "cache_schema")}
    frontend["extractor_sha256"] = sha256_file(Path(__file__).parent / "data/extract_hrnet.py")
    frontend["hrnet_config_sha256"] = sha256_file(Path(config["hrnet_config"]))
    hrnet_root = Path(config["hrnet_module_root"]) / "lib/model/backbone/hrnet"
    frontend["hrnet_source_files"] = {str(p.relative_to(hrnet_root)): sha256_file(p)
                                      for p in sorted(hrnet_root.rglob("*.py"))}
    values = {"manifest_hash": meta["manifest_sha256"], "split_hash": sha256_file(Path(config["split_csv"])),
              "stats_hash": stats["stats_sha256"], "stats_file_hash": sha256_file(Path(config["stats_path"])),
              "cache_index_hash": sha256_file(index_path) if index_path.is_file() else None,
              "frontend_hash": index.get("frontend_sha256"),
              "frontend_runtime_hash": canonical_hash(frontend), "source_hash": visual_source_hash(),
              "smpl_runtime_model_hash": sha256_file(Path(config["smpl_model_path"]))}
    return values, records, stats, index


def visual_preflight(config: dict):
    """Pure reads: no Dataset, stats fit, contact generation, cache or run writes."""
    from anysole.types import validate_view_config
    from anysole.data.workspace_adapter import grouped_window_inventory
    validate_view_config(config)
    errors = []
    values = records = stats = index = None
    try:
        values, records, stats, index = visual_fingerprints(config)
    except (OSError, ValueError, KeyError) as exc:
        errors.append(type(exc).__name__ + ": " + str(exc))
    assets = {}
    for key in ("hrnet_checkpoint", "hrnet_config", "detector_checkpoint", "smpl_model_path"):
        path = Path(config[key])
        assets[key] = {"path": str(path.resolve()), "exists": path.is_file(),
                       "sha256": sha256_file(path) if path.is_file() else None}
        expected = config.get(key + "_hash")
        if not path.is_file() or (expected and assets[key]["sha256"] != expected):
            errors.append(f"missing or changed {key}")
    missing = []
    if records is not None:
        extractor_hash = sha256_file(Path(__file__).parent / "data/extract_hrnet.py")
        crop_hash = sha256_file(Path(__file__).parent / "data/crop.py")
        hrnet_root = Path(config["hrnet_module_root"]) / "lib/model/backbone/hrnet"
        hrnet_files = {str(p.relative_to(hrnet_root)): sha256_file(p) for p in sorted(hrnet_root.rglob("*.py"))}
        if not index.get("complete") or not index.get("frontend_sha256"):
            errors.append("canonical cache index is not complete/versioned")
        if index.get("manifest_sha256") != values["manifest_hash"]:
            errors.append("cache index manifest hash mismatch")
        for key, entry in index.get("entries", {}).items():
            provenance = entry.get("provenance", {})
            if provenance.get("frontend_sha256") != index.get("frontend_sha256"):
                errors.append(f"mixed or unversioned cache frontend: {key}")
                break
            frontend_assets = provenance.get("asset_hashes", {})
            if any(frontend_assets.get(k) != assets[k]["sha256"] for k in ("hrnet_checkpoint", "hrnet_config", "detector_checkpoint")):
                errors.append(f"cache/current asset hash mismatch: {key}")
                break
            if (provenance.get("extractor_sha256") != extractor_hash or
                    provenance.get("crop_source_sha256") != crop_hash or
                    provenance.get("hrnet_source_files") != hrnet_files):
                errors.append(f"cache/current frontend source hash mismatch: {key}")
                break
        for record in records:
            if record["eligible"]:
                for view in record["views"]:
                    if view["sources"]:
                        key = f"{record['recording_id']}/cam{view['camera_id']}"
                        entry = index.get("entries", {}).get(key)
                        if entry is None or not (Path(config["feature_cache_root"]) / entry["path"]).is_file():
                            missing.append(key)
        if missing:
            errors.append(f"missing {len(missing)} real frontend caches")
        for key, current in values.items():
            if config.get(key) is not None and current != config[key]:
                errors.append(f"configured {key} does not match actual contents")
    return {"schema": "anysole.visual-preflight.v1", "ready": not errors, "errors": errors,
            "assets": assets, "fingerprints": values, "missing_caches": missing,
            "unfrozen": [k for k, v in config.items() if v is None],
            "candidate_inventory": grouped_window_inventory(Path(config["manifest_path"])) if records is not None else None,
            "formal_plan_frozen": bool(config.get("plan_frozen")),
            "smpl_source_runtime_equivalence": config.get("smpl_source_runtime_equivalence")}


def _move_grouped(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_grouped(v, device) for k, v in value.items()}
    return value


def visual_health(device, run_dir):
    import resource
    values = {"health/peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "health/disk_free_bytes": shutil.disk_usage(run_dir).free}
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())
        values.update({"health/io_" + name: int(fields[name]) for name in ("read_bytes", "write_bytes")})
    except (OSError, KeyError, ValueError):
        pass
    if device.type == "cuda":
        values.update({"health/gpu_allocated_bytes": torch.cuda.memory_allocated(device),
                       "health/gpu_reserved_bytes": torch.cuda.memory_reserved(device),
                       "health/gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
    return values


def microbatch_groups(indices, weights, limit, *, keys=None, require_pairs=False):
    if require_pairs:
        # Same deterministic packing for every grouped experiment. A legal
        # derangement exists iff no physical key occupies more than half.
        solution = {len(indices): []}
        for start in reversed(range(len(indices))):
            total = 0
            for end in range(start + 1, len(indices) + 1):
                total += weights[indices[end-1]]
                if total > limit:
                    break
                chunk = indices[start:end]
                names = [keys[i] for i in chunk]
                eligible = len(chunk) >= 2 and max(names.count(k) for k in set(names)) * 2 <= len(chunk)
                if eligible and end in solution:
                    solution[start] = [chunk] + solution[end]
                    break
        if 0 not in solution:
            raise ValueError("common grouped microbatch packing cannot fit a distinct-key derangement; increase microbatch RGB budget")
        return solution[0]
    result, current, total = [], [], 0
    for index in indices:
        cost = weights[index]
        if cost > limit:
            raise ValueError("microbatch budget would split a complete group")
        if current and total + cost > limit:
            result.append(current)
            current, total = [], 0
        current.append(index)
        total += cost
    if current:
        result.append(current)
    return result


def visual_accumulation_step(model, batches, config, optimizer, scaler, device,
                             update, horizon):
    """One successful optimizer update, separately normalized for every term."""
    from anysole.utils.losses import (
        compute_visual_loss_terms, compute_pair_loss_terms,
        compute_visual_loss_counts, compute_pair_loss_counts, loss_term_weights,
    )
    sigma_frozen = update < int(float(config.get("sigma_freeze_frac", .1)) * horizon)
    if hasattr(model, "pose_head"):
        model.pose_head.set_sigma_frozen(sigma_frozen)
    peak = float(config.get("lambda_mv") or 0.)
    ramp = max(1, int(float(config.get("mv_ramp_frac", .1)) * horizon))
    lambda_mv = peak * min(float(update) / ramp, 1.)
    weights = loss_term_weights(config, int(config["tw"]), lambda_mv=lambda_mv)
    prepared, denominators = [], {}
    regularizer_used = False
    for number, batch in enumerate(batches):
        index = grouped_view_index(batch)
        counts = compute_visual_loss_counts(batch, index, config, include_assignment=False,
                                            sigma_frozen=sigma_frozen)
        supervised = counts.get("pose", 0) + counts.get("pressure", 0) > 0
        include_assignment = supervised and not regularizer_used
        if include_assignment:
            regularizer_used = True
            counts = compute_visual_loss_counts(batch, index, config, include_assignment=True,
                                                sigma_frozen=sigma_frozen)
        if config.get("pair_shuffle") and (len(batch["meta"]["sample_key"]) < 2 or len(set(batch["meta"]["sample_key"])) < 2):
            raise ValueError("pair_shuffle packing needs two distinct complete groups per microbatch")
        permutation = shuffle_pair_groups(batch, int(config["seed"]), update, number) if config.get("pair_shuffle") and lambda_mv else None
        if lambda_mv:
            counts.update(compute_pair_loss_counts(batch, index, config, pair_group_index=permutation))
        for key, count in counts.items():
            denominators[key] = denominators.get(key, 0) + int(count)
        prepared.append((batch, index, permutation, include_assignment, counts))
    if denominators.get("pose", 0) + denominators.get("pressure", 0) == 0:
        return {"stepped": False, "reason": "zero_supervision", "counts": denominators,
                "lambda_mv": lambda_mv, "sigma_frozen": sigma_frozen}
    optimizer.zero_grad(set_to_none=True)
    raw, weighted, shuffle_maps = {}, {}, []
    amp = bool(config.get("amp")) and device.type == "cuda"
    for batch, index, permutation, include_assignment, expected_counts in prepared:
        batch = _move_grouped(batch, device)
        index = index.to(device)
        permutation = permutation.to(device) if permutation is not None else None
        if not len(index):
            continue
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            outputs = visual_forward(model, batch, index)
            terms = compute_visual_loss_terms(outputs, batch, index, config,
                                               include_assignment=include_assignment)
            if lambda_mv:
                terms.update(compute_pair_loss_terms(outputs, batch, index, config,
                                                       pair_group_index=permutation))
            objective = outputs["x0_hat"].reshape(-1)[:0].sum()
            contributions = {}
            for key, term in terms.items():
                numerator = term["raw_sum"]
                if not torch.isfinite(numerator):
                    raise FloatingPointError(f"nonfinite loss term {key}")
                count = int(term["valid_count"])
                if count != expected_counts.get(key, 0):
                    raise ValueError(f"loss/count preparation disagrees for {key}")
                raw[key] = raw.get(key, 0.) + float(numerator.detach())
                contribution = numerator / denominators[key] if denominators.get(key, 0) else numerator * 0
                contribution = contribution * float(weights.get(key, 0.))
                weighted[key] = weighted.get(key, 0.) + float(contribution.detach())
                objective = objective + contribution
                contributions[key] = contribution
        if config.get("a2_version") is not None and (
                config.get("pressure_pair_route") != "all" or config.get("supervised_pressure_route") != "all"):
            from anysole.utils.a2_gradients import routed_backward
            routed_backward(contributions, model, config, scaler)
        else:
            scaler.scale(objective).backward()
        shuffle_maps.append(permutation.cpu().tolist() if permutation is not None else None)
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise FloatingPointError("absent training gradients")
    if any(not torch.isfinite(p.grad).all() for p in parameters):
        # Finite raw losses can overflow only after FP16 loss scaling. Let
        # GradScaler consume its recorded found_inf without calling the
        # optimizer. The caller retries this same batch and dropout RNG.
        scale_before = float(scaler.get_scale())
        if not scaler.is_enabled() or scale_before <= 1.:
            raise FloatingPointError("nonfinite training gradients at unscaled precision")
        scaler.update()
        scale_after = float(scaler.get_scale())
        if not math.isfinite(scale_after) or not 1. <= scale_after < scale_before:
            raise FloatingPointError("GradScaler did not safely reduce its overflowing scale")
        optimizer.zero_grad(set_to_none=True)
        return {"stepped": False, "reason": "amp_overflow", "counts": denominators,
                "amp_scale_before": scale_before, "amp_scale_after": scale_after,
                "lambda_mv": lambda_mv, "sigma_frozen": sigma_frozen}
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(config.get("grad_clip", 5.)),
                                                     error_if_nonfinite=True))
    scaler.step(optimizer)
    scaler.update()
    return {"stepped": True, "counts": denominators,
            "raw": {k: raw[k] / denominators[k] if denominators.get(k, 0) else 0. for k in raw},
            "weighted": weighted, "grad_norm": grad_norm, "lambda_mv": lambda_mv,
            "sigma_frozen": sigma_frozen, "shuffle_maps": shuffle_maps}


def visual_update_with_amp_retries(model, batches, config, optimizer, scaler,
                                   device, update, horizon, state, on_overflow=None):
    """Retry scaling overflow with unchanged data, model RNG and update budget."""
    rng = capture_training_rng(device)
    retries = 0
    limit = config.get("max_amp_backoffs_per_update", 16)
    if type(limit) is not int or not 0 <= limit <= 16:
        raise ValueError("max_amp_backoffs_per_update must be an integer from 0 through 16")
    state.setdefault("amp_overflow_attempts", 0)
    while True:
        state["attempt_steps"] += 1
        result = visual_accumulation_step(model, batches, config, optimizer,
                                          scaler, device, update, horizon)
        if result.get("reason") != "amp_overflow":
            return result
        retries += 1
        state["amp_overflow_attempts"] += 1
        if on_overflow is not None:
            on_overflow({"update": update, "attempt_steps": state["attempt_steps"],
                         "retry": retries, "scale_before": result["amp_scale_before"],
                         "scale_after": result["amp_scale_after"],
                         "sample_keys": list(state["last_sample_keys"]),
                         "successful_updates": state["successful_updates"],
                         "rgb_windows": state["rgb_windows"]})
        if retries > limit:
            raise FloatingPointError("AMP backoff limit exceeded for the same update")
        restore_training_rng(rng, device)


def beta_zero_offsets(model_path: Path):
    from anysole.data.smpl_io import _rest_joints
    from anysole.types import JOINT_PARENTS
    rest = _rest_joints({"betas": np.zeros(10)}, model_path=Path(model_path))
    offsets = np.zeros((24, 3), np.float32)
    for j, parent in enumerate(JOINT_PARENTS):
        if parent >= 0:
            offsets[j] = rest[j] - rest[parent]
    return torch.from_numpy(offsets)


def visual_sample_plot(model, item, config, device, *, split):
    """One fixed sample, strictly visual prediction and train-only GT display."""
    from anysole.data.dataset import grouped_collate
    from anysole.types import JOINT_PARENTS
    rng, training = capture_training_rng(device), model.training
    model.eval()
    try:
        with torch.no_grad():
            batch = _move_grouped(grouped_collate([item]), device)
            index = grouped_view_index(batch)
            if not len(index):
                return None
            out = visual_forward(model, batch, index)
            g, c = index[0].tolist()
            offsets = beta_zero_offsets(Path(config["smpl_model_path"]))
            pose = out["x0_hat"][:1].float().cpu()
            pred = fk_pose6d(pose, torch.zeros(1, pose.shape[1], 3), offsets,
                             torch.tensor(JOINT_PARENTS))[0]
            gt = batch["labels"]["kp_gt"][g].cpu()
            motion_error = torch.linalg.vector_norm((pred-pred[:, :1])-(gt-gt[:, :1]), dim=-1).mean(-1)
            valid = batch["visual"]["view_valid"][g, c].cpu()
            motion = batch["masks"]["motion_valid"][g].cpu() & valid
            motion_error[~motion] = float('nan')
            velocity = out["v_hat"][0].float().cpu()
            times = batch["visual"]["target_time_s"][g].cpu()
            edges = batch["visual"]["view_edges"][g, c].cpu()
            trajectory = torch.zeros_like(velocity)
            for t in range(1, len(times)):
                if bool(edges[t-1] & valid[t-1] & valid[t]):
                    trajectory[t] = trajectory[t-1] + velocity[t] * (times[t]-times[t-1]).float()
            gt_root = batch["labels"]["root"][g].cpu()
            truth_trajectory = torch.zeros_like(gt_root)
            anchor = 0
            for t in range(len(times)):
                if t == 0 or not bool(edges[t-1] & motion[t-1] & motion[t]):
                    anchor = t
                truth_trajectory[t] = gt_root[t]-gt_root[anchor]
            return {"prediction": out["pressure_hat"][0].float().cpu().numpy(),
                    "truth": batch["labels"]["pressure"][g].cpu().numpy(),
                    "mask": (batch["masks"]["pressure_valid"][g] & valid.to(device)[:, None]).cpu().numpy(),
                    "motion_error_m": motion_error.numpy(), "trajectory_m": trajectory.numpy(),
                    "truth_trajectory_m": truth_trajectory.numpy(), "times_s": (times-times[0]).numpy(),
                    "key": batch["meta"]["sample_key"][g], "split": split,
                    "unit": "normalized_sensor_reading"}
    finally:
        model.train(training)
        restore_training_rng(rng, device)


def evaluate_visual(model, dataset, config, device, *, offsets=None):
    """Shared strict-V old-val protocol; metrics have one canonical owner."""
    from anysole.utils.eval_protocol import evaluate_visual_dataset
    return evaluate_visual_dataset(model, dataset, config, device, offsets=offsets)


class VisualMonitor:
    """Worker-owned short heartbeat and disk guard while main sleeps."""
    def __init__(self, run_dir, config):
        self.run_dir, self.config = Path(run_dir), config
        self.stop, self.failure = threading.Event(), None
        self.progress = {"update": 0}
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            try:
                free = shutil.disk_usage(self.run_dir).free
                if free < int(self.config.get("disk_min_free_bytes") or 0):
                    raise OSError("insufficient disk headroom")
                atomic_json(self.run_dir / "heartbeat.json", {
                    **self.progress, "pid": os.getpid(), "pgid": os.getpgrp(),
                    "starttime": process_starttime(os.getpid()), "unix_time": time.time(),
                    "disk_free_bytes": free})
            except Exception as exc:
                self.failure = type(exc).__name__ + ": " + str(exc)
                return
            self.stop.wait(float(self.config.get("heartbeat_sec", 10)))

    def __enter__(self):
        self.thread.start()
        return self

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)


class VisualLogger:
    """Local JSONL/TensorBoard always; online SDK when auth is already present."""
    def __init__(self, run_dir, config, saved=None):
        from torch.utils.tensorboard import SummaryWriter
        self.run_dir, self.config = Path(run_dir), config
        self.history_step = int((saved or {}).get("history_step", 0))
        metrics_path = self.run_dir / "metrics.jsonl"
        if metrics_path.is_file():
            for line in metrics_path.read_text().splitlines():
                try:
                    self.history_step = max(self.history_step, int(json.loads(line).get("history_step", -1)) + 1)
                except (ValueError, TypeError):
                    pass
            if metrics_path.stat().st_size and not metrics_path.read_bytes().endswith(b"\n"):
                with metrics_path.open("a") as handle:
                    handle.write("\n")
        self.run, self.run_id, self.mode, self.url = None, (saved or {}).get("wandb_id"), "disabled", None
        self.run_id = (saved or {}).get("wandb_logical_id") or self.run_id
        self.online_id = (saved or {}).get("online_wandb_id") or (
            (saved or {}).get("wandb_id") if (saved or {}).get("wandb_mode") == "online" else None)
        self.offline_segments = list((saved or {}).get("offline_segments", []))
        self.tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        requested = config.get("wandb_mode", "online")
        if requested != "disabled":
            import wandb
            shared = Path(config["wandb_shared_dir"])
            for name in ("cache", "config", "data"):
                (shared / name).mkdir(parents=True, exist_ok=True)
                os.environ[f"WANDB_{name.upper()}_DIR"] = str(shared / name)
            os.environ["WANDB_DIR"] = str(self.run_dir)
            auth = False
            if requested == "online":
                try:
                    auth = bool(wandb.Api(timeout=5).api_key)
                except Exception:
                    pass
            self.mode = "online" if requested == "online" and auth else "offline"
            self.run_id = self.run_id or uuid.uuid4().hex[:8]
            # Offline attempts get their own segment. They share the explicit
            # logical training identity, not a false online resume claim.
            sdk_id = self.run_id if self.mode == "online" else None
            options = dict(
                id=sdk_id, resume="must" if sdk_id and self.online_id else None,
                project=config.get("wandb_project", "anysole"), entity=config.get("wandb_entity"),
                group=config.get("wandb_group", "view-consistency-20261016"),
                job_type=config.get("wandb_job_type", "train"), name=self.run_dir.name,
                dir=str(self.run_dir), mode=self.mode,
                config={k: v for k, v in config.items() if k not in ("wandb_entity",)},
                settings=wandb.Settings(disable_git=True, save_code=False, console="off", quiet=True, init_timeout=30))
            fallback = "authentication_unavailable" if requested == "online" and not auth else None
            try:
                self.run = wandb.init(**options)
            except Exception as exc:
                if self.mode != "online":
                    raise
                # Keep only the error class; messages may contain private URLs.
                fallback = "online_init_failed:" + type(exc).__name__
                self.mode = "offline"
                options.update(mode="offline", id=None, resume=None)
                self.run = wandb.init(**options)
            self.run.define_metric("progress/update")
            self.run.define_metric("*", step_metric="progress/update")
            self.url = self.run.url if self.mode == "online" else None
            if self.mode == "online":
                self.online_id = self.run.id
            if self.mode == "offline":
                self.offline_segments.append(str(Path(self.run.dir).parent))
        self.metadata = {"wandb_id": self.run.id if self.run is not None else None,
                         "wandb_logical_id": self.run_id, "online_wandb_id": self.online_id,
                         "wandb_mode": self.mode, "wandb_url": self.url,
                         "offline_dir": str(Path(self.run.dir).parent) if self.run is not None and self.mode == "offline" else None,
                         "wandb_files_dir": self.run.dir if self.run is not None else None,
                         "sync_state": "online" if self.mode == "online" else "not_synced",
                         "offline_segments": self.offline_segments,
                         "online_fallback_reason": fallback if requested != "disabled" else None}

    def log(self, values, *, plot=None):
        record = {**values, "history_step": self.history_step, "unix_time": time.time()}
        with (self.run_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        update = int(values["progress/update"])
        for key, value in values.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                self.tb.add_scalar(key, value, update)
        media = {}
        plots = plot if isinstance(plot, list) else [plot]
        for plot in plots:
            if plot is None:
                continue
            # Display clipping is explicit; scalar losses/metrics stay linear.
            for foot, part in (("left", slice(0, 48)), ("right", slice(48, 96))):
                truth = np.where(plot["mask"][:, part], plot["truth"][:, part], np.nan)
                prediction = np.where(plot["mask"][:, part], plot["prediction"][:, part], np.nan)
                image = np.concatenate([truth, prediction], axis=0)[None]
                image = np.where(np.isfinite(image), image, 0)
                tag = f"{plot['split']}/{foot}_pressure_display_clip01_truth_prediction"
                self.tb.add_image(tag, np.clip(image, 0, 1), update, dataformats="CHW")
                if self.run is not None:
                    import wandb
                    media[tag] = wandb.Image(np.clip(image[0], 0, 1), caption=plot["key"])
                    traces = []
                    for t, (gt_row, pred_row) in enumerate(zip(truth, prediction)):
                        gt_finite, pred_finite = gt_row[np.isfinite(gt_row)], pred_row[np.isfinite(pred_row)]
                        traces.append([t, float(gt_finite.mean()) if len(gt_finite) else None,
                                       float(pred_finite.mean()) if len(pred_finite) else None,
                                       float(gt_finite.max()) if len(gt_finite) else None,
                                       float(pred_finite.max()) if len(pred_finite) else None])
                    media[f"{plot['split']}/{foot}_pressure_time_traces"] = wandb.Table(
                        columns=["window_frame", "truth_mean", "prediction_mean", "truth_peak", "prediction_peak"], data=traces)
                # Low-frequency fixed sample traces, indexed by relative time.
                for kind, readings in (("truth", truth), ("prediction", prediction)):
                    for t, reading in enumerate(readings):
                        finite = reading[np.isfinite(reading)]
                        if len(finite):
                            self.tb.add_scalar(f"{plot['split']}/{foot}_{kind}_mean_u{update}", float(finite.mean()), t)
                            self.tb.add_scalar(f"{plot['split']}/{foot}_{kind}_peak_u{update}", float(finite.max()), t)
            if "motion_error_m" in plot:
                columns = ["relative_time_s", "motion_error_m", "pred_x_m", "pred_y_m", "pred_z_m",
                           "gt_x_m", "gt_y_m", "gt_z_m"]
                data = np.column_stack([plot["times_s"], plot["motion_error_m"],
                                        plot["trajectory_m"], plot["truth_trajectory_m"]])
                for t, row in enumerate(data):
                    for column, value in zip(columns[1:], row[1:]):
                        if math.isfinite(value):
                            self.tb.add_scalar(f"{plot['split']}/{column}_u{update}", value, t)
                if self.run is not None:
                    import wandb
                    media[f"{plot['split']}/fixed_motion_trajectory"] = wandb.Table(
                        columns=columns, data=[[float(x) if math.isfinite(x) else None for x in row] for row in data])
        if self.run is not None:
            self.run.log({**values, **media}, step=self.history_step)
        self.history_step += 1
        self.tb.flush()

    def state_dict(self):
        return {**self.metadata, "history_step": self.history_step}

    def close(self):
        self.tb.close()
        if self.run is not None:
            self.run.finish()


def restore_visual_runtime_state(model, checkpoint):
    """Restore functional flags that a torch state_dict does not serialize."""
    if not hasattr(model, "pose_head") or not hasattr(model.pose_head, "set_sigma_frozen"):
        return
    flags = []
    for name in ("model_runtime", "training", "schedule"):
        value = checkpoint.get(name, {}).get("sigma_frozen")
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError("checkpoint sigma-frozen state must be boolean")
            flags.append(value)
    if flags and any(flag != flags[0] for flag in flags):
        raise ValueError("checkpoint sigma-frozen functional/schedule states disagree")
    if not flags and checkpoint.get("training", {}).get("successful_updates", 0) > 0:
        raise ValueError("trained checkpoint is missing sigma-frozen functional state")
    # A pure random/untrained checkpoint predating this field used False.
    model.pose_head.set_sigma_frozen(flags[0] if flags else False)


def _visual_checkpoint(model, optimizer, scheduler, scaler, sampler, selector,
                       config, state, stats, logger, device):
    training = copy.deepcopy(state)
    runtime = {}
    if hasattr(model, "pose_head") and getattr(model.pose_head, "gate", None) == "sigma":
        flags = {bool(layer.sigma_frozen) for layer in model.pose_head.decoder.layers}
        if len(flags) != 1:
            raise ValueError("model decoder sigma-frozen flags disagree")
        runtime["sigma_frozen"] = flags.pop()
        if training.get("sigma_frozen") not in (None, runtime["sigma_frozen"]):
            raise ValueError("model and recorded training sigma-frozen states disagree")
        training["sigma_frozen"] = runtime["sigma_frozen"]
    return {"schema_version": VIEW_CHECKPOINT_SCHEMA, "semantic_hash": visual_run_identity(config),
            "config": dict(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "sampler": sampler.state_dict(), "selector": selector.state_dict(),
            "training": training, "model_runtime": runtime, "rng": capture_training_rng(device),
            "stats": stats, "observability": logger.state_dict(),
            "schedule": {"horizon": config["target_updates"], "lr": scheduler.get_last_lr(),
                         "lambda_mv": state.get("lambda_mv", 0.), "sigma_frozen": training.get("sigma_frozen")}}


def load_visual_checkpoint(path, config, model, optimizer, scheduler, scaler, sampler,
                           selector, device):
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("checkpoint is corrupt or incomplete") from exc
    if (checkpoint.get("schema_version") != VIEW_CHECKPOINT_SCHEMA or
            checkpoint.get("semantic_hash") != visual_run_identity(config) or
            checkpoint.get("semantic_hash") != visual_run_identity(checkpoint.get("config", {}))):
        raise ValueError("checkpoint schema/semantic identity mismatch")
    # State-dict shape equality alone cannot distinguish legacy from time PE.
    entry = get_builder("visual")
    saved = entry.effective_config(checkpoint["config"])
    if saved["structure_hash"] != config["structure_hash"]:
        raise ValueError("checkpoint pressure-head/PE structure mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    restore_visual_runtime_state(model, checkpoint)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    sampler.load_state_dict(checkpoint["sampler"])
    selector.load_state_dict(checkpoint["selector"])
    if config.get("a2_version") is not None:
        from anysole.utils.a2_training import validate_exposure
        validate_exposure(checkpoint["training"], config, keys=sampler.keys, weights=sampler.weights)
        if (sampler.epoch < 0 or not 0 <= sampler.cursor <= len(sampler.keys) or
                any(type(i) is not int or not 0 <= i < len(sampler.keys) for i in sampler.pending)):
            raise ValueError("invalid A2 checkpoint sampler cursor/pending state")
    restore_training_rng(checkpoint["rng"], device)
    return checkpoint


def _validate_visual_experiment(config):
    experiment = config.get("experiment")
    if experiment not in ("base_legacy", "base_time", "base_pool", "view_consist",
                           "pose_pair", "press_pair", "pair_shuffle"):
        raise ValueError("explicit grouped experiment required")
    baseline = experiment.startswith("base_")
    from anysole.utils.a2_config import is_a2, validate_a2_config
    a2 = is_a2(config)
    validate_a2_config(config, formal=config.get("stage") == "formal")
    if not a2 and config["pressure_head"] != ("legacy" if experiment == "base_legacy" else "time_pe"):
        raise ValueError("experiment name and frozen pressure head disagree")
    if baseline and float(config.get("lambda_mv") or 0) != 0:
        raise ValueError("baseline cannot enable pair loss")
    if bool(config.get("pair_shuffle")) != (experiment == "pair_shuffle"):
        raise ValueError("pair_shuffle routing must match the experiment")
    if experiment == "pose_pair" and config.get("mv_pressure_weight") != 0:
        raise ValueError("pose_pair must disable only pressure pair")
    if experiment == "press_pair" and config.get("mv_pose_weight") != 0:
        raise ValueError("press_pair must disable only pose pair")
    if int(config.get("num_workers") or 0) != 0:
        raise ValueError("deterministic cached P2 loader currently requires num_workers=0")
    retries = config.get("max_retries", 2)
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 2:
        raise ValueError("at most two documented failed-run recovery attempts are allowed")
    if int(config["target_updates"]) * int(config["effective_batch_rgb_windows"]) != int(config["target_rgb_windows"]):
        raise ValueError("successful-update and exact RGB-exposure budgets disagree")


def reference_visual_budget(n_cam3_windows: int, batch_rgb_windows: int = 256,
                            *, drop_last: bool = False, epochs: int = 740):
    """Pure arithmetic; P3 supplies measured post-frontend cam3 coverage."""
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1
           for x in (n_cam3_windows, batch_rgb_windows, epochs)):
        raise ValueError("positive integer reference coverage/batch/epochs required")
    steps = n_cam3_windows // batch_rgb_windows if drop_last else math.ceil(n_cam3_windows/batch_rgb_windows)
    if steps < 1:
        raise ValueError("reference batch rule yields zero optimizer steps")
    updates = epochs * steps
    return {"n_cam3_train_windows": n_cam3_windows, "reference_batch_rgb_windows": batch_rgb_windows,
            "effective_batch_rgb_windows": batch_rgb_windows,
            "reference_steps_per_epoch": steps, "reference_epochs": epochs,
            "target_updates": updates, "target_rgb_windows": updates * batch_rgb_windows}


def restrict_pilot_windows(dataset, config):
    """Use a predeclared subset of real windows without bypassing preflight."""
    keys = config.get("pilot_train_sample_keys")
    if keys is None:
        return
    if config.get("stage") != "pilot" or not keys or len(keys) != len(set(keys)):
        raise ValueError("only pilot runs may declare a nonempty unique window subset")
    lookup = {dataset.sample_key(i): window for i, window in enumerate(dataset.windows)}
    if any(key not in lookup for key in keys):
        raise ValueError("declared pilot sample key is absent from the real training domain")
    dataset.windows = [lookup[key] for key in keys]


def visual_selection_references(metrics, *, floor=1.e-8):
    """Freeze only a qualified base_pool pilot's shared old-val references."""
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("positive finite selection-reference floor required")
    m, p = float(metrics["val/mpjpe"]), float(metrics["val/pressure_rmse"])
    if not all(math.isfinite(x) and x >= 0 for x in (m, p)) or min(
            metrics["val/motion_frames"], metrics["val/pressure_cells"]) <= 0:
        raise ValueError("selection references need finite metrics and nonempty old-val coverage")
    return {"selection_m_ref": max(m, floor), "selection_p_ref": max(p, floor),
            "selection_coverage_hash": metrics["val/coverage_hash"], "selection_reference_floor": floor}


def train_visual_run(config, run_dir, *, resume=None, stop_after_updates=None,
                     dataset=None, validation_dataset=None, model=None, validation_fn=None):
    """Shared production controller; injectable pilot-only IO for state tests."""
    from anysole.types import validate_view_config
    from anysole.data.dataset import GroupedViewDataset, grouped_collate
    config = copy.deepcopy(config)
    formal = config.get("stage") == "formal"
    injected = dataset is not None or model is not None or validation_fn is not None
    if formal and (injected or stop_after_updates is not None):
        raise ValueError("formal runs cannot use test IO or stop/truncation controls")
    if config.get("stage") not in ("pilot", "formal"):
        raise ValueError("training requires explicit pilot or formal stage")
    validate_view_config(config, formal=formal)
    config = get_builder("visual").effective_config(config)
    stats = config.get("_test_stats")
    if not injected:
        report = visual_preflight(config)
        if not report["ready"]:
            raise ValueError("preflight failed: " + "; ".join(report["errors"]))
        values, _, stats, _ = visual_fingerprints(config)
        for key, value in values.items():
            if config.get(key) is not None and config[key] != value:
                raise ValueError(f"frozen input hash changed: {key}")
            config[key] = value
    _validate_visual_experiment(config)
    target = int(config["target_updates"])
    if target < 1 or int(config["effective_batch_rgb_windows"]) < 1:
        raise ValueError("positive pilot budget required")
    if not stats:
        raise ValueError("frozen train-only stats required")
    run_dir = Path(run_dir).resolve()
    workspace = Path("/private/Tactile").resolve() if config.get("a2_version") is not None else Path(__file__).resolve().parents[2]
    if not run_dir.is_relative_to(workspace):
        raise ValueError("run directory must stay inside /private/Tactile")
    run_dir.mkdir(parents=True, exist_ok=True)
    config["run_dir"] = str(run_dir)
    identity = visual_run_identity(config)
    device = resolve_device(str(config.get("device", "cpu")))
    if formal and device.type != "cuda":
        raise ValueError("formal run requires the measured CUDA resource envelope")
    if formal and not config.get("resource_frozen"):
        raise ValueError("formal run requires measured and frozen GPU resource envelope")
    if dataset is None:
        cameras = tuple(config.get("train_cameras", (3,) if config["experiment"] in ("base_legacy", "base_time") else (1, 2, 3, 4)))
        common = dict(manifest=Path(config["manifest_path"]), dataset_root=Path(config["dataset_root"]),
                      feature_cache_root=Path(config["feature_cache_root"]), smpl_model_path=Path(config["smpl_model_path"]),
                      window_length=int(config["window_length"]), stride=int(config["train_stride"]), cameras=cameras)
        dataset = GroupedViewDataset(**common, split="train", cover_tail=False)
        validation_dataset = GroupedViewDataset(**{**common, "cameras": tuple(config["validation_cameras"])}, split="val", cover_tail=True)
        if formal and config.get("a2_version") == "anysole.a3.v1":
            if (len(dataset) != config["n_train_group_windows"] or
                    sum(len(w[3]) for w in dataset.windows) != config["n_train_rgb_windows"] or
                    canonical_hash([dataset.sample_key(i) for i in range(len(dataset))]) != config["train_sample_keys_hash"]):
                raise ValueError("A3 actual train windows differ from the frozen train-only budget")
        elif formal:
            reference = dataset if cameras == (3,) else GroupedViewDataset(
                **{**common, "cameras": (3,)}, split="train", cover_tail=False)
            if len(reference) != config["n_cam3_train_windows"]:
                raise ValueError("actual post-frontend cam3 train windows differ from the frozen reference budget")
            if reference is not dataset:
                del reference
    if not len(dataset) or validation_dataset is None:
        raise ValueError("nonempty train and old-val datasets required")
    restrict_pilot_windows(dataset, config)
    weights = [len(window[3]) for window in dataset.windows]
    keys = [dataset.sample_key(i) for i in range(len(dataset))]
    sampler = GroupBudgetSampler(weights, keys, int(config["seed"]))
    a2 = config.get("a2_version") is not None
    if a2 and config["budget_mode"] == "cam3_exact":
        if len(keys) != 1765 or any(w != 4 for w in weights) or any(set(w[3]) != {1, 2, 3, 4} for w in dataset.windows):
            raise ValueError("exact cam3 matching requires all 1765 physical groups to contain all four cameras")
        if config.get("cam3_sample_keys_hash") != canonical_hash(keys):
            raise ValueError("B-CAM3 keys/order differ from the verified original cam3 sampler")
    previous = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").is_file() else {}
    if previous and previous.get("semantic_hash") != identity:
        raise ValueError("run directory already belongs to another semantic identity")
    if previous and resume is None:
        raise ValueError("existing run requires explicit --resume, even when a checkpoint exists")
    recovery_attempts = int(previous.get("recovery_attempts", 0))
    if resume and previous.get("state") == "failed":
        if not config.get("recovery_note"):
            raise ValueError("failed run requires a recorded fix/recovery_note before retry")
        recovery_attempts += 1
        if recovery_attempts > int(config.get("max_retries", 2)):
            raise ValueError("failed-run recovery limit exceeded")
    metadata = {"schema_version": VIEW_CHECKPOINT_SCHEMA, "semantic_hash": identity,
                "run_id": previous.get("run_id", uuid.uuid4().hex),
                "attempt_id": uuid.uuid4().hex, "recovery_attempts": recovery_attempts,
                "pid": os.getpid(), "pgid": os.getpgrp(), "starttime": process_starttime(os.getpid()),
                "stage": config["stage"], "state": "running", "started_at": time.time(),
                "target_updates": target, "target_rgb_windows": config["target_rgb_windows"],
                "formal_complete": False, "exit_code": None}
    for field in ("last_checkpoint_hash", "best_checkpoint_hash"):
        if previous.get(field):
            metadata[field] = previous[field]
    if config.get("recovery_note"):
        metadata["recovery_note"] = config["recovery_note"]
    logger = None
    state = {"successful_updates": 0, "attempt_steps": 0, "rgb_windows": 0,
             "amp_overflow_attempts": 0,
             "physical_group_windows": 0, "zero_supervision_attempts": 0,
             "consecutive_zero_supervision_attempts": 0,
             "unique_sample_keys": [], "last_sample_keys": [], "last_shuffle_maps": []}
    stop_requested = threading.Event()
    handlers = {}
    global_lease = Path(config["control_dir"]) / "run_locks" / (identity + ".lock")
    with ProcessLease(global_lease, identity), ProcessLease(run_dir / "writer.lock", identity), VisualMonitor(run_dir, config) as monitor:
        try:
            atomic_json(run_dir / "run.json", metadata)
            import yaml
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
            seed = int(config["seed"])
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            torch.use_deterministic_algorithms(bool(config.get("deterministic_algorithms", False)))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            model = model if model is not None else get_builder("visual").build(config)
            model = model.to(device).train()
            if hasattr(model, "pose_head"):
                model.pose_head.set_stats(torch.tensor(stats["pose_mean"], device=device),
                                          torch.tensor(stats["pose_std"], device=device))
            if a2:
                from anysole.utils.a2_training import initialization_identity
                if resume is None:
                    initial_identity = initialization_identity(model)
                    expected_common = config.get("common_initial_state_hash")
                    if expected_common is not None and initial_identity["common_state_hash"] != expected_common:
                        raise ValueError("controlled model common initial weights differ")
                    atomic_json(run_dir / "initialization.json", initial_identity)
                probe_indices = config.get("gradient_probe_indices", list(range(min(2, len(dataset)))))
                if len(probe_indices) < 2 or any(type(i) is not int or not 0 <= i < len(dataset) for i in probe_indices):
                    raise ValueError("A2 fixed train gradient probe requires two valid physical groups")
                probe_batch = grouped_collate([dataset[i] for i in probe_indices])
                atomic_json(run_dir / "gradient_probe_support.json", {
                    "indices": probe_indices, "sample_keys": [keys[i] for i in probe_indices],
                    "view_valid_hash": canonical_hash(probe_batch["visual"]["view_valid"].tolist()),
                    "target_time_hash": canonical_hash(probe_batch["visual"]["target_time_s"].tolist()), "fit_split": "train"})
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
            warmup = max(1, int(float(config.get("lr_warmup_frac", .05)) * target))
            def lr_factor(step):
                warm = min((step + 1) / warmup, 1.)
                if config.get("lr_schedule", "constant") == "cosine":
                    warm *= .5 * (1 + math.cos(math.pi * min(step, target) / target))
                return warm
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
            scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("amp")) and device.type == "cuda")
            selector = JointSelector(config.get("selection_m_ref"), config.get("selection_p_ref"),
                                     config.get("selection_coverage_hash"),
                                     motion_weight=config.get("selection_motion_weight", .5),
                                     pressure_weight=config.get("selection_pressure_weight", .5))
            loaded = None
            if resume is not None:
                loaded = load_visual_checkpoint(resume, config, model, optimizer, scheduler,
                                                scaler, sampler, selector, device)
                state = loaded["training"]
                if loaded["stats"] != stats:
                    raise ValueError("checkpoint normalization statistics changed")
            if a2 and resume is None:
                state["exposure_ledger"] = {"schema": "anysole.a2.exposure.v1", "per_key": {}, "per_camera": {}, "per_camera_key": {}}
            rng = capture_training_rng(device)
            logger = VisualLogger(run_dir, config, loaded.get("observability") if loaded else None)
            restore_training_rng(rng, device)
            metadata.update(logger.metadata)
            metadata["replay_from_update"] = state["successful_updates"] if resume else None
            metadata["replay_to_update"] = previous.get("successful_updates") if resume else None
            if config.get("save_initial_checkpoint") and resume is None:
                initial = _visual_checkpoint(model, optimizer, scheduler, scaler, sampler, selector,
                                             config, state, stats, logger, device)
                metadata["initial_checkpoint_hash"] = atomic_checkpoint(run_dir / "initial.pt", initial)
            for sig in (signal.SIGTERM, signal.SIGINT):
                if threading.current_thread() is threading.main_thread():
                    handlers[sig] = signal.signal(sig, lambda *_: stop_requested.set())
            unique = set(state["unique_sample_keys"])
            last_val = state.get("last_validation")
            validation_every = int(config["validation_every_updates"])
            checkpoint_every = int(config["checkpoint_every_updates"])
            if min(validation_every, checkpoint_every) < 1:
                raise ValueError("positive validation/checkpoint cadence required")
            while state["successful_updates"] < target:
                monitor.check()
                selected = sampler.next_update(int(config["effective_batch_rgb_windows"]))
                parts = microbatch_groups(selected, weights, int(config["microbatch_rgb_windows"]),
                                         keys=keys, require_pairs=config["experiment"] not in ("base_legacy", "base_time"))
                batches = [grouped_collate([dataset[i] for i in part]) for part in parts]
                started = time.monotonic()
                state["last_sample_keys"] = [keys[i] for i in selected]
                def record_overflow(event):
                    with (run_dir / "amp_overflows.jsonl").open("a") as handle:
                        handle.write(json.dumps({**event, "time": time.time()}, sort_keys=True) + "\n")
                result = visual_update_with_amp_retries(
                    model, batches, config, optimizer, scaler, device,
                    state["successful_updates"] + int(config.get("pilot_schedule_update_offset", 0)),
                    target, state, record_overflow)
                if not result["stepped"]:
                    state["zero_supervision_attempts"] += 1
                    state["consecutive_zero_supervision_attempts"] += 1
                    if state["consecutive_zero_supervision_attempts"] >= int(config.get("max_zero_supervision_attempts", 3)):
                        raise ValueError("persistent zero-supervision batches")
                    continue
                state["consecutive_zero_supervision_attempts"] = 0
                scheduler.step()
                state["successful_updates"] += 1
                state["rgb_windows"] += sum(weights[i] for i in selected)
                state["physical_group_windows"] += len(selected)
                if a2:
                    from anysole.utils.a2_training import record_exposure
                    record_exposure(state, selected, dataset, keys)
                unique.update(state["last_sample_keys"])
                state["unique_sample_keys"] = sorted(unique)
                state["lambda_mv"], state["sigma_frozen"] = result["lambda_mv"], result["sigma_frozen"]
                state["last_shuffle_maps"] = result["shuffle_maps"]
                step = state["successful_updates"]
                elapsed = max(time.monotonic() - started, 1e-9)
                monitor.progress = {"update": step, "target_updates": target,
                                    "rgb_windows": state["rgb_windows"]}
                if a2:
                    ledger_row = {"schema": "anysole.a2.update.v1", "attempt_id": metadata["attempt_id"],
                                  "successful_update": step, "attempt_steps": state["attempt_steps"],
                                  "rgb_windows": state["rgb_windows"], "sample_keys": state["last_sample_keys"],
                                  "cameras": [list(dataset.windows[i][3]) for i in selected],
                                  "seconds": elapsed, "lambda_mv": result["lambda_mv"],
                                  "sigma_frozen": result["sigma_frozen"], "counts": result["counts"],
                                  "raw": result["raw"], "weighted": result["weighted"]}
                    with (run_dir / "update_ledger.jsonl").open("a") as handle:
                        handle.write(json.dumps(ledger_row, sort_keys=True, allow_nan=False) + "\n")
                    timing_warmup = int(config.get("timing_warmup_updates", 0))
                    if step > timing_warmup:
                        timing = state.setdefault("timing", {"successful_updates": 0, "rgb_windows": 0, "seconds": 0.})
                        timing["successful_updates"] += 1
                        timing["rgb_windows"] += sum(weights[i] for i in selected)
                        timing["seconds"] += elapsed
                        metadata["timing_summary"] = {**timing, "rgb_windows_per_second": timing["rgb_windows"]/timing["seconds"]}
                    metadata["exposure_per_camera"] = state["exposure_ledger"]["per_camera"]
                    if step in config.get("gradient_probe_updates", [100, 500, 1300, 2600, 3900, 5180]):
                        from anysole.utils.a2_training import fixed_gradient_probe
                        probe = fixed_gradient_probe(model, probe_batch, config, device, update=step, horizon=target)
                        atomic_json(run_dir / "gradient_probes" / ("step_%06d.json" % step), probe)
                values = {"progress/update": step, "progress/target_updates": target,
                          "progress/completion": step / target, "progress/rgb_windows": state["rgb_windows"],
                          "progress/attempt_steps": state["attempt_steps"],
                          "progress/amp_overflow_attempts": state["amp_overflow_attempts"],
                          "progress/zero_supervision_attempts": state["zero_supervision_attempts"],
                          "progress/physical_group_windows": state["physical_group_windows"],
                          "progress/unique_sample_keys": len(unique),
                          "health/recovery_attempts": recovery_attempts,
                          "train/lr": scheduler.get_last_lr()[0], "train/lambda_mv": result["lambda_mv"],
                          "train/grad_norm": result["grad_norm"], "health/step_seconds": elapsed,
                          "train/amp_scale": float(scaler.get_scale()),
                          "health/rgb_windows_per_second": config["effective_batch_rgb_windows"] / elapsed,
                          "progress/eta_seconds": elapsed * (target - step)}
                for prefix, terms in (("loss_raw", result["raw"]), ("loss_weighted", result["weighted"]), ("valid_count", result["counts"])):
                    values.update({f"train/{prefix}/{key}": value for key, value in terms.items()})
                selected_best, plot = False, None
                if step % validation_every == 0 or step == target:
                    evaluator = validation_fn or evaluate_visual
                    last_val, plot = evaluator(model, validation_dataset, config, device)
                    state["last_validation"] = last_val
                    selected_best = selector.consider(last_val, step)
                    values.update(last_val)
                    values["val/best_step"] = selector.best_step or 0
                    if selector.best_step is not None:
                        values["val/selection_score"] = next(c["score"] for c in selector.candidates if c["step"] == selector.best_step)
                    else:
                        values["val/selection_frozen"] = 0
                    if not injected:
                        plot = [visual_sample_plot(model, dataset[0], config, device, split="train"),
                                visual_sample_plot(model, validation_dataset[0], config, device, split="val")]
                if step % int(config.get("log_every_updates", 50)) == 0 or last_val and (step % validation_every == 0 or step == target) or step == 1:
                    values.update(visual_health(device, run_dir))
                    if metadata.get("checkpoint_seconds") is not None:
                        values["health/checkpoint_seconds"] = metadata["checkpoint_seconds"]
                    logging_rng = capture_training_rng(device)
                    logger.log({**values, "attempt_id": metadata["attempt_id"]}, plot=plot)
                    restore_training_rng(logging_rng, device)
                metadata.update(successful_updates=step, rgb_windows=state["rgb_windows"],
                                attempt_steps=state["attempt_steps"], selection=selector.state_dict(),
                                observability=logger.state_dict(), last_validation=last_val)
                if not injected and (step % checkpoint_every == 0 or step == target):
                    current, *_ = visual_fingerprints(config)
                    if any(current[k] != config[k] for k in current):
                        raise ValueError("input/source hash drift during training")
                stopping = stop_requested.is_set() or stop_after_updates is not None and step >= stop_after_updates
                snapshot = a2 and (step in config.get("snapshot_updates", []) or
                                   step in config.get("gradient_probe_updates", [100, 500, 1300, 2600, 3900, 5180]))
                if selected_best or step % checkpoint_every == 0 or step == target or stopping or snapshot:
                    saving_started = time.monotonic()
                    checkpoint = _visual_checkpoint(model, optimizer, scheduler, scaler, sampler, selector,
                                                    config, state, stats, logger, device)
                    if snapshot:
                        snap = run_dir / "snapshots" / ("step_%06d.pt" % step)
                        snapshot_hash = atomic_checkpoint(snap, checkpoint)
                        metadata.setdefault("snapshots", {})[str(step)] = {"path": str(snap), "sha256": snapshot_hash}
                    if selected_best:
                        metadata["best_checkpoint_hash"] = atomic_checkpoint(run_dir / "best.pt", checkpoint)
                    metadata["last_checkpoint_hash"] = atomic_checkpoint(run_dir / "last.pt", checkpoint)
                    metadata["checkpoint_seconds"] = time.monotonic()-saving_started
                atomic_json(run_dir / "run.json", metadata)
                if stopping and step < target:
                    metadata.update(state="paused", exit_code=0)
                    atomic_json(run_dir / "run.json", metadata)
                    return metadata
            if a2:
                from anysole.utils.a2_training import validate_exposure
                validate_exposure(state, config, terminal=True)
                atomic_json(run_dir / "exposure_ledger.json", state["exposure_ledger"])
            if state["rgb_windows"] != int(config["target_rgb_windows"]) or last_val is None and not metadata.get("last_validation"):
                raise ValueError("terminal successful budget or complete validation missing")
            if formal and (selector.best_step is None or not (run_dir / "best.pt").is_file()):
                raise ValueError("formal run has no qualified jointly selected checkpoint")
            metadata.update(state="completed" if formal else "pilot_completed", exit_code=0,
                            formal_complete=formal, finished_at=time.time(),
                            successful_updates=state["successful_updates"], rgb_windows=state["rgb_windows"],
                            attempt_steps=state["attempt_steps"],
                            last_validation=last_val, selection=selector.state_dict())
            atomic_json(run_dir / "run.json", metadata)
            return metadata
        except BaseException as exc:
            metadata.update(state="failed", exit_code=1, finished_at=time.time(),
                            error_type=type(exc).__name__, error=str(exc),
                            successful_updates=state["successful_updates"],
                            attempt_steps=state["attempt_steps"], rgb_windows=state["rgb_windows"])
            atomic_json(run_dir / "run.json", metadata)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if logger is not None:
                logger.close()


def _visual_main(args, raw_config):
    config = dict(raw_config)
    defaults = vars(parse_args([]))
    supported = {"config", "preflight_only", "run_dir", "resume", "stop_after_updates",
                 "device", "wandb_mode", "wandb_entity"}
    unsupported = [key for key, value in vars(args).items()
                   if key not in supported and value != defaults[key]]
    if unsupported:
        raise ValueError("grouped settings must be explicit in the hashed config; unsupported CLI overrides: " + ", ".join(unsupported))
    if args.preflight_only:
        report = visual_preflight(config)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 2
    if args.init_from is not None or args.epochs is not None or args.limit_sessions is not None:
        raise ValueError("grouped training cannot use warm-start, epoch or session truncation overrides")
    if args.device != "auto":
        config["device"] = args.device
    for key in ("wandb_mode", "wandb_entity"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    directory = args.run_dir or config.get("run_dir")
    if directory is None:
        raise ValueError("grouped training requires --run-dir")
    train_visual_run(config, directory, resume=args.resume, stop_after_updates=args.stop_after_updates)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    import yaml
    raw_config = yaml.safe_load(args.config.read_text()) or {}
    if raw_config.get("schema_version") == "view_consistency.v1":
        return _visual_main(args, raw_config)
    if args.preflight_only or args.run_dir is not None or args.resume is not None or args.stop_after_updates is not None:
        raise ValueError("these runtime controls require the view-consistency schema")
    config = load_config(args.config)
    config["input_adapter"] = "AnySole"
    config["adapter_version"] = ADAPTER_VERSION
    config["shared_schema_version"] = "shared.session.v1"
    config["split_hash"] = sha256_file(Path(config["split_csv"]))
    config["contact_method"] = str(config["contact_method"])
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = int(args.seed)
    seed = int(config.get("seed", 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("training seed: %d" % seed)
    if args.tw is not None:
        if args.tw < 1:
            raise ValueError("--tw must be positive")
        config["tw"] = int(args.tw)
    if args.dropout is not None:
        config["dropout"] = float(args.dropout)
    if args.config_probs is not None:
        config["config_probs"] = args.config_probs
    if args.dropout_vt is not None:
        tokens = [item for item in " ".join(args.dropout_vt).replace(",", " ").split() if item]
        if len(tokens) != 2:
            raise ValueError("--dropoutVT needs two percentages (e.g. '--dropoutVT 20 30' or '--dropoutVT 20,30')")
        drop_v, drop_t = float(tokens[0]) / 100.0, float(tokens[1]) / 100.0
        if drop_v < 0.0 or drop_t < 0.0 or drop_v + drop_t > 1.0:
            raise ValueError("--dropoutVT percentages must be non-negative and sum to at most 100")
        config["config_probs"] = [1.0 - drop_v - drop_t, drop_t, drop_v]
        config["dropout_vt"] = [drop_v * 100.0, drop_t * 100.0]
    if args.modal is not None:
        config["modal"] = args.modal
    if args.contact_method is not None:
        config["contact_method"] = args.contact_method
    if args.out_dir is not None:
        config["out_dir"] = str(args.out_dir)
    for key in ("wandb_mode", "wandb_project", "wandb_entity", "wandb_eval_interval", "wandb_experiment_tag"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    # Registered experiment names identify the model branch; the architecture
    # comes from the model's entry script (anysole/models/<module>.py), which
    # fixes the structure — structure flags are not a CLI choice (2026-09-24
    # independence restructure).  Every run starts from random init: model +
    # hyperparameters is the full identity of a run.
    entry = None
    if args.model_name is not None:
        config["modal"] = MODEL_ANYSOLEV2
        entry = get_builder(args.model_name)
        config["model_name"] = args.model_name
        config.update(entry.STRUCTURE)
        config.update(getattr(entry, "CONFIG_EXTRA", {}))
    if config.get("modal", MODEL_ANYSOLEV1) not in MODEL_NAMES:
        raise ValueError("Unknown modal %r; expected one of %s" % (config.get("modal"), MODEL_NAMES))
    if config.get("modal") == MODEL_ANYSOLEV2 and (
        args.tau_max is not None
        or args.tau_fixed is not None
        or bool(config.get("warm_start", False))
        or bool(config.get("continuation", False))
    ):
        # F0b: the regression model has no diffusion chain; these E-era knobs
        # would silently do nothing, so say so instead.
        print("WARNING: --tau-max/--tau-fixed/warm_start/continuation are diffusion-era "
              "knobs and are ignored by modal anysolev2 (regression)")

    session_ids = None
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_ids = load_split_ids(Path(config["split_csv"]), "train")[: args.limit_sessions]

    if args.tau_max is not None:
        if not 1 <= args.tau_max <= int(config["diffusion_train_steps"]):
            raise ValueError("--tau-max must be in [1, diffusion_train_steps]")
        config["tau_max"] = int(args.tau_max)
    if args.tau_fixed is not None:
        if not 0 <= args.tau_fixed < int(config["diffusion_train_steps"]):
            raise ValueError("--tau-fixed must be in [0, diffusion_train_steps)")
        config["tau_fixed"] = int(args.tau_fixed)
    if args.loss_cap is not None:
        if not args.loss_cap > 0.0:
            raise ValueError("--loss-cap must be positive")
        config["loss_cap"] = float(args.loss_cap)
    if args.grad_clip is not None:
        if not args.grad_clip > 0.0:
            raise ValueError("--grad-clip must be positive")
        config["grad_clip"] = float(args.grad_clip)
    if args.lr is not None:
        if not args.lr > 0.0:
            raise ValueError("--lr must be positive")
        config["lr"] = float(args.lr)
    if args.stride is not None:
        if not 1 <= args.stride <= int(config["tw"]):
            raise ValueError("--stride must be in [1, tw]")
        config["stride"] = int(args.stride)
    if args.part_json is not None:
        # V4B: learned hard partition (probe_part_cluster_b.py output or a
        # plain list of joint-index lists).  The partition is structural data
        # for the V4B entry; every other model rejects it.
        if args.model_name != "V4B":
            raise ValueError("--part-json is only valid for --model-name V4B")
        import json as _json
        part_json_path = Path(args.part_json)
        if not part_json_path.is_file():
            raise FileNotFoundError(
                "V4B partition file missing: %s\n"
                "Re-derive it (data-only, no trained model needed):\n"
                "  python z_note/probes/probe_part_cluster_b_data.py\n"
                "  -> results/AnySole/partitions/partitions_kinematic_b_K*.json\n"
                "then pass one of the exported candidates here."
                % part_json_path
            )
        doc = _json.loads(part_json_path.read_text())
        groups = doc["partition"] if isinstance(doc, dict) and "partition" in doc else doc
        if isinstance(groups, dict):
            # probe format: {"slot_xx": ["joint_name", ...]} -> index lists
            from anysole.types import JOINT_NAMES as _JN
            name_to_idx = {n: i for i, n in enumerate(_JN)}
            groups = [[name_to_idx[n] for n in names] for names in groups.values()]
        part_joints = tuple(tuple(int(j) for j in g) for g in groups)
        config["pose_parts"] = len(part_joints)
        config["part_joints"] = [list(g) for g in part_joints]
        print("part-json: %d groups %s" % (len(part_joints), [list(g) for g in part_joints]))
    if args.lambda_assign is not None:
        config["lambda_assign"] = float(args.lambda_assign)
    if args.lambda_sigma is not None:
        config["lambda_sigma"] = float(args.lambda_sigma)
    if args.sigma_freeze_frac is not None:
        config["sigma_freeze_frac"] = float(args.sigma_freeze_frac)
    if args.assign_cluster:
        # assign_cluster is part of the V4A entry's identity (its STRUCTURE
        # already carries it); the CLI flag stays only to pass the runtime
        # annealing/loss hyperparameters.
        if args.model_name != "V4A":
            raise ValueError("--assign-cluster is only valid for --model-name V4A "
                             "(its entry fixes 24 slots + soft A)")
        for key in ("lambda_assign_ent", "lambda_assign_conc", "lambda_assign_dead",
                    "assign_dead_beta", "assign_temp_init", "assign_temp_final",
                    "assign_anneal_frac", "assign_lock_frac", "assign_lr_mult"):
            config[key] = float(getattr(args, key))
        if float(config["lambda_assign"]) != 1.0:
            print("NOTE: V4A clustering expects --lambda-assign 1.0 (the cluster "
                  "terms carry their own --lambda-assign-* weights); got %s"
                  % config["lambda_assign"])
    # V4A: the anneal progress consumed by the cluster L_assign each step;
    # defaults to fully-annealed (1.0) for every non-cluster run.
    config.setdefault("assign_anneal", 1.0)
    if args.lr_warmup_frac is not None:
        config["lr_warmup_frac"] = float(args.lr_warmup_frac)
    for key in ("lambda_pose", "lambda_kp", "lambda_traj", "lambda_trec", "lambda_vrec",
                "lambda_con", "lambda_pose_vel", "lambda_bone"):
        value = getattr(args, key)
        if value is not None:
            config[key] = float(value)
    config["motion_protocol"] = MOTION_PROTOCOL
    config["motion_n_joints"] = int(N_JOINTS)
    config["motion_pose_dim"] = int(POSE_DIM)
    config["joint_protocol_checksum"] = JOINT_PROTOCOL_CHECKSUM

    try:
        dataset = AnySoleDataset(
            mode="train",
            seq_root=Path(config["seq_root"]),
            split_csv=Path(config["split_csv"]),
            cache_root=Path(config["cache_root"]),
            window_length=int(config["tw"]),
            session_ids=session_ids,
            contact_method=str(config["contact_method"]),
            stride=config.get("stride"),
            tactile_input=str(config.get("tactile_input", "raw108")),
            no_imu=bool(config.get("no_imu", False)),
            v_input=str(config.get("v_input", "hrnet")),
            f2_repr=bool(config.get("f2_repr", False)),
            smpl_roots=config.get("smpl_roots"),
            cam_id=int(config.get("cam_id", 3)),
        )
    except FileNotFoundError as exc:
        if "HRNet cache" in str(exc):
            raise FileNotFoundError(
                "%s Run `python -m anysole.data.extract_hrnet --cam-id %d`; "
                "training never extracts features on demand."
                % (exc, int(config["cam_id"]))
            ) from exc
        raise
    if len(dataset) == 0:
        raise RuntimeError("Training dataset contains no valid windows")

    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    try:
        val_dataset = AnySoleDataset(mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]), cache_root=Path(config["cache_root"]), window_length=int(config["tw"]), contact_method=str(config["contact_method"]), tactile_input=str(config.get("tactile_input", "raw108")), no_imu=bool(config.get("no_imu", False)), v_input=str(config.get("v_input", "hrnet")), f2_repr=bool(config.get("f2_repr", False)), smpl_roots=config.get("smpl_roots"), cam_id=int(config.get("cam_id", 3)))
        if len(val_dataset):
            val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), collate_fn=collate_windows, pin_memory=device.type == "cuda")
    except (FileNotFoundError, RuntimeError) as exc:
        print("validation disabled: %s" % exc)
    wandb_run = None
    if str(config.get("wandb_mode", "disabled")) != "disabled":
        try:
            import wandb
            # Local run data lives under results/logs/wandb (not a repo-root wandb/).
            os.environ.setdefault("WANDB_DIR", str(Path(os.environ.get("ANYSOLE_RESULTS", str(GAIT_ROOT / "results"))) / "logs" / "wandb"))
            # Identity fields consumed by Wandb_Analyzer: experiment_tag groups a
            # training batch, run_kind/model/stage/fold decide the raw/ directory
            # tree. Single-stage, non-cross-validation runs use stage=single,
            # fold=fold0.
            wandb_identity = {
                "experiment_tag": str(config.get("wandb_experiment_tag") or "anysole_v1"),
                "run_kind": "train",
                "model": str(config.get("modal", MODEL_ANYSOLEV1)),
                "stage": "single",
                "fold": "fold0",
            }
            wandb_run = wandb.init(
                mode=str(config["wandb_mode"]),
                project=str(config.get("wandb_project") or "Anysole"),
                entity=config.get("wandb_entity"),
                name="%s_%s_%s" % (wandb_identity["model"], wandb_identity["stage"], wandb_identity["fold"]),
                group="%s_%s" % (wandb_identity["experiment_tag"], wandb_identity["run_kind"]),
                job_type="train",
                tags=[wandb_identity["experiment_tag"], wandb_identity["run_kind"], wandb_identity["model"], wandb_identity["stage"], wandb_identity["fold"]],
                config={**dict(config), **wandb_identity},
            )
        except ImportError as exc:
            raise RuntimeError("W&B monitoring requested but wandb is not installed") from exc
    model_kw = {}
    modal = str(config.get("modal", MODEL_ANYSOLEV1))
    regress_mode = modal == MODEL_ANYSOLEV2
    # Keep model variants independent in the centralized results tree. An
    # explicit --out-dir remains authoritative for custom experiments.
    # Otherwise the dir is INFERRED (2026-09-22): model-name + contact-method
    # + stacked hyperparameter fields built from the effective config (or
    # pinned by --variant); every run trains from random init — no warm-start
    # source is ever inferred (--init-from = debug override only).
    if args.model_name is not None:
        contact_method = str(config["contact_method"])
        # Uniform hierarchy + complete hyperparameter record (2026-09-24):
        # the variant name = the full stacked fields of the effective config
        # (fixed order, no default omission) — every run gets its own dir,
        # including all-default ones; --variant pins it explicitly instead.
        variant = args.variant
        if variant is None:
            variant = variant_from_config(config)
        if variant is not None:
            config["variant"] = str(variant)
        inferred = infer_anysole_paths(args.model_name, variant=variant, contact=contact_method)
        if args.out_dir is None:
            config["out_dir"] = str(inferred["model_dir"] / "checkpoints")
        # Independence (2026-09-24): no warm-start is ever inferred — every
        # run starts from random init; --init-from remains a debug override.
    else:
        if args.out_dir is None:
            raise ValueError(
                "--out-dir omitted: pass --model-name (dir identifier, e.g. V3_4a) so the "
                "output dir is inferred as <model-name>_<contact>[/<stacked fields>]/checkpoints, "
                "or pass --out-dir explicitly for a custom experiment."
            )
    if modal == MODEL_ANYSOLEV2:
        # Registered models are built from their entry script — the structure
        # is fixed by the model name (anysole/models/<module>.py), and the
        # entry's STRUCTURE/CONFIG_EXTRA were recorded into config above.
        # Regression pose head, no diffusion pair in forward.
        if entry is None:
            raise ValueError("anysolev2 requires --model-name (entry script defines the structure)")
        if bool(config.get("use_insole_drift", False)):
            raise ValueError("anysolev2 does not support insole drift compensation")
        part_joints = tuple(tuple(int(j) for j in g) for g in config["part_joints"]) \
            if config.get("part_joints") else None
        model = entry.build(config, part_joints=part_joints).to(device)
    else:
        if modal == "anysolev1_insole_drift" or bool(config.get("use_insole_drift", False)):
            templates, subject_map = load_template_bank(config["template_path"])
            model_kw.update(templates=templates, subject_to_index=subject_map)
        model = AnySoleModel(
            d=int(config["d_model"]),
            tw=int(config["tw"]),
            dropout=float(config.get("dropout", 0.1)),
            modal=modal,
            use_insole_drift=bool(config.get("use_insole_drift", False)),
            pose_layers=int(config.get("pose_layers", 6)),
            tactile_input=str(config.get("tactile_input", "raw108")),
            tactile_direct=bool(config.get("tactile_direct", False)),
            no_imu=bool(config.get("no_imu", False)),
            v_input=str(config.get("v_input", "hrnet")),
            t_encoder=str(config.get("t_encoder", "linear")),
            f2_repr=bool(config.get("f2_repr", False)),
            **model_kw,
        ).to(device)
    if args.init_from is not None:
        # Debug-only warm-start (independent-models rule: never inferred).
        # Loaded before fitting pose stats, so the normalization buffers are
        # refit on the current dataset regardless of what the source
        # checkpoint carried.
        init_from_checkpoint(
            model, args.init_from,
            drop_prefixes=tuple(p for p in args.init_drop.split(",") if p),
        )
    # Frozen pose-normalization stats (recorded so eval/infer rebuild the same depth).
    config["pose_layers"] = int(config.get("pose_layers", 6))
    # E6.1: pose_repr follows the modal so losses/eval agree with the head.
    # F0b: anysolev2 has no position-space variant — the regression head is 6D only.
    if modal == MODEL_ANYSOLEV2:
        config["pose_repr"] = "6d"
    else:
        config["pose_repr"] = "pos" if modal == MODEL_ANYSOLEV1_POS else str(config.get("pose_repr", "6d"))
    pos_mode = config["pose_repr"] == "pos"
    if pos_mode and float(config.get("lambda_pose_vel", 0.0)) == 0.0 and float(config.get("lambda_bone", 0.0)) == 0.0:
        # Guard against repeating the 2026-09-16 misconfigured E6.2 launches:
        # three runs went out with the E6.2 losses at 0 (yaml default) and
        # were indistinguishable from E6.1.  E6.2 without these losses is
        # E6.1 - make that impossible to miss.
        print("=" * 79)
        print("WARNING: pos modal with lambda_pose_vel=0 AND lambda_bone=0")
        print("         this trains plain E6.1, not E6.2 (smooth/rigidity off).")
        print("         pass --lambda-pose-vel 10.0 --lambda-bone 1.0 for E6.2.")
        print("=" * 79)
    # noise_scaled is a config knob now: E3 base = False (unscaled N(0,1)
    # noise, the regime whose output was verified good), E4+ = True.  Saved in
    # every checkpoint so eval/infer match the sampling init scale.
    config["noise_scaled"] = bool(config.get("noise_scaled", True))
    model.noise_scaled = config["noise_scaled"]
    # Pose stats live on the pose head; guard kept defensively (the V3
    # structure always has one, but a missing head must fail loudly elsewhere,
    # not here).
    if model.pose_head is not None:
        pose_mean, pose_std = fit_pose_stats(
            loader, device, key="pose_gt_pos" if pos_mode else "pose_gt"
        )
        model.pose_head.set_stats(pose_mean, pose_std)
        print("pose stats: mean %.4f +/- %.4f, std %.4f +/- %.4f (dims floored at 1e-2: %d/%d)"
              % (pose_mean.mean().item(), pose_mean.std().item(), pose_std.mean().item(),
                 pose_std.std().item(), int((pose_std <= 1e-2).sum().item()), pose_mean.shape[0]))
    if bool(config.get("f2_repr", False)):
        # F2a: per-dim traj stats balance psi_dot / v_h / h inside the
        # trajectory loss; saved with the checkpoint for eval reference.
        config["traj_f2_stats"] = fit_traj_stats(loader, device)
        print("traj f2 stats: mean %s std %s"
              % (config["traj_f2_stats"]["mean"], config["traj_f2_stats"]["std"]))
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    # V4A clustering: part_logits (A) gets its own optimizer group with an lr
    # multiplier, so the assignment can move faster than the heads while they
    # co-train (soft EM).  The per-epoch/per-step lr setters below multiply
    # by each group's "assign_mult".
    assign_lr_mult = float(config.get("assign_lr_mult", 1.0))
    if assign_lr_mult != 1.0 and bool(config.get("soft_parts", False)):
        part_logits = model.pose_head.part_logits
        rest = [p for p in model.parameters() if p is not part_logits]
        optimizer = torch.optim.Adam(
            [
                {"params": rest, "assign_mult": 1.0},
                {"params": [part_logits], "assign_mult": assign_lr_mult},
            ],
            lr=float(config["lr"]),
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    warn_legacy_output_checkpoint(out_dir)

    # V3 structural steps (fix_plan_v3 §2.2): linear LR warmup over the first
    # lr_warmup_frac of the total planned steps, multiplying the epoch cosine
    # LR. 0/absent = no warmup (F0b/F4a behavior).
    lr_warmup_frac = float(config.get("lr_warmup_frac", 0.0))
    warmup_steps = 0
    if lr_warmup_frac > 0:
        warmup_steps = int(lr_warmup_frac * int(config["epochs"]) * len(loader))
        print("lr warmup: %.4f x %d epochs x %d steps/epoch = %d steps"
              % (lr_warmup_frac, int(config["epochs"]), len(loader), warmup_steps))

    global_step = 0
    # V4A clustering schedule (方案 A): temperature/entropy anneal linearly
    # over the first assign_anneal_frac of total steps, then A is frozen for
    # the last (1 - assign_lock_frac) of the run so the heads settle against
    # a fixed grouping.  Non-cluster runs keep the defaults (temp 1.0, no
    # lock) so V3-3 behavior is byte-identical.
    cluster_mode = bool(config.get("assign_cluster", False))
    if cluster_mode:
        total_steps = int(config["epochs"]) * len(loader)
        anneal_steps = int(float(config.get("assign_anneal_frac", 0.7)) * total_steps)
        lock_steps = int(float(config.get("assign_lock_frac", 0.7)) * total_steps)
        temp_init = float(config.get("assign_temp_init", 1.0))
        temp_final = float(config.get("assign_temp_final", 0.2))
        print("V4A cluster schedule: anneal %d steps (temp %.2f -> %.2f, entropy -> 0), "
              "lock A after step %d, A lr mult %.1f"
              % (anneal_steps, temp_init, temp_final, lock_steps, assign_lr_mult))
    else:
        anneal_steps = lock_steps = 0
        temp_init = temp_final = 1.0
    # Best-ckpt tracking (loss-explosion protection): the val eval cadence is
    # the only visibility into val quality, so the best-so-far model by
    # val/tau0_mpjpe/VT is snapshotted to ckpt_best.pt — a finite-gradient
    # explosion (F0b_warm ep370 / V3_2_part9 ep271, no nonfinite guard fires)
    # can destroy ckpt_last.pt in ~10 epochs; the snapshot survives.
    # V3-3 sigma freeze schedule (plan §3, "first sigma_freeze_frac sigma
    # fixed"): sigma stays at 0 while the tau_p prior bias shapes the gate,
    # then comes online for the remainder of the run.
    if str(config.get("gate", "none")) != "none":
        sigma_freeze_steps = int(
            float(config.get("sigma_freeze_frac", 0.1))
            * len(loader) * int(config["epochs"])
        )
    else:
        sigma_freeze_steps = 0
    best_vt = None
    for epoch in range(int(config["epochs"])):
        model.train()
        # E4 cosine LR decay over the whole run; E3 base uses lr_schedule
        # "constant" (E3's regime, whose output was verified good).
        if str(config.get("lr_schedule", "cosine")) == "constant":
            epoch_lr = float(config["lr"])
        else:
            progress = epoch / float(config["epochs"])
            epoch_lr = float(config["lr"]) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr * float(group.get("assign_mult", 1.0))
        epoch_start_step = global_step
        nonfinite_skips = 0
        epoch_sums = {"loss_total": 0.0, "loss_pose": 0.0, "loss_traj": 0.0, "loss_con": 0.0, "loss_kp": 0.0, "loss_pose_vel": 0.0, "loss_bone": 0.0, "loss_assign": 0.0, "loss_sigma": 0.0, "loss_Trec_Tmissing": 0.0, "loss_Vrec_Vmissing": 0.0, "grad_norm": 0.0, "n": 0, "n_tmissing": 0, "n_vmissing": 0}
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            batch_size = batch["pose_gt"].shape[0]
            config_id = sample_config_ids(batch_size, probs=config["config_probs"]).to(device)
            batch["config_id"] = config_id
            assert_batch_shapes(batch, batch_size, tw=int(config["tw"]))
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
            # F1: hmr_gvhmr checkpoints consume the GVHMR channel; V1 models
            # have no V_hmr parameter.
            v_kw = {"V_hmr": batch.get("V_hmr")} if str(config.get("v_input", "hrnet")) == "hmr_gvhmr" else {}

            if cluster_mode:
                # V4A per-step schedule: anneal the assignment temperature
                # (soft -> sharp) and the L_assign entropy scale, and freeze
                # part_logits (A) after the lock point (the M step).
                anneal_prog = min(1.0, global_step / float(max(anneal_steps, 1)))
                model.pose_head.set_assign_temp(temp_init + (temp_final - temp_init) * anneal_prog)
                config["assign_anneal"] = 1.0 - anneal_prog
                if lock_steps > 0:
                    model.pose_head.part_logits.requires_grad_(global_step < lock_steps)

            optimizer.zero_grad(set_to_none=True)
            if sigma_freeze_steps:
                model.pose_head.set_sigma_frozen(global_step < sigma_freeze_steps)
            if regress_mode:
                # F0b: regression head — no tau sampling, no q_sample, no
                # x_tau. L_pose applies directly to the regression output
                # (losses.py unchanged).
                out = model(v_feat, t_raw, t_phys, config_id, batch.get("session_id"), T_s2m=t_s2m, **v_kw)
            else:
                # Test11 τ training-regime knobs: --tau-fixed pins every step to one
                # noise level (0 = identity-mapping training), --tau-max narrows the
                # sampling band to U(0, tau_max) (low-noise region training).
                if "tau_fixed" in config:
                    tau = torch.full(
                        (batch_size,),
                        int(config["tau_fixed"]),
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    tau = torch.randint(
                        0,
                        int(config.get("tau_max", config["diffusion_train_steps"])),
                        (batch_size,),
                        device=device,
                        dtype=torch.long,
                    )
                # E4: diffuse with per-dim noise scaled by the pose std (Step2Motion
                # normalizes the data before diffusing; scaling the noise is the
                # same thing here since the head normalizes its input internally).
                # Without this, N(0,1) noise lands on a signal of std ~0.118 (8.5x
                # SNR mismatch) and the DDIM chain injects per-frame jitter.
                pose_target = batch["pose_gt_pos"] if pos_mode else batch["pose_gt"]
                noise = torch.randn_like(pose_target)
                if model.noise_scaled:
                    noise = noise * model.pose_head.pose_std.view(1, 1, -1)
                x_tau = diffusion.q_sample(pose_target, tau, noise=noise)
                out = model(v_feat, t_raw, t_phys, x_tau, tau, config_id,
                            batch.get("session_id"), T_s2m=t_s2m, **v_kw)
            losses = compute_losses(out, batch, config_id, config)

            finite = torch.isfinite(losses["loss"]) and torch.isfinite(out["F"]).all()
            if not bool(finite):
                for group in optimizer.param_groups:
                    group["lr"] = 1.0e-4
                optimizer.zero_grad(set_to_none=True)
                print("non-finite fusion/loss at step %d; lr set to 1e-4, batch skipped" % global_step)
                nonfinite_skips += 1
                global_step += 1
                continue

            losses["loss"].backward()
            # NaN-grad guard (fix_plan_v3 §2.2): check every parameter BEFORE
            # the clip, because clip_grad_norm_ mixes one non-finite gradient
            # into every parameter (F5B_gated failure mode).  Same treatment
            # as the loss-level guard above: skip the step, drop the lr.
            grads_ok = all(
                p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            )
            if not grads_ok:
                for group in optimizer.param_groups:
                    group["lr"] = 1.0e-4
                optimizer.zero_grad(set_to_none=True)
                print("non-finite gradient at step %d; lr set to 1e-4, batch skipped" % global_step)
                nonfinite_skips += 1
                global_step += 1
                continue
            grad_clip = config.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            grad_sq = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_sq = grad_sq + parameter.grad.detach().square().sum()
            epoch_sums["grad_norm"] += float(torch.sqrt(grad_sq).item()) * batch_size
            if warmup_steps > 0:
                # Per-step linear ramp over the cosine epoch LR (the epoch
                # value is set on the groups at the top of the loop).
                ramp = min(1.0, global_step / float(warmup_steps))
                for group in optimizer.param_groups:
                    group["lr"] = epoch_lr * ramp * float(group.get("assign_mult", 1.0))
            optimizer.step()
            n = batch_size
            epoch_sums["n"] += n
            for key, name in (("loss", "loss_total"), ("L_pose", "loss_pose"), ("L_traj", "loss_traj"), ("L_con", "loss_con"), ("L_kp", "loss_kp"), ("L_pose_vel", "loss_pose_vel"), ("L_bone", "loss_bone"), ("L_assign", "loss_assign"), ("L_sigma", "loss_sigma")):
                epoch_sums[name] += float(losses[key].detach().item()) * n
            n_tmissing = int((config_id == CONFIG_T).sum().item())
            n_vmissing = int((config_id == CONFIG_V).sum().item())
            if n_tmissing:
                epoch_sums["loss_Trec_Tmissing"] += float(losses["L_Trec"].detach().item()) * n_tmissing
                epoch_sums["n_tmissing"] += n_tmissing
            if n_vmissing:
                epoch_sums["loss_Vrec_Vmissing"] += float(losses["L_Vrec"].detach().item()) * n_vmissing
                epoch_sums["n_vmissing"] += n_vmissing
            if global_step % 50 == 0:
                values = " ".join("%s=%.6f" % (key, value.detach().item()) for key, value in losses.items())
                print("epoch=%d step=%d %s" % (epoch + 1, global_step, values))
            global_step += 1

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": dict(config),
        }
        # V4A: assign_anneal is per-step scratch for the L_assign entropy
        # scale, not a static config value — don't persist it.
        checkpoint["config"].pop("assign_anneal", None)
        # Atomic replace: a reader (auto-eval torch.load) must never see a
        # half-written file when several runs share one out_dir.  Each process
        # writes its own temp file, so a concurrent save cannot corrupt it.
        # Re-mkdir so an externally deleted results dir cannot kill a long run.
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "ckpt_last.pt"
        tmp_path = out_dir / ("ckpt_last.pt.tmp%d" % os.getpid())
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, ckpt_path)
        print("saved %s (epoch %d)" % (ckpt_path, epoch + 1))
        # Val eval cadence is the wall-clock knob: one _evaluate pass (3
        # condition configs x full val set, DDIM sampling) costs ~16 normal
        # epochs of wall-clock.  Triggered by --wandb_eval_interval but runs
        # regardless of wandb mode; results go to the console and, when a
        # wandb run is active, to its log.
        eval_results = None
        eval_interval = max(int(config.get("wandb_eval_interval", 5)), 1)
        if val_loader is not None and (epoch + 1) % eval_interval == 0:
            model.eval()
            eval_results = _evaluate(model, diffusion, val_loader, config, device)
            print("epoch %d eval: %s" % (epoch + 1,
                  "  ".join("%s=%.4f" % (key, value) for key, value in sorted(eval_results.items()))))
            vt_now = eval_results.get("val/tau0_mpjpe/VT")
            if vt_now is not None and (best_vt is None or vt_now < best_vt):
                best_vt = vt_now
                best_ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "config": dict(config),
                    "best_vt": best_vt,
                }
                best_path = out_dir / "ckpt_best.pt"
                best_tmp = out_dir / ("ckpt_best.pt.tmp%d" % os.getpid())
                torch.save(best_ckpt, best_tmp)
                os.replace(best_tmp, best_path)
                print("saved best %s (epoch %d, val/tau0 VT %.4f)" % (best_path, epoch + 1, best_vt))
        if wandb_run is not None:
            denom = max(epoch_sums["n"], 1)
            log = {"epoch": epoch + 1}
            log["train/global_step"] = global_step
            log["train/steps_per_epoch"] = global_step - epoch_start_step
            log["train/nonfinite_skips"] = nonfinite_skips
            for key, value in epoch_sums.items():
                if key in ("n", "n_tmissing", "n_vmissing"):
                    continue
                metric_denom = denom
                if key == "loss_Trec_Tmissing": metric_denom = max(epoch_sums["n_tmissing"], 1)
                if key == "loss_Vrec_Vmissing": metric_denom = max(epoch_sums["n_vmissing"], 1)
                log[f"train/{key}"] = value / metric_denom
            log["train/lr"] = optimizer.param_groups[0]["lr"]
            # V3-3 gate probes (plan §2.3 F4): per-slot mean [g_V, g_T, g_∅]
            # from the last decoder layer, under wandb prefix "gate/".  The
            # assignment matrix itself stays auditable via the smoke/probe
            # scripts off the checkpoint's part_logits parameter.
            if str(config.get("gate", "none")) != "none":
                summary = model.pose_head.gate_summary()  # (N_PARTS, 3)
                for p, pname in enumerate(PART_NAMES):
                    log["gate/%s_gV" % pname] = float(summary[p, 0])
                    log["gate/%s_gT" % pname] = float(summary[p, 1])
                    log["gate/%s_gE" % pname] = float(summary[p, 2])
            # V4A cluster probes: per-slot loads + effective K so the grouping
            # emergence is watchable live (wandb prefix "assign/").  The final
            # grouping itself is audited off the checkpoint via
            # z_note/probes/probe_v4a_readout.py.
            if cluster_mode:
                assign = model.pose_head.assignment().detach()
                loads = assign.sum(dim=1)
                log["assign/effective_K_ge0.5"] = float((loads >= 0.5).sum())
                log["assign/effective_K_ge1.0"] = float((loads >= 1.0).sum())
                log["assign/max_load"] = float(loads.max())
                log["assign/mean_entropy"] = float(
                    -(assign * (assign + 1e-12).log()).sum(dim=0).mean()
                )
                for k in range(int(loads.shape[0])):
                    log["assign/load_%02d" % k] = float(loads[k])
            # Optional capped logging (--loss-cap): the first epochs spike far
            # above the converged range (loss_total 30 -> 0.1, loss_pose 1.5 ->
            # 0.01), which stretches the raw charts' y-axis and hides
            # convergence.  When set, clip the train losses in place under
            # their original names so the wandb view stays within 0-cap and
            # Wandb_Analyzer picks up the clipped series directly.
            loss_cap = config.get("loss_cap")
            if loss_cap is not None:
                for key in ("loss_total", "loss_pose", "loss_traj", "loss_con", "loss_kp",
                            "loss_Trec_Tmissing", "loss_Vrec_Vmissing", "grad_norm"):
                    full_key = "train/%s" % key
                    if full_key in log:
                        log[full_key] = min(float(log[full_key]), float(loss_cap))
            if eval_results is not None:
                log.update(eval_results)
            wandb_run.log(log, step=epoch + 1)
    if wandb_run is not None:
        wandb_run.finish()
    try:
        from anysole.eval import main as eval_main
        # Training-time auto-eval stays lean: metrics only and no protocol pass.
        eval_main(["--config", str(args.config), "--ckpt", str(out_dir / "ckpt_last.pt"),
                   "--split", "test", "--device", str(device),
                   "--contact-method", str(config["contact_method"]), "--no-write-motion"])
    except Exception as exc:
        print("automatic test evaluation failed: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
