"""Train AnySole V1."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from anysole.data.dataset import (
    AnySoleDataset,
    collate_windows,
    load_split_ids,
    sample_config_ids,
)
from anysole.diffusion import GaussianDiffusion
from anysole.losses import compute_losses
from anysole.models import AnySoleModel, AnySoleModelV2, MODEL_NAMES, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_POS, MODEL_ANYSOLEV2
from anysole.types import CONFIG_PROBS, CONFIG_T, CONFIG_V, GAIT_ROOT, POSE_DIM, T_S2M_DIM, anysole_model_dir, assert_batch_shapes
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.geometry import f2_to_world, fk_pose6d
from anysole.losses import soft_contact_from_keypoints
from anysole.types import CONFIG_NAMES


DEFAULT_CONFIG = {
    "d_model": 256,
    "dropout": 0.1,
    "tw": 20,
    "batch_size": 256,
    "lr": 1.0e-3,
    "epochs": 200,
    "num_workers": 4,
    "diffusion_train_steps": 1000,
    "diffusion_sample_steps": 50,
    "pose_layers": 6,
    # ---- E6.x 系列开关（默认值保持既有行为；E3 基础用 noise_scaled=false +
    # lr_schedule=constant，详见 model_fix_note/E6x_series_plan.md）----
    "pose_repr": "6d",          # E6.1: "pos" = 根局部位置表示
    "stride": None,             # E6.3: 训练窗 stride（None = 非重叠）
    "lambda_pose_vel": 0.0,     # E6.2: 帧间平滑损失权重（0 = 关闭）
    "lambda_bone": 0.0,         # E6.2: 骨长刚性损失权重（0 = 关闭，仅 pos 模式）
    "continuation": False,      # E6.4: eval/infer 续写式推理
    "warm_start": False,        # E6.5: 采样起点对齐训练边缘分布
    "noise_scaled": True,       # E4 起的噪声缩放；E3 基础 = False
    "lr_schedule": "cosine",    # E4 起的余弦衰减；E3 基础 = "constant"
    "lambda_pose": 1.0,
    "lambda_traj": 1.0,
    "lambda_trec": 0.1,
    "lambda_vrec": 0.1,
    "lambda_kp": 1.0,
    "lambda_con": 0.1,
    "traj_velocity_w": 1.0,
    "traj_delta_w": 1.0,
    "traj_delta_weight_power": 1.0,
    "traj_deltas": [2, 4, 8, 19],
    "config_probs": list(CONFIG_PROBS),
    "cam_id": 3,
    "seq_root": "/data/fangyuxuan/projects/gait/AnysoleWorkspace/derived/MotionPRO/sequences/cam3",
    "split_csv": "/data/fangyuxuan/projects/gait/AnysoleWorkspace/splits/default/splits.csv",
    "cache_root": "/data/fangyuxuan/projects/gait/AnysoleWorkspace/derived/AnySole/hrnet_cache/cam3",
    "out_dir": "/data/fangyuxuan/projects/gait/AnySole/outputs/v1",
    "use_insole_drift": False,
    "contact_method": "tactile_abs",
    "mocap_format": "smpl",
    "smpl_roots": ["/data/lizhe/projects/Tactile/Mocap/0804", "/data/lizhe/projects/Tactile/Mocap/0807", "/data/lizhe/projects/Tactile/Mocap/0808", "/data/lizhe/projects/Tactile/Mocap/0810"],
    "modal": MODEL_ANYSOLEV1,
    "template_path": "/data/fangyuxuan/projects/gait/AnysoleWorkspace/calibration/insole_templates.json",
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
    src = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(src, dict) or "model" not in src:
        raise ValueError("--init-from checkpoint has no 'model' state dict: %s" % path)
    src = src["model"]
    dst = model.state_dict()
    loaded, skipped = [], []
    for key, value in src.items():
        if any(key.startswith(p) for p in drop_prefixes) or key not in dst or dst[key].shape != value.shape:
            skipped.append(key)
            continue
        dst[key] = value.to(dst[key].dtype)
        loaded.append(key)
    model.load_state_dict(dst)
    print("init-from %s: %d/%d keys copied%s" % (
        path, len(loaded), len(dst),
        " (skipped: %s)" % ", ".join(skipped) if skipped else ""))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AnySole V1.")
    parser.add_argument("--config", type=Path, default=GAIT_ROOT / "configs" / "v1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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
        "--tactile-input",
        choices=("raw108", "s2m50"),
        default=None,
        help="E6.6a: tactile encoder input. raw108 = cat([T_raw, T_phys]) (E3); "
        "s2m50 = Step2Motion-口径 50-dim channel (16 pooled pressure + synthesized "
        "IMU + force + CoP, anysole/data/tactile_s2m.py). Saved into the checkpoint.",
    )
    parser.add_argument(
        "--t-encoder",
        choices=("linear", "foot_conv"),
        default=None,
        help="F4a: tactile stream encoder (raw108 only). linear = flat "
        "LinearTemporalEncoder (E3 口径, default); foot_conv = FootConvEncoder — "
        "per-foot 4×12 grid conv (right foot mirrored), [left/right/global] "
        "tokens + temporal transformer. The input DATA is unchanged (108-dim); "
        "only the encoding changes (anysole/models/foot_encoder.py). "
        "Saved into the checkpoint.",
    )
    parser.add_argument(
        "--tactile-direct",
        action="store_true",
        default=None,
        help="E6.6b: per-group tactile encoder (TactileEncoder) + direct pose-head "
        "cross-attention to cat([F, t_tok]). Requires --tactile-input s2m50.",
    )
    parser.add_argument(
        "--no-imu",
        action="store_true",
        default=None,
        help="E6.8: delete the synthesized IMU channels (acc3+gyro3 per foot) from "
        "the s2m50 tactile input — the data is built as 38-dim (pressure16+force+CoP "
        "per foot, no IMU values computed or stored) and the encoder's IMU groups are "
        "removed. Requires --tactile-input s2m50. Saved into the checkpoint.",
    )
    parser.add_argument(
        "--v-input",
        choices=("hrnet", "hmr_gvhmr"),
        default=None,
        help="F1: visual encoder input. hrnet = V_feat (HRNet 2048 + CLIFF bbox, "
        "E3 口径); hmr_gvhmr = V_hmr (1156-dim GVHMR channel: aa body_pose 63 + "
        "global_orient 3 + betas 10 + COCO-17 kp2d 51 + HMR2 img features 1024 + "
        "bbox/q 5). Requires the hmr cache (anysole/data/extract_hmr.py). "
        "Saved into the checkpoint.",
    )
    parser.add_argument(
        "--mocap-format", choices=("smpl", "bvh"), default=None,
        help="Ground-truth motion source. smpl (default) reads motion_neutral_smpl.npz; "
        "bvh preserves the legacy Skeleton3 baseline path. Saved into the checkpoint.",
    )
    parser.add_argument(
        "--f2-repr",
        action="store_true",
        default=None,
        help="F2a: heading/tilt representation. pose target root 6D = tilt "
        "(yaw removed), trajectory target = 4-dim [psi_dot, v_hx, v_hz, h] "
        "(geometry.py f2_to_world recovers the world pose/trans). The traj "
        "head outputs 4-dim; traj stats are fitted on the training set and "
        "saved into the checkpoint. Saved into the checkpoint.",
    )
    parser.add_argument(
        "--pose-parts",
        choices=(3, 9),
        type=int,
        default=None,
        help="V3-2: pose-head query granularity. 3 = 23 joint queries in body/"
        "left/right groups (F0b baseline); 9 = one query + one unembed head "
        "per part (PART_JOINTS, fix_plan_v3.md §V3-2). Saved into the checkpoint.",
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
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None, help="Model variant to train.")
    parser.add_argument(
        "--contact-method",
        default=None,
        help="Test5 contact-label scheme for contact_gt (see results_display/README.md). "
        "Default: config contact_method, falling back to tactile_abs (原 contact.npy).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Override checkpoint output directory.")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="F0b fast track: warm-start from an existing checkpoint. "
                             "Only state-dict keys whose names AND shapes match the current "
                             "model are copied (everything else keeps its fresh init), so "
                             "e.g. an E3 anysolev1 ckpt can seed anysolev2 — the V1 pose-head "
                             "diffusion projections (proj_*/timestep_token) have no V2 "
                             "counterpart and are skipped automatically. --init-drop adds "
                             "explicit prefixes to skip.")
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
                    # E6.1: compare in the metric space directly (22 joints,
                    # root-local; root covered by root_ate below).
                    err = torch.linalg.vector_norm(
                        pred_pose.reshape(bsz, -1, 22, 3)
                        - batch["pose_gt_pos"].reshape(bsz, -1, 22, 3),
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
                            out0["x0_hat"].reshape(bsz, -1, 22, 3)
                            - batch["pose_gt_pos"].reshape(bsz, -1, 22, 3),
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
                    pred_contact = soft_contact_from_keypoints(pred_kp) > 0.5
                    gt_contact = batch["contact_gt"] > 0.5
                    tp = (pred_contact & gt_contact).sum().item()
                    fp = (pred_contact & ~gt_contact).sum().item()
                    fn = (~pred_contact & gt_contact).sum().item()
                    totals.setdefault("tp", 0); totals.setdefault("fp", 0); totals.setdefault("fn", 0)
                    totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn
                    foot = pred_kp[:, :, (17, 18, 21, 22), :][:, :, :, [0, 2]]
                    # Keypoints are meters; report horizontal contact speed in cm/s.
                    speed = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1).mean(dim=-1) * 40.0 * 100.0
                    contact_frames = pred_contact[:, 1:].any(dim=-1)
                    slide = speed[contact_frames]
                    if slide.numel():
                        totals["foot_slide"] += float(slide.mean().item()) * n
                totals["n"] += n
            count = max(totals["n"], 1)
            p = totals.get("tp", 0) / max(totals.get("tp", 0) + totals.get("fp", 0), 1)
            r = totals.get("tp", 0) / max(totals.get("tp", 0) + totals.get("fn", 0), 1)
            if regress_mode:
                # F0b: no denoising chain — the direct regression output is
                # both the final prediction and the "clean-input" value.
                tau0_mpjpe = totals["mpjpe"] / count
            result.update({f"val/mpjpe/{config_name}": totals["mpjpe"] / count,
                           f"val/root_ate/{config_name}": totals["root_ate"] / count,
                           f"val/contact_f1/{config_name}": 2 * p * r / max(p + r, 1e-8),
                           f"val/foot_slide/{config_name}": totals["foot_slide"] / count,
                           f"val/tau0_mpjpe/{config_name}": tau0_mpjpe})
    result["val/gap_mpjpe_dropT"] = result["val/mpjpe/V"] - result["val/mpjpe/VT"]
    result["val/gap_mpjpe_dropV"] = result["val/mpjpe/T"] - result["val/mpjpe/VT"]
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
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
    if args.stride is not None:
        if not 1 <= args.stride <= int(config["tw"]):
            raise ValueError("--stride must be in [1, tw]")
        config["stride"] = int(args.stride)
    if args.tactile_input is not None:
        config["tactile_input"] = args.tactile_input
    if args.t_encoder is not None:
        config["t_encoder"] = args.t_encoder
    if args.v_input is not None:
        config["v_input"] = args.v_input
    if args.mocap_format is not None:
        config["mocap_format"] = args.mocap_format
    if args.f2_repr:
        config["f2_repr"] = True
    if args.pose_parts is not None:
        config["pose_parts"] = int(args.pose_parts)
    if args.lr_warmup_frac is not None:
        config["lr_warmup_frac"] = float(args.lr_warmup_frac)
    if args.tactile_direct:
        config["tactile_direct"] = True
    if args.no_imu:
        if str(config.get("tactile_input", "raw108")) != "s2m50":
            raise ValueError("--no-imu requires --tactile-input s2m50 (raw108 has no IMU channels)")
        config["no_imu"] = True
    for key in ("lambda_pose", "lambda_kp", "lambda_traj", "lambda_trec", "lambda_vrec",
                "lambda_con", "lambda_pose_vel", "lambda_bone"):
        value = getattr(args, key)
        if value is not None:
            config[key] = float(value)

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
            no_imu=bool(config.get("no_imu", False)),
            v_input=str(config.get("v_input", "hrnet")),
            f2_repr=bool(config.get("f2_repr", False)),
            mocap_format=str(config.get("mocap_format", "smpl")),
            smpl_roots=config.get("smpl_roots"),
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
        val_dataset = AnySoleDataset(mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]), cache_root=Path(config["cache_root"]), window_length=int(config["tw"]), contact_method=str(config["contact_method"]), no_imu=bool(config.get("no_imu", False)), v_input=str(config.get("v_input", "hrnet")), f2_repr=bool(config.get("f2_repr", False)), mocap_format=str(config.get("mocap_format", "smpl")), smpl_roots=config.get("smpl_roots"))
        if len(val_dataset):
            val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), collate_fn=collate_windows, pin_memory=device.type == "cuda")
    except (FileNotFoundError, RuntimeError) as exc:
        print("validation disabled: %s" % exc)
    wandb_run = None
    if str(config.get("wandb_mode", "disabled")) != "disabled":
        try:
            import wandb
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
    if args.out_dir is None:
        # Model dir naming: anysole_{version/ablation}_{contact_method}. Every
        # contact-label scheme gets its own dir so sweeps never clobber each
        # other's checkpoints/metrics.
        contact_method = str(config.get("contact_method", "tactile_abs"))
        config["out_dir"] = str(anysole_model_dir(modal, contact_method) / "checkpoints")
    if modal == MODEL_ANYSOLEV2:
        # F0b: regression model (model_v2.py) — V1 structure, regress pose
        # head, no diffusion pair in forward.
        if bool(config.get("use_insole_drift", False)):
            raise ValueError("anysolev2 does not support insole drift compensation")
        model = AnySoleModelV2(
            d=int(config["d_model"]),
            tw=int(config["tw"]),
            dropout=float(config.get("dropout", 0.1)),
            pose_layers=int(config.get("pose_layers", 6)),
            tactile_input=str(config.get("tactile_input", "raw108")),
            tactile_direct=bool(config.get("tactile_direct", False)),
            no_imu=bool(config.get("no_imu", False)),
            v_input=str(config.get("v_input", "hrnet")),
            t_encoder=str(config.get("t_encoder", "linear")),
            f2_repr=bool(config.get("f2_repr", False)),
            pose_parts=int(config.get("pose_parts", 3)),
        ).to(device)
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
            **model_kw,
        ).to(device)
    if args.init_from is not None:
        # F0b fast track: warm-start before fitting pose stats, so the
        # normalization buffers are refit on the current dataset regardless
        # of what the source checkpoint carried.
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
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

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
    # Best-ckpt tracking (loss-explosion protection): the val eval cadence is
    # the only visibility into val quality, so the best-so-far model by
    # val/tau0_mpjpe/VT is snapshotted to ckpt_best.pt — a finite-gradient
    # explosion (F0b_warm ep370 / V3_2_part9 ep271, no nonfinite guard fires)
    # can destroy ckpt_last.pt in ~10 epochs; the snapshot survives.
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
            group["lr"] = epoch_lr
        epoch_start_step = global_step
        nonfinite_skips = 0
        epoch_sums = {"loss_total": 0.0, "loss_pose": 0.0, "loss_traj": 0.0, "loss_con": 0.0, "loss_kp": 0.0, "loss_pose_vel": 0.0, "loss_bone": 0.0, "loss_Trec_Tmissing": 0.0, "loss_Vrec_Vmissing": 0.0, "grad_norm": 0.0, "n": 0, "n_tmissing": 0, "n_vmissing": 0}
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

            optimizer.zero_grad(set_to_none=True)
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
                out = model(v_feat, t_raw, t_phys, x_tau, tau, config_id, batch.get("session_id"), T_s2m=t_s2m)
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
                    group["lr"] = epoch_lr * ramp
            optimizer.step()
            n = batch_size
            epoch_sums["n"] += n
            for key, name in (("loss", "loss_total"), ("L_pose", "loss_pose"), ("L_traj", "loss_traj"), ("L_con", "loss_con"), ("L_kp", "loss_kp"), ("L_pose_vel", "loss_pose_vel"), ("L_bone", "loss_bone")):
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
        # Training-time auto-eval stays lean: metrics only, no BVH export and
        # no F0a protocol pass (run eval again later to refresh
        # predictions/eval_bvh and the fseries protocol JSON).
        eval_main(["--config", str(args.config), "--ckpt", str(out_dir / "ckpt_last.pt"),
                   "--modal", modal, "--split", "test", "--device", str(device),
                   "--contact-method", str(config["contact_method"]), "--no-write-bvh",
                   "--no-protocol"])
    except Exception as exc:
        print("automatic test evaluation failed: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
