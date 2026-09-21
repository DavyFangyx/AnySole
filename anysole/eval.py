"""Evaluate AnySole V1 under VT, V-only, and T-only conditioning.

Besides the motion metrics, every mode also reports tactile aux-head
reconstruction quality (T_mae / T_rmse / T_corr over the 96 cells).  Under
V-only conditioning the tactile input is zeroed, so that row measures V2T
(vision-to-tactile) generation and is mirrored by the top-level "v2t"
field of the metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.data.smpl_io import pelvis_to_smpl_trans, smpl24_pose6d_to_poses, smpl_archive_metadata
from anysole.diffusion import GaussianDiffusion
from anysole.geometry import f2_to_world, fk_pose6d, positions_to_6d_np, rot6d_to_rotmat, rot6d_to_rotmat_np, rotmat_to_6d, rotmat_to_6d_np
from anysole.losses import soft_contact_from_keypoints
from anysole.models import AnySoleModel, AnySoleModelV2, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT, MODEL_ANYSOLEV1_POS, MODEL_ANYSOLEV2, MODEL_NAMES
from anysole.train import condition_inputs, load_config, move_batch, resolve_device
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.types import (
    ANYSOLE_ROOT,
    CONFIG_MODE_NAMES,
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    DEFAULT_CONFIG_PATH,
    FPS,
    GAIT_ROOT,
    JOINT_PROTOCOL_CHECKSUM,
    MOTION_PROTOCOL,
    N_JOINTS,
    POSE_DIM,
    anysole_model_dir,
    assert_batch_shapes,
)

# Quasi-strict load allowance (fix_plan_v3 §3.1): state-dict keys added as
# code hygiene AFTER the baseline checkpoints were trained.  Only these may
# be missing when loading F4a/F2p4-era checkpoints.
_MISSING_OK = frozenset({
    "encoders.t_enc.stream_norm.weight",
    "encoders.t_enc.stream_norm.bias",
})


def _pa_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-joint errors after per-frame similarity Procrustes alignment."""
    n_joints = pred.shape[-2]
    x = pred.reshape(-1, n_joints, 3)
    y = target.reshape(-1, n_joints, 3)
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x0 = x - x_mean
    y0 = y - y_mean
    covariance = x0.transpose(1, 2) @ y0
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones_like(singular)
    correction[:, -1] = torch.where(
        torch.det(u @ vh) < 0,
        correction.new_tensor(-1.0),
        correction.new_tensor(1.0),
    )
    rotation = u @ torch.diag_embed(correction) @ vh
    variance = x0.square().sum(dim=(1, 2)).clamp_min(1.0e-8)
    scale = (singular * correction).sum(dim=1) / variance
    aligned = scale[:, None, None] * (x0 @ rotation) + y_mean
    return torch.linalg.vector_norm(aligned - y, dim=-1)


def _tactile_corr(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame Pearson correlation over the 96 pressure cells.

    Returns (correlation, valid): frames where either side is constant
    (std ~ 0, e.g. a fully off insole) have no defined correlation and are
    masked out instead of contributing garbage values.
    """
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)
    denom = (pred_c.square().sum(dim=-1) * target_c.square().sum(dim=-1)).sqrt()
    corr = (pred_c * target_c).sum(dim=-1) / denom.clamp_min(1.0e-8)
    return corr, denom > 1.0e-8


def _rotation_error_deg(pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
    pred_rot = rot6d_to_rotmat(pred_pose.reshape(*pred_pose.shape[:2], N_JOINTS, 6))
    gt_rot = rot6d_to_rotmat(gt_pose.reshape(*gt_pose.shape[:2], N_JOINTS, 6))
    relative = pred_rot.transpose(-1, -2) @ gt_rot
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


class MetricSums:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {
            name: 0.0
            for name in ("MPJPE", "PA-MPJPE", "MPJRE", "traj_ATE", "contact_acc",
                         "contact_recall", "air_recall", "T_mae", "T_mse", "T_corr")
        }
        self.counts: Dict[str, int] = {name: 0 for name in self.sums}

    def add(self, name: str, values: torch.Tensor, scale: float = 1.0) -> None:
        self.sums[name] += float(values.double().sum().item()) * scale
        self.counts[name] += values.numel()

    def means(self) -> Dict[str, float]:
        out = {name: self.sums[name] / max(self.counts[name], 1) for name in self.sums}
        out["contact_balanced_acc"] = 0.5 * (out["contact_recall"] + out["air_recall"])
        out["T_rmse"] = float(out.pop("T_mse") ** 0.5)
        return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AnySole V1 with DDIM sampling.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint .pt. Omitted when --modal and --contact-method name a model dir "
        "under results/AnySole (checkpoints/ckpt_last.pt is used).",
    )
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None)
    parser.add_argument(
        "--contact-method",
        default=None,
        help="Test5 contact-label scheme for contact_gt (see results_display/README.md), "
        "and the model-dir suffix when --ckpt is omitted. "
        "Default: the checkpoint's training contact_method, then config, then tactile_abs.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--config-id",
        default=None,
        metavar="MODE[,MODE...]",
        help="Evaluation mode(s): VT2M, V2M, T2M; omitted means all three.",
    )
    parser.add_argument(
        "--write-motion",
        type=Path,
        default=None,
        metavar="DIR",
        help="Standard SMPL motion NPZ export dir (default: <model>/predictions/eval_motion).",
    )
    parser.add_argument("--no-write-motion", action="store_true", help="Disable SMPL motion export.")
    parser.add_argument("--metrics-out", type=Path, default=None, metavar="FILE",
                        help="Write VT2M/V2M/T2M metrics as JSON (default: model results metrics directory).")
    parser.add_argument(
        "--no-protocol",
        action="store_true",
        help="F0a: skip the extended protocol pass (eval_protocol.py, "
        "metrics/<split>_fseries.json).",
    )
    parser.add_argument(
        "--protocol-seed",
        type=int,
        default=0,
        help="F0a: RNG seed for the protocol pass (matters for diffusion sampling). "
        "Sigma-seed needs three runs with seeds 0/1/2.",
    )
    parser.add_argument(
        "--no-robustness",
        action="store_true",
        help="F0a: skip the protocol robustness rows (contiguous 20-40%% V / T frame "
        "dropout under VT2M).",
    )
    return parser.parse_args(argv)


def _parse_config_values(value: Optional[str]) -> List[int]:
    mode_to_config = dict(zip(CONFIG_MODE_NAMES, (CONFIG_VT, CONFIG_V, CONFIG_T)))
    if value is None:
        return [CONFIG_VT, CONFIG_V, CONFIG_T]
    modes = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("--config-id must contain at least one mode")
    unknown = [item for item in modes if item not in mode_to_config]
    if unknown:
        raise ValueError("Unknown mode(s) %s; choose from VT2M,V2M,T2M" % ",".join(unknown))
    return list(dict.fromkeys(mode_to_config[item] for item in modes))


def _discover_model_targets(
    modal: Optional[str], contact_method: Optional[str]
) -> List[tuple[str, str, Path]]:
    """Scan results/AnySole for model dirs named <modal>_<contact_method>.

    Returns (modal, contact_method, ckpt) triples sorted by dir name, so a
    bare ``python -m anysole.eval`` re-evaluates every trained model.
    Longest modal names match first (``anysolev1_insole_drift_tactile_abs``
    is the drift modal plus ``tactile_abs``, not ``anysolev1`` plus a long
    contact method), and dirs without ``checkpoints/ckpt_last.pt`` are
    skipped with a warning.
    """
    anysole_root = Path(os.environ.get("ANYSOLE_RESULTS", str(GAIT_ROOT / "results"))) / "AnySole"
    targets: List[tuple[str, str, Path]] = []
    if not anysole_root.is_dir():
        return targets
    for child in sorted(anysole_root.iterdir()):
        if not child.is_dir():
            continue
        for name in sorted(MODEL_NAMES, key=len, reverse=True):
            prefix = name + "_"
            if not (child.name.startswith(prefix) and len(child.name) > len(prefix)):
                continue
            contact = child.name[len(prefix):]
            if (modal is not None and name != modal) or (
                contact_method is not None and contact != contact_method
            ):
                break
            ckpt = child / "checkpoints" / "ckpt_last.pt"
            if ckpt.is_file():
                targets.append((name, contact, ckpt))
            else:
                print("skip %s: missing %s" % (child.name, ckpt))
            break
    return targets


def _load_model(checkpoint: dict, config: dict, device: torch.device) -> AnySoleModel:
    saved_config = checkpoint.get("config", {})
    if not isinstance(saved_config, dict):
        raise RuntimeError("checkpoint config is missing or is not a mapping")
    state_pose_dim = getattr(checkpoint.get("model", {}).get("pose_head.pose_mean"), "shape", (POSE_DIM,))[0]
    saved_joints = int(saved_config.get("motion_n_joints", state_pose_dim // 6 if state_pose_dim else N_JOINTS))
    saved_protocol = str(saved_config.get("motion_protocol", ""))
    saved_checksum = str(saved_config.get("joint_protocol_checksum", ""))
    if (saved_joints != N_JOINTS or state_pose_dim != POSE_DIM
            or saved_protocol != MOTION_PROTOCOL
            or saved_checksum != JOINT_PROTOCOL_CHECKSUM):
        raise RuntimeError(
            "checkpoint motion protocol is incompatible with the current native SMPL-24 "
            "joint tree (joints=%s, pose_dim=%s, protocol=%r, joint checksum=%s). "
            "This includes BVH-23 checkpoints and early 24/144 SMPL checkpoints trained "
            "with the incorrect collar-parent tree. Retrain under the current protocol; "
            "an old checkpoint may only be used through --init-from, which resets its "
            "pose/traj layers."
            % (saved_joints, state_pose_dim, saved_protocol,
               saved_checksum[:12] if saved_checksum else "missing")
        )
    d_model = int(saved_config.get("d_model", config["d_model"]))
    # tw always comes from the checkpoint, so evaluating an older model stays
    # correct even after anysole/configs/v1.yaml has moved on.
    tw = int(saved_config.get("tw", config["tw"]))
    modal = str(saved_config.get("modal", MODEL_ANYSOLEV1))
    if modal not in MODEL_NAMES:
        raise ValueError("Unknown checkpoint modal %r" % modal)
    use_drift = modal == MODEL_ANYSOLEV1_INSOLE_DRIFT
    model_kw = {}
    if use_drift:
        template_path = saved_config.get("template_path", config.get("template_path"))
        if not template_path:
            raise ValueError("template_path is required for drift-enabled evaluation")
        templates, subject_map = load_template_bank(template_path)
        model_kw.update(templates=templates, subject_to_index=subject_map)
    dropout = float(saved_config.get("dropout", config.get("dropout", 0.1)))
    pose_layers = int(saved_config.get("pose_layers", 6))
    tactile_input = str(saved_config.get("tactile_input", "raw108"))
    tactile_direct = bool(saved_config.get("tactile_direct", False))
    no_imu = bool(saved_config.get("no_imu", False))
    v_input = str(saved_config.get("v_input", "hrnet"))
    t_encoder = str(saved_config.get("t_encoder", "linear"))
    f2_repr = bool(saved_config.get("f2_repr", False))
    pose_parts = int(saved_config.get("pose_parts", 3))
    # F5 part9 is archived (fix_plan_v3.md supersedes v2 §F5): its checkpoints
    # are kept on disk but the V3 codebase does not load them.
    if str(saved_config.get("decoder", "v1")) == "part9":
        raise ValueError(
            "F5 part9 checkpoint is archived (v2 §F5 作废，fix_plan_v3 取代)；"
            "V3 代码不回载。基线用 F4a_footconv / F2p4_combo。"
        )
    if modal == MODEL_ANYSOLEV2:
        # F0b: regression model (model_v2.py); the regress pose head is
        # implied by the modal, and forward takes no diffusion pair.
        model = AnySoleModelV2(
            d=d_model, tw=tw, dropout=dropout, pose_layers=pose_layers,
            tactile_input=tactile_input, tactile_direct=tactile_direct, no_imu=no_imu,
            v_input=v_input, t_encoder=t_encoder, f2_repr=f2_repr,
            pose_parts=pose_parts,
        ).to(device)
    else:
        model = AnySoleModel(
            d=d_model, tw=tw, modal=modal, use_insole_drift=use_drift, dropout=dropout,
            pose_layers=pose_layers, tactile_input=tactile_input, tactile_direct=tactile_direct,
            no_imu=no_imu,
            v_input=v_input, t_encoder=t_encoder, f2_repr=f2_repr,
            **model_kw,
        ).to(device)
    # Quasi-strict load: the only tolerated missing keys are the hygiene-only
    # additions (foot_encoder stream_norm, fix_plan_v3 §3.1) that pre-date the
    # checkpoint.  Anything else — missing OR unexpected — is a real mismatch.
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or not set(missing) <= _MISSING_OK:
        raise RuntimeError(
            "state_dict mismatch: missing=%s unexpected=%s" % (missing, unexpected)
        )
    # E4: sampling init scale must match the checkpoint's training noise scale
    # (train.py saves noise_scaled=True when it scales q_sample noise by
    # pose_std; older checkpoints lack the flag and sample unscaled).
    model.noise_scaled = bool(saved_config.get("noise_scaled", False))
    model.eval()
    return model


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_values = _parse_config_values(args.config_id)
    config = load_config(args.config)
    if args.modal is not None:
        config["modal"] = args.modal
    if args.ckpt is not None:
        targets: List[tuple[Optional[str], Optional[str], Path]] = [
            (args.modal, args.contact_method, args.ckpt)
        ]
    elif args.modal is not None and args.contact_method is not None:
        targets = [
            (
                args.modal,
                args.contact_method,
                anysole_model_dir(args.modal, args.contact_method) / "checkpoints" / "ckpt_last.pt",
            )
        ]
    else:
        # Bare `python -m anysole.eval`: auto-scan results/AnySole and
        # evaluate every model whose dir is named <modal>_<contact_method>.
        targets = _discover_model_targets(args.modal, args.contact_method)
        if not targets:
            raise ValueError(
                "--ckpt is required unless results/AnySole contains model dirs named "
                "<modal>_<contact_method> (e.g. anysolev1_tactile_abs) with "
                "checkpoints/ckpt_last.pt"
            )
    multi = len(targets) > 1
    if multi and args.metrics_out is not None:
        raise ValueError(
            "--metrics-out is per-model; omit it when evaluating several models "
            "(each model's metrics go to its own metrics/test.json)"
        )
    device = resolve_device(args.device)
    for modal, contact_method, ckpt in targets:
        _evaluate_one(args, config, config_values, modal, contact_method, ckpt, device, multi=multi)
    return 0


def _positions_to_6d_batch(pred_pos: torch.Tensor, batch: dict, device: torch.device) -> torch.Tensor:
    """E6.1 export conversion: root-local positions -> WORLD 6D, per sample.

    positions_to_6d_np recovers rest-frame rotations; pre-multiplying the root
    by the session frame-0 rotation (dataset root_rot_init) yields world
    rotations for FK metrics and standard SMPL motion export.
    """
    out = []
    for i in range(pred_pos.shape[0]):
        sixd = positions_to_6d_np(
            pred_pos[i].cpu().numpy(),
            batch["offsets"][i].cpu().numpy(),
            batch["parents"][i].cpu().numpy(),
        )
        root_init = batch["root_rot_init"][i].cpu().numpy()  # (3,3)
        root_local = rot6d_to_rotmat_np(sixd.reshape(-1, N_JOINTS, 6)[:, 0:1, :])[:, 0]
        world_root = np.einsum("ij,tjk->tik", root_init, root_local)
        sixd[:, 0:6] = rotmat_to_6d_np(world_root)
        out.append(torch.from_numpy(sixd))
    return torch.stack(out, dim=0).to(device)


def _session_window_groups(dataset: AnySoleDataset) -> dict:
    """{session_id: [dataset indices]} in dataset order (consecutive windows)."""
    groups: dict = {}
    for i in range(len(dataset)):
        groups.setdefault(dataset[i]["session_id"], []).append(i)
    return groups


def _sample_x_t_init(model, bsz, tw, device, prior=None):
    """Sampling init: pure noise, or E6.5 marginal start with a prior."""
    x_T = torch.randn(bsz, tw, model.pose_head.pose_dim, device=device)
    if getattr(model, "noise_scaled", False):
        x_T = x_T * model.pose_head.pose_std.to(device).view(1, 1, -1)
    if prior is not None:
        sqrt_abar = float(np.sqrt(model.diffusion_abar_top))
        x_T = sqrt_abar * prior.to(device) + float(np.sqrt(1.0 - model.diffusion_abar_top)) * x_T
    return x_T


def _evaluate_one(
    args: argparse.Namespace,
    config: dict,
    config_values: List[int],
    modal: Optional[str],
    contact_method: Optional[str],
    ckpt: Path,
    device: torch.device,
    multi: bool = False,
) -> None:
    """Evaluate one checkpoint under each requested conditioning mode."""
    ckpt_parent = ckpt.parent
    model_root = ckpt_parent.parent if ckpt_parent.name == "checkpoints" else ckpt_parent
    print("== evaluating %s (ckpt %s)" % (model_root.name, ckpt))
    motion_out = None
    if not args.no_write_motion:
        motion_out = args.write_motion or model_root / "predictions" / "eval_motion"
        if multi and args.write_motion is not None:
            # Scan mode with an explicit export dir: nest per model so
            # different checkpoints don't clobber each other's exports.
            motion_out = args.write_motion / model_root.name
    checkpoint = torch.load(ckpt, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % ckpt)
    checkpoint_modal = str(checkpoint.get("config", {}).get("modal", MODEL_ANYSOLEV1))
    if modal is not None and modal != checkpoint_modal:
        raise ValueError("--modal %s does not match checkpoint modal %s" % (modal, checkpoint_modal))
    ckpt_contact = str(checkpoint.get("config", {}).get("contact_method") or "")
    contact_method = contact_method or ckpt_contact or str(config.get("contact_method", "tactile_abs"))
    model = _load_model(checkpoint, config, device)
    # --no-imu checkpoints: the eval dataset must build the 38-dim T_s2m
    # (IMU channels deleted) to match the checkpoint's encoder input width.
    no_imu = bool(checkpoint.get("config", {}).get("no_imu", False))
    tw = int(checkpoint.get("config", {}).get("tw", config["tw"]))
    diffusion = GaussianDiffusion(n_train_steps=int(checkpoint.get("config", {}).get("diffusion_train_steps", config["diffusion_train_steps"])))
    sample_steps = int(args.sample_steps or config["diffusion_sample_steps"])
    model.diffusion_abar_top = float(diffusion.alphas_cumprod[diffusion.n_train_steps - 1])
    pos_mode = checkpoint_modal == MODEL_ANYSOLEV1_POS
    regress_mode = checkpoint_modal == MODEL_ANYSOLEV2
    # F2a: heading/tilt representation — world pose/trans recovered via
    # geometry.f2_to_world before FK metrics and SMPL export.
    f2_repr = bool(checkpoint.get("config", {}).get("f2_repr", False))
    # F1: hmr_gvhmr checkpoints consume the GVHMR channel (V_hmr).
    v_hmr_mode = str(checkpoint.get("config", {}).get("v_input", "hrnet")) == "hmr_gvhmr"
    continuation = bool(config.get("continuation", False))
    warm_start = bool(config.get("warm_start", False))

    # --split selects the evaluated session set even without --limit-sessions.
    # (AnySoleDataset mode="eval" otherwise defaults to the val column, which
    # silently made every non-val metrics file report val numbers.)
    session_ids = load_split_ids(Path(config["split_csv"]), args.split)
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_ids = session_ids[: args.limit_sessions]
    dataset = AnySoleDataset(
        mode="eval",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=tw,
        session_ids=session_ids,
        contact_method=contact_method,
        tactile_input=str(checkpoint.get("config", {}).get("tactile_input", "raw108")),
        no_imu=no_imu,
        v_input=str(checkpoint.get("config", {}).get("v_input", "hrnet")),
        f2_repr=bool(checkpoint.get("config", {}).get("f2_repr", False)),
        smpl_roots=checkpoint.get("config", {}).get("smpl_roots", config.get("smpl_roots")),
        # E6.4: continuation eval uses overlapping windows (stride = tw/2);
        # without it the stride stays non-overlapping (E3 behavior).
        stride=(tw // 2) if (pos_mode and continuation) else None,
    )
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset contains no valid windows")
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )

    all_values = {}
    for config_value in config_values:
        metrics = MetricSums()
        exports = {}
        with torch.inference_mode():
            if pos_mode:
                # E6.1 position mode: sample positions, convert to world 6D
                # per window (R_init composition), then reuse all 6D metrics.
                if continuation:
                    # E6.4: each session's windows share one DDIM chain.
                    half = tw // 2
                    for session_id, idxs in _session_window_groups(dataset).items():
                        raw = [dataset[i] for i in idxs]
                        batch = move_batch(collate_windows(raw), device)
                        bsz = batch["pose_gt"].shape[0]
                        config_id = torch.full((bsz,), config_value, device=device, dtype=torch.long)
                        batch["config_id"] = config_id
                        assert_batch_shapes(batch, bsz, tw=tw)
                        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                        v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
                        cond = {"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                                "T_s2m": t_s2m, "V_hmr": batch.get("V_hmr") if v_hmr_mode else None,
                                "config_id": config_id,
                                "session_id": batch.get("session_id")}
                        prior = model.pose_head.pose_mean.view(1, 1, -1).expand(bsz, tw, -1) if warm_start else None
                        x_T = _sample_x_t_init(model, bsz, tw, device, prior=prior)
                        pred_pose = diffusion.ddim_sample_loop_continue(
                            model, x_T=x_T, tau_related_kwargs=cond,
                            steps=sample_steps, half=half)[0]
                        tau_zero = torch.zeros(bsz, device=device, dtype=torch.long)
                        out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                        anchor = batch["trans_anchor"][:, None, :]
                        pred_trans = out["trans_hat"] + anchor
                        gt_trans = batch["trans_gt"] + anchor
                        pred6d = _positions_to_6d_batch(pred_pose, batch, device)
                        t_ae = (out["pressure_hat"] - batch["T_raw"]).abs()
                        corr, corr_valid = _tactile_corr(out["pressure_hat"], batch["T_raw"])
                        for w in range(bsz):
                            sl = slice(0, tw) if w == 0 else slice(half, tw)
                            p6 = pred6d[w : w + 1, sl]
                            pt = pred_trans[w : w + 1, sl]
                            gtt = gt_trans[w : w + 1, sl]
                            kp = fk_pose6d(p6, pt, batch["offsets"][w : w + 1], batch["parents"][w : w + 1])
                            kp_gt = batch["kp_gt"][w : w + 1, sl]
                            metrics.add("MPJPE", torch.linalg.vector_norm(kp - kp_gt, dim=-1), 1000.0)
                            metrics.add("PA-MPJPE", _pa_error(kp, kp_gt), 1000.0)
                            metrics.add("MPJRE", _rotation_error_deg(p6, batch["pose_gt"][w : w + 1, sl]))
                            metrics.add("traj_ATE", torch.linalg.vector_norm(pt - gtt, dim=-1), 1000.0)
                            pred_contact = soft_contact_from_keypoints(
                                kp, batch["floor_y"][w : w + 1]
                            ) > 0.5
                            gt_contact = batch["contact_gt"][w : w + 1, sl] > 0.5
                            metrics.add("contact_acc", (pred_contact == gt_contact).float())
                            metrics.add("contact_recall", pred_contact[gt_contact].float())
                            metrics.add("air_recall", (~pred_contact[~gt_contact]).float())
                            metrics.add("T_mae", t_ae[w : w + 1, sl])
                            metrics.add("T_mse", t_ae[w : w + 1, sl].square())
                            if bool(corr_valid[w : w + 1, sl].any()):
                                metrics.add("T_corr", corr[w : w + 1, sl][corr_valid[w : w + 1, sl]])
                            if motion_out is not None:
                                exports.setdefault(session_id, []).append(
                                    (
                                        int(raw[w]["frame_start"]) + (0 if w == 0 else half),
                                        p6[0].cpu().numpy(),
                                        pt[0].cpu().numpy(),
                                        gtt[0].cpu().numpy(),
                                    )
                                )
                else:
                    for raw_batch in loader:
                        batch = move_batch(raw_batch, device)
                        batch_size = batch["pose_gt"].shape[0]
                        config_id = torch.full((batch_size,), config_value, device=device, dtype=torch.long)
                        batch["config_id"] = config_id
                        assert_batch_shapes(batch, batch_size, tw=tw)
                        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                        v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
                        cond = {"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                                "T_s2m": t_s2m, "V_hmr": batch.get("V_hmr") if v_hmr_mode else None,
                                "config_id": config_id,
                                "session_id": batch.get("session_id")}
                        prior = model.pose_head.pose_mean.view(1, 1, -1).expand(batch_size, tw, -1) if warm_start else None
                        x_T = _sample_x_t_init(model, batch_size, tw, device, prior=prior)
                        pred_pose = diffusion.ddim_sample_loop(
                            model, x_T=x_T, tau_related_kwargs=cond,
                            steps=sample_steps, eta=0.0, device=device)
                        tau_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
                        out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                        anchor = batch["trans_anchor"][:, None, :]
                        pred_trans = out["trans_hat"] + anchor
                        gt_trans = batch["trans_gt"] + anchor
                        pred6d = _positions_to_6d_batch(pred_pose, batch, device)
                        pred_kp = fk_pose6d(pred6d, pred_trans, batch["offsets"], batch["parents"])
                        metrics.add("MPJPE", torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1), 1000.0)
                        metrics.add("PA-MPJPE", _pa_error(pred_kp, batch["kp_gt"]), 1000.0)
                        metrics.add("MPJRE", _rotation_error_deg(pred6d, batch["pose_gt"]))
                        metrics.add("traj_ATE", torch.linalg.vector_norm(pred_trans - gt_trans, dim=-1), 1000.0)
                        pred_contact = soft_contact_from_keypoints(
                            pred_kp, batch["floor_y"]
                        ) > 0.5
                        gt_contact = batch["contact_gt"] > 0.5
                        metrics.add("contact_acc", (pred_contact == gt_contact).float())
                        metrics.add("contact_recall", pred_contact[gt_contact].float())
                        metrics.add("air_recall", (~pred_contact[~gt_contact]).float())
                        t_ae = (out["pressure_hat"] - batch["T_raw"]).abs()
                        metrics.add("T_mae", t_ae)
                        metrics.add("T_mse", t_ae.square())
                        corr, corr_valid = _tactile_corr(out["pressure_hat"], batch["T_raw"])
                        if bool(corr_valid.any()):
                            metrics.add("T_corr", corr[corr_valid])
                        if motion_out is not None:
                            for idx, session_id in enumerate(raw_batch["session_id"]):
                                exports.setdefault(session_id, []).append(
                                    (
                                        int(raw_batch["frame_start"][idx]),
                                        pred6d[idx].cpu().numpy(),
                                        pred_trans[idx].cpu().numpy(),
                                        gt_trans[idx].cpu().numpy(),
                                    )
                                )
            else:
                for raw_batch in loader:
                    batch = move_batch(raw_batch, device)
                    batch_size = batch["pose_gt"].shape[0]
                    config_id = torch.full((batch_size,), config_value, device=device, dtype=torch.long)
                    batch["config_id"] = config_id
                    assert_batch_shapes(batch, batch_size, tw=tw)
                    v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                    v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
                    if regress_mode:
                        # F0b: single regression forward — no DDIM chain.
                        # E6.4 continuation is diffusion-era and never applies
                        # here (F3 replaces it with Hann blending); the FADE=4
                        # crossfade below stays.
                        out = model(v_feat, t_raw, t_phys, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                        pred_pose = out["x0_hat"]
                    else:
                        cond = {
                            "V_feat": v_feat,
                            "T_raw": t_raw,
                            "T_phys": t_phys,
                            "T_s2m": t_s2m,
                            "V_hmr": batch.get("V_hmr") if v_hmr_mode else None,
                            "config_id": config_id,
                            "session_id": batch.get("session_id"),
                        }
                        prior = model.pose_head.pose_mean.view(1, 1, -1).expand(batch_size, tw, -1) if warm_start else None
                        pred_pose = diffusion.ddim_sample_loop(
                            model,
                            tau_related_kwargs=cond,
                            shape=(batch_size, tw, POSE_DIM),
                            steps=sample_steps,
                            eta=0.0,
                            device=device,
                            prior=prior,
                        )
                        tau_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
                        out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                    pred_trans = out["trans_hat"]
                    anchor = batch["trans_anchor"][:, None, :]
                    if f2_repr:
                        # F2a: world pose/trans from the tilt pose + heading
                        # trajectory; the GT world pose is recovered the same
                        # way for the rotation metric.
                        pred_pose_w, pred_trans_world = f2_to_world(
                            pred_pose, out["v_hat"], batch["psi_anchor"], batch["trans_anchor"]
                        )
                        gt_pose_w, _ = f2_to_world(
                            batch["pose_gt"], batch["traj_gt_f2"], batch["psi_anchor"],
                            batch["trans_anchor"],
                        )
                    else:
                        pred_trans_world = pred_trans + anchor
                        pred_pose_w = pred_pose
                        gt_pose_w = batch["pose_gt"]
                    gt_trans_world = batch["trans_gt"] + anchor
                    pred_kp = fk_pose6d(pred_pose_w, pred_trans_world, batch["offsets"], batch["parents"])

                    metrics.add("MPJPE", torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1), 1000.0)
                    metrics.add("PA-MPJPE", _pa_error(pred_kp, batch["kp_gt"]), 1000.0)
                    metrics.add("MPJRE", _rotation_error_deg(pred_pose_w, gt_pose_w))
                    metrics.add("traj_ATE", torch.linalg.vector_norm(pred_trans_world - gt_trans_world, dim=-1), 1000.0)
                    pred_contact = soft_contact_from_keypoints(
                        pred_kp, batch["floor_y"]
                    ) > 0.5
                    gt_contact = batch["contact_gt"] > 0.5
                    metrics.add("contact_acc", (pred_contact == gt_contact).float())
                    metrics.add("contact_recall", pred_contact[gt_contact].float())
                    metrics.add("air_recall", (~pred_contact[~gt_contact]).float())

                    # Tactile aux-head metrics (normalized 0-1 units).  Under
                    # V-only conditioning the T input is zeroed, so these numbers
                    # measure V2T generation quality (the V2M row in the report).
                    # F5 part9 drops the aux heads — those rows are skipped.
                    if out.get("pressure_hat") is not None:
                        t_ae = (out["pressure_hat"] - batch["T_raw"]).abs()
                        metrics.add("T_mae", t_ae)
                        metrics.add("T_mse", t_ae.square())
                        corr, corr_valid = _tactile_corr(out["pressure_hat"], batch["T_raw"])
                        if bool(corr_valid.any()):
                            metrics.add("T_corr", corr[corr_valid])

                    if motion_out is not None:
                        for idx, session_id in enumerate(raw_batch["session_id"]):
                            exports.setdefault(session_id, []).append(
                                (
                                    int(raw_batch["frame_start"][idx]),
                                    pred_pose_w[idx].cpu().numpy(),
                                    pred_trans_world[idx].cpu().numpy(),
                                    gt_trans_world[idx].cpu().numpy(),
                                )
                            )

        values = metrics.means()
        all_values[CONFIG_MODE_NAMES[config_value]] = values
        print(
            "%s MPJPE=%.3fmm PA-MPJPE=%.3fmm MPJRE=%.3fdeg traj_ATE=%.3fmm contact_acc=%.4f contact_bal=%.4f air_recall=%.4f T_mae=%.3f T_rmse=%.3f T_corr=%.3f"
            % (
                CONFIG_MODE_NAMES[config_value],
                values["MPJPE"],
                values["PA-MPJPE"],
                values["MPJRE"],
                values["traj_ATE"],
                values["contact_acc"],
                values["contact_balanced_acc"],
                values["air_recall"],
                values["T_mae"],
                values["T_rmse"],
                values["T_corr"],
            )
        )
        if motion_out is not None:
            beta_by_session = {
                s["session_id"]: np.asarray(s.get("betas", np.zeros(10)), dtype=np.float32)
                for s in loader.dataset.sessions
            }
            for session_id, windows in exports.items():
                windows.sort(key=lambda item: item[0])
                pose = np.concatenate([item[1] for item in windows], axis=0)
                trans = np.concatenate([item[2] for item in windows], axis=0)
                gt_trans = np.concatenate([item[3] for item in windows], axis=0)
                # E4: windows are non-overlapping (stride = window_length), so
                # the seam between consecutive windows is a hard cut between
                # two independent predictions (measured 213mm jump vs 8.8mm
                # GT frame-to-frame).  Crossfade F frames on both sides of
                # every seam; 6D poses are re-orthonormalized after blending.
                FADE = 0 if pos_mode else 4
                n_windows = len(windows)
                if n_windows > 1 and FADE > 0:
                    for w in range(1, n_windows):
                        seam = w * tw
                        if seam + FADE > pose.shape[0]:
                            break
                        for j in range(FADE):
                            alpha = (j + 1) / (FADE + 1)
                            a, b = seam - FADE + j, seam + j
                            old_a, old_b = pose[a].copy(), pose[b].copy()
                            pose[a] = (1 - alpha) * old_a + alpha * old_b
                            pose[b] = (1 - alpha) * old_b + alpha * old_a
                            old_ta, old_tb = trans[a].copy(), trans[b].copy()
                            trans[a] = (1 - alpha) * old_ta + alpha * old_tb
                            trans[b] = (1 - alpha) * old_tb + alpha * old_ta
                # Re-orthonormalize blended 6D vectors before writing the
                # Standard SMPL motion archive.
                R = rot6d_to_rotmat(torch.from_numpy(pose.reshape(-1, N_JOINTS, 6)).float())
                pose = rotmat_to_6d(R).reshape(-1, POSE_DIM).numpy()
                output_path = motion_out / ("%s_%s.npz" % (session_id, CONFIG_MODE_NAMES[config_value]))
                motion_out.mkdir(parents=True, exist_ok=True)
                smpl_poses = smpl24_pose6d_to_poses(pose)
                np.savez_compressed(
                    output_path,
                    poses=smpl_poses,
                    trans=pelvis_to_smpl_trans(pose, trans, beta_by_session.get(session_id)),
                    # Standard SMPL ``trans`` above is model-origin
                    # translation.  Keep the exact pelvis-space arrays used
                    # by traj_ATE separately so visualization never compares
                    # model origin against pelvis GT.
                    pred_pelvis_trans=trans.astype(np.float32),
                    gt_pelvis_trans=gt_trans.astype(np.float32),
                    gt_trans=gt_trans.astype(np.float32),  # legacy alias
                    betas=beta_by_session.get(session_id, np.zeros((10,), dtype=np.float32)),
                    betas_source=np.asarray("ground_truth_session"),
                    root_orient=smpl_poses[:, :3],
                    pose_body=smpl_poses[:, 3:],
                    mocap_frame_rate=np.asarray(FPS, dtype=np.float32),
                    source_frame_times_s=np.arange(len(pose), dtype=np.float32) / float(FPS),
                    **smpl_archive_metadata(),
                )
                print("wrote %s" % output_path)
    metrics_out = args.metrics_out
    if metrics_out is None:
        metrics_out = model_root / "metrics" / ("%s.json" % args.split)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(ckpt), "modal": str(checkpoint.get("config", {}).get("modal", MODEL_ANYSOLEV1)),
               "contact_method": contact_method,
               "split": args.split, "sample_steps": sample_steps, "metrics": all_values}
    if "V2M" in all_values:
        payload["v2t"] = {name: all_values["V2M"][name] for name in ("T_mae", "T_rmse", "T_corr")}
    metrics_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s" % metrics_out)
    # F0a: extended protocol (per-part aggregation, W-MPJPE/RTE/yaw,
    # jitter/accel/seam, contact F1 + foot slide, V2M pressure quality, T2M
    # upper-body plausibility, robustness rows) -> metrics/<split>_fseries.json.
    # Runs after the frozen per-window metrics so the classic JSON stays the
    # single source for the pre-F0 numbers.
    if not args.no_protocol:
        from anysole.eval_protocol import run_protocol
        run_protocol(
            checkpoint=checkpoint,
            config=config,
            model=model,
            dataset=dataset,
            device=device,
            checkpoint_path=str(ckpt),
            split=args.split,
            config_values=config_values,
            regress_mode=regress_mode,
            diffusion=diffusion,
            sample_steps=sample_steps,
            warm_start=warm_start,
            seed=args.protocol_seed,
            robustness=not args.no_robustness,
            contact_method=contact_method,
            tw=tw,
            out_path=model_root / "metrics" / ("%s_fseries.json" % args.split),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
