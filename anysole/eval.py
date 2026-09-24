"""Evaluate AnySole with the canonical session-level metric protocol."""

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
from anysole.utils.diffusion import GaussianDiffusion
from anysole.utils.geometry import f2_to_world, fk_pose6d, positions_to_6d_np, rot6d_to_rotmat, rot6d_to_rotmat_np, rotmat_to_6d, rotmat_to_6d_np
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT, MODEL_ANYSOLEV1_POS, MODEL_ANYSOLEV2, MODEL_NAMES
from anysole.train import condition_inputs, load_config, move_batch, resolve_device
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.registry import get_builder, infer_anysole_paths
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
    JOINT_NAMES,
    MOTION_PROTOCOL,
    N_JOINTS,
    POSE_DIM,
    assert_batch_shapes,
)

# Quasi-strict load allowance (fix_plan_v3 §3.1): state-dict keys added as
# code hygiene AFTER the baseline checkpoints were trained.  Only these may
# be missing when loading F4a/F2p4-era checkpoints.
_MISSING_OK = frozenset({
    "encoders.t_enc.stream_norm.weight",
    "encoders.t_enc.stream_norm.bias",
})


def _smpl_yup_to_display(points: np.ndarray) -> np.ndarray:
    """Convert native SMPL (+Y up) joints to the display/world (+Z up) frame."""
    points = np.asarray(points, dtype=np.float32)
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)


def _pack_motion_export(session: dict, windows: list[tuple], pos_mode: bool) -> dict:
    """Place window predictions back on the original session frame grid.

    Dataset windows are intentionally skipped when they contain fake frames.
    Concatenating those windows would silently shift every later prediction in
    time.  The unified archive therefore always has the original ``n_frames``
    length and carries ``valid_mask``/``frame_indices`` explicitly.
    """
    n_frames = int(np.asarray(session["V_feat"]).shape[0])
    pose_dim = int(windows[0][1].shape[-1]) if windows else POSE_DIM
    identity = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        N_JOINTS,
    )
    pose = np.repeat(identity[None], n_frames, axis=0)
    if pose_dim != pose.shape[1]:
        pose = np.zeros((n_frames, pose_dim), dtype=np.float32)
    trans = np.zeros((n_frames, 3), dtype=np.float32)
    seen = np.zeros((n_frames,), dtype=bool)
    ordered = sorted(windows, key=lambda item: int(item[0]))

    # Preserve the historical crossfade only across genuinely adjacent
    # prediction ranges.  Never crossfade over a skipped/fake-frame gap.
    fade = 0 if pos_mode else 4
    if fade > 0:
        for previous, current in zip(ordered, ordered[1:]):
            previous_start = int(previous[0])
            current_start = int(current[0])
            previous_len = int(np.asarray(previous[1]).shape[0])
            current_len = int(np.asarray(current[1]).shape[0])
            if current_start != previous_start + previous_len:
                continue
            if previous_len < fade or current_len < fade:
                continue
            previous_pose = np.asarray(previous[1])
            current_pose = np.asarray(current[1])
            previous_trans = np.asarray(previous[2])
            current_trans = np.asarray(current[2])
            for j in range(fade):
                alpha = (j + 1) / (fade + 1)
                a = previous_len - fade + j
                b = j
                old_a, old_b = previous_pose[a].copy(), current_pose[b].copy()
                previous_pose[a] = (1 - alpha) * old_a + alpha * old_b
                current_pose[b] = (1 - alpha) * old_b + alpha * old_a
                old_a, old_b = previous_trans[a].copy(), current_trans[b].copy()
                previous_trans[a] = (1 - alpha) * old_a + alpha * old_b
                current_trans[b] = (1 - alpha) * old_b + alpha * old_a

    for start, window_pose, window_trans, _window_gt_trans in ordered:
        start = int(start)
        window_pose = np.asarray(window_pose, dtype=np.float32)
        window_trans = np.asarray(window_trans, dtype=np.float32)
        if start >= n_frames:
            continue
        length = min(len(window_pose), len(window_trans), n_frames - start)
        if length <= 0:
            continue
        pose[start:start + length] = window_pose[:length]
        trans[start:start + length] = window_trans[:length]
        seen[start:start + length] = True

    fake_mask = np.asarray(
        session.get("fake_mask", np.zeros(n_frames, dtype=np.uint8))
    ).reshape(-1)
    valid = seen.copy()
    if len(fake_mask) == n_frames:
        valid &= fake_mask == 0

    pose_tensor = torch.from_numpy(pose).float().unsqueeze(0)
    trans_tensor = torch.from_numpy(trans).float().unsqueeze(0)
    offsets = torch.from_numpy(np.asarray(session["offsets"])).float()
    parents = torch.from_numpy(np.asarray(session["parents"])).long()
    joints_native = fk_pose6d(pose_tensor, trans_tensor, offsets, parents)[0].numpy()
    return {
        "pose": pose,
        "trans": trans,
        "gt_trans": np.asarray(session["trans_global"], dtype=np.float32),
        "valid": valid,
        "joint_xyz_world": _smpl_yup_to_display(joints_native),
    }


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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AnySole V1 with DDIM sampling.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint .pt. Omitted when --model-name and --contact-method name a model dir "
        "under results/AnySole (checkpoints/ckpt_last.pt is used).",
    )
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--model-name", default=None,
        help="Registered model identifier (e.g. V4A / V3_4a). The checkpoint "
        "architecture is read from the checkpoint config.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Stacked hyperparameter fields under the model dir "
        "(the full field stack, e.g. tw40_st40_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_"
        "ep740_bs256_sd1); model-name+contact+variant "
        "resolves to <model dir>/<variant>/checkpoints/ckpt_last.pt — the "
        "same address --ckpt would name (2026-09-22 stacked naming).",
    )
    parser.add_argument(
        "--which",
        choices=("last", "best"),
        default="last",
        help="Which checkpoint to address when inferred from --model-name. "
        "last = ckpt_last (end of every run); best = ckpt_best (val-metric-best "
        "intermediate, may be several epochs older — use for final reports).",
    )
    parser.add_argument(
        "--contact-method",
        default=None,
        help="D_Test3 contact-label scheme for contact_gt (labels generated by "
        "AnysoleWorkspace/tool/contact_labels.py), "
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
        "--protocol-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write extended protocol metrics to this JSON path. Defaults to the checkpoint model directory.",
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
    # F5 part9 is archived (fix_plan_v3.md supersedes v2 §F5): its checkpoints
    # are kept on disk but the V3 codebase does not load them.
    if str(saved_config.get("decoder", "v1")) == "part9":
        raise ValueError(
            "F5 part9 checkpoint is archived (v2 §F5 作废，fix_plan_v3 取代)；"
            "V3 代码不回载。基线用 F4a_footconv / F2p4_combo。"
        )
    if modal == MODEL_ANYSOLEV2:
        # Registered models are rebuilt from their entry script — the
        # structure is fixed by the recorded model_name, and the saved
        # structure flags are validated against the entry (2026-09-24
        # independence restructure; no warm-start, no flag reconstruction).
        model_name = str(saved_config.get("model_name", "") or "")
        if not model_name:
            raise RuntimeError(
                "checkpoint config has no model_name; registered-model "
                "checkpoints must record it (train --model-name writes it)"
            )
        entry = get_builder(model_name)
        saved_struct = {k: saved_config.get(k) for k in entry.STRUCTURE}
        if saved_config.get("part_joints") is not None and "pose_parts" in saved_struct:
            # V4B: pose_parts is derived from the partition at build time.
            saved_struct["pose_parts"] = len(saved_config["part_joints"])
        if saved_struct != dict(entry.STRUCTURE):
            raise RuntimeError(
                "checkpoint structure does not match the %s entry: saved=%s entry=%s"
                % (model_name, saved_struct, dict(entry.STRUCTURE))
            )
        for key, value in getattr(entry, "CONFIG_EXTRA", {}).items():
            if saved_config.get(key) != value:
                raise RuntimeError(
                    "checkpoint flag %s=%r does not match the %s entry (%r)"
                    % (key, saved_config.get(key), model_name, value)
                )
        build_cfg = dict(config)
        build_cfg["d_model"] = d_model
        build_cfg["tw"] = tw
        build_cfg["dropout"] = dropout
        build_cfg["pose_layers"] = pose_layers
        part_joints = saved_config.get("part_joints")
        model = entry.build(
            build_cfg,
            part_joints=tuple(tuple(int(j) for j in g) for g in part_joints)
            if part_joints else None,
        ).to(device)
    else:
        model = AnySoleModel(
            d=d_model, tw=tw, modal=modal, use_insole_drift=use_drift, dropout=dropout,
            pose_layers=pose_layers,
            tactile_input=str(saved_config.get("tactile_input", "raw108")),
            tactile_direct=bool(saved_config.get("tactile_direct", False)),
            no_imu=bool(saved_config.get("no_imu", False)),
            v_input=str(saved_config.get("v_input", "hrnet")),
            t_encoder=str(saved_config.get("t_encoder", "linear")),
            f2_repr=bool(saved_config.get("f2_repr", False)),
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
    # V4A: the assignment softmax temperature must match the checkpoint's
    # final annealed value (default 1.0 = the plain V3-3 softmax).
    if getattr(model.pose_head, "soft_parts", False) and bool(saved_config.get("assign_cluster", False)):
        model.pose_head.set_assign_temp(float(saved_config.get("assign_temp_final", 1.0)))
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
    elif args.model_name is not None and args.contact_method is not None:
        if args.variant and args.model_name is None:
            raise ValueError("--variant requires --model-name (the dir identifier, e.g. V3_4a)")
        dir_name = args.model_name
        targets = [
            (
                None,
                args.contact_method,
                infer_anysole_paths(dir_name, variant=args.variant,
                                    contact=args.contact_method, which=args.which)["ckpt"],
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
    if multi and args.protocol_out is not None:
        raise ValueError(
            "--protocol-out is per-model; omit it when evaluating several models"
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
        raise ValueError("legacy architecture override %s does not match checkpoint architecture %s" % (modal, checkpoint_modal))
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
    window_session_ids = {
        dataset.sessions[file_index]["session_id"]
        for file_index, _left, _right in dataset.valid_windows
    }
    skipped_no_window = [sid for sid in session_ids if sid not in window_session_ids]
    if skipped_no_window:
        print(
            "skip sessions with no valid evaluation windows: "
            + ", ".join(skipped_no_window)
        )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )

    for config_value in config_values:
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
                        for w in range(bsz):
                            sl = slice(0, tw) if w == 0 else slice(half, tw)
                            p6 = pred6d[w : w + 1, sl]
                            pt = pred_trans[w : w + 1, sl]
                            gtt = gt_trans[w : w + 1, sl]
                            kp = fk_pose6d(p6, pt, batch["offsets"][w : w + 1], batch["parents"][w : w + 1])
                            kp_gt = batch["kp_gt"][w : w + 1, sl]
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

        if motion_out is not None:
            beta_by_session = {
                s["session_id"]: np.asarray(s.get("betas", np.zeros(10)), dtype=np.float32)
                for s in loader.dataset.sessions
            }
            session_by_id = {
                s["session_id"]: s for s in loader.dataset.sessions
            }
            for session_id, windows in exports.items():
                session = session_by_id[session_id]
                packed = _pack_motion_export(session, windows, pos_mode)
                pose = packed["pose"]
                trans = packed["trans"]
                gt_trans = packed["gt_trans"]
                # Re-orthonormalize blended 6D vectors before writing the
                # Standard SMPL motion archive.
                R = rot6d_to_rotmat(torch.from_numpy(pose.reshape(-1, N_JOINTS, 6)).float())
                pose = rotmat_to_6d(R).reshape(-1, POSE_DIM).numpy()
                joints_native = fk_pose6d(
                    torch.from_numpy(pose).float().unsqueeze(0),
                    torch.from_numpy(trans).float().unsqueeze(0),
                    torch.from_numpy(np.asarray(session["offsets"])).float(),
                    torch.from_numpy(np.asarray(session["parents"])).long(),
                )[0].numpy()
                packed["joint_xyz_world"] = _smpl_yup_to_display(joints_native)
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
                    joint_xyz_world=packed["joint_xyz_world"].astype(np.float32),
                    joint_names=np.asarray(JOINT_NAMES),
                    valid_mask=packed["valid"].astype(bool),
                    frame_indices=np.arange(len(pose), dtype=np.int64),
                    joint_coordinate_system=np.asarray("world_z_up"),
                    session_id=np.asarray(session_id),
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
    from anysole.utils.eval_protocol import run_protocol
    protocol_out = args.protocol_out
    if protocol_out is None:
        protocol_out = model_root / "metrics" / ("%s_fseries.json" % args.split)
    final_metrics = run_protocol(
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
        out_path=protocol_out,
        v2t_out_dir=motion_out,
    )
    if not final_metrics:
        raise RuntimeError(
            "canonical session evaluation requires non-overlapping windows; "
            "this checkpoint/config uses an unsupported continuation stride"
        )
    payload = {"checkpoint": str(ckpt), "modal": str(checkpoint.get("config", {}).get("modal", MODEL_ANYSOLEV1)),
               "contact_method": contact_method,
               "split": args.split, "sample_steps": sample_steps, "metrics": final_metrics}
    if "V2M" in final_metrics:
        v2t_names = (
            "T_mae", "T_rmse", "T_corr", "pressure_force_mae",
            "pressure_force_rmse", "pressure_force_r2",
            "pressure_cop_error_left", "pressure_cop_error_right",
            "pressure_cop_error_mean", "contact_f1", "contact_acc",
            "contact_recall", "air_recall",
        )
        payload["v2t"] = {name: final_metrics["V2M"][name]
                          for name in v2t_names if name in final_metrics["V2M"]}
    metrics_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s" % metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
