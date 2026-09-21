"""Run AnySole on one aligned session and export a standard SMPL NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from anysole.data.smpl_io import load_smpl, resolve_smpl_path
from anysole.data.smpl_io import pelvis_to_smpl_trans, smpl24_pose6d_to_poses, smpl_archive_metadata
from anysole.data.dataset import (
    assemble_v_hmr,
    find_session_dir,
    hmr_cache_path,
    hrnet_cache_path,
    session_time_grid,
)
from anysole.data.pressure import load_session_pressure, normalize_raw
from anysole.data.tactile_s2m import build_t_s2m, resolve_legacy_bvh_path
from anysole.diffusion import GaussianDiffusion
from anysole.geometry import (
    f2_to_world_np,
    heading_from_root_np,
    positions_to_6d_np,
    rot6d_to_rotmat_np,
    rotmat_to_6d_np,
)
from anysole.models import AnySoleModel, AnySoleModelV2, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT, MODEL_ANYSOLEV1_POS, MODEL_ANYSOLEV2, MODEL_NAMES
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.train import load_config, resolve_device
from anysole.types import (
    ANYSOLE_ROOT,
    CONFIG_MODE_NAMES,
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    FAKE_MARKED_ROOT,
    FPS,
    GAIT_ROOT,
    JOINT_PROTOCOL_CHECKSUM,
    MOTION_PROTOCOL,
    N_JOINTS,
    POSE_DIM,
    T_PHYS_DIM,
    T_RAW_DIM,
    T_S2M_DIM,
    T_S2M_NOIMU_DIM,
    SMPL_ROOTS,
    V_FEAT_DIM,
    V_HMR_DIM,
    anysole_model_dir,
)

# Quasi-strict load allowance (fix_plan_v3 §3.1) — keep in sync with the
# identical _MISSING_OK in eval.py.
_MISSING_OK = frozenset({
    "encoders.t_enc.stream_norm.weight",
    "encoders.t_enc.stream_norm.bias",
})


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer standard SMPL motion for one session.")
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
        help="Model-dir suffix when --ckpt is omitted (anysolev1_{ablation}_{contact_method}).",
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", type=Path, default=GAIT_ROOT / "configs" / "v1.yaml")
    parser.add_argument(
        "--config-id",
        default=None,
        metavar="MODE[,MODE...]",
        help="Inference mode(s): VT2M, V2M, T2M; comma-separate to export multiple modes.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def _right_pad_windows(values: np.ndarray, tw: int) -> torch.Tensor:
    """Split into non-overlapping windows; repeat-pad the tail, which is cut after inference."""
    n_frames = values.shape[0]
    n_padded = ((n_frames + tw - 1) // tw) * tw
    if n_padded > n_frames:
        pad = np.repeat(values[-1:], n_padded - n_frames, axis=0)
        values = np.concatenate([values, pad], axis=0)
    return torch.from_numpy(np.ascontiguousarray(values.reshape(-1, tw, values.shape[-1]))).float()


def _stride_windows(values: np.ndarray, tw: int, half: int):
    """E6.4: overlapping windows at stride=half; tail repeat-padded."""
    n_frames = values.shape[0]
    starts = list(range(0, max(n_frames - tw + 1, 1), half))
    needed = starts[-1] + tw
    if needed > n_frames:
        pad = np.repeat(values[-1:], needed - n_frames, axis=0)
        values = np.concatenate([values, pad], axis=0)
    out = np.stack([values[s : s + tw] for s in starts], axis=0)
    return torch.from_numpy(np.ascontiguousarray(out)).float(), starts


def _pressure_paths(meta: dict) -> tuple:
    root = FAKE_MARKED_ROOT / meta["date"] / meta["subject"] / meta["rec_name"]
    return root / "pressure_left.csv", root / "pressure_right.csv"


_MODE_TO_CONFIG = dict(zip(CONFIG_MODE_NAMES, (CONFIG_VT, CONFIG_V, CONFIG_T)))
_CONFIG_TO_MODE = {value: key for key, value in _MODE_TO_CONFIG.items()}


def _parse_modes(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    modes = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("--config-id must contain at least one mode")
    unknown = [item for item in modes if item not in _MODE_TO_CONFIG]
    if unknown:
        raise ValueError("Unknown mode(s) %s; choose from VT2M,V2M,T2M" % ",".join(unknown))
    # Preserve command-line order while avoiding duplicate exports.
    return list(dict.fromkeys(modes))


def _run_one(args: argparse.Namespace, config_value: int, output_override: Optional[Path] = None) -> int:
    config = load_config(args.config)
    if args.modal is not None:
        config["modal"] = args.modal
    device = resolve_device(args.device)
    seq_dir = find_session_dir(Path(config["seq_root"]), args.session)
    meta = json.loads((seq_dir / "align_meta.json").read_text())
    n_frames = int(meta["n_frames"])
    if n_frames <= 0:
        raise ValueError("Session %s has no frames" % args.session)

    cache_path = hrnet_cache_path(args.session, Path(config["cache_root"]))
    has_video = cache_path.is_file()
    left_path, right_path = _pressure_paths(meta)
    has_pressure = left_path.is_file() and right_path.is_file()
    if config_value in (CONFIG_VT, CONFIG_V) and not has_video:
        raise FileNotFoundError(
            "Missing %s. Run `python -m anysole.data.extract_hrnet --cam-id %d --session %s`."
            % (cache_path, int(config["cam_id"]), args.session)
        )
    if config_value in (CONFIG_VT, CONFIG_T) and not has_pressure:
        raise FileNotFoundError("Missing pressure input: %s or %s" % (left_path, right_path))

    if config_value in (CONFIG_VT, CONFIG_V):
        loaded_v = torch.load(cache_path, map_location="cpu")
        v_feat = loaded_v.float().numpy() if torch.is_tensor(loaded_v) else np.asarray(loaded_v, dtype=np.float32)
        if v_feat.shape != (n_frames, V_FEAT_DIM):
            raise ValueError("V_feat shape %s != (%d, %d)" % (v_feat.shape, n_frames, V_FEAT_DIM))
    else:
        v_feat = np.zeros((n_frames, V_FEAT_DIM), dtype=np.float32)
    if config_value in (CONFIG_VT, CONFIG_T):
        pressure = load_session_pressure(meta, session_time_grid(meta))
        t_raw = normalize_raw(pressure["T_raw"])
        t_phys = np.asarray(pressure["T_phys"], dtype=np.float32)
    else:
        t_raw = np.zeros((n_frames, T_RAW_DIM), dtype=np.float32)
        t_phys = np.zeros((n_frames, T_PHYS_DIM), dtype=np.float32)
    # E6.6: unused for raw108 checkpoints (zeros are ignored). Historical
    # s2m50 checkpoints obtain synthetic IMU only from the companion BVH.
    t_s2m = np.zeros((n_frames, T_S2M_DIM), dtype=np.float32)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % args.ckpt)
    saved_config = checkpoint.get("config", {})
    if not isinstance(saved_config, dict):
        raise RuntimeError("checkpoint config is missing or is not a mapping")
    checkpoint_modal = str(saved_config.get("modal", MODEL_ANYSOLEV1))
    if args.modal is not None and args.modal != checkpoint_modal:
        raise ValueError("--modal %s does not match checkpoint modal %s" % (args.modal, checkpoint_modal))
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
    tw = int(saved_config.get("tw", config["tw"]))
    modal = str(saved_config.get("modal", MODEL_ANYSOLEV1))
    if modal not in MODEL_NAMES:
        raise ValueError("Unknown checkpoint modal %r" % modal)
    model_kw = {}
    if modal == MODEL_ANYSOLEV1_INSOLE_DRIFT:
        template_path = saved_config.get("template_path", config.get("template_path"))
        templates, subject_map = load_template_bank(template_path)
        model_kw.update(templates=templates, subject_to_index=subject_map)
    dropout = float(saved_config.get("dropout", config.get("dropout", 0.1)))
    pose_layers = int(saved_config.get("pose_layers", 6))
    tactile_input = str(saved_config.get("tactile_input", "raw108"))
    tactile_direct = bool(saved_config.get("tactile_direct", False))
    no_imu = bool(saved_config.get("no_imu", False))
    if tactile_input == "s2m50" and no_imu:
        # --no-imu checkpoint: the tactile channel is 38-dim. Rebuild the zero
        # placeholder (used for V-only rows) at the deleted-channel width.
        t_s2m = np.zeros((n_frames, T_S2M_NOIMU_DIM), dtype=np.float32)
    # F1: hmr_gvhmr checkpoints consume the GVHMR channel.
    v_hmr = np.zeros((n_frames, V_HMR_DIM), dtype=np.float32)
    if str(saved_config.get("v_input", "hrnet")) == "hmr_gvhmr" and config_value in (CONFIG_VT, CONFIG_V):
        hmr_path = hmr_cache_path("gvhmr", args.session)
        if not hmr_path.is_file():
            raise FileNotFoundError(
                "Missing GVHMR cache %s. Run `python -m anysole.data.extract_hmr`." % hmr_path
            )
        v_hmr = assemble_v_hmr(torch.load(hmr_path, map_location="cpu"), n_frames)
    if modal == MODEL_ANYSOLEV2:
        # F0b: regression model — the regress pose head is implied by the
        # modal; forward takes no diffusion pair.
        if str(saved_config.get("decoder", "v1")) == "part9":
            raise ValueError(
                "F5 part9 checkpoint is archived (v2 §F5 作废，fix_plan_v3 取代)；"
                "V3 代码不回载。基线用 F4a_footconv / F2p4_combo。"
            )
        model = AnySoleModelV2(
            d=d_model, tw=tw, dropout=dropout, pose_layers=pose_layers,
            tactile_input=tactile_input, tactile_direct=tactile_direct, no_imu=no_imu,
            v_input=str(saved_config.get("v_input", "hrnet")),
            t_encoder=str(saved_config.get("t_encoder", "linear")),
            f2_repr=bool(saved_config.get("f2_repr", False)),
            pose_parts=int(saved_config.get("pose_parts", 3)),
        ).to(device)
    else:
        model = AnySoleModel(
            d=d_model, tw=tw, modal=modal, dropout=dropout, pose_layers=pose_layers,
            tactile_input=tactile_input, tactile_direct=tactile_direct, no_imu=no_imu,
            v_input=str(saved_config.get("v_input", "hrnet")),
            t_encoder=str(saved_config.get("t_encoder", "linear")),
            f2_repr=bool(saved_config.get("f2_repr", False)),
            **model_kw,
        ).to(device)
    # Quasi-strict load — same allowance as eval.py _load_model
    # (fix_plan_v3 §3.1); keep both _MISSING_OK sets in sync.
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or not set(missing) <= _MISSING_OK:
        raise RuntimeError(
            "state_dict mismatch: missing=%s unexpected=%s" % (missing, unexpected)
        )
    # E4: match the checkpoint's training noise scale (see train.py noise_scaled).
    model.noise_scaled = bool(saved_config.get("noise_scaled", False))
    model.eval()
    diffusion = GaussianDiffusion(n_train_steps=int(saved_config.get("diffusion_train_steps", config["diffusion_train_steps"])))
    sample_steps = int(args.sample_steps or config["diffusion_sample_steps"])
    pos_mode = modal == MODEL_ANYSOLEV1_POS
    continuation = bool(config.get("continuation", False)) and pos_mode
    warm_start = bool(config.get("warm_start", False))
    f2_repr = bool(saved_config.get("f2_repr", False))
    uses_synthetic_imu = tactile_input == "s2m50" and config_value in (CONFIG_VT, CONFIG_T)
    # Standard F4 (6D + raw108) must not read its SMPL target during inference
    # merely to obtain betas for serialization. Position/F2 need explicit SMPL
    # anchors; historical synthetic IMU is a separate BVH-derived input.
    needs_gt_motion = bool(pos_mode or f2_repr)
    motion = None
    if needs_gt_motion:
        t_mocap = session_time_grid(meta) - float(meta["offset_s"])
        smpl_path = resolve_smpl_path(
            meta,
            tuple(Path(p) for p in saved_config.get(
                "smpl_roots", config.get("smpl_roots", SMPL_ROOTS)
            )),
        )
        motion = load_smpl(smpl_path, query_t=t_mocap)
        print(
            "WARNING: this checkpoint branch uses GT-SMPL-derived inference state "
            "(pos/f2); it is an evaluation protocol, not mocap-free deployment."
        )
    if uses_synthetic_imu:
        # E6.6a: preserve the original Step2Motion BVH-23/ToeBase IMU source.
        # This GT-BVH-derived input is experimental and never used by raw108 F4.
        t_mocap = session_time_grid(meta) - float(meta["offset_s"])
        t_s2m = build_t_s2m(
            t_raw,
            bvh_path=None if no_imu else resolve_legacy_bvh_path(meta),
            query_t=None if no_imu else t_mocap,
            no_imu=no_imu,
        )
        if not no_imu:
            print(
                "WARNING: s2m50 uses synthetic IMU derived from the directly exported GT BVH; "
                "it is an evaluation protocol, not mocap-free deployment."
            )
    psi_array = None
    trans_world_array = None
    if f2_repr:
        # F2a: per-frame heading + world root position for the window anchors.
        root_rot = rot6d_to_rotmat_np(motion["pose_6d"].reshape(-1, N_JOINTS, 6)[:, 0])
        psi_array = heading_from_root_np(root_rot)
        trans_world_array = motion["trans_m"]

    if continuation:
        # E6.4: overlapping windows (stride = tw/2) with chain continuation;
        # only the fresh half of each window is exported.
        half = tw // 2
        v_windows, _ = _stride_windows(v_feat, tw, half)
        traw_windows, _ = _stride_windows(t_raw, tw, half)
        tphys_windows, _ = _stride_windows(t_phys, tw, half)
        ts2m_windows, _ = _stride_windows(t_s2m, tw, half)
        vhmr_windows, _ = _stride_windows(v_hmr, tw, half)
        root_init = rot6d_to_rotmat_np(motion["pose_6d"].reshape(-1, N_JOINTS, 6)[:1, 0])[0]
    else:
        v_windows = _right_pad_windows(v_feat, tw)
        traw_windows = _right_pad_windows(t_raw, tw)
        tphys_windows = _right_pad_windows(t_phys, tw)
        ts2m_windows = _right_pad_windows(t_s2m, tw)
        vhmr_windows = _right_pad_windows(v_hmr, tw)
    pose_parts, trans_parts = [], []
    # Each window is relative to the preceding frame.  Stitch windows in
    # order by carrying forward the last predicted world-space position
    # (continuation: local frame half-1).
    stitch_anchor = torch.zeros(3)
    carry = None
    with torch.inference_mode():
        for left in range(0, v_windows.shape[0], args.batch_size):
            right = min(left + args.batch_size, v_windows.shape[0])
            v_batch = v_windows[left:right].to(device)
            traw_batch = traw_windows[left:right].to(device)
            tphys_batch = tphys_windows[left:right].to(device)
            ts2m_batch = ts2m_windows[left:right].to(device)
            vhmr_batch = vhmr_windows[left:right].to(device)
            config_id = torch.full((right - left,), config_value, device=device, dtype=torch.long)
            v_kw = {"V_hmr": vhmr_batch} if str(saved_config.get("v_input", "hrnet")) == "hmr_gvhmr" else {}
            cond = {
                "V_feat": v_batch,
                "T_raw": traw_batch,
                "T_phys": tphys_batch,
                "T_s2m": ts2m_batch,
                "V_hmr": vhmr_batch if str(saved_config.get("v_input", "hrnet")) == "hmr_gvhmr" else None,
                "config_id": config_id,
                "session_id": [args.session] * (right - left),
            }
            if modal == MODEL_ANYSOLEV2:
                # F0b: one regression forward per window, no sampling chain.
                out = model(v_batch, traw_batch, tphys_batch, config_id,
                            [args.session] * (right - left), T_s2m=ts2m_batch, **v_kw)
                pred_pose = out["x0_hat"]
            else:
                prior = model.pose_head.pose_mean.view(1, 1, -1).expand(right - left, tw, -1) if warm_start else None
                if continuation:
                    x_T = torch.randn(right - left, tw, model.pose_head.pose_dim, device=device)
                    if model.noise_scaled:
                        x_T = x_T * model.pose_head.pose_std.to(device).view(1, 1, -1)
                    pred_pose, carry = diffusion.ddim_sample_loop_continue(
                        model, x_T=x_T, tau_related_kwargs=cond,
                        steps=sample_steps, half=half, carry=carry)
                else:
                    pred_pose = diffusion.ddim_sample_loop(
                        model,
                        tau_related_kwargs=cond,
                        shape=(right - left, tw, model.pose_head.pose_dim),
                        steps=sample_steps,
                        eta=0.0,
                        device=device,
                        prior=prior,
                    )
                tau_zero = torch.zeros(right - left, device=device, dtype=torch.long)
                out = model(v_batch, traw_batch, tphys_batch, pred_pose, tau_zero, config_id,
                            [args.session] * (right - left), T_s2m=ts2m_batch, **v_kw)
            trans_rel = out["trans_hat"].cpu()
            if pos_mode:
                for w in range(right - left):
                    first = (left == 0 and w == 0)
                    sl = slice(0, tw) if (first or not continuation) else slice(half, tw)
                    sixd = positions_to_6d_np(
                        pred_pose[w, sl].cpu().numpy(), motion["offsets_m"], motion["parents"]
                    )
                    root_local = rot6d_to_rotmat_np(sixd.reshape(-1, N_JOINTS, 6)[:, 0:1, :])[:, 0]
                    world_root = np.einsum("ij,tjk->tik", root_init, root_local)
                    sixd[:, 0:6] = rotmat_to_6d_np(world_root)
                    pose_parts.append(torch.from_numpy(sixd))
                    window_world = trans_rel[w, sl] + stitch_anchor.view(1, 3)
                    trans_parts.append(window_world)
                    if continuation:
                        stitch_anchor = (trans_rel[w, half - 1] + stitch_anchor).clone()
                    else:
                        stitch_anchor = window_world[-1].clone()
            else:
                if f2_repr:
                    # F2a: recover the world pose/trans per window from the
                    # tilt pose + heading trajectory and its GT anchors.
                    for w in range(right - left):
                        window_idx = left + w
                        # Window k starts at frame k*tw; match Dataset's
                        # anchor=max(left-1, 0), not the window ordinal k-1.
                        anchor_idx = min(max(window_idx * tw - 1, 0), n_frames - 1)
                        world_pose, world_trans = f2_to_world_np(
                            pred_pose[w].cpu().numpy(),
                            out["v_hat"][w].cpu().numpy(),
                            float(psi_array[anchor_idx]),
                            trans_world_array[anchor_idx],
                        )
                        pose_parts.append(torch.from_numpy(world_pose))
                        trans_parts.append(torch.from_numpy(world_trans))
                else:
                    pose_parts.append(pred_pose.cpu())
                    world_windows = []
                    for window_rel in trans_rel:
                        window_world = window_rel + stitch_anchor.view(1, 3)
                        world_windows.append(window_world)
                        stitch_anchor = window_world[-1].clone()
                    trans_world = torch.stack(world_windows, dim=0)
                    trans_parts.append(trans_world)

    pred_pose_np = torch.cat(pose_parts, dim=0).reshape(-1, POSE_DIM)[:n_frames].numpy()
    pred_trans_np = torch.cat(trans_parts, dim=0).reshape(-1, 3)[:n_frames].numpy()
    export_betas = (
        np.asarray(motion.get("betas"), dtype=np.float32)
        if motion is not None
        else np.zeros((10,), dtype=np.float32)
    )
    betas_source = "ground_truth_session" if motion is not None else "neutral_zero"
    if output_override is not None:
        output_path = output_override
    elif args.out is not None:
        output_path = args.out
    else:
        # Standard results layout: checkpoints live beside predictions.
        model_root = args.ckpt.parent.parent if args.ckpt.parent.name == "checkpoints" else args.ckpt.parent
        output_path = model_root / "predictions" / ("%s_%s.npz" % (args.session, _CONFIG_TO_MODE[config_value]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smpl_poses = smpl24_pose6d_to_poses(pred_pose_np)
    np.savez_compressed(
        output_path,
        poses=smpl_poses,
        trans=pelvis_to_smpl_trans(pred_pose_np, pred_trans_np, export_betas),
        betas=export_betas,
        betas_source=np.asarray(betas_source),
        root_orient=smpl_poses[:, :3],
        pose_body=smpl_poses[:, 3:],
        mocap_frame_rate=np.asarray(FPS, dtype=np.float32),
        source_frame_times_s=np.arange(len(smpl_poses), dtype=np.float32) / float(FPS),
        **smpl_archive_metadata(),
    )
    print("wrote %s (%d frames, config=%s)" % (output_path, n_frames, _CONFIG_TO_MODE[config_value]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    requested_modes = _parse_modes(args.config_id)

    if args.ckpt is None:
        if args.modal is None or args.contact_method is None:
            raise ValueError(
                "--ckpt is required unless both --modal and --contact-method name a "
                "model dir under results/AnySole (anysolev1_{ablation}_{contact_method})"
            )
        args.ckpt = anysole_model_dir(args.modal, args.contact_method) / "checkpoints" / "ckpt_last.pt"

    if not requested_modes:
        # Automatic selection remains available when the mode is omitted.
        config = load_config(args.config)
        seq_dir = find_session_dir(Path(config["seq_root"]), args.session)
        meta = json.loads((seq_dir / "align_meta.json").read_text())
        cache_path = hrnet_cache_path(args.session, Path(config["cache_root"]))
        left_path, right_path = _pressure_paths(meta)
        if cache_path.is_file() and left_path.is_file() and right_path.is_file():
            requested_modes = ["VT2M"]
        elif cache_path.is_file():
            requested_modes = ["V2M"]
        elif left_path.is_file() and right_path.is_file():
            requested_modes = ["T2M"]
        else:
            raise FileNotFoundError("Session %s has neither HRNet cache nor pressure CSVs" % args.session)

    for mode in requested_modes:
        output_override = None
        if args.out is not None and len(requested_modes) > 1:
            output_override = args.out / ("%s_%s.npz" % (args.session, mode))
        _run_one(args, _MODE_TO_CONFIG[mode], output_override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
