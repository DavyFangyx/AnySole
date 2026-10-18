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
from anysole.data.workspace_adapter import load_shared_session, shared_smpl_path
from anysole.data.tactile_s2m import build_t_s2m, resolve_legacy_bvh_path
from anysole.utils.diffusion import GaussianDiffusion
from anysole.utils.geometry import (
    f2_to_world_np,
    heading_from_root_np,
    positions_to_6d_np,
    rot6d_to_rotmat_np,
    rotmat_to_6d_np,
)
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT, MODEL_ANYSOLEV1_POS, MODEL_ANYSOLEV2, MODEL_NAMES
from anysole.registry import get_builder, infer_anysole_paths
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.train import load_config, resolve_device
from anysole.types import (
    ANYSOLE_ROOT,
    CONFIG_MODE_NAMES,
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    DEFAULT_CONFIG_PATH,
    FAKE_MARKED_ROOT,
    FPS,
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
        help="Checkpoint .pt. Omitted when --model-name and --contact-method name a model dir "
        "under results/AnySole (checkpoints/ckpt_last.pt is used).",
    )
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Registered model identifier (e.g. V4A / V3_4a). The checkpoint "
        "architecture is read from the checkpoint config.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Stacked hyperparameter fields under the model dir (the full field "
        "stack, e.g. tw40_st40_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1). Requires "
        "--model-name.",
    )
    parser.add_argument(
        "--which",
        choices=("last", "best"),
        default="last",
        help="Which checkpoint to address when inferred: last = ckpt_last (end of every "
        "run), best = ckpt_best (val-metric-best intermediate, for final reports).",
    )
    parser.add_argument(
        "--contact-method",
        default=None,
        help="Model-dir suffix when --ckpt is omitted (anysolev1_{ablation}_{contact_method}).",
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
    if "pressure_48_path" in meta:
        path = Path(meta["pressure_48_path"])
        return path, path
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
    shared = None
    if (seq_dir / "session.json").is_file():
        shared = load_shared_session(args.session, Path(config["seq_root"]))
        meta = dict(shared["meta"])
        meta.update({
            "n_frames": len(shared["frames"]["frame_id"]),
            "visual_time_s": np.asarray(shared["frames"]["visual_time_s"]),
            "mocap_time_s": np.asarray(shared["frames"]["mocap_time_s"]),
            "bvh_path": shared["meta"].get("source_files", {}).get("bvh", ""),
        })
        pressure_path = shared["dir"] / "pressure_48.npz"
        meta["pressure_48_path"] = str(pressure_path)
    else:
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
        if shared is not None:
            left48 = np.asarray(shared["pressure"]["left48"], dtype=np.float32)
            right48 = np.asarray(shared["pressure"]["right48"], dtype=np.float32)
            t_raw = normalize_raw(np.concatenate([left48, right48], axis=1))
            from anysole.data.pressure import physical_tokens
            t_phys = physical_tokens(normalize_raw(left48), normalize_raw(right48))
        else:
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
        raise ValueError("legacy architecture override %s does not match checkpoint architecture %s" % (args.modal, checkpoint_modal))
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
        # Registered models are rebuilt from their entry script — structure
        # fixed by the recorded model_name, saved flags validated against it
        # (2026-09-24 independence restructure).
        if str(saved_config.get("decoder", "v1")) == "part9":
            raise ValueError(
                "F5 part9 checkpoint is archived (v2 §F5 作废，fix_plan_v3 取代)；"
                "V3 代码不回载。基线用 F4a_footconv / F2p4_combo。"
            )
        model_name = str(saved_config.get("model_name", "") or "")
        if not model_name:
            raise RuntimeError(
                "checkpoint config has no model_name; registered-model "
                "checkpoints must record it (train --model-name writes it)"
            )
        entry = get_builder(model_name)
        saved_struct = {k: saved_config.get(k) for k in entry.STRUCTURE}
        if saved_config.get("part_joints") is not None and "pose_parts" in saved_struct:
            saved_struct["pose_parts"] = len(saved_config["part_joints"])
        if saved_struct != dict(entry.STRUCTURE):
            raise RuntimeError(
                "checkpoint structure does not match the %s entry: saved=%s entry=%s"
                % (model_name, saved_struct, dict(entry.STRUCTURE))
            )
        build_cfg = dict(config)
        build_cfg["d_model"] = d_model
        build_cfg["tw"] = tw
        build_cfg["dropout"] = dropout
        build_cfg["pose_layers"] = pose_layers
        model = entry.build(
            build_cfg,
            part_joints=tuple(tuple(int(j) for j in g) for g in saved_config["part_joints"])
            if saved_config.get("part_joints") else None,
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
    # V4A: restore the checkpoint's final assignment temperature (see eval.py).
    if getattr(model.pose_head, "soft_parts", False) and bool(saved_config.get("assign_cluster", False)):
        model.pose_head.set_assign_temp(float(saved_config.get("assign_temp_final", 1.0)))
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
        smpl_roots = tuple(Path(p) for p in saved_config.get(
            "smpl_roots", config.get("smpl_roots", SMPL_ROOTS)
        ))
        smpl_path = (
            shared_smpl_path(shared, smpl_roots) if shared is not None
            else resolve_smpl_path(meta, smpl_roots)
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
    import sys
    arguments=list(sys.argv[1:] if argv is None else argv)
    if arguments in (['--help'],['-h']) or any(flag in arguments for flag in ('--checkpoint','--input','--rgb-meta')):
        return visual_infer_main(arguments)
    args = parse_args(argv)
    requested_modes = _parse_modes(args.config_id)

    if args.ckpt is None:
        if args.model_name is None or args.contact_method is None:
            raise ValueError(
                "--ckpt is required unless --model-name and --contact-method "
                "(plus optional --variant) name a model dir under results/AnySole"
            )
        if args.variant and args.model_name is None:
            raise ValueError("--variant requires --model-name (the dir identifier, e.g. V3_4a)")
        dir_name = args.model_name
        args.ckpt = infer_anysole_paths(dir_name, variant=args.variant,
                                        contact=args.contact_method, which=args.which)["ckpt"]

    if not requested_modes:
        # Automatic selection remains available when the mode is omitted.
        config = load_config(args.config)
        seq_dir = find_session_dir(Path(config["seq_root"]), args.session)
        if (seq_dir / "session.json").is_file():
            shared = load_shared_session(args.session, Path(config["seq_root"]))
            meta = {"pressure_48_path": str(shared["dir"] / "pressure_48.npz")}
        else:
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


def load_rgb_only_meta(path):
    """Closed RGB-only schema; reject label/identity/calibration references."""
    value=json.loads(Path(path).read_text())
    allowed={'schema_version','camera_id','source_time_s','frame_files','target_start_s','target_end_s','target_hz'}
    if not isinstance(value,dict) or set(value)-allowed:
        raise ValueError('RGB metadata contains forbidden or unknown fields')
    if value.get('schema_version')!='anysole.rgb-only.v1':
        raise ValueError('expected anysole.rgb-only.v1 metadata')
    if value.get('target_hz',40.)!=40.:
        raise ValueError('student target grid is explicitly 40Hz')
    if not isinstance(value.get('camera_id',1),int):
        raise ValueError('camera_id must be an integer metadata tag')
    return value


def iter_rgb_input(input_path,meta,assume_fps=None):
    """Yield only RGB pixels, source ID and physical capture/container time."""
    import cv2
    import re
    from anysole.data.workspace_adapter import read_rgb_sources
    path=Path(input_path)
    if assume_fps is not None and (not np.isfinite(assume_fps) or assume_fps<=0):
        raise ValueError('--assume-fps must be positive')
    if path.is_dir():
        files=meta.get('frame_files')
        if files is None:
            files=sorted(p.name for p in path.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
        if not files:
            raise ValueError('no RGB frames')
        if any(Path(name).name!=name or Path(name).suffix.lower() not in ('.jpg','.jpeg','.png') for name in files):
            raise ValueError('frame_files must be local RGB filenames')
        if 'source_time_s' in meta:
            times=np.asarray(meta['source_time_s'],np.float64)
            if times.shape!=(len(files),):
                raise ValueError('one physical source timestamp is required per RGB filename')
            rows=[{'file':name,'frame_id':i,'time_s':float(t)} for i,(name,t) in enumerate(zip(files,times))]
        elif all(re.fullmatch(r'frame\d+_\d+_\d{6}\.\d+\.jpg',name) for name in files):
            camera=int(meta.get('camera_id',re.fullmatch(r'frame\d+_(\d+)_.*',files[0])[1]))
            rows=read_rgb_sources(path,camera)
            meta.setdefault('camera_id',camera)
            if set(files)!=set(r['file'] for r in rows):
                raise ValueError('physical timestamp directory and frame_files disagree')
        elif assume_fps is not None:
            rows=[{'file':name,'frame_id':i,'time_s':i/assume_fps} for i,name in enumerate(files)]
        else:
            raise ValueError('frames need physical source_time_s or timestamp filenames; unknown source fps requires --assume-fps')
        previous=-np.inf
        for row in rows:
            if not np.isfinite(row['time_s']) or row['time_s']<previous:
                raise ValueError('source timestamps must be finite and nondecreasing')
            previous=row['time_s']
            rgb=cv2.imread(str(path/row['file']))
            if rgb is None:
                raise ValueError('unreadable RGB frame '+row['file'])
            yield row,cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    else:
        if path.suffix.lower() not in ('.mp4','.mov','.avi','.mkv','.webm','.m4v'):
            raise ValueError('input must be a single RGB video or frame directory')
        if 'source_time_s' in meta or 'frame_files' in meta:
            raise ValueError('video uses container timestamps, not sidecar frame labels')
        capture=cv2.VideoCapture(str(path)); previous=-np.inf; index=0
        try:
            while True:
                ok,frame=capture.read()
                if not ok:
                    break
                time_s=float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
                if not np.isfinite(time_s) or time_s<=previous:
                    if assume_fps is None:
                        raise ValueError('container timestamp unavailable/nonmonotone; explicitly pass --assume-fps to replace timebase')
                    time_s=index/assume_fps
                if assume_fps is not None:
                    time_s=index/assume_fps
                previous=time_s
                yield {'file':path.name,'frame_id':index,'time_s':time_s},cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                index+=1
        finally:
            capture.release()
        if not index:
            raise ValueError('video contains no decodable RGB frames')


def visual_feature_windows(model,features,source_valid,sources,meta,device,*,window=20,stride=10):
    """RGB timebase -> declared target axis; no labels consulted at any point."""
    from anysole.data.workspace_adapter import match_rgb_sources
    from anysole.utils.eval_protocol import aggregate_visual_windows
    from anysole.utils.geometry import rot6d_to_rotmat
    first=float(meta.get('target_start_s',sources[0]['time_s']))
    last=float(meta.get('target_end_s',sources[-1]['time_s']))
    if not np.isfinite([first,last]).all() or last<first:
        raise ValueError('invalid RGB target range')
    target=first+np.arange(int(np.floor((last-first)*40+1e-7))+1,dtype=np.float64)/40
    matched=match_rgb_sources(target,sources,25.)
    rows=np.asarray(matched['source_row'],int)
    valid=np.asarray(matched['view_valid'],bool)&np.asarray(source_valid,bool)[rows]
    values=np.zeros((len(target),2051),np.float32);values[valid]=np.asarray(features)[rows[valid]]
    source_time=np.asarray(matched['source_time_s'],np.float64)
    source_id=np.asarray(matched['source_frame_id'],np.int64)
    edges=(valid[:-1]&valid[1:]&(np.diff(source_id)>0)&(np.diff(source_time)>0)&(np.diff(source_time)<=.050000001))
    starts=list(range(0,max(len(target)-window+1,0),stride))
    tail=max(len(target)-window,0)
    if not starts or starts[-1]!=tail:
        starts.append(tail)
    windows=[]
    for start in starts:
        stop=min(start+window,len(target));n=stop-start
        vv=np.zeros(window,bool);vv[:n]=valid[start:stop]
        feat=np.zeros((window,2051),np.float32);feat[:n]=values[start:stop]
        times=target[start]+np.arange(window)/40
        time_actual=target[start:stop];times[:n]=time_actual
        rotation=np.zeros((n,24,3,3));pressure=np.zeros((n,96));velocity=np.zeros((n,3))
        if vv.any():
            with torch.inference_mode():
                out=model.forward_visual(torch.from_numpy(feat[None]).to(device),torch.from_numpy(vv[None]).to(device),
                                         torch.tensor(times[None],dtype=torch.float64,device=device))
            rotation=rot6d_to_rotmat(out['x0_hat'][0,:n].float().reshape(n,24,6)).cpu().numpy()
            pressure=out['pressure_hat'][0,:n].float().cpu().numpy();velocity=out['v_hat'][0,:n].float().cpu().numpy()
        windows.append({'recording_id':'rgb_input','camera_id':int(meta.get('camera_id',1)),
            'frame_id':np.arange(start,stop),'time_s':time_actual,'valid':valid[start:stop],
            'view_edges':edges[start:max(stop-1,start)],'rotation':rotation,'pressure':pressure,'velocity':velocity,
            **{name:np.asarray(matched[name])[start:stop] for name in ('source_frame_id','source_time_s','reuse','time_residual_ms')}})
    return aggregate_visual_windows(windows),matched


def visual_infer_main(argv):
    import os
    import time
    from anysole.data.extract_hrnet import RGBFeatureFrontend,load_encoder
    from anysole.data.workspace_adapter import sha256_file
    from anysole.utils import eval_protocol as protocol
    parser=argparse.ArgumentParser(description='Pure single-camera RGB inference; no motion/pressure/bbox/calibration labels.')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--rgb-meta',type=Path,required=True,help='Closed anysole.rgb-only.v1 JSON; source_time_s for ordinary frame names.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assume-fps',type=float,help='Explicit source timebase replacement; never silently assume 40Hz source.')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--resource-plan',type=Path,help='Frozen common training/evaluation slot resource plan, required for CUDA.')
    parser.add_argument('--frontend-batch',type=int,default=16)
    args=parser.parse_args(argv)
    if args.frontend_batch<1:
        raise ValueError('positive frontend batch required')
    output=args.output.resolve()
    if not output.is_relative_to(Path('/private/Tactile')):
        raise ValueError('output must stay inside /private/Tactile')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('inference output must be new or empty')
    meta=load_rgb_only_meta(args.rgb_meta)
    os.environ['YOLO_CONFIG_DIR']=str(output/'.ultralytics')
    identity=protocol.visual_hash({'checkpoint':sha256_file(args.checkpoint),'input':str(args.input.resolve()),'output':str(output)})
    with protocol.visual_gpu_slot(args.device,args.resource_plan,identity) as resource:
        model,config,stats,checkpoint=protocol.load_visual_student(args.checkpoint,args.device)
        assets={}
        for key,hashkey in (('hrnet_checkpoint','hrnet_checkpoint_hash'),('detector_checkpoint','detector_checkpoint_hash')):
            asset=Path(config[key]);actual=sha256_file(asset)
            if actual!=config[hashkey]:
                raise ValueError('RGB frontend checkpoint hash mismatch for '+key)
            assets[key]={'path':str(asset),'sha256':actual}
        frontend_fingerprint={key:config[key] for key in ('hrnet_checkpoint_hash','detector_checkpoint_hash',
                                             'detector_version','preprocessing_version','cache_schema')}
        frontend_fingerprint['extractor_sha256']=sha256_file(Path(__file__).parent/'data/extract_hrnet.py')
        frontend_fingerprint['hrnet_config_sha256']=sha256_file(Path(config['hrnet_config']))
        hrnet_root=Path(config['hrnet_module_root'])/'lib/model/backbone/hrnet'
        frontend_fingerprint['hrnet_source_files']={str(path.relative_to(hrnet_root)):sha256_file(path)
                                      for path in sorted(hrnet_root.rglob('*.py'))}
        if config.get('frontend_runtime_hash') not in (None,protocol.visual_hash(frontend_fingerprint)):
            raise ValueError('RGB frontend source/config differs from frozen training frontend')
        import ultralytics
        if str(ultralytics.__version__)!=str(config['detector_version']):
            raise ValueError('RGB detector version differs from frozen cache frontend')
        from ultralytics import YOLO
        encoder=load_encoder(torch.device(args.device),Path(config['hrnet_module_root']),Path(config['hrnet_config']),Path(config['hrnet_checkpoint']))
        timing={name:0. for name in ('decode','detection','crop','hrnet','student','total')}
        detector=YOLO(config['detector_checkpoint'])
        def synchronize():
            if torch.device(args.device).type=='cuda':
                torch.cuda.synchronize(torch.device(args.device))
        class TimedDetector:
            def predict(self,**kwargs):
                synchronize();start=time.perf_counter();value=detector.predict(**kwargs);synchronize()
                timing['detection']+=time.perf_counter()-start
                return value
        frontend=RGBFeatureFrontend(encoder,TimedDetector(),torch.device(args.device))
        sources=[];features=[];source_valid=[];prepared=[];prepared_indices=[];crop_metadata=[];source_hashes=[]
        total=time.perf_counter();iterator=iter(iter_rgb_input(args.input,meta,args.assume_fps))
        def flush():
            if not prepared:
                return
            synchronize();start=time.perf_counter();encoded=frontend.encode_prepared(prepared);synchronize()
            timing['hrnet']+=time.perf_counter()-start
            for index,feature in zip(prepared_indices,encoded.numpy()):
                features[index]=feature;source_valid[index]=True
            prepared.clear();prepared_indices.clear()
        while True:
            begin=time.perf_counter()
            try:
                row,rgb=next(iterator)
            except StopIteration:
                break
            timing['decode']+=time.perf_counter()-begin
            index=len(sources);sources.append(row);features.append(np.zeros(2051,np.float32));source_valid.append(False)
            import hashlib
            source_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            before=timing['detection'];begin=time.perf_counter();value=frontend.prepare(rgb)
            timing['crop']+=max(0,time.perf_counter()-begin-(timing['detection']-before))
            crop_metadata.append(value[2] if value is not None else None)
            if value is not None:
                prepared.append(value);prepared_indices.append(index)
            if len(prepared)>=args.frontend_batch:
                flush()
        flush()
        synchronize();begin=time.perf_counter()
        series,matching=visual_feature_windows(model,np.stack(features),source_valid,sources,meta,torch.device(args.device))
        synchronize();timing['student']=time.perf_counter()-begin
        timing['total']=time.perf_counter()-total
        smpl_hash=sha256_file(Path(config['smpl_model_path']))
        if config.get('smpl_runtime_model_hash') not in (None,smpl_hash):
            raise ValueError('current SMPL hash differs from saved checkpoint')
        provenance={'checkpoint_sha256':sha256_file(args.checkpoint),'structure_hash':config['structure_hash'],
            'protocol_sha256':sha256_file(Path(protocol.__file__)),'smpl_runtime_model_hash':smpl_hash,
            'stats_sha256':stats.get('stats_sha256'),'rgb_metadata_sha256':sha256_file(args.rgb_meta),
            'input_rgb_sha256':protocol.visual_hash({'pixels':source_hashes,'sources':sources,'assume_fps':args.assume_fps}),
            'input_file_sha256':sha256_file(args.input) if args.input.is_file() else None,
            'frontend_assets':assets,'GT_access':'none'}
        predictions=protocol.export_visual_series(series,output/'predictions',config,provenance)
        count=sum(int(item['valid'].sum()) for item in series.values())
        report={'schema_version':protocol.VISUAL_EVAL_PROTOCOL,**provenance,'predictions':predictions,
            'source_rows':sources,'source_valid':source_valid,'target_matching':matching,'rgb_crop_track':crop_metadata,
            'resource':resource,'timing_seconds':timing,'timing_scope':'offline clip after asset/model load; excludes serialization',
            'frontend_runtime_sha256':protocol.visual_hash(frontend_fingerprint),'decoded_frames':len(sources),'predicted_target_frames':count,
            'frontend_fingerprint':frontend_fingerprint,'prediction_status':'ok' if count else 'no valid RGB detections',
            'offline_frames_per_second':count/max(timing['total'],1e-9),'lookahead_seconds':19/40,
            'source_timebase':'explicit --assume-fps replacement' if args.assume_fps else ('physical JPEG timestamps/sidecar' if args.input.is_dir() else 'container CAP_PROP_POS_MSEC'),
            'source_fps_assumption':args.assume_fps,'target_hz':40.,'causal':False,'pred_beta_source':'zero',
            'pressure_out_of_range_fraction':float(np.mean(np.concatenate([((s['pressure'][s['valid']]<0)|(s['pressure'][s['valid']]>1)).reshape(-1) for s in series.values()]))) if count else None}
        output.mkdir(parents=True,exist_ok=True)
        (output/'inference.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
