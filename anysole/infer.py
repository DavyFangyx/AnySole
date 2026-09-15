"""Run AnySole on one aligned session and export a BVH file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from anysole.data.bvh_io import load_bvh, pose_trans_to_motion, write_bvh
from anysole.data.dataset import find_session_dir, hrnet_cache_path, resolve_bvh_path, session_time_grid
from anysole.data.pressure import load_session_pressure, normalize_raw
from anysole.diffusion import GaussianDiffusion
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT, MODEL_NAMES
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
    POSE_DIM,
    T_PHYS_DIM,
    T_RAW_DIM,
    V_FEAT_DIM,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer a Skeleton3 BVH for one AnySole session.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None)
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", type=Path, default=ANYSOLE_ROOT / "configs" / "v1.yaml")
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

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % args.ckpt)
    checkpoint_modal = str(checkpoint.get("config", {}).get("modal", MODEL_ANYSOLEV1))
    if args.modal is not None and args.modal != checkpoint_modal:
        raise ValueError("--modal %s does not match checkpoint modal %s" % (args.modal, checkpoint_modal))
    saved_config = checkpoint.get("config", {})
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
    model = AnySoleModel(d=d_model, tw=tw, modal=modal, dropout=dropout, **model_kw).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    sample_steps = int(args.sample_steps or config["diffusion_sample_steps"])

    v_windows = _right_pad_windows(v_feat, tw)
    traw_windows = _right_pad_windows(t_raw, tw)
    tphys_windows = _right_pad_windows(t_phys, tw)
    pose_parts, trans_parts = [], []
    # Each window is relative to the preceding frame.  Stitch windows in
    # order by carrying forward the last predicted world-space position.
    stitch_anchor = torch.zeros(3)
    with torch.inference_mode():
        for left in range(0, v_windows.shape[0], args.batch_size):
            right = min(left + args.batch_size, v_windows.shape[0])
            v_batch = v_windows[left:right].to(device)
            traw_batch = traw_windows[left:right].to(device)
            tphys_batch = tphys_windows[left:right].to(device)
            config_id = torch.full((right - left,), config_value, device=device, dtype=torch.long)
            cond = {
                "V_feat": v_batch,
                "T_raw": traw_batch,
                "T_phys": tphys_batch,
                "config_id": config_id,
                "session_id": [args.session] * (right - left),
            }
            pred_pose = diffusion.ddim_sample_loop(
                model,
                tau_related_kwargs=cond,
                shape=(right - left, tw, POSE_DIM),
                steps=sample_steps,
                eta=0.0,
                device=device,
            )
            tau_zero = torch.zeros(right - left, device=device, dtype=torch.long)
            out = model(v_batch, traw_batch, tphys_batch, pred_pose, tau_zero, config_id, [args.session] * (right - left))
            pose_parts.append(pred_pose.cpu())
            trans_rel = out["trans_hat"].cpu()
            world_windows = []
            for window_rel in trans_rel:
                window_world = window_rel + stitch_anchor.view(1, 3)
                world_windows.append(window_world)
                stitch_anchor = window_world[-1].clone()
            trans_world = torch.stack(world_windows, dim=0)
            trans_parts.append(trans_world)

    pred_pose_np = torch.cat(pose_parts, dim=0).reshape(-1, POSE_DIM)[:n_frames].numpy()
    pred_trans_np = torch.cat(trans_parts, dim=0).reshape(-1, 3)[:n_frames].numpy()
    hierarchy = load_bvh(resolve_bvh_path(meta)).hierarchy
    if output_override is not None:
        output_path = output_override
    elif args.out is not None:
        output_path = args.out
    else:
        # Standard results layout: checkpoints live beside predictions.
        model_root = args.ckpt.parent.parent if args.ckpt.parent.name == "checkpoints" else args.ckpt.parent
        output_path = model_root / "predictions" / ("%s_%s.bvh" % (args.session, _CONFIG_TO_MODE[config_value]))
    write_bvh(output_path, hierarchy, pose_trans_to_motion(pred_pose_np, pred_trans_np), 1.0 / FPS)
    print("wrote %s (%d frames, config=%s)" % (output_path, n_frames, _CONFIG_TO_MODE[config_value]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    requested_modes = _parse_modes(args.config_id)

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
            output_override = args.out / ("%s_%s.bvh" % (args.session, mode))
        _run_one(args, _MODE_TO_CONFIG[mode], output_override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
