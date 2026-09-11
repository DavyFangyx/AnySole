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
from anysole.models import AnySoleModel
from anysole.train import load_config, resolve_device
from anysole.types import (
    ANYSOLE_ROOT,
    CONFIG_NAMES,
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
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", type=Path, default=ANYSOLE_ROOT / "configs" / "v1.yaml")
    parser.add_argument("--config-id", type=int, choices=(CONFIG_VT, CONFIG_V, CONFIG_T), default=None)
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


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
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
    if args.config_id is None:
        if has_video and has_pressure:
            config_value = CONFIG_VT
        elif has_video:
            config_value = CONFIG_V
        elif has_pressure:
            config_value = CONFIG_T
        else:
            raise FileNotFoundError("Session %s has neither HRNet cache nor pressure CSVs" % args.session)
    else:
        config_value = args.config_id
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
    saved_config = checkpoint.get("config", {})
    d_model = int(saved_config.get("d_model", config["d_model"]))
    tw = int(saved_config.get("tw", config["tw"]))
    model = AnySoleModel(d=d_model, tw=tw).to(device)
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
            out = model(v_batch, traw_batch, tphys_batch, pred_pose, tau_zero, config_id)
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
    output_path = args.out or args.ckpt.parent / ("%s_%s.bvh" % (args.session, CONFIG_NAMES[config_value]))
    write_bvh(output_path, hierarchy, pose_trans_to_motion(pred_pose_np, pred_trans_np), 1.0 / FPS)
    print("wrote %s (%d frames, config=%s)" % (output_path, n_frames, CONFIG_NAMES[config_value]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
