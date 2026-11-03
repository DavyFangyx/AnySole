"""Compare cold T, V->T and VT->T with predicted state carried across windows.

Only the common T suffix is scored. Visual history tokens stay masked after
the switch: this experiment isolates predicted-state initialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.eval import _load_model, _session_window_groups
from anysole.train import condition_inputs, model_observation_kwargs, move_batch
from anysole.types import CONFIG_T, CONFIG_V, CONFIG_VT, FPS
from anysole.utils.geometry import (fk_pose6d_np, rot6d_to_rotmat,
                                    rot6d_to_rotmat_np, rotmat_to_6d,
                                    yaw_rotmat, yaw_rotmat_np)
from anysole.utils.metrics import (mean_point_error, pa_mpjpe, pelvis_align,
                                   root_orientation_drift_deg,
                                   root_orientation_error_deg, temporal_metrics)

ARMS = ("cold_T", "V_to_T", "VT_to_T")


@torch.inference_mode()
def rollout(dataset, model, indices, device, prefix_windows=2, progress=None,
            init_height=0.0):
    """Run three paired arms without accessing GT pose, velocity or position."""
    tw = model.tw
    relative = bool(getattr(model, "ar2", False))
    if not (relative or getattr(model, "ar", False)):
        raise ValueError("This runner requires AR/ARSS/AR2/AR2M state interfaces")
    if prefix_windows < 1:
        raise ValueError("prefix_windows must be positive")
    width = 148 if relative else 147
    prev = torch.zeros(3, width, device=device)
    psi = torch.zeros(3, dtype=torch.float64, device=device)
    # 先验初始高度（2026-11-03 修复，与 eval_protocol 同键）：自由积分
    # 无高度参照 → 骨盆贴地；默认 0 = 旧口径。SMPL y-up：y 通道 = 高度。
    pos = torch.zeros(3, 3, dtype=torch.float64, device=device)
    pos[:, 1] = float(init_height)
    ema = None
    previous = None
    segment = -1
    pose_chunks, trans_chunks, frames, scored, segments = [], [], [], [], []
    for number, index in enumerate(indices):
        raw = dataset[index]
        start = int(raw["frame_start"])
        if previous is None or start != previous + tw:
            prev.zero_()
            psi.zero_()
            pos.zero_()
            ema = None
            segment += 1
            segment_start = start
        in_prefix = start < segment_start + prefix_windows * tw
        cid = torch.tensor([CONFIG_T, CONFIG_V if in_prefix else CONFIG_T,
                            CONFIG_VT if in_prefix else CONFIG_T], device=device)
        batch = move_batch(collate_windows([raw] * 3), device)
        v, tr, tp, ts = condition_inputs(batch, cid)
        obs_kw = model_observation_kwargs(batch)
        if obs_kw:
            histories = [{key: value.clone() for key, value in history.items()}
                         for history in batch["obs_history"]]
            # T was unavailable before the switch in the V-only prefix arm.
            length = histories[1]["T_raw"].shape[0]
            unavailable = torch.arange(start - length, start) < segment_start + prefix_windows * tw
            for key in ("T_raw", "T_phys"):
                histories[1][key][unavailable] = 0
            obs_kw = {"obs_history": histories}
        ps = torch.zeros(3, tw, width, device=device)
        ps[:, 0] = prev
        poses, positions = [], []
        cached = None
        no_feedback = relative and not bool(getattr(model, "ar2_fb", True))
        for t in range(tw):
            if cached is None:
                out = model(v, tr, tp, cid, batch["session_id"], T_s2m=ts,
                            prev_state=ps, causal=True, **obs_kw)
                if no_feedback:
                    cached = out
            else:
                out = cached
            pose = out["x0_hat"][:, t]
            velocity = out["v_hat"][:, t]
            if relative:
                dyaw = out["dyaw_hat"][:, t, 0]
                psi = psi + dyaw.double()
                visual = cid != CONFIG_T
                if bool(visual.any()):
                    observed = out["vis_yaw"][:, t, 0].double()
                    ema = observed.clone() if ema is None else 0.5 * ema + 0.5 * observed
                    error = (ema - psi + torch.pi) % (2 * torch.pi) - torch.pi
                    psi = psi + visual * out["vis_gain"][:, t, 0].double() * error
                rotation = yaw_rotmat(psi).to(pose.dtype)
                world_root = rotation @ rot6d_to_rotmat(pose[:, :6])
                world_pose = torch.cat([rotmat_to_6d(world_root), pose[:, 6:]], dim=-1)
                world_velocity = (rotation @ velocity[..., None])[..., 0]
                state = torch.cat([dyaw[:, None], pose, velocity], dim=-1)
            else:
                world_pose, world_velocity = pose, velocity
                state = torch.cat([pose, velocity], dim=-1)
            pos = pos + world_velocity.double() / FPS
            poses.append(world_pose.cpu())
            positions.append(pos.float().cpu().clone())
            if t + 1 < tw:
                ps[:, t + 1] = state
            if not bool(torch.isfinite(world_pose).all() & torch.isfinite(pos).all()):
                raise ValueError("Non-finite generated motion")
        prev = state.clone()
        previous = start
        pose_chunks.append(torch.stack(poses, dim=1))
        trans_chunks.append(torch.stack(positions, dim=1))
        frames.extend(range(start, start + tw))
        scored.extend([not in_prefix] * tw)
        segments.extend([segment] * tw)
        if progress and (number % 5 == 0 or number + 1 == len(indices)):
            progress(number + 1, len(indices))
    return {"pose": torch.cat(pose_chunks, dim=1).numpy(),
            "trans": torch.cat(trans_chunks, dim=1).numpy(),
            "frame_ids": np.asarray(frames), "score_mask": np.asarray(scored),
            "segments": np.asarray(segments)}


def score(session, generated, relative):
    mask = generated["score_mask"]
    if not mask.any():
        return {"frames": 0, "arms": {}}
    ids = generated["frame_ids"][mask]
    gt = session["kp_gt"][ids]
    gt_rot = rot6d_to_rotmat_np(session["pose_gt"][ids, :6])
    if relative:
        gt_rot = yaw_rotmat_np(session["psi"][ids]) @ gt_rot
    groups = generated["segments"][mask]
    rows = {}
    for arm, label in enumerate(ARMS):
        pose = generated["pose"][arm, mask]
        pred = fk_pose6d_np(pose, generated["trans"][arm, mask],
                            session["offsets"], session["parents"])
        pred_local, _ = pelvis_align(pred)
        gt_local, _ = pelvis_align(gt)
        root = rot6d_to_rotmat_np(pose[:, :6])
        pa = pa_mpjpe(pred, gt)
        heading = root_orientation_error_deg(root, gt_rot)
        drift = np.zeros(len(ids))
        motion = gt_motion = 0.
        first, last = [], []
        for group in np.unique(groups):
            take = np.flatnonzero(groups == group)
            drift[take] = root_orientation_drift_deg(root[take], gt_rot[take])
            for points, rotations, target in ((pred_local, root, False),
                                               (gt_local, gt_rot, True)):
                local = np.matmul(points[take], rotations[take])[:, 1:]
                rms = np.sqrt(np.mean(np.sum((local - local.mean(0)) ** 2, axis=-1))) * 1000
                if target:
                    gt_motion += rms * len(take)
                else:
                    motion += rms * len(take)
            first.extend(take[:20])
            last.extend(take[-20:])
        temporal = temporal_metrics(pred_local, gt_local, ids / FPS)
        rows[label] = {"pa_mpjpe_mm": float(pa.mean()),
                       "mpjpe_mm": float(mean_point_error(pred_local, gt_local, scale=1000).mean()),
                       "root_orientation_deg": float(heading.mean()),
                       "root_orientation_drift_deg": float(drift.mean()),
                       "pa_first20_mm": float(pa[first].mean()),
                       "pa_last20_mm": float(pa[last].mean()),
                       "jitter_pred_m_s3": float(temporal["jitter_pred_m_s3"]),
                       "body_motion_rms_mm": float(motion / len(ids)),
                       "gt_body_motion_rms_mm": float(gt_motion / len(ids))}
    return {"frames": int(len(ids)), "frame_ids": ids.tolist(), "arms": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--sessions", nargs="+")
    parser.add_argument("--prefix-windows", type=int, default=2)
    parser.add_argument("--init-height", type=float, default=0.0,
                        help="先验初始高度（m，SMPL y-up；默认 0 = 旧口径）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    report = {"protocol": "predicted_state_only_visual_prefix_then_T",
              "prefix_windows": args.prefix_windows, "device": args.device,
              "GT_state_initialization": False, "visual_history_after_switch": False,
              "models": {}}
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        if int(checkpoint["epoch"]) != int(cfg["epochs"]):
            raise ValueError(f"Incomplete training checkpoint: {path}")
        model = _load_model(checkpoint, cfg, torch.device(args.device)).eval()
        name = f"{cfg['model_name']}_sd{cfg['seed']}_oc{cfg.get('obs_context', cfg['tw'])}"
        if name in report["models"]:
            raise ValueError(f"Duplicate model label: {name}")
        sessions = args.sessions or load_split_ids(Path(cfg["split_csv"]), "test")
        ds = AnySoleDataset(mode="eval", session_ids=sessions,
                            seq_root=Path(cfg["seq_root"]), split_csv=Path(cfg["split_csv"]),
                            cache_root=Path(cfg["cache_root"]), window_length=int(cfg["tw"]),
                            stride=int(cfg["tw"]), contact_method=cfg["contact_method"],
                            smpl_support_policy=cfg.get("smpl_support_policy", "exclude"),
                            smpl_roots=cfg.get("smpl_roots"),
                            v7_ar2=bool(getattr(model, "ar2", False)),
                            v7_ar=bool(getattr(model, "ar", False)),
                            obs_context=cfg.get("obs_context"))
        rows = {}
        for sid, indices in _session_window_groups(ds).items():
            generated = rollout(ds, model, indices, torch.device(args.device), args.prefix_windows,
                                init_height=args.init_height,
                                progress=lambda n, total: print(f"{name} {sid}: {n}/{total}", flush=True))
            session = next(s for s in ds.sessions if s["session_id"] == sid)
            rows[sid] = score(session, generated, bool(getattr(model, "ar2", False)))
        total = sum(row["frames"] for row in rows.values())
        if not total:
            raise ValueError("No T suffix to score")
        summary = {arm: {key: sum(row["frames"] * row["arms"][arm][key]
                                 for row in rows.values() if row["frames"]) / total
                         for key in next(row["arms"][arm] for row in rows.values() if row["frames"])}
                   for arm in ARMS}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report["models"][name] = {"checkpoint": str(path), "sha256": digest,
                                  "epoch": checkpoint["epoch"], "frames": total,
                                  "sessions": rows, "summary": summary}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(name, json.dumps(summary), flush=True)
        del checkpoint, model, ds
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
