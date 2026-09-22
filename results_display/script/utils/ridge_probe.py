"""Formalized ridge-on-F probe (F0a): the linear readout ceiling of the fused
memory F.  Formalizes z_note/r_test10_stage_probes.py — fix_plan_v2.md §2 makes
this a per-step mandatory probe (E3 reference ≈ 85 mm MPJPE); when a step
improves, the ridge ceiling should move with it.

ridge F (512 per frame = the two fused tokens of each frame, the test12 口径)
-> native SMPL-24 pose_gt (144), fitted on the train split under VT conditioning, evaluated
on the requested split with each window's GT trajectory.  Reports MPJPE,
PA-MPJPE (per-frame Procrustes, eval 口径), per-part MPJPE (9 parts +
upper/lower/ankle-foot/hands) and the raw L_pose MSE.

Works for both anysolev1 (diffusion) and anysolev2 (regression) checkpoints —
only the encoders + fusion are used.

Usage (touch_gait env):
  python results_display/script/utils/ridge_probe.py \
    --ckpt results/AnySole/<STEP>/checkpoints/ckpt_last.pt [--split val] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.eval import _load_model, load_config, resolve_device
from anysole.utils.eval_protocol import (
    ANKLE_FOOT_JOINTS,
    HAND_JOINTS,
    LOWER_JOINTS,
    PART_JOINTS,
    PART_NAMES,
    UPPER_JOINTS,
    _pa_align,
)
from anysole.utils.geometry import fk_pose6d


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge-on-F readout ceiling probe (F0a).")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "anysole" / "configs" / "v1.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Evaluation split (the ridge is always fitted on train).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ridge-lam", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON output (default: <model>/metrics/ridge_probe_<split>.json).")
    return parser.parse_args(argv)


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float, device: torch.device) -> torch.Tensor:
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, device=device)], dim=1)
    A = Xb.T @ Xb + lam * torch.eye(d + 1, device=device)
    return torch.linalg.solve(A, Xb.T @ Y)


def ridge_apply(W: torch.Tensor, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.cat([X, torch.ones(X.shape[0], 1, device=device)], dim=1) @ W


def collect_f(model, dataset, device: torch.device, batch_size: int) -> tuple:
    """(F per frame, native SMPL-24 pose_gt) under VT conditioning.

    The 512 = 2 fused tokens x 256 dims, the test12_stage_probes 口径: F is
    [v_fused(20) | t_fused(20)], so per frame there are exactly two tokens.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_windows)
    f_list, pose_list = [], []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
            B = batch["pose_gt"].shape[0]
            config_id = torch.zeros(B, dtype=torch.long, device=device)
            if model.tactile_input == "s2m50":
                t_tac = batch["T_s2m"]
            else:
                t_tac = torch.cat([batch["T_raw"], batch["T_phys"]], dim=-1)
            v_tok, t_tok = model.encoders(batch["V_feat"], t_tac, config_id,
                                          V_hmr=batch.get("V_hmr"))
            F = model.fusion(v_tok, t_tok)  # (B, 40, d)
            tw_runtime = batch["pose_gt"].shape[1]
            if F.shape[1] == 2 * tw_runtime:
                # v1 fusion: two tokens per frame -> 512-dim per frame.
                f_list.append(F.reshape(-1, 2, F.shape[-1]).reshape(-1, 2 * F.shape[-1]))
            elif F.shape[1] == tw_runtime:
                # F5 part decoder: one part-mean token per frame.
                f_list.append(F.reshape(-1, F.shape[-1]))
            else:
                raise ValueError("unexpected F tokens %d for tw %d" % (F.shape[1], tw_runtime))
            pose_list.append(batch["pose_gt"].reshape(-1, batch["pose_gt"].shape[-1]))
    return torch.cat(f_list), torch.cat(pose_list)


def mpjpe_table(pose_pred: torch.Tensor, dataset, device: torch.device, batch_size: int) -> tuple:
    """Per-frame MPJPE (mm, 24 joints) on the split with GT trajectories."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_windows)
    err_list, pa_list = [], []
    idx = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
            B, TW = batch["pose_gt"].shape[0], batch["pose_gt"].shape[1]
            pred = pose_pred[idx: idx + B * TW].reshape(B, TW, -1)
            idx += B * TW
            gt_trans = batch["trans_gt"] + batch["trans_anchor"][:, None, :]
            kp = fk_pose6d(pred, gt_trans, batch["offsets"], batch["parents"])
            err_list.append(torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1).reshape(-1, kp.shape[-2]))
            pa_list.append(torch.linalg.vector_norm(_pa_align(kp, batch["kp_gt"]) - batch["kp_gt"], dim=-1).reshape(-1, kp.shape[-2]))
    return torch.cat(err_list), torch.cat(pa_list)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    config = load_config(args.config)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % args.ckpt)
    saved = checkpoint.get("config", {})
    tw = int(saved.get("tw", config["tw"]))
    contact_method = str(saved.get("contact_method", config.get("contact_method", "joint_and")))
    no_imu = bool(saved.get("no_imu", False))
    model = _load_model(checkpoint, config, device)

    def build_dataset(mode):
        return AnySoleDataset(
            mode=mode, seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
            cache_root=Path(config["cache_root"]), window_length=tw,
            session_ids=(load_split_ids(Path(config["split_csv"]),
                                        {"val": "val", "test": "test"}[args.split])
                         if mode in ("eval", "test") else None),
            contact_method=contact_method,
            tactile_input=str(saved.get("tactile_input", "raw108")),
            no_imu=no_imu,
            v_input=str(saved.get("v_input", "hrnet")),
            f2_repr=bool(saved.get("f2_repr", False)),
        )

    train_ds = build_dataset("train")
    split_ds = build_dataset("test" if args.split == "test" else "eval")
    train_f, train_p = collect_f(model, train_ds, device, args.batch_size)
    split_f, split_p = collect_f(model, split_ds, device, args.batch_size)
    print("ridge-on-F: train %d frames, %s %d frames (lam=%.2f)"
          % (train_f.shape[0], args.split, split_f.shape[0], args.ridge_lam))

    W = ridge_fit(train_f, train_p, args.ridge_lam, device)
    pred = ridge_apply(W, split_f, device)
    err, pa_err = mpjpe_table(pred, split_ds, device, args.batch_size)
    l_pose = float(((pred - split_p) ** 2).mean().item())

    out: dict = {
        "checkpoint": str(args.ckpt),
        "modal": str(saved.get("modal", "anysolev1")),
        "split": args.split,
        "ridge_lam": args.ridge_lam,
        "L_pose": l_pose,
        "MPJPE": float(err.mean().item()) * 1000.0,
        "PA-MPJPE": float(pa_err.mean().item()) * 1000.0,
    }
    for pname, joints in zip(PART_NAMES, PART_JOINTS):
        out["MPJPE_part_%s" % pname] = float(err[:, joints].mean().item()) * 1000.0
        out["PA-MPJPE_part_%s" % pname] = float(pa_err[:, joints].mean().item()) * 1000.0
    for agg, joints in (("upper", UPPER_JOINTS), ("lower", LOWER_JOINTS),
                        ("anklefoot", ANKLE_FOOT_JOINTS), ("hands", HAND_JOINTS)):
        out["MPJPE_%s" % agg] = float(err[:, joints].mean().item()) * 1000.0
        out["PA-MPJPE_%s" % agg] = float(pa_err[:, joints].mean().item()) * 1000.0

    print("ridge-on-F -> pose: %s MPJPE = %.1f mm, PA-MPJPE = %.1f mm, L_pose = %.4f"
          % (args.split, out["MPJPE"], out["PA-MPJPE"], l_pose))
    for pname in PART_NAMES:
        print("  %-9s MPJPE %6.1f mm   PA-MPJPE %6.1f mm"
              % (pname, out["MPJPE_part_%s" % pname], out["PA-MPJPE_part_%s" % pname]))
    print("  upper %.1f / lower %.1f / ankle-foot %.1f / hands %.1f mm"
          % (out["MPJPE_upper"], out["MPJPE_lower"], out["MPJPE_anklefoot"], out["MPJPE_hands"]))

    out_path = args.out
    if out_path is None:
        model_root = args.ckpt.parent.parent if args.ckpt.parent.name == "checkpoints" else args.ckpt.parent
        out_path = model_root / "metrics" / ("ridge_probe_%s.json" % args.split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
