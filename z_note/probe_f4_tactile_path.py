"""Decompose the tactile information path on the F4a checkpoint (2026-09-18).

Question (user thesis): F4a's foot_conv pipeline should have more potential than
the observed VT gain (-8mm lower body); is the bottleneck the FUSION mechanism
(which dilutes the T stream) or the T tokens themselves (encoder/data limit)?

Method: fit a ridge readout (lam=1.0, the formalized ridge_probe 口径) from each
stage of the pipeline, all under VT conditioning, fitted on train / evaluated on
val with GT trajectories:

  F        : fused memory (2 tokens x 256 per frame)  — what the pose head sees
  v_tok    : V encoder tokens (1 x 256 per frame)
  t_tok1   : foot_conv forward() 1-token stream (1 x 256, LayerNorm'd)
  t_tok3   : foot_conv encode_stream 3-token stream (3 x 256, NO norm)
  v+t concat : pre-fusion concatenation (2 x 256)

Readouts: per-part MPJPE (mm).  If ridge(t_tok1) lower-body << ridge(F)
lower-body, the fusion destroys T information and a fusion-side fix has real
headroom; if they are similar, the T stream itself is the ceiling and the
tactile potential must be unlocked on the encoder/data side instead.

Usage (touch_gait env):
  python z_note/probe_f4_tactile_path.py --ckpt results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt --device cuda:1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.eval import _load_model, load_config, resolve_device
from anysole.eval_protocol import (
    ANKLE_FOOT_JOINTS,
    HAND_JOINTS,
    LOWER_JOINTS,
    PART_JOINTS,
    PART_NAMES,
    UPPER_JOINTS,
)
from anysole.geometry import fk_pose6d

PART_AGG = {
    "lower": LOWER_JOINTS,
    "anklefoot": ANKLE_FOOT_JOINTS,
    "hands": HAND_JOINTS,
    "upper": UPPER_JOINTS,
    **{f"part_{p}": j for p, j in zip(PART_NAMES, PART_JOINTS)},
}


def collect(model, dataset, device, batch_size):
    """Per-frame features for every pipeline stage + pose_gt, VT conditioning."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_windows)
    feats = {k: [] for k in ("F", "v", "t1", "t3", "vt")}
    poses = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
            B = batch["pose_gt"].shape[0]
            config_id = torch.zeros(B, dtype=torch.long, device=device)
            t_tac = torch.cat([batch["T_raw"], batch["T_phys"]], dim=-1)
            v_tok = model.encoders.v_enc(batch["V_feat"])          # (B, tw, d)
            t_tok1 = model.encoders.t_enc(t_tac)                    # (B, tw, d)  forward(), LN'd
            t_tok3 = model.encoders.t_enc.encode_stream(t_tac)      # (B, tw, 3, d) NO norm
            F = model.fusion(v_tok, t_tok1)                         # (B, 40, d)
            tw = v_tok.shape[1]
            feats["F"].append(F.reshape(-1, 2, F.shape[-1]).reshape(-1, 2 * F.shape[-1]))
            feats["v"].append(v_tok.reshape(-1, v_tok.shape[-1]))
            feats["t1"].append(t_tok1.reshape(-1, t_tok1.shape[-1]))
            feats["t3"].append(t_tok3.reshape(-1, 3, t_tok3.shape[-1]).reshape(-1, 3 * t_tok3.shape[-1]))
            feats["vt"].append(torch.cat([v_tok, t_tok1], dim=-1).reshape(-1, 2 * v_tok.shape[-1]))
            poses.append(batch["pose_gt"].reshape(-1, batch["pose_gt"].shape[-1]))
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(poses)


def ridge_fit(X, Y, lam, device):
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, device=device)], dim=1)
    A = Xb.T @ Xb + lam * torch.eye(d + 1, device=device)
    return torch.linalg.solve(A, Xb.T @ Y)


def ridge_apply(W, X, device):
    return torch.cat([X, torch.ones(X.shape[0], 1, device=device)], dim=1) @ W


def eval_mpjpe(model, dataset, device, batch_size, preds):
    """preds: dict stage -> (N_all, 138) aligned to the dataset frame order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_windows)
    idx = 0
    out = {k: {**{p: 0.0 for p in PART_AGG}, **{"all": 0.0, "n": 0}} for k in preds}
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
            B, TW = batch["pose_gt"].shape[0], batch["pose_gt"].shape[1]
            gt_trans = batch["trans_gt"] + batch["trans_anchor"][:, None, :]
            for k, P in preds.items():
                pred = P[idx: idx + B * TW].reshape(B, TW, -1)
                kp = fk_pose6d(pred, gt_trans, batch["offsets"], batch["parents"])
                err = torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1)  # (B, TW, 23)
                for p, joints in PART_AGG.items():
                    out[k][p] += float(err[..., joints].mean().item()) * B * TW
                out[k]["all"] += float(err.mean().item()) * B * TW
                out[k]["n"] += B * TW
            idx += B * TW
    for k in out:
        for p in out[k]:
            if p != "n":
                out[k][p] = out[k][p] / out[k]["n"] * 1000.0
        del out[k]["n"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                    default=Path("/data/fangyuxuan/projects/gait/AnysoleWorkspace/splits/default/splits.csv"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ridge-lam", type=float, default=1.0)
    args = ap.parse_args(argv)
    device = resolve_device(args.device)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    base_cfg = load_config(Path("/data/fangyuxuan/projects/gait/configs/v1.yaml"))
    model = _load_model(ck, base_cfg, device).eval()

    def make_ds(mode):
        return AnySoleDataset(
            mode=mode, seq_root=cfg["seq_root"], split_csv=args.split_csv,
            cache_root=cfg["cache_root"], window_length=int(cfg.get("tw", 20)),
            contact_method=cfg.get("contact_method", "tactile_abs"),
            no_imu=bool(cfg.get("no_imu", False)),
            v_input=cfg.get("v_input", "hrnet"),
            f2_repr=False,
        )

    print("collecting train features ...")
    Xtr, Ytr = collect(model, make_ds("train"), device, args.batch_size)
    print("train frames:", Ytr.shape[0], "| collecting val features ...")
    Xva, Yva = collect(model, make_ds("eval"), device, args.batch_size)
    print("val frames:", Yva.shape[0])

    preds = {}
    for k in Xtr:
        W = ridge_fit(Xtr[k], Ytr, args.ridge_lam, device)
        preds[k] = ridge_apply(W, Xva[k], device)

    table = eval_mpjpe(model, make_ds("eval"), device, args.batch_size, preds)
    names = {"F": "F (fused, 512d)", "v": "v_tok (256d)", "t1": "t_tok1 (256d, forward)",
             "t3": "t_tok3 (768d, stream)", "vt": "v+t concat (512d, pre-fusion)"}
    cols = ["all", "lower", "anklefoot", "part_l_leg", "part_r_leg",
            "part_l_foot", "part_r_foot", "part_torso", "part_headneck",
            "part_l_arm", "part_r_arm", "hands", "upper"]
    print("\nRidge readout ceiling per pipeline stage (val, mm, GT trajectory):")
    header = "%-24s" % "stage"
    for c in cols:
        header += "%10s" % c
    print(header)
    for k in preds:
        row = "%-24s" % names[k]
        for c in cols:
            row += "%10.1f" % table[k][c]
        print(row)


if __name__ == "__main__":
    main()
