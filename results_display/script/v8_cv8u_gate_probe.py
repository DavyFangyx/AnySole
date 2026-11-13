#!/usr/bin/env python3
"""Inspect the V8_Cv8U per-slot precision gate K on a checkpoint.

Usage:
  python results_display/script/v8_cv8u_gate_probe.py \
      --ckpt <ckpt_last.pt> --split-csv <splits_test_..._val_...csv> --split val

Outputs a CSV with mean±std K per slot for VT, plus stance/swing split
(contact_gt>0.5 over either foot). V-only/T-only K are hard-masked to 0/1 by
construction, so only VT is informative.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anysole.data.dataset import AnySoleDataset
from anysole.registry import get_builder
from anysole.types import PART_NAMES

REPO = Path(__file__).resolve().parents[2]
SLOTS = ("root_vel",) + tuple(PART_NAMES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-csv", required=True,
                    help="actual fold CSV (e.g. .../splits_test_S5S6S11_val_S12.csv)")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--session-ids", default="")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    entry = get_builder(cfg["model_name"])
    model = entry.build({**cfg, "d_model": int(cfg["d_model"]),
                         "tw": int(cfg["tw"]),
                         "pose_layers": int(cfg.get("pose_layers", 6)),
                         "dropout": float(cfg.get("dropout", 0.1))})
    model.load_state_dict(ck["model"], strict=True)
    model.eval().cuda()

    ids = [s for s in args.session_ids.split(",") if s] or None
    ds = AnySoleDataset(mode=("eval" if args.split == "val" else "test"),
                        window_length=int(cfg["tw"]),
                        split_csv=Path(args.split_csv), session_ids=ids,
                        contact_method=str(cfg["contact_method"]),
                        tactile_input=str(cfg["tactile_input"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    k_all, stance_all = [], []
    with torch.no_grad():
        for batch in loader:
            cid = torch.zeros(batch["V_feat"].shape[0], dtype=torch.long, device="cuda")
            out = model(batch["V_feat"].cuda(), batch["T_raw"].cuda(),
                        batch["T_phys"].cuda(), cid, T_s2m=batch["T_s2m"].cuda())
            k_slots = out["unc_gate"][..., 0].cpu()        # (B,tw,10)
            k_all.append(k_slots)
            stance = (batch["contact_gt"].mean(dim=-1) > 0.5).cpu()  # (B,tw)
            stance_all.append(stance)
    k = torch.cat(k_all, dim=0)              # (N,tw,10)
    stance = torch.cat(stance_all, dim=0)    # (N,tw)
    rows = []
    for p, name in enumerate(SLOTS):
        vals = k[..., p].reshape(-1).numpy()
        st = k[..., p][stance].numpy()
        sw = k[..., p][~stance].numpy()
        rows.append([name, len(vals), float(np.mean(vals)), float(np.std(vals)),
                     float(np.mean(st)) if len(st) else float("nan"),
                     float(np.mean(sw)) if len(sw) else float("nan")])
    out = Path(args.out) if args.out else (
        REPO / "results" / "diagnostics" / "20261112_v8_cv8u_pilot"
        / ("gate_%s.csv" % args.split))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "n", "K_mean", "K_std", "K_stance", "K_swing"])
        w.writerows(rows)
    print("[csv] %s" % out)
    for r in rows:
        print("%-10s K=%.3f±%.3f stance=%.3f swing=%.3f"
              % (r[0], r[2], r[3], r[4], r[5]))


if __name__ == "__main__":
    main()
