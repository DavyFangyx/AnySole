#!/usr/bin/env python3
"""2-D fixed gate scan on a frozen Cv7_PhysIMU base (validation split).

Gates are independent: g_V scales A_V, g_T scales A_T.
Grid includes (1,1), the original no-gate base.  Error is pose_std-normalized
6-D MSE, matching the NLL/diagnostic convention (not plain 6D MSE).

Usage:
  PYTHONPATH=. python results_display/script/frozen_gate_2d_scan.py \
    --ckpt <V8_Cv7_PhysIMU fold ckpt> \
    --split-csv <fold csv> --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anysole.data.dataset import AnySoleDataset
from anysole.registry import get_builder

REPO = Path(__file__).resolve().parents[2]
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--session-ids", default="")
    ap.add_argument("--cache-root", default="",
                    help="optional V cache root (e.g. cam3_occ_lower050)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = get_builder(cfg["model_name"]).build(
        {**cfg, "d_model": int(cfg["d_model"]), "tw": int(cfg["tw"]),
         "pose_layers": int(cfg.get("pose_layers", 6)),
         "dropout": float(cfg.get("dropout", 0.1))})
    model.load_state_dict(ck["model"], strict=True)
    model.eval().cuda()

    ids = [x for x in args.session_ids.split(",") if x] or None
    cache_root = Path(args.cache_root) if args.cache_root else None
    ds = AnySoleDataset(mode=("eval" if args.split == "val" else "test"),
                        window_length=int(cfg["tw"]), session_ids=ids,
                        split_csv=Path(args.split_csv),
                        **({"cache_root": cache_root} if cache_root else {}),
                        contact_method=str(cfg["contact_method"]),
                        tactile_input=str(cfg["tactile_input"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # (gV,gT) -> list of per-window errors
    errs = {(gv, gt): [] for gv in GRID for gt in GRID}
    errs_no_override = []
    with torch.no_grad():
        for batch in loader:
            bsz = batch["V_feat"].shape[0]
            cid = torch.zeros(bsz, dtype=torch.long, device="cuda")
            gt = batch["pose_gt"].cuda()
            pose_std = None
            for gv in GRID:
                for gt_ in GRID:
                    gate_v = torch.full((bsz, int(cfg["tw"]), 10, 1), gv,
                                        device="cuda", dtype=torch.float32)
                    gate_t = torch.full((bsz, int(cfg["tw"]), 10, 1), gt_,
                                        device="cuda", dtype=torch.float32)
                    out = model(batch["V_feat"].cuda(), batch["T_raw"].cuda(),
                                batch["T_phys"].cuda(), cid,
                                T_s2m=batch["T_s2m"].cuda(),
                                gate_override=gate_t,
                                gate_v_override=gate_v)
                    if pose_std is None:
                        pose_std = out["pose_std"].to(gt.dtype).clamp(min=1e-2)
                    err = (((out["x0_hat"] - gt) / pose_std) ** 2).mean(dim=(1, 2))
                    errs[(gv, gt_)].extend(err.cpu().tolist())
            out = model(batch["V_feat"].cuda(), batch["T_raw"].cuda(),
                        batch["T_phys"].cuda(), cid, T_s2m=batch["T_s2m"].cuda())
            err = (((out["x0_hat"] - gt) / pose_std) ** 2).mean(dim=(1, 2))
            errs_no_override.extend(err.cpu().tolist())

    keys = list(errs)
    stack = np.stack([np.array(errs[k]) for k in keys], axis=1)  # (N, 25)
    means = {str(k): float(np.mean(v)) for k, v in errs.items()}
    best_key = min(keys, key=lambda k: np.mean(errs[k]))
    oracle_idx = stack.argmin(axis=1)
    oracle = stack[np.arange(len(stack)), oracle_idx]
    base_11 = float(np.mean(errs[(1.0, 1.0)]))
    report = {
        "n_windows": int(len(stack)),
        "error_metric": "pose_std-normalized 6D MSE",
        "split": args.split,
        "grid": list(GRID),
        "mean_error_by_gate": {"%.2f,%.2f" % k: v for k, v in
                               zip(keys, [np.mean(errs[k]) for k in keys])},
        "base_1_1": base_11,
        "no_override_error": float(np.mean(errs_no_override)),
        "override_1_1_equals_no_override": bool(
            abs(base_11 - float(np.mean(errs_no_override))) < 1e-9),
        "best_fixed_gate": list(best_key),
        "best_fixed_error": float(np.mean(errs[best_key])),
        "best_fixed_gain_vs_1_1": base_11 - float(np.mean(errs[best_key])),
        "oracle_error": float(oracle.mean()),
        "oracle_gain_vs_1_1": base_11 - float(oracle.mean()),
        "oracle_gate_hist": {
            "%.2f,%.2f" % keys[i]: float((oracle_idx == i).mean())
            for i in range(len(keys))
        },
    }
    out = Path(args.out) if args.out else (
        REPO / "results" / "diagnostics" / "20261112_v8_gate_frozen"
        / ("fixed_gate_2d_%s.json" % args.split))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
