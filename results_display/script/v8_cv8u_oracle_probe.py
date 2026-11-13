#!/usr/bin/env python3
"""Zero-training oracle probe for the V8_Cv8U gate on PhysIMU.

For each VT window, override K with a grid and compute the pose error:
  K=0 -> pure V branch; K=1 -> pure T branch; 0<K<1 -> weighted mix.
Report the best fixed K, the per-window oracle K, and the oracle gain over
K=0 / K=1.  This bounds what any gate can achieve on this checkpoint.
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
    ap.add_argument("--split", default="test", choices=("val", "test"))
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

    ds = AnySoleDataset(mode=("eval" if args.split == "val" else "test"),
                        window_length=int(cfg["tw"]),
                        split_csv=Path(args.split_csv),
                        contact_method=str(cfg["contact_method"]),
                        tactile_input=str(cfg["tactile_input"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    errs = {k: [] for k in GRID}
    with torch.no_grad():
        for batch in loader:
            bsz = batch["V_feat"].shape[0]
            cid = torch.zeros(bsz, dtype=torch.long)
            gt = batch["pose_gt"].cuda()
            for k in GRID:
                gate = torch.full((bsz, int(cfg["tw"]), 10, 1), k,
                                  device="cuda", dtype=torch.float32)
                out = model(batch["V_feat"].cuda(), batch["T_raw"].cuda(),
                            batch["T_phys"].cuda(), cid.cuda(),
                            T_s2m=batch["T_s2m"].cuda(),
                            gate_override=gate,
                            gate_v_override=1.0 - gate)
                err = ((out["x0_hat"] - gt) ** 2).mean(dim=(1, 2))
                errs[k].extend(err.cpu().tolist())
    errs = {k: np.array(v) for k, v in errs.items()}
    stack = np.stack([errs[k] for k in GRID], axis=1)  # (N,5)
    best_fixed = min(GRID, key=lambda k: errs[k].mean())
    oracle_idx = stack.argmin(axis=1)
    oracle = stack[np.arange(len(stack)), oracle_idx]
    report = {
        "n_windows": int(len(stack)),
        "mean_error_by_K": {str(k): float(errs[k].mean()) for k in GRID},
        "best_fixed_K": float(best_fixed),
        "best_fixed_error": float(errs[best_fixed].mean()),
        "oracle_error": float(oracle.mean()),
        "oracle_gain_vs_K0": float(errs[0.0].mean() - oracle.mean()),
        "oracle_gain_vs_K1": float(errs[1.0].mean() - oracle.mean()),
        "oracle_K_hist": {str(k): float((oracle_idx == i).mean())
                          for i, k in enumerate(GRID)},
    }
    out = Path(args.out) if args.out else (
        REPO / "results" / "diagnostics" / "20261112_v8_cv8u_pilot"
        / ("oracle_%s.json" % args.split))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
