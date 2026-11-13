#!/usr/bin/env python3
"""Coordinate-descent per-group fixed gate on validation (frozen base).

Groups (10 query slots): root=[0,1], upper=[2,3,4,5], left=[6,8], right=[7,9].
Scans (gV,gT) in {0,.25,.5,.75,1}^2 per group, holding the others fixed.
Metric: pose_std-normalized 6D MSE.
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
GROUP_IDX = (0, 0, 1, 1, 1, 1, 2, 3, 2, 3)  # slots: vel, root, torso, head, arms, legs/feet


def evaluate(model, loader, gv_slots, gt_slots, tw):
    vals = []
    with torch.no_grad():
        for batch in loader:
            bsz = batch["V_feat"].shape[0]
            cid = torch.zeros(bsz, dtype=torch.long, device="cuda")
            gt = batch["pose_gt"].cuda()
            gate_v = torch.tensor(gv_slots, device="cuda", dtype=torch.float32)
            gate_t = torch.tensor(gt_slots, device="cuda", dtype=torch.float32)
            gate_v = gate_v.view(1, 1, 10, 1).expand(bsz, tw, -1, -1)
            gate_t = gate_t.view(1, 1, 10, 1).expand(bsz, tw, -1, -1)
            out = model(batch["V_feat"].cuda(), batch["T_raw"].cuda(),
                        batch["T_phys"].cuda(), cid,
                        T_s2m=batch["T_s2m"].cuda(),
                        gate_override=gate_t, gate_v_override=gate_v)
            std = out["pose_std"].to(gt.dtype).clamp(min=1e-2)
            err = (((out["x0_hat"] - gt) / std) ** 2).mean(dim=(1, 2))
            vals.extend(err.cpu().tolist())
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--session-ids", default="")
    ap.add_argument("--cache-root", default="",
                    help="optional V cache root (e.g. cam3_occ_lower050)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    tw = int(cfg["tw"])
    model = get_builder(cfg["model_name"]).build(
        {**cfg, "d_model": int(cfg["d_model"]), "tw": tw,
         "pose_layers": int(cfg.get("pose_layers", 6)),
         "dropout": float(cfg.get("dropout", 0.1))})
    model.load_state_dict(ck["model"], strict=True)
    model.eval().cuda()
    ids = [x for x in args.session_ids.split(",") if x] or None
    cache_root = Path(args.cache_root) if args.cache_root else None
    ds = AnySoleDataset(mode=("eval" if args.split == "val" else "test"),
                        window_length=tw, split_csv=Path(args.split_csv), session_ids=ids,
                        **({"cache_root": cache_root} if cache_root else {}),
                        contact_method=str(cfg["contact_method"]),
                        tactile_input=str(cfg["tactile_input"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # start from global best found by the 2-D scan; fallback (1, 0.75)
    gv = [1.0] * 10
    gt = [0.75] * 10
    base_err = evaluate(model, loader, gv, gt, tw)
    history = []
    for _ in range(args.rounds):
        for g in range(4):
            slots = [i for i, x in enumerate(GROUP_IDX) if x == g]
            best = (gv[slots[0]], gt[slots[0]])
            best_err = evaluate(model, loader, gv, gt, tw)
            for cand_v in GRID:
                for cand_t in GRID:
                    tv, tt = list(gv), list(gt)
                    for s in slots:
                        tv[s], tt[s] = cand_v, cand_t
                    err = evaluate(model, loader, tv, tt, tw)
                    if err < best_err:
                        best_err, best = err, (cand_v, cand_t)
            for s in slots:
                gv[s], gt[s] = best
            history.append({"group": g, "gV": best[0], "gT": best[1],
                            "err": best_err})
    final_err = evaluate(model, loader, gv, gt, tw)
    report = {
        "split": args.split, "n_windows": len(ds),
        "metric": "pose_std-normalized 6D MSE",
        "base_1_1_error": evaluate(model, loader, [1.0]*10, [1.0]*10, tw),
        "start_1_0.75_error": base_err,
        "per_group_gate": [
            {"group": ["root", "upper", "left", "right"][g],
             "slots": [i for i, x in enumerate(GROUP_IDX) if x == g],
             "gV": gv[[i for i, x in enumerate(GROUP_IDX) if x == g][0]],
             "gT": gt[[i for i, x in enumerate(GROUP_IDX) if x == g][0]]}
            for g in range(4)],
        "per_group_error": final_err,
        "gain_vs_1_1": evaluate(model, loader, [1.0]*10, [1.0]*10, tw) - final_err,
        "history": history,
    }
    out = Path(args.out) if args.out else (
        REPO / "results" / "diagnostics" / "20261112_v8_gate_frozen"
        / ("fixed_gate_pergroup_%s.json" % args.split))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
