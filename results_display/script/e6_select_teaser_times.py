#!/usr/bin/env python3
"""Select teaser moments that show the T benefit growing with occlusion.

B_T(t,p,c) = E_V(t,p,c) - E_VT(t,p,c)   (positive = T helps)
Score lower-body frames by B_T(occ100) - B_T(clean) and B_T(occ100).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
import e6_ft_contribution as e6
import e6_part_error_teaser as P
from anysole.types import PART_JOINTS, PART_NAMES

LOWER = (5, 6, 7, 8)  # l_leg,r_leg,l_foot,r_foot

def part_matrix(full):
    out = np.full((full.shape[0], len(PART_NAMES)), np.nan, np.float32)
    for i, joints in enumerate(PART_JOINTS):
        out[:, i] = np.nanmean(full[:, list(joints)], axis=1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--session-id", default="S11051")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    split_csv = e6.SPLITS / e6.FOLDS[args.fold]
    model, cfg, _ = e6._load_model(e6.DIAG / ("router_ft_only_%s_v2rerun.pt" % args.fold))
    device = torch.device(args.device)
    model.to(device).eval()
    raw = P.collect(model, cfg, split_csv, args.session_id, e6.DEFAULT_VISUAL,
                    device, 16)
    part = {m: {c: part_matrix(P.part_values(raw[m][c])) for c in raw[m]}
            for m in raw}
    conds = [c for c, _ in e6.CONDS]
    # benefit B = E_V - E_VT
    B = {c: part["V"][c] - part["VT"][c] for c in conds}
    n = min(v.shape[0] for v in B.values())
    trend = np.nanmean(B["occ100"][:n][:, LOWER] - B["clean"][:n][:, LOWER], axis=1)
    b100 = np.nanmean(B["occ100"][:n][:, LOWER], axis=1)
    b50 = np.nanmean(B["occ050"][:n][:, LOWER], axis=1)
    b20 = np.nanmean(B["occ020"][:n][:, LOWER], axis=1)
    score = trend + 0.5 * b100
    rows = []
    for t in range(n):
        if not np.isfinite(score[t]):
            continue
        rows.append((t, float(score[t]), float(trend[t]), float(b100[t]),
                     float(b50[t]), float(b20[t])))
    rows.sort(key=lambda r: -r[1])
    print("top frames: frame score trend B100 B50 B20")
    for r in rows[:20]:
        print("  %4d %.2f %.2f %.2f %.2f %.2f" % r)
    # greedy spacing >= 40 frames (1 s)
    picks, used = [], []
    for r in rows:
        if all(abs(r[0] - u) >= 40 for u in used):
            picks.append(r); used.append(r[0])
        if len(picks) == 6:
            break
    picks.sort()
    print("picked:", [p[0] for p in picks])
    out = e6.DEFAULT_OUT.parents[1] / "detail" / "teaser_selection"
    out.mkdir(parents=True, exist_ok=True)
    p = out / ("E6_TeaserSelection_%s.csv" % args.session_id)
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "score", "trend", "B_occ100", "B_occ050",
                    "B_occ020", "selected"])
        sel = {x[0] for x in picks}
        for r in rows:
            w.writerow(list(r) + [int(r[0] in sel)])
    print("wrote", p)
    (out / ("E6_TeaserSelection_%s.json" % args.session_id)).write_text(
        json.dumps({"picks": [p[0] for p in picks],
                    "rows": picks}, indent=1))

if __name__ == "__main__":
    main()
