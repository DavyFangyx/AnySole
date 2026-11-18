#!/usr/bin/env python3
"""Merge our V2M/VT2M per-session rows with visual-baseline per-session rows
and print demo candidates for E5 occlusion / time-gap conditions."""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/fangyuxuan/projects/gait")
SCREEN = ROOT / "results/diagnostics/20261116_e5_session_detail/screening"
COND_FILE = {
    "occ020": "baseline_e5_occ_lower020.csv",
    "occ090": "baseline_e5_occ_lower090.csv",
    "gap060": "baseline_e5_gap_r060.csv",
    "gap080": "baseline_e5_gap_r080.csv",
}
ACTION = {
    "01": "stand-shift", "02": "single-stand", "03": "shuttle-walk",
    "04": "shuttle-walk-fast", "05": "run", "06": "squat",
    "07": "jump-jack", "08": "shuttle-run", "09": "lateral-walk",
    "10": "sit-stand", "11": "stairs",
}


def load_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def load_manifests() -> dict[str, dict]:
    out = {}
    for p in glob.glob(str(ROOT / "AnysoleWorkspace/protocol/splits/crossfit_woS13/*_manifest.jsonl")):
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["session_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conds", nargs="+", default=["occ020", "occ090"])
    ap.add_argument("--vt-mpjpe-max", type=float, default=100.0)
    ap.add_argument("--vt-pa-max", type=float, default=85.0)
    ap.add_argument("--min-baseline", type=float, default=120.0,
                    help="at least one visual baseline MPJPE above this")
    args = ap.parse_args()
    mans = load_manifests()

    for cond in args.conds:
        ours = load_csv(SCREEN / ("E5_screen_%s.csv" % cond))
        bpath = SCREEN / COND_FILE[cond]
        base = load_csv(bpath) if bpath.is_file() else []
        bm = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in base:
            bm[r["session_id"]][r["model"]].append(r)

        def pick(sid: str, model: str, fold: str) -> dict | None:
            cand = bm.get(sid, {}).get(model)
            if not cand:
                return None
            if model == "MotionPRO":
                match = [x for x in cand if x["fold"] == fold]
                if match:
                    return match[0]
            return cand[0]

        cands = []
        for r in ours:
            sid, fold = r["session_id"], r["fold"]
            vt_mp, vt_pa = fnum(r, "MPJPE_VT"), fnum(r, "PA_anklefoot_VT")
            v_mp, v_pa = fnum(r, "MPJPE_V"), fnum(r, "PA_anklefoot_V")
            if not (vt_mp <= args.vt_mpjpe_max and vt_pa <= args.vt_pa_max):
                continue
            bs = {m: pick(sid, m, fold) for m in ("WHAM", "GVHMR", "MotionPRO")}
            b_mp = {m: fnum(x, "mpjpe_mm") for m, x in bs.items() if x}
            if not b_mp or max(b_mp.values()) < args.min_baseline:
                continue
            max_b = max(b_mp.values())
            drop = (v_mp - vt_mp) + (v_pa - vt_pa)
            cands.append(dict(r, fold=fold, action=ACTION.get(mans[sid]["action"], "?"),
                              trial=mans[sid]["trial"], subject=mans[sid]["subject_id"],
                              b_mp=b_mp, max_b=max_b, drop=drop,
                              b_pa={m: fnum(x, "mpjpe_anklefoot_mm")
                                    for m, x in bs.items() if x}))
        cands.sort(key=lambda c: (-(c["max_b"] / max(fnum(c, "MPJPE_VT"), 1.0)),
                                  -c["drop"]))
        print("\n" + "=" * 30, cond, "candidates:", len(cands))
        hdr = ("sid     fold       action            n   | OURS V->VT MPJPE | OURS V->VT PAaf | "
               "WHAM MPJPE PAaf | GVHMR | MPRO | GP V->VT | FS V->VT")
        print(hdr)
        for c in cands[:20]:
            def bstr(m):
                return ("%5.0f/%5.0f" % (c["b_mp"].get(m, float("nan")),
                                         c["b_pa"].get(m, float("nan"))))
            print("%-7s %-10s %-17s %4s | %6.1f -> %5.1f | %6.1f -> %5.1f | %s %s %s | %5.1f->%4.1f | %4.1f->%4.1f" % (
                c["session_id"], c["fold"], "%s t%s %s" % (c["action"], c["trial"], c["subject"]),
                c["n_frames"],
                fnum(c, "MPJPE_V"), fnum(c, "MPJPE_VT"),
                fnum(c, "PA_anklefoot_V"), fnum(c, "PA_anklefoot_VT"),
                bstr("WHAM"), bstr("GVHMR"), bstr("MotionPRO"),
                fnum(c, "GroundPen_V"), fnum(c, "GroundPen_VT"),
                fnum(c, "FootSliding_V"), fnum(c, "FootSliding_VT")))


if __name__ == "__main__":
    main()
