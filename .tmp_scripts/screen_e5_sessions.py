#!/usr/bin/env python3
"""Screening: per-session V2M vs VT2M for E5 occlusion/gap demo selection.

Reads the --dump-detail session_detail_test.json produced by
.tmp_scripts/run_e5_session_detail.sh and ranks sessions where the
vision-only row (V2M) is bad and the VT row is good.

Usage: python .tmp_scripts/screen_e5_sessions.py occ020 [occ090 ...]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/fangyuxuan/projects/gait")
DETAIL = ROOT / "results/diagnostics/20261116_e5_session_detail"
OUT = ROOT / "results/diagnostics/20261116_e5_session_detail/screening"

METRICS = [
    ("mpjpe_mm", "MPJPE"),
    ("pa_mpjpe_mm", "PA-MPJPE"),
    ("PA-MPJPE_anklefoot", "PA_anklefoot"),
    ("MPJPE_anklefoot", "MPJPE_anklefoot"),
    ("global_mpjpe_mm", "GlobalMPJPE"),
    ("ground_pen_mm", "GroundPen"),
    ("foot_sliding_mm", "FootSliding"),
    ("jitter_pred_m_s3", "Jitter_pred"),
]


def load_cond(cond: str) -> list[dict]:
    """Rows keyed by (fold, session): crossfit folds re-use the same session."""
    out: list[dict] = []
    for fold_dir in sorted((DETAIL / "V8_IMU_Residual" / cond).glob("f*")):
        p = fold_dir / "session_detail_test.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        rows = {mode: {r["session_id"]: r for r in d["conditions"].get(mode, [])}
                for mode in ("V2M", "VT2M")}
        for sid, rv in rows["V2M"].items():
            rt = rows["VT2M"].get(sid)
            if rt is None:
                continue
            out.append({
                "fold": fold_dir.name,
                "session_id": sid,
                "n_frames": rv["n_frames"],
                "V": rv["metrics"],
                "VT": rt["metrics"],
            })
    return out


def fnum(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def build_rows(cond: str, sessions: list[dict]) -> list[dict]:
    rows = []
    for rec in sessions:
        sid = rec["session_id"]
        row = {"session_id": sid, "fold": rec["fold"], "cond": cond,
               "n_frames": rec["n_frames"]}
        for key, name in METRICS:
            v = fnum(rec["V"].get(key))
            vt = fnum(rec["VT"].get(key))
            row[name + "_V"] = v
            row[name + "_VT"] = vt
            row[name + "_d"] = v - vt
        # rank score: emphasise MPJPE + PA anklefoot, plus veto on huge VT error
        mp_d = row["MPJPE_V"] - row["MPJPE_VT"]
        pa_d = row["PA_anklefoot_V"] - row["PA_anklefoot_VT"]
        af_d = row["MPJPE_anklefoot_V"] - row["MPJPE_anklefoot_VT"]
        row["score_gap"] = np.nanmean([mp_d, pa_d, af_d])
        row["vt_bad"] = np.nanmean([row["MPJPE_VT"], row["PA_anklefoot_VT"]])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["session_id", "fold", "cond", "n_frames"]
    for _, name in METRICS:
        cols += [name + "_V", name + "_VT", name + "_d"]
    cols += ["score_gap", "vt_bad"]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v)
                            else (round(v, 3) if isinstance(v, float) else v))
                        for k, v in row.items()})
    print("[csv]", path, len(rows))


def main() -> None:
    conds = sys.argv[1:] or ["occ020", "occ090"]
    for cond in conds:
        sessions = load_cond(cond)
        rows = build_rows(cond, sessions)
        if not rows:
            print("no sessions for", cond)
            continue
        write_csv(OUT / ("E5_screen_%s.csv" % cond), rows)
        # console top list with a VT absolute-error sanity filter
        good = [r for r in rows if np.isfinite(r["score_gap"])]
        good.sort(key=lambda r: r["score_gap"], reverse=True)
        print("\n== %s == sessions=%d  top by (V-VT) gap" % (cond, len(rows)))
        hdr = ("sid fold n | MPJPE V/VT/d | PA_af V/VT/d | AF V/VT/d | "
               "GP V/VT | FS V/VT")
        print(hdr)
        for r in good[:15]:
            print("%s %s %4d | %6.1f %5.1f %6.1f | %6.1f %5.1f %6.1f | "
                  "%6.1f %5.1f %6.1f | %5.1f %4.1f | %5.1f %4.1f" % (
                      r["session_id"], r["fold"], r["n_frames"],
                      r["MPJPE_V"], r["MPJPE_VT"], r["MPJPE_d"],
                      r["PA_anklefoot_V"], r["PA_anklefoot_VT"],
                      r["PA_anklefoot_d"],
                      r["MPJPE_anklefoot_V"], r["MPJPE_anklefoot_VT"],
                      r["MPJPE_anklefoot_d"],
                      r["GroundPen_V"], r["GroundPen_VT"],
                      r["FootSliding_V"], r["FootSliding_VT"]))


if __name__ == "__main__":
    main()
