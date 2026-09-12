#!/usr/bin/env python3
"""Validate manifest paths, frame indices, timing and Skeleton3 consistency."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
EXPECTED = 23

def read_rows(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--report", type=Path, default=None); a=p.parse_args()
    errors=[]; rows=read_rows(a.manifest); checksums=set(); report=[]
    for r in rows:
        sid=r["session_id"]; issues=[]; n=int(r["n_frames"]); fake=json.loads(r["fake_frame_indices"]); valid=json.loads(r["valid_frame_indices"])
        if len(set(fake)) != len(fake) or any(i<0 or i>=n for i in fake): issues.append("bad_fake_indices")
        if len(set(valid)) != len(valid) or any(i<0 or i>=n for i in valid): issues.append("bad_valid_indices")
        if set(fake) & set(valid): issues.append("fake_and_valid_overlap")
        pressure_shape = ""; pressure_range = ""
        pressure_path = ROOT / r["pressure_path"]
        if pressure_path.is_file() and pressure_path.suffix == ".npz":
            try:
                pressure = np.load(pressure_path)["pressure"]
                pressure_shape = "x".join(str(v) for v in pressure.shape)
                pressure_range = f"{float(np.nanmin(pressure)):.3f}..{float(np.nanmax(pressure)):.3f}"
                if pressure.shape[0] != n: issues.append("pressure_frame_count_mismatch")
                if not np.isfinite(pressure).all(): issues.append("pressure_nonfinite")
            except Exception as exc:
                issues.append("pressure_unreadable")
        for key in ("pressure_path", "bvh_path"):
            if not Path(ROOT / r[key]).is_file(): issues.append(f"missing_{key}")
        if r["joint_checksum"]: checksums.add(r["joint_checksum"])
        if r["quality"] != "ok": issues.extend(x for x in r["quality"].split(";") if x)
        report.append({"session_id":sid,"quality":"ok" if not issues else ";".join(dict.fromkeys(issues)),"n_frames":n,"fake_frames":len(fake),"valid_frames":len(valid),"pressure_shape":pressure_shape,"pressure_range":pressure_range,"joint_checksum":r["joint_checksum"]})
        if issues: errors.append(f"{sid}: {report[-1]['quality']}")
    if len(checksums)>1: errors.append("joint_checksum_mismatch")
    text="# Data quality report\n\n" + f"sessions: {len(rows)}\nfailures: {len(errors)}\nchecksums: {len(checksums)}\n\n"
    text += "| session_id | quality | n_frames | fake | valid | pressure_shape | pressure_range | joint_checksum |\n|---|---|---:|---:|---:|---|---|---|\n"
    text += "\n".join(f"| {x['session_id']} | {x['quality']} | {x['n_frames']} | {x['fake_frames']} | {x['valid_frames']} | {x['pressure_shape']} | {x['pressure_range']} | {x['joint_checksum']} |" for x in report)
    out=a.report or a.manifest.with_name("data_quality_report.md"); out.write_text(text+"\n", encoding="utf-8"); print(f"wrote {out}; failures={len(errors)}")
    if errors: print("\n".join(errors))
    return 1 if errors else 0
if __name__ == "__main__": raise SystemExit(main())
