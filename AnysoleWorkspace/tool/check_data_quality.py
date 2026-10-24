#!/usr/bin/env python3
"""Validate manifest paths, frame indices, timing and Skeleton3 consistency."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402
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
        try:
            pressure_dir = resolve_uri(r["pressure_path"])  # hfraw://…/dataset/{date}/{subject}/{sid}
            left = np.load(pressure_dir / "left.npz")["processed48"]
            right = np.load(pressure_dir / "right.npz")["processed48"]
            n_rows = min(len(left), len(right))  # 左右脚行数可不等，取交集对齐
            pressure = np.concatenate([left[:n_rows], right[:n_rows]], axis=1)
            pressure_shape = "x".join(str(v) for v in pressure.shape)
            pressure_range = f"{float(np.nanmin(pressure)):.3f}..{float(np.nanmax(pressure)):.3f}"
            # 行级 npz 与视频帧数不同空间（build_shared 按墙钟重采样到帧），
            # 不再对比帧数；仅查值域与有限性
            if not np.isfinite(pressure).all(): issues.append("pressure_nonfinite")
        except Exception:
            issues.append("pressure_unreadable")
        for key in ("pressure_path", "bvh_path", "smpl_path"):
            if key not in r:
                continue  # manifests generated before the SMPL migration
            try:
                resolved = resolve_uri(r[key])
            except Exception:
                issues.append(f"missing_{key}")
                continue
            ok = resolved.is_file() if key != "pressure_path" else (resolved / "left.npz").is_file()
            if not ok: issues.append(f"missing_{key}")
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
