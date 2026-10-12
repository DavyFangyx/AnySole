#!/usr/bin/env python3
"""Per-recording x camera frame-axis statistics for the four-camera extension.

Reads only JPEG file names under the raw rgb farm (no image decode) and
reports, per recording x camera (四相机决策记录 D1 执行路线 #4):

    jpeg_count / first,last frame name / frame interval median, max (ms) /
    gaps > 60ms / first-frame offset vs cam3 (ms) / end drift vs a 40fps axis
    (frames) / manifest n_frames (when the recording maps to a manifest row)

Outputs a CSV + JSON table under ``AnysoleWorkspace/reports/`` plus a compact
console summary.  The verdict gate is "some camera has widespread dropped
frames" -- only then does D1 need re-opening.

Usage:
    python3 AnysoleWorkspace/tool/frame_stats.py
    python3 AnysoleWorkspace/tool/frame_stats.py --dates 20260804,20260807,20260808,20260810
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

DEFAULT_DATES = ("20260804", "20260807", "20260808", "20260810")
REPORTS = ROOT / "AnysoleWorkspace/reports"
MANIFEST = ROOT / "AnysoleWorkspace/protocol/manifests/session_manifest.jsonl"
FPS = 40.0
GAP_S = 0.060


def _day_s(name: str) -> float:
    core = name.split("_")[-1][:-4]
    return int(core[0:2]) * 3600 + int(core[2:4]) * 60 + float(core[4:])


def manifest_n_frames() -> dict[str, int]:
    """recording basename -> manifest n_frames (from the main cam3 manifest)."""
    import json as _json
    lookup: dict[str, int] = {}
    if not MANIFEST.is_file():
        return lookup
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = _json.loads(line)
        try:
            recording = resolve_uri(row["video_path"], must_exist=False)
        except Exception:
            continue
        lookup[recording.name] = int(row["n_frames"])
    return lookup


def camera_row(rec: Path, cam: int, cam3_first_s: float | None,
               manifest_n: dict[str, int]) -> dict:
    names = sorted(p.name for p in (rec / str(cam)).glob("*.jpg"))
    row: dict = {
        "date": rec.parts[-3], "subject": rec.parts[-2], "recording": rec.name,
        "camera": cam, "jpeg_count": len(names),
        "manifest_n_frames": manifest_n.get(rec.name, ""),
    }
    if not names:
        row.update({"first_frame": "", "last_frame": "", "interval_median_ms": "",
                    "interval_max_ms": "", "gaps_over_60ms": "", "offset_vs_cam3_ms": "",
                    "end_drift_40fps_frames": ""})
        return row
    times = np.asarray([_day_s(name) for name in names], dtype=np.float64)
    intervals = np.diff(times) * 1000.0
    row.update({
        "first_frame": names[0], "last_frame": names[-1],
        "interval_median_ms": round(float(np.median(intervals)), 2),
        "interval_max_ms": round(float(intervals.max()), 2),
        "gaps_over_60ms": int((intervals > GAP_S * 1000.0).sum()),
        "offset_vs_cam3_ms": round((times[0] - cam3_first_s) * 1000.0, 2)
        if cam3_first_s is not None and cam != 3 else "",
        "end_drift_40fps_frames": round(float((times[-1] - times[0]) - (len(names) - 1) / FPS) * FPS, 2),
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES))
    args = parser.parse_args()
    dates = [item.strip() for item in args.dates.split(",") if item.strip()]
    rgb_root = resolve_uri("raw://rgb", must_exist=True)
    manifest_n = manifest_n_frames()
    rows: list[dict] = []
    missing_cam3 = []
    for date in dates:
        for rec in sorted((rgb_root / date).glob("*/*")):
            if not rec.is_dir():
                continue
            cam3_names = sorted((rec / "3").glob("*.jpg"))
            cam3_first_s = _day_s(cam3_names[0].name) if cam3_names else None
            if cam3_first_s is None:
                missing_cam3.append(str(rec))
            for cam in (1, 2, 3, 4):
                rows.append(camera_row(rec, cam, cam3_first_s, manifest_n))
    REPORTS.mkdir(parents=True, exist_ok=True)
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS / f"frame_stats_by_camera_{stamp}.csv"
    json_path = REPORTS / f"frame_stats_by_camera_{stamp}.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # compact console summary
    print(f"recs={len(rows) // 4} rows={len(rows)}; wrote {csv_path} / {json_path}")
    if missing_cam3:
        print(f"recs missing cam3: {missing_cam3}", file=sys.stderr)
    for cam in (1, 2, 3, 4):
        subset = [row for row in rows if row["camera"] == cam]
        empty = sum(1 for row in subset if not row["jpeg_count"])
        worst = max((row["gaps_over_60ms"] for row in subset if row["jpeg_count"]),
                    default=0)
        med = round(float(np.median([row["interval_median_ms"] for row in subset
                                     if row["jpeg_count"]])), 2)
        drift = max((abs(row["end_drift_40fps_frames"]) for row in subset
                     if row["jpeg_count"]), default=0.0)
        print(f"cam{cam}: empty_recs={empty} median_interval={med}ms "
              f"max_gaps_over_60ms={worst} max_|end_drift|={drift:.2f}frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
