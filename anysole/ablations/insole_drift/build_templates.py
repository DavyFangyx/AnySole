"""Build one automatically selected left/right template per subject."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from anysole.data.pressure import cop_from_grid


def pressure_to_feet(pressure):
    """Convert MotionPRO raster pressure frames to left/right 48-cell arrays."""
    pressure = np.asarray(pressure, dtype=np.float32)
    if pressure.ndim == 2 and pressure.shape[1] == 96:
        return pressure[:, :48], pressure[:, 48:]
    if pressure.ndim != 3 or pressure.shape[1:] != (160, 120):
        raise ValueError("pressure must have shape (frames, 96) or (frames, 160, 120), got %s" % (pressure.shape,))
    left = pressure[:, 40:120, 6:54].reshape(-1, 4, 20, 12, 4).mean(axis=(2, 4))
    right = pressure[:, 40:120, 66:114].reshape(-1, 4, 20, 12, 4).mean(axis=(2, 4))
    return left.reshape(-1, 48), right.reshape(-1, 48)


def select_template(left, right, valid):
    force_l, force_r = left.sum(1), right.sum(1)
    cop_l, cop_r = cop_from_grid(left), cop_from_grid(right)
    center = np.linalg.norm(cop_l - .5, axis=1) + np.linalg.norm(cop_r - .5, axis=1)
    score = np.minimum(force_l, force_r) / (1.0 + center)
    score[~valid] = -np.inf
    best = int(np.argmax(score))
    lo, hi = max(0, best - 40), min(len(left), best + 41)
    return np.quantile(left[lo:hi], .95, axis=0), np.quantile(right[lo:hi], .95, axis=0), best, (lo, hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import csv
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    grouped = {}
    for row in rows:
        path = Path(row["pressure_path"])
        if not path.is_absolute(): path = args.manifest.parents[2] / path
        if path.is_file(): grouped.setdefault(row["subject_id"], []).append((row["session_id"], path))
    templates, records = [], []
    for subject, sessions in sorted(grouped.items()):
        candidates = []
        for sid, path in sessions:
            p = np.load(path)["pressure"].astype(np.float32)
            try:
                left, right = pressure_to_feet(p)
            except ValueError:
                continue
            valid = np.isfinite(left).all(1) & np.isfinite(right).all(1)
            l, r, frame, span = select_template(left, right, valid)
            candidates.append((float(np.minimum(l.sum(), r.sum())), sid, l, r, frame, span))
        if not candidates: continue
        _, sid, l, r, frame, span = max(candidates, key=lambda x: x[0])
        templates.append(np.stack((l / max(l.max(), 1e-6), r / max(r.max(), 1e-6))))
        records.append({"subject_id": subject, "session_id": sid, "center_frame": frame, "frame_range": span})
    if not templates:
        raise RuntimeError("No valid pressure sessions found; cannot build insole templates")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"subject_to_index": {r["subject_id"]: i for i, r in enumerate(records)}, "templates": np.asarray(templates).tolist(), "records": records}, indent=2))


if __name__ == "__main__": main()
