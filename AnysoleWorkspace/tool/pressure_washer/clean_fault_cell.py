#!/usr/bin/env python3
"""Zero the right-foot fault channel (cell 35) in the final fake-marked tactile tree.

Implements R3 option A on the raw data-processing side (Baselines/决策/
05_R1-R3_风险裁定.md §四): per recording, evaluate the D7 fault criterion on
the right-foot channel 34 (1-based 35) over `valid_mask=1 & fake!=1` rows and,
for recordings that hit, zero the whole channel for the whole recording
(session-level zeroing, not per-frame).  Left-foot values are never touched.

Criterion (D7, ON => >1023 reading; identical to the R3 fleet scan):
    CV(cell[hi]) < 0.4  AND  longest consecutive hi-run > 10  AND
    max(cell) > 3 * p99(other 47 channels)

On top of the criterion, three protocol sessions are cleaned by explicit
override (the CV knife-edge quartet of the R3 follow-up, 2026-10-02: their
fault signatures are unambiguous -- max 6.1k-7.7k, max/p99 8.0-12.6, runs
11-25 -- but CV lands on either side of 0.4 depending on resampling):
    S13072, S14033, S6101
(S6071, the fourth member, already hits the CSV-side criterion and needs no
override.)

The tree is modified in place by default (`--output` writes a copy instead);
every modification is recorded in `fault_cell_manifest.csv` at the tree root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
DEFAULT_INPUT = WORKSPACE / "work/data_pipeline/pressure_washer/final_fake_marked"
MANIFEST_PATH = WORKSPACE / "protocol/manifests/session_manifest.jsonl"

FAULT_CHANNEL = 34          # 0-based index among the 48 channels == 1-based 35
CHANNEL_COLUMN = str(FAULT_CHANNEL + 1)
HI_THRESHOLD = 1023.0
CV_LIMIT = 0.4
RUN_LIMIT = 10
RATIO_LIMIT = 3.0
# CV knife-edge sessions cleaned by explicit ruling (see module docstring).
OVERRIDE_SESSIONS = ("S13072", "S14033", "S6101")


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for b in mask:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return int(best)


def cv_of(x: np.ndarray) -> float:
    if not x.size or not x.mean():
        return float("nan")
    return float(x.std() / x.mean())


def evaluate_criterion(cell: np.ndarray, others: np.ndarray) -> dict:
    """D7 criterion on one foot's channel 34 (cell: (rows,), others: (rows, 47))."""
    hi = cell > HI_THRESHOLD
    p99_other = float(np.percentile(others, 99)) if others.size else 0.0
    if hi.sum() == 0:
        return dict(max=float(cell.max()), frac_gt1023=0.0, cv=np.nan, run=0,
                    p99_other=p99_other, ratio=np.nan, hit=False)
    hit = bool(cv_of(cell[hi]) < CV_LIMIT and longest_run(hi) > RUN_LIMIT
               and p99_other > 0 and cell.max() > RATIO_LIMIT * p99_other)
    return dict(max=float(cell.max()), frac_gt1023=float(hi.mean()),
                cv=cv_of(cell[hi]), run=longest_run(hi), p99_other=p99_other,
                ratio=float(cell.max() / p99_other) if p99_other else np.inf, hit=hit)


def read_foot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (values (rows,48), valid, fake, keep) for one foot CSV."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        empty = np.zeros((0, 48), dtype=np.float64)
        return empty, np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=bool)
    values = np.array([[float(row[str(c)]) for c in range(1, 49)] for row in rows], dtype=np.float64)
    valid = np.array([int(float(row.get("valid_mask", 1))) for row in rows], dtype=np.uint8)
    fake = np.array([int(float(row.get("fake", 0))) for row in rows], dtype=np.uint8)
    keep = (valid == 1) & (fake != 1)
    return values, valid, fake, keep


def flagged_channels(right: np.ndarray, left: np.ndarray, keep_r: np.ndarray, keep_l: np.ndarray) -> list[str]:
    """Audit-only: which channels would hit the criterion anywhere (R3 fleet audit)."""
    flagged = []
    for foot, keep, tag in ((right, keep_r, "R"), (left, keep_l, "L")):
        x = foot[keep]
        if not x.shape[0]:
            continue
        for c in range(48):
            cell = x[:, c]
            o = np.delete(x, c, axis=1)
            p99o = np.percentile(o, 99)
            hi = cell > HI_THRESHOLD
            if (p99o > 0 and cell.max() > RATIO_LIMIT * p99o and longest_run(hi) > RUN_LIMIT
                    and cv_of(cell[hi]) < CV_LIMIT):
                flagged.append(f"{tag}{c}")
    return flagged


def override_rec_dirs() -> set[str]:
    """Rec-dir names for OVERRIDE_SESSIONS, resolved via the session manifest."""
    names = set()
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("session_id") in OVERRIDE_SESSIONS and row.get("pressure_path"):
            names.add(row["pressure_path"].rstrip("/").split("/")[-1])
    return names


def zero_column(path: Path, column: str) -> None:
    """Text-level zeroing of one column; every other byte is preserved."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        lines = handle.read().splitlines(keepends=True)
    header = lines[0].rstrip("\r\n").split(",")
    if column not in header:
        raise ValueError(f"{path}: column {column!r} not found in header {header}")
    index = header.index(column)
    for i in range(1, len(lines)):
        if not lines[i].strip():
            continue
        fields = lines[i].split(",")
        fields[index] = "0.0"
        lines[i] = ",".join(fields)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def process_recording(rec_dir: Path, output_dir: Path, in_place: bool, dry_run: bool,
                      overrides: set[str], manifest_rows: list[dict]) -> dict:
    date, subject = rec_dir.parts[-3], rec_dir.parts[-2]
    sid = ""
    protocol = 0
    for row in manifest_rows:
        if row["pressure_path"].rstrip("/").split("/")[-1] == rec_dir.name:
            sid, protocol = row["session_id"], 1
            break
    left_path, right_path = rec_dir / "pressure_left.csv", rec_dir / "pressure_right.csv"
    right, _, _, keep_r = read_foot(right_path)
    left, _, _, keep_l = read_foot(left_path)
    if not in_place and not dry_run:
        out_rec = output_dir / rec_dir.parent.parent.name / rec_dir.parent.name / rec_dir.name
        out_rec.mkdir(parents=True, exist_ok=True)
        shutil.copy2(left_path, out_rec / left_path.name)
        shutil.copy2(right_path, out_rec / right_path.name)
        right_path = out_rec / right_path.name

    base = dict(rec_dir=rec_dir.name, date=date, subject=subject, sid=sid, protocol=protocol)
    if not keep_r.any():
        base.update(action="none", note="no keep rows")
        return base
    eval_r = evaluate_criterion(right[keep_r][:, FAULT_CHANNEL], np.delete(right[keep_r], FAULT_CHANNEL, axis=1))
    if keep_l.any():
        eval_l = evaluate_criterion(left[keep_l][:, FAULT_CHANNEL], np.delete(left[keep_l], FAULT_CHANNEL, axis=1))
    else:
        eval_l = dict(max=np.nan, cv=np.nan, hit=False)
    override = rec_dir.name in overrides
    hit = bool(eval_r["hit"]) or override
    if hit and not dry_run:
        zero_column(right_path, CHANNEL_COLUMN)
    row = dict(
        **base,
        n_kept=int(keep_r.sum()), max=eval_r["max"], frac_gt1023=eval_r["frac_gt1023"],
        cv_gt1023=eval_r["cv"], longest_gt1023=eval_r["run"], p99_other=eval_r["p99_other"],
        max_over_p99others=eval_r["ratio"], hit_criterion=int(eval_r["hit"]),
        hit_override=int(override), action="zero_R34" if hit else "none",
        l_max=eval_l["max"], l_cv=eval_l["cv"], l_hit=int(eval_l["hit"]),
        flagged_channels=",".join(flagged_channels(right, left, keep_r, keep_l)),
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-input", "--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="Final fake-marked tactile tree (flat: <date>/<S*>/<rec...>).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write a cleaned copy here instead of modifying the input in place.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate criteria and write the manifest only; no zeroing.")
    parser.add_argument("--manifest-only-path", type=Path, default=None,
                        help="Override manifest output path (testing).")
    args = parser.parse_args()

    manifest_rows = [json.loads(line) for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    overrides = override_rec_dirs()
    in_place = args.output_dir is None
    if args.dry_run:
        in_place = False
        args.output_dir = args.output_dir or Path("/tmp/clean_fault_cell_dryrun")

    rec_dirs = sorted(p.parent for p in args.input_dir.glob("*/S*/rec*/pressure_right.csv"))
    if not rec_dirs:
        raise SystemExit(f"No recordings found under {args.input_dir}")
    rows = []
    hits = skips = over = 0
    for rec_dir in rec_dirs:
        row = process_recording(rec_dir, args.output_dir or args.input_dir, in_place,
                                args.dry_run, overrides, manifest_rows)
        rows.append(row)
        if row["action"] == "zero_R34":
            hits += 1
            over += bool(row["hit_override"]) and not row["hit_criterion"]
        else:
            skips += 1

    header = ("# fault_cell_manifest -- R3 cell-35 cleaning (option A, raw data side)\n"
              f"# input_tree: {args.input_dir}\n"
              "# criterion (D7, ON => >1023 reading): CV(cell>1023) < 0.4 AND longest run > 10\n"
              "#   AND max > 3 * p99(other 47 channels), over valid_mask=1 & fake!=1 rows.\n"
              f"# override sessions (CV knife-edge ruling 2026-10-02): {', '.join(OVERRIDE_SESSIONS)}\n"
              "# action zero_R34 = whole right-foot channel 34 set to 0 for this recording; left foot untouched.\n")
    fieldnames = ["rec_dir", "date", "subject", "sid", "protocol", "n_kept", "max",
                  "frac_gt1023", "cv_gt1023", "longest_gt1023", "p99_other",
                  "max_over_p99others", "hit_criterion", "hit_override", "action",
                  "l_max", "l_cv", "l_hit", "flagged_channels", "note"]
    manifest_out = args.manifest_only_path or (args.output_dir or args.input_dir) / "fault_cell_manifest.csv"
    if not args.dry_run or args.manifest_only_path:
        with manifest_out.open("w", encoding="utf-8", newline="") as handle:
            handle.write(header)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"recordings: {len(rows)} | zeroed: {hits} (override-only: {over}) | untouched: {skips}")
    print(f"manifest: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
