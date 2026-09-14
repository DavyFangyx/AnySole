#!/usr/bin/env python3
"""Write one trial-level MotionPRO split CSV.

Complete sequences are used unless --exclude drops their subjects.
Default assignment is IID: group by subject + action, keep one take in
train, and send the largest trial to val/test when there are multiple
takes. --ood SUBJECTS moves those subjects entirely to val/test.
--exclude SUBJECTS removes those subjects from the CSV. val equals test.
The only output is AnysoleWorkspace/splits/default/splits.csv.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

MOTIONPRO_ROOT = Path(__file__).resolve().parents[1]
if str(MOTIONPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONPRO_ROOT))

from lib.util.workspace import WORKSPACE_ROOT, resolve_path

SEQ_ROOT = WORKSPACE_ROOT / "derived/MotionPRO/sequences"
SPLIT_ROOT = WORKSPACE_ROOT / "splits/default"
SESSION_RE = re.compile(r"^(S\d+?)(\d{2})(\d)$")
REQUIRED_FILES = (
    "align_meta.json",
    "pressure.npz",
    "smpl.npy",
    "contact.npy",
    "keypoints.npy",
    "feature_hrnet.pth",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write MotionPRO split CSV.")
    parser.add_argument(
        "--ood",
        nargs="+",
        metavar="SUBJECT",
        help="Hold-out subjects, e.g. --ood S11,S10. Everyone else stays IID.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        metavar="SUBJECT",
        help="Drop subjects from the split, e.g. --exclude S11,S10.",
    )
    parser.add_argument("--seq-root", type=str, default=None, help="Override the centralized sequence root.")
    parser.add_argument("--out-csv", type=str, default=str(SPLIT_ROOT / "splits.csv"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true", help="Skip when the split CSV exists.")
    mode.add_argument("--force", action="store_true", help="Recompute and overwrite the split CSV.")
    return parser.parse_args()


def parse_session(session_id: str) -> tuple[str, str, int]:
    match = SESSION_RE.match(session_id)
    if match is None:
        raise ValueError(f"Unrecognized session id: {session_id}")
    subject, action, trial = match.groups()
    return subject, action, int(trial)


def sequence_complete(seq_dir: Path) -> bool:
    color_dir = seq_dir / "color"
    if not color_dir.is_dir() or not any(color_dir.iterdir()):
        return False
    return all((seq_dir / name).is_file() for name in REQUIRED_FILES)


def list_sequences(seq_root: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(seq_root.glob("*/*/*/align_meta.json")):
        seq_dir = meta_path.parent
        session_id = seq_dir.name
        subject, action, trial = parse_session(session_id)
        rows.append(
            {
                "session_id": session_id,
                "subject": subject,
                "action": action,
                "trial": trial,
                "complete": sequence_complete(seq_dir),
            }
        )
    return rows


def known_subjects(rows: list[dict]) -> list[str]:
    return sorted({row["subject"] for row in rows}, key=subject_sort_key)


def subject_sort_key(subject: str) -> tuple[int, str]:
    match = re.match(r"^S(\d+)$", subject, flags=re.IGNORECASE)
    if match is None:
        return (10**9, subject)
    return (int(match.group(1)), subject)


def normalize_subject(raw: str, known: list[str]) -> str:
    token = raw.strip()
    if not token:
        raise ValueError("Empty subject name")
    known_map = {name.lower(): name for name in known}
    candidates = [token, token.upper(), token if token.upper().startswith("S") else f"S{token}"]
    for candidate in candidates:
        hit = known_map.get(candidate.lower())
        if hit is not None:
            return hit
        if candidate.upper().startswith("S") and candidate[1:].isdigit():
            compact = "S" + str(int(candidate[1:]))
            hit = known_map.get(compact.lower())
            if hit is not None:
                return hit
    raise ValueError(f"Unknown subject {raw!r}. Known: {', '.join(known)}")


def parse_subjects(values: list[str] | None, known: list[str]) -> list[str]:
    if not values:
        return []
    subjects = []
    for value in values:
        for part in value.replace(";", ",").split(","):
            part = part.strip()
            if part:
                subjects.append(normalize_subject(part, known))
    unique = []
    seen = set()
    for subject in subjects:
        if subject not in seen:
            seen.add(subject)
            unique.append(subject)
    if not unique:
        raise ValueError("No subjects parsed")
    return unique


def assign_splits(
    rows: list[dict],
    ood_subjects: list[str],
    exclude_subjects: list[str] | None = None,
) -> list[dict]:
    held_out = set(ood_subjects)
    excluded = set(exclude_subjects or [])
    assigned = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not row["complete"] or row["subject"] in excluded:
            continue
        if row["subject"] in held_out:
            assigned.append({**row, "split": "test"})
        else:
            groups[(row["subject"], row["action"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (row["trial"], row["session_id"]))
        if len(group) <= 1:
            for row in group:
                assigned.append({**row, "split": "train"})
        else:
            for row in group[:-1]:
                assigned.append({**row, "split": "train"})
            assigned.append({**group[-1], "split": "test"})
    assigned.sort(
        key=lambda row: (
            subject_sort_key(row["subject"]),
            row["action"],
            row["trial"],
            row["session_id"],
        )
    )
    return assigned


def write_split_csv(path: Path, assigned: list[dict]) -> None:
    train = [row["session_id"] for row in assigned if row["split"] == "train"]
    test = [row["session_id"] for row in assigned if row["split"] == "test"]
    n = max(len(train), len(test), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "train", "val", "test"])
        for idx in range(n):
            writer.writerow(
                [
                    idx,
                    train[idx] if idx < len(train) else "",
                    test[idx] if idx < len(test) else "",
                    test[idx] if idx < len(test) else "",
                ]
            )


def remove_stale_files(out_dir: Path) -> None:
    for path in out_dir.glob("split_index.csv"):
        path.unlink()
    for path in out_dir.glob("split_manifest_*.csv"):
        path.unlink()
    for path in out_dir.glob("splits_iid*.csv"):
        path.unlink()
    for path in out_dir.glob("splits_ood*.csv"):
        path.unlink()


def resolve_seq_root(args: argparse.Namespace) -> Path:
    if args.seq_root is not None:
        return Path(resolve_path(args.seq_root, MOTIONPRO_ROOT))
    # Camera selection belongs to sequence preparation; the unified protocol
    # uses cam3 for split generation.
    return SEQ_ROOT / "cam3"


def main() -> int:
    args = parse_args()
    args.out_csv = Path(resolve_path(args.out_csv, MOTIONPRO_ROOT))
    if args.skip_existing and args.out_csv.is_file():
        print(f"skip {args.out_csv} (exists)")
        return 0
    seq_root = resolve_seq_root(args)
    rows = list_sequences(seq_root)
    if not rows:
        raise SystemExit(f"No sequences under {seq_root}")
    known = known_subjects(rows)
    ood_subjects = parse_subjects(args.ood, known)
    exclude_subjects = parse_subjects(args.exclude, known)
    overlap = sorted(set(ood_subjects) & set(exclude_subjects), key=subject_sort_key)
    if overlap:
        ood_subjects = [subject for subject in ood_subjects if subject not in set(exclude_subjects)]
    assigned = assign_splits(rows, ood_subjects, exclude_subjects)
    if not assigned:
        raise SystemExit("No complete sequences to split")
    write_split_csv(args.out_csv, assigned)
    remove_stale_files(args.out_csv.parent)
    n_train = sum(row["split"] == "train" for row in assigned)
    n_test = sum(row["split"] == "test" for row in assigned)
    ood_text = ",".join(ood_subjects) if ood_subjects else "none"
    exclude_text = ",".join(exclude_subjects) if exclude_subjects else "none"
    print(
        f"wrote {args.out_csv} ood={ood_text} exclude={exclude_text} "
        f"train={n_train} val/test={n_test}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
