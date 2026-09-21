"""Audit BVH-derived contact labels and class balance for AnySole.

This probe is intentionally label-only and cheap.  It verifies that every
stored label agrees frame-for-frame with the unchanged BVH contact generator,
reports contact/air class balance per split, and compares trivial predictors.
With heavily imbalanced ``joint_and`` labels, ordinary accuracy or
contact-positive F1 can look strong for a model that has learned no airborne
phase; balanced accuracy and air recall expose it.

Run from the repository root::

    python results_display/script/probe_contact_balance.py --method joint_and
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from anysole.data.dataset import find_session_dir
from anysole.types import SEQ_ROOT, SPLIT_CSV
from contact_methods import apply_method, foot_signal, load_aligned, soft_prob


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    gt = np.asarray(gt, dtype=bool).reshape(-1)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    precision = tp / max(tp + fp, 1)
    contact_recall = tp / max(tp + fn, 1)
    air_recall = tn / max(tn + fp, 1)
    return {
        "acc": (tp + tn) / max(len(gt), 1),
        "contact_f1": 2.0 * precision * contact_recall / max(precision + contact_recall, 1e-12),
        "contact_recall": contact_recall,
        "air_recall": air_recall,
        "balanced_acc": 0.5 * (contact_recall + air_recall),
    }


def _fmt(values: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in values.items())


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AnySole contact labels and trivial baselines.")
    parser.add_argument("--method", default="joint_and")
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV)
    parser.add_argument("--seq-root", type=Path, default=SEQ_ROOT)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--skip-kinematic-ceiling", action="store_true",
        help="Skip the slower BVH soft-contact versus label comparison.",
    )
    args = parser.parse_args()

    with args.split_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_split: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    all_labels: list[np.ndarray] = []
    all_contact_sessions = 0
    bad_sidecars: list[str] = []
    mismatched_bvh_labels: list[str] = []
    session_records: list[tuple[Path, np.ndarray]] = []
    session_count = 0
    for split in ("train", "val", "test"):
        for row in rows:
            sid = str(row.get(split, "")).strip()
            if not sid:
                continue
            session_count += 1
            seq_dir = find_session_dir(args.seq_root, sid)
            label_path = seq_dir / f"contact_{args.method}.npy"
            sidecar_path = label_path.with_suffix(".json")
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            raw = np.load(label_path)
            if raw.ndim != 2 or raw.shape[1] < 8:
                raise ValueError(f"{label_path}: expected (T,>=8), got {raw.shape}")
            labels = raw[:, [6, 7]] > 0.5
            by_split[split].append(labels)
            all_labels.append(labels)
            session_records.append((seq_dir, labels))
            all_contact_sessions += int(bool(labels.all()))
            try:
                sidecar = json.loads(sidecar_path.read_text())
            except (OSError, json.JSONDecodeError):
                sidecar = {}
            if sidecar.get("method") != args.method:
                bad_sidecars.append(sid)

            # Do not trust provenance metadata alone: recompute through the
            # original BVH pipeline and compare the actual label values.
            ctx = load_aligned(seq_dir)
            regenerated, _ = apply_method(ctx, args.method, None)
            regenerated_feet = regenerated[:, [6, 7]]
            if regenerated_feet.shape != labels.shape or not np.array_equal(
                regenerated_feet > 0.5, labels
            ):
                mismatched_bvh_labels.append(sid)

    if bad_sidecars:
        raise AssertionError(f"missing/wrong contact method sidecars: {bad_sidecars}")
    if mismatched_bvh_labels:
        raise AssertionError(
            "stored labels differ from the original BVH generator: "
            f"{mismatched_bvh_labels}"
        )
    labels = np.concatenate(all_labels, axis=0).reshape(-1)
    prevalence = float(labels.mean())
    rng = np.random.RandomState(args.seed)
    uniform = rng.rand(labels.size) < 0.5
    matched = rng.rand(labels.size) < prevalence

    print(
        f"method={args.method} motion_source=direct_export_bvh "
        f"sidecars={session_count} bad=0 bvh_value_mismatches=0"
    )
    for split in ("train", "val", "test"):
        values = np.concatenate(by_split.get(split, []), axis=0).reshape(-1)
        print(f"{split}: labels={values.size} contact={values.mean():.4%} air={(~values).mean():.4%}")
    print(
        f"all: labels={labels.size} contact={prevalence:.4%} air={(~labels).mean():.4%} "
        f"all-contact_sessions={all_contact_sessions}/{session_count}"
    )
    print("always_contact: " + _fmt(_metrics(np.ones_like(labels), labels)))
    print("random_p0.5:    " + _fmt(_metrics(uniform, labels)))
    print("random_matched: " + _fmt(_metrics(matched, labels)))

    if not args.skip_kinematic_ceiling:
        ceiling_pred = []
        ceiling_gt = []
        for seq_dir, target in session_records:
            ctx = load_aligned(seq_dir)
            pred_feet = []
            for side in ("left", "right"):
                height, speed = foot_signal(ctx, side)
                # losses.soft_contact_from_keypoints assigns the first frame
                # the first available forward-difference speed.
                if speed.size > 1:
                    speed[0] = speed[1]
                pred_feet.append(soft_prob(height, speed) > 0.5)
            pred = np.stack(pred_feet, axis=1)
            n = min(len(pred), len(target))
            ceiling_pred.append(pred[:n])
            ceiling_gt.append(target[:n])
        gt_pred = np.concatenate(ceiling_pred, axis=0)
        gt_target = np.concatenate(ceiling_gt, axis=0)
        print("GT_BVH_soft_contact_comparison: " + _fmt(_metrics(gt_pred, gt_target)))

    if prevalence > 0.90:
        print(
            "WARNING: labels are >90% contact. contact_acc/contact_f1 alone are not valid "
            "health checks; require air_recall and balanced_acc."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
