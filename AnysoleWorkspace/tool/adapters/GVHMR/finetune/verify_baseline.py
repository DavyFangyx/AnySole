"""Reproduce the original GVHMR baseline through the new checkpoint entry.

The new ``baseline`` variant must match the archived ``local_gen`` prediction
on the same session/frame set before any fine-tuned checkpoint is compared.

    python -m GVHMR.finetune.verify_baseline --session S11022 \
        --variant baseline_smoke --fold fS5S6S11
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODEL_DIRS = {"gvhmr": "GVHMR", "wham": "WHAM"}


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _diff(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return {"shape_mismatch": [list(left.shape), list(right.shape)]}
    error = np.abs(left - right)
    return {
        "max_abs": float(error.max()) if error.size else 0.0,
        "mean_abs": float(error.mean()) if error.size else 0.0,
        "p99_abs": float(np.percentile(error, 99)) if error.size else 0.0,
    }


def _fold_for(session: str, explicit: str | None) -> list[str]:
    import csv

    if explicit:
        return [explicit]
    folds = []
    for fold in FOLDS:
        path = ROOT / "AnysoleWorkspace/protocol/splits/crossfit_woS13" / f"{fold}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            if any((row.get(fold) or "").strip() == session for row in csv.DictReader(handle)):
                folds.append(fold)
    return folds


def _check_session(model: str, session: str, variant: str, folds: list[str],
                   joint_tol: float, vertex_tol: float) -> dict:
    results = ROOT / "results/baselines" / MODEL_DIRS[model]
    old = results / "predictions/local_gen/eval_motion" / f"{session}.npz"
    new = results / "predictions" / variant / "eval_motion" / f"{session}.npz"
    if not old.is_file() or not new.is_file():
        return {"session_id": session, "pass": False, "missing": [str(old), str(new)]}
    old_data = _load(old)
    new_data = _load(new)
    report = {"session_id": session, "prediction": {}}
    for key in ("joint_xyz_world", "vertices_world"):
        report["prediction"][key] = _diff(old_data[key], new_data[key])
    for key in ("valid_mask", "frame_indices"):
        left = np.asarray(old_data.get(key, []))
        right = np.asarray(new_data.get(key, []))
        report["prediction"][key] = {
            "equal": bool(left.shape == right.shape and np.array_equal(left, right)),
        }
    metrics_report = {}
    for fold in folds or [""]:
        if not fold:
            continue
        metrics_old = results / "metrics" / f"{fold}_comparison.json"
        metrics_new = results / "metrics" / f"{fold}_{variant}_comparison.json"
        if not metrics_old.is_file() or not metrics_new.is_file():
            metrics_report[fold] = {"missing": True}
            continue
        old_payload = json.loads(metrics_old.read_text(encoding="utf-8"))
        new_payload = json.loads(metrics_new.read_text(encoding="utf-8"))
        old_row = next((row for row in old_payload.get("sessions", [])
                        if str(row.get("session_id")) == session), None)
        new_row = next((row for row in new_payload.get("sessions", [])
                        if str(row.get("session_id")) == session), None)
        if old_row is None or new_row is None:
            metrics_report[fold] = {"missing": True}
            continue
        diff = {}
        for key in sorted(set(old_row) & set(new_row)):
            left = old_row[key]
            right = new_row[key]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if np.isfinite(left) and np.isfinite(right):
                    diff[key] = float(abs(left - right))
        metrics_report[fold] = diff
    report["metrics_abs_diff"] = metrics_report
    joint_ok = report["prediction"]["joint_xyz_world"].get("max_abs", 1e9) * 1000.0 <= joint_tol
    vertex_ok = report["prediction"]["vertices_world"].get("max_abs", 1e9) * 1000.0 <= vertex_tol
    mask_ok = all(report["prediction"].get(k, {}).get("equal", False)
                  for k in ("valid_mask", "frame_indices"))
    metric_ok = all(
        not item.get("missing") and max(item.values(), default=0.0) <= 1e-6
        for item in metrics_report.values() if item
    )
    report["pass"] = bool(joint_ok and vertex_ok and mask_ok and metric_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", default="", help="single session or comma-separated")
    parser.add_argument("--all", action="store_true", help="all archived local_gen sessions")
    parser.add_argument("--crossfit", action="store_true",
                        help="union of the four crossfit test sets (intersected with local_gen)")
    parser.add_argument("--model", choices=sorted(MODEL_DIRS), default="gvhmr")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--fold", default="", choices=("",) + FOLDS)
    parser.add_argument("--joint-tolerance-mm", type=float, default=1e-6)
    parser.add_argument("--vertex-tolerance-mm", type=float, default=1e-6)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    results = ROOT / "results/baselines" / MODEL_DIRS[args.model]
    archived = {path.stem for path in (results / "predictions/local_gen/eval_motion").glob("*.npz")}
    if args.crossfit:
        import csv

        sessions = set()
        split_root = ROOT / "AnysoleWorkspace/protocol/splits/crossfit_woS13"
        for fold in FOLDS:
            with (split_root / f"{fold}.csv").open(encoding="utf-8-sig", newline="") as handle:
                sessions.update(
                    (row.get(fold) or "").strip() for row in csv.DictReader(handle)
                    if (row.get(fold) or "").strip()
                )
        sessions = sorted(sessions & archived)
    elif args.all:
        sessions = sorted(archived)
    else:
        sessions = [value.strip() for value in args.session.split(",") if value.strip()]
    if not sessions:
        raise SystemExit("provide --session or --all")
    reports = []
    for session in sessions:
        folds = _fold_for(session, args.fold or None)
        reports.append(_check_session(
            args.model, session, args.variant, folds,
            args.joint_tolerance_mm, args.vertex_tolerance_mm,
        ))
    passed = [item for item in reports if item.get("pass")]
    failed = [item for item in reports if not item.get("pass")]
    summary = {
        "model": args.model,
        "variant": args.variant,
        "sessions": len(reports),
        "passed": len(passed),
        "failed": len(failed),
        "failed_sessions": [item.get("session_id") for item in failed],
        "pass": not failed,
        "details": reports,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("model", "variant", "sessions", "passed", "failed", "failed_sessions", "pass")},
                     indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
