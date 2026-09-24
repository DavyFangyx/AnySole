#!/usr/bin/env python3
"""Generate train+eval tasks for every registered AnySole base model.

One conf per model (TASK=model_run): the runner trains the model from
random init — the 2026-09-24 independence restructure has no warm-start
lineage — then runs the formal val/test evaluation.  Checkpoints land in
the registry addresses under results/AnySole/, so a model already trained
by any experiment is reused instead of retrained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import CONFIG_DIR, REPO, emit_experiment
from model_registry import MODELS


EXPERIMENT_ID = "all_models"

# States that block re-emission.  Re-running this generator while a model
# is queued, running or already finished must not enqueue a duplicate
# 740-epoch training.  failed/ deliberately does NOT block: the recovery
# path for a failed task is to fix the cause and regenerate (e.g. V4B once
# its partition input has been exported).
_BLOCKING_STATES = ("queue", "running", "done")


def _existing_conf(model_id: str) -> list[Path]:
    hits: list[Path] = []
    for state in _BLOCKING_STATES:
        directory = CONFIG_DIR / state
        if directory.is_dir():
            hits.extend(sorted(directory.glob(f"*__{EXPERIMENT_ID}__{model_id}.conf")))
    return hits


def _warn_v4b_partition(selected: list[str]) -> None:
    """V4B's --part-json is the data-derived kinematic partition
    (model-independent: probe_part_cluster_b_data.py).  The conf is still
    emitted if the file is missing — it fails fast at train startup — but
    the user should know how to regenerate it."""
    if "V4B" not in selected:
        return
    part_json = (MODELS["V4B"].get("train_args") or {}).get("part_json")
    if not part_json:
        return
    path = REPO / str(part_json)
    if path.is_file():
        return
    print("[warn] V4B structural input missing: %s" % path)
    print("[warn] the V4B conf is still emitted, but its training will fail at startup")
    print("[warn] until the file exists.  Recovery chain:")
    print("[warn]   1. regenerate the partition (z_note/probes/probe_part_cluster_b_data.py)")
    print("[warn]   2. python configs/z_gen/all_models.py --models V4B")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs (default: every registered model)")
    parser.add_argument("--force", action="store_true",
                        help="emit confs even when a model already has one queued/running/done")
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()] or list(MODELS)
    unknown = [model_id for model_id in models if model_id not in MODELS]
    if unknown:
        raise SystemExit("unknown model IDs: %s (registered: %s)"
                         % (",".join(unknown), ",".join(MODELS)))

    _warn_v4b_partition(models)

    emitted = 0
    for model_id in models:
        if not args.force:
            existing = _existing_conf(model_id)
            if existing:
                print("[skip] %s: conf already in %s/ (%s; use --force to re-emit)"
                      % (model_id, existing[0].parent.name, existing[0].name))
                continue
        emitted += emit_experiment(EXPERIMENT_ID, [model_id])
    print(f"generated {emitted} queue task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
