#!/usr/bin/env python3
"""Generate the V-degradation (V削弱) experiment group queue tasks.

Two suites, each 8 configs self-trained and self-tested per config
(V削弱实验报告.md):

  V3_3B  融合主线 -- singlemodal_eval_cams (cam1/2/4 main + vonly specialists)
                    + singlemodal_eval_occl (cam3+occ010/020/030/050 + vonly);
  V3_3D  干净通道 -- position_test_vdegrad (7 per-config position-head models).

The cam3 baseline rows of both suites reuse existing runs (singlemodal_eval
V3_3B / position_test V3_3D) and are NOT re-emitted.
"""

from __future__ import annotations

import argparse

from common import emit_experiment
from model_registry import MODELS

# Suite -> (experiment id, model ids).  The task rows derive from the
# experiment tables in common.EXPERIMENTS (specialists included there).
SUITES = {
    "V3_3B": [
        ("singlemodal_eval_cams",
         ["V3_3B_cam1", "V3_3B_cam2", "V3_3B_cam4"]),
        ("singlemodal_eval_occl",
         ["V3_3B_cam3_occ010", "V3_3B_cam3_occ020",
          "V3_3B_cam3_occ030", "V3_3B_cam3_occ050"]),
    ],
    "V3_3D": [
        ("position_test_vdegrad",
         ["V3_3D_cam1", "V3_3D_cam2", "V3_3D_cam4",
          "V3_3D_cam3_occ010", "V3_3D_cam3_occ020",
          "V3_3D_cam3_occ030", "V3_3D_cam3_occ050"]),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="V3_3B,V3_3D",
                        help="comma-separated suite IDs (default: both)")
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs to subset the group")
    args = parser.parse_args()
    suites = [item.strip() for item in args.suite.split(",") if item.strip()]
    unknown_suite = [s for s in suites if s not in SUITES]
    if unknown_suite:
        raise SystemExit("unknown suite IDs: %s (registered: %s)"
                         % (",".join(unknown_suite), ",".join(SUITES)))
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    if selected:
        unknown = [model_id for model_id in selected if model_id not in MODELS]
        if unknown:
            raise SystemExit("unknown model IDs: %s (registered: %s)"
                             % (",".join(unknown), ",".join(MODELS)))
    selected_set = set(selected) or None

    total = 0
    for suite in suites:
        for experiment_id, model_ids in SUITES[suite]:
            models = [m for m in model_ids
                      if selected_set is None or m in selected_set]
            if models:
                total += emit_experiment(experiment_id, models)
    print(f"generated {total} queue task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
