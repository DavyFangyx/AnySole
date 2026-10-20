#!/usr/bin/env python3
"""Generate the V4_1 first-group queue tasks (2026-10-19, V4_1A方案).

Two configs, self-trained and self-tested per config:
  V4_1               cam3 base (v1.yaml roots);
  V4_1_cam3_occ010   cam3 + bottom 10% bbox occlusion (v1_cam3_occ010.yaml).

V6_1A occ_mix is an optional training-side augmentation carried by the conf
(OCC_MIX=true / P_OCC / P_NULLV / P_NULLT, honored by configs/tools/runner.py)
— default off, never a standalone run (V6_1A方案 §3: 对照 = 同配方 p 全 0).
"""

from __future__ import annotations

import argparse

from common import emit_experiment
from model_registry import MODELS

SUITES = {
    "V4_1": [
        ("v4_1_test", ["V4_1", "V4_1_cam3_occ010", "V4_1_occmix_cam3_occ010"]),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="V4_1",
                        help="comma-separated suite IDs (default: V4_1)")
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
