#!/usr/bin/env python3
"""Generate the V3_Hv1 first-group queue tasks (2026-10-17).

Two configs, self-trained and self-tested per config:
  V3_Hv1               cam3 base (v1.yaml roots);
  V3_Hv1_cam3_occ010   cam3 + bottom 10% bbox occlusion (v1_cam3_occ010.yaml).
"""

from __future__ import annotations

import argparse

from common import emit_experiment
from model_registry import MODELS

SUITES = {
    "V3_Hv1": [
        ("v3h_test", ["V3_Hv1", "V3_Hv1_cam3_occ010"]),
    ],
    "V3_Hv2": [
        ("v3h2_test", ["V3_Hv2", "V3_Hv2_cam3_occ010"]),
    ],
    "V3_Hv3": [
        ("v3h3_test", ["V3_Hv3", "V3_Hv3_cam3_occ010"]),
    ],
    "V3_Hv5": [
        ("v3h5_test", ["V3_Hv5", "V3_Hv5_cam3_occ010"]),
    ],
    "V3_Hv6": [
        ("v3h6_test", ["V3_Hv6", "V3_Hv6_cam3_occ010"]),
    ],
    "V3_Hv5_vdegrad": [
        ("v3h5_test", ["V3_Hv5_cam1", "V3_Hv5_cam2",
                       "V3_Hv5_cam4", "V3_Hv5_cam3_occ020"]),
    ],
    "V3_Hv5_grid": [
        ("v3h5_test", ["V3_Hv5_cam1_occ010", "V3_Hv5_cam1_occ020",
                       "V3_Hv5_cam2_occ010", "V3_Hv5_cam2_occ020",
                       "V3_Hv5_cam4_occ010", "V3_Hv5_cam4_occ020"]),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="V3_Hv1",
                        help="comma-separated suite IDs (default: V3_Hv1)")
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
