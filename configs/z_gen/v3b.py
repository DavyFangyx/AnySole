#!/usr/bin/env python3
"""Generate the V3_Bv1 occ010 production-eval queue task (2026-10-18).

ckpt 已训（2026-10-13，ckpt_best/last 齐），metrics 缺失——只补评估。
数据根按名称尾缀推导 v1_cam3_occ010.yaml（与削弱实验同基座）。
"""

from __future__ import annotations

import argparse

from common import emit_experiment
from model_registry import MODELS

SUITES = {
    "V3_Bv1_cam3_occ010": [
        ("v3b_test", ["V3_Bv1_cam3_occ010"]),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="V3_Bv1_cam3_occ010",
                        help="comma-separated suite IDs")
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs to subset the group")
    args = parser.parse_args()
    suites = [item.strip() for item in args.suite.split(",") if item.strip()]
    unknown_suite = [s for s in suites if s not in SUITES]
    if unknown_suite:
        raise SystemExit("unknown suite IDs: %s" % ",".join(unknown_suite))
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    if selected:
        unknown = [m for m in selected if m not in MODELS]
        if unknown:
            raise SystemExit("unknown model IDs: %s" % ",".join(unknown))
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
