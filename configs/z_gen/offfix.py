#!/usr/bin/env python3
"""Generate the offset-resync experiment group queue tasks (2026-10-20).

offsetfix (default): V3_Bv1 cam3 基准 + cam3 occ010/020/030/050, all with
corrected_offset=True (B 类 session GT 采样按 z_note/data_offset_fixpipe
修正表平移；A/C/D 不动)。--experiment 可选其他组：
  offsetfix_v45_v2 — V5_Av1/V4_Av1 × cam3/cam3_occ010，触觉 = denoised108_v2。
"""

from __future__ import annotations

import argparse

from common import emit_experiment

EXPERIMENT_ID = "offsetfix"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=EXPERIMENT_ID,
                        help="experiment group id in EXPERIMENTS (default: %(default)s)")
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs to subset the group")
    args = parser.parse_args()
    return emit_experiment(args.experiment,
                           [m.strip() for m in args.models.split(",")
                            if m.strip()] or None)


if __name__ == "__main__":
    raise SystemExit(main())
