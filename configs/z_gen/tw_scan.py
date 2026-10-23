#!/usr/bin/env python3
"""Generate the time-window scan queue tasks (2026-10-22).

Five families x tw{5,10,15,20} x cam3/cam3_occ010 on the production v3+offset
facts (v3off axis); see z_note/multimodal_fusion/时间窗扫描实验.md.

Two waves (the E cells of wave 1 are the carriers of V4_Av1/V5_Av1 wave 2):
  --wave 1   E (V3_Ev1) + V3_Bv1 + V5_Av4 cells            (21 confs)
  --wave 2   V4_Av1 + V5_Av1 cells (gate: wave-1 E done)   (14 confs)
"""

from __future__ import annotations

import argparse

from common import emit_experiment

WAVES = {
    1: ["V3_Ev1_cam3_v3off_tw5", "V3_Ev1_cam3_v3off_tw10",
        "V3_Ev1_cam3_v3off_tw15", "V3_Ev1_cam3_occ010_v3off",
        "V3_Ev1_cam3_occ010_v3off_tw5", "V3_Ev1_cam3_occ010_v3off_tw10",
        "V3_Ev1_cam3_occ010_v3off_tw15",
        "V3_Bv1_cam3_v3off_tw5", "V3_Bv1_cam3_v3off_tw10",
        "V3_Bv1_cam3_v3off_tw15", "V3_Bv1_cam3_occ010_v3off",
        "V3_Bv1_cam3_occ010_v3off_tw5", "V3_Bv1_cam3_occ010_v3off_tw10",
        "V3_Bv1_cam3_occ010_v3off_tw15",
        "V5_Av4_cam3_v3off_tw5", "V5_Av4_cam3_v3off_tw10",
        "V5_Av4_cam3_v3off_tw15", "V5_Av4_cam3_occ010_v3off",
        "V5_Av4_cam3_occ010_v3off_tw5", "V5_Av4_cam3_occ010_v3off_tw10",
        "V5_Av4_cam3_occ010_v3off_tw15"],
    2: ["V4_Av1_cam3_v3off_tw5", "V4_Av1_cam3_v3off_tw10",
        "V4_Av1_cam3_v3off_tw15", "V4_Av1_cam3_occ010_v3off",
        "V4_Av1_cam3_occ010_v3off_tw5", "V4_Av1_cam3_occ010_v3off_tw10",
        "V4_Av1_cam3_occ010_v3off_tw15",
        "V5_Av1_cam3_v3off_tw5", "V5_Av1_cam3_v3off_tw10",
        "V5_Av1_cam3_v3off_tw15", "V5_Av1_cam3_occ010_v3off",
        "V5_Av1_cam3_occ010_v3off_tw5", "V5_Av1_cam3_occ010_v3off_tw10",
        "V5_Av1_cam3_occ010_v3off_tw15"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, default=1, choices=sorted(WAVES),
                        help="wave to emit (default: 1)")
    args = parser.parse_args()
    return emit_experiment("tw_scan", WAVES[args.wave])


if __name__ == "__main__":
    raise SystemExit(main())
