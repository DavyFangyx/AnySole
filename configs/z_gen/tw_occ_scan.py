#!/usr/bin/env python3
"""Generate the tw x occlusion interaction queue tasks (2026-10-24).

occ020/030 x tw{5,10,15,20} on the production v3+offset facts (v3off axis),
V3_Bv1 / V5_Av4 (from-scratch V streams, same recipe as the tw_scan occ010
leg).  Response = V2M rows; occlusion penalty vs tw tests the phase-context
hypothesis.  See z_note/multimodal_fusion/Time_Windows/时间窗扫描实验.md §9.
"""

from __future__ import annotations

from common import emit_experiment


def main() -> int:
    return emit_experiment("tw_occ_scan", None)


if __name__ == "__main__":
    raise SystemExit(main())
