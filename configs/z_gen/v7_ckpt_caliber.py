#!/usr/bin/env python3
"""Generate the V7 ckpt 口径对照 queue tasks (2026-10-26 用户裁定).

E2/E5 对照补 best/last 双口径行：last occ 12 格（重跑批次权重）+
best cam3 3 格 + best occ 12 格，全部 eval-only 零重训。

用法：
  python configs/z_gen/v7_ckpt_caliber.py [--models a,b,...]
"""

from __future__ import annotations

import argparse

from common import emit_experiment

EXPERIMENT_ID = "v7_ckpt_caliber"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs to subset the group")
    args = parser.parse_args()
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    n = emit_experiment(EXPERIMENT_ID, selected or None)
    print(f"generated {n} queue task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
