#!/usr/bin/env python3
"""Generate the V7 first-round training queue tasks (2026-10-25 计划步骤 2).

三个 U（候选 A/B/C）各一个 seed，wo05 划分、740 epoch 共同上限、
multi_metric 选点（300–400 epoch 作观察节点）——条目定义见
model_registry 同名块。occ010 独立训练注册但不发射（计划 §2）。

用法：
  python configs/z_gen/v7_first_round.py [--models a,b,...]
"""

from __future__ import annotations

import argparse

from common import emit_experiment

EXPERIMENT_ID = "v7_first_round"


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
