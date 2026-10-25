#!/usr/bin/env python3
"""Generate the V4_Av4 B-finetune round queue tasks (2026-10-25).

V4_Av4（V4 修正分支挂 B 骨架双阶段精调，carrier_lr_mult=0.1）×（cam3 / cam3_occ010）
2 格，训练+评估一体事务——条目定义见 model_registry 同名块。

用法：
  python configs/z_gen/v4b_finetune.py [--models a,b,...]

--models：只发射指定条目（补发用）。
"""

from __future__ import annotations

import argparse

from common import emit_experiment

EXPERIMENT_ID = "v4b_finetune"


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
