#!/usr/bin/env python3
"""Generate the V1–V3 clean training queue tasks (2026-10-26).

V1（C 单门基线）/ V2（根姿态双门）/ V3（T-only 适配层）/ V1_enum
（枚举训练）四格 740-epoch 正式训练。条目定义见 model_registry 同名块。

用法：
  python configs/z_gen/v7_variants_clean.py [--models a,b,...]
"""

from __future__ import annotations

import argparse

from common import emit_experiment

EXPERIMENT_ID = "v7_variants_clean"


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
