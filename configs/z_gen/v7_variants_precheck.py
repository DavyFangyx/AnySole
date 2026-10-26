#!/usr/bin/env python3
"""Generate the V1–V3 precheck queue tasks (2026-10-26).

V1 / V2 / V3 / V1_enum 四格 6-epoch 端到端预检（结构冒烟后全训
v7_variants_clean）。条目定义见 model_registry 同名块。

用法：
  python configs/z_gen/v7_variants_precheck.py [--models a,b,...]
"""

from __future__ import annotations

import argparse

from common import emit_experiment

EXPERIMENT_ID = "v7_variants_precheck"


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
