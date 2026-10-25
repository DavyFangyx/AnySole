#!/usr/bin/env python3
"""Generate the conditional-cooperation round queue tasks (2026-10-25).

V5_Av5 / V4_Av2（框架保留、通路换 E 式双流同训 + occ_mix + 条件契约
λ_cond） ×（契约/去契约对照）×（cam3 / cam3_occ010）8 格，训练+评估
一体事务——条目定义见 model_registry 同名块。

用法：
  python configs/z_gen/cond_coop.py [--models a,b,...]

--models：只发射指定条目（补发用）。
"""

from __future__ import annotations

import argparse

from common import emit_experiment

EXPERIMENT_ID = "cond_coop"


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
