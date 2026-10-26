#!/usr/bin/env python3
"""Generate the V7 E1 expert training queue tasks (2026-10-26 用户裁定).

S_V（仅视觉 [0,1,0]）/ S_T（仅触觉 [0,0,1]）两格——同架构 = 候选 A 骨架
（v7_a 入口），checkpoint_role auto 推断为对应专家；E1 判据 = 统一模型
U 在对应设定下优于专家（T 设定看 ATE/Global、V 设定看足部指标），专家
固定各自完整权重、不参与 U 的选点判定（ckpt 规则 §2）。

用法：
  python configs/z_gen/v7_experts.py [--models a,b,...]
"""

from __future__ import annotations

import argparse

from common import emit_experiment

EXPERIMENT_ID = "v7_experts"


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
