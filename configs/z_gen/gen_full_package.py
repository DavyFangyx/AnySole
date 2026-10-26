#!/usr/bin/env python3
"""全量训练套餐生成器（2026-10-26 用户裁定：训练套餐与实验解耦）。

训练内容：U + S_V + S_T（同架构、同数据、同预算 740 epoch 单 seed），一次训齐。
E1 的五行口径（U@T/U@V/U@VT vs S_T@T/S_V@V，附录表 A）与 E2/E5 的 U 行
数据 = 本套餐各模型的正式 test.json。

DEFAULT_MODELS：U = V7_Cv5（双门 + 枚举三条件 [0.5,0.25,0.25]）；
S_V/S_T = V7_Cv5_SV1 / V7_Cv5_ST1（v7_c2 同骨架，config_probs 分别为
[0,1,0] / [0,0,1]）。换 U 家族时改 DEFAULT_MODELS 或用 --models 覆盖。

底层条目：v7_variants_clean（U 训练）+ v7_cv5_experts（专家训练）。

用法：
  python configs/z_gen/gen_full_package.py
  python configs/z_gen/gen_full_package.py --models V7_Cv5,V7_Cv5_SV1
  python configs/z_gen/gen_full_package.py --wandb true --checkpoint-all true --dry-run
"""

from __future__ import annotations

import argparse

from common import add_gen_args, emit_subset, parse_bool

# 全量 = U + 同架构专家（换家族时改这里或 --models）
DEFAULT_MODELS = ["V7_Cv5", "V7_Cv5_SV1", "V7_Cv5_ST1"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gen_args(parser)
    args = parser.parse_args()
    models = ([item.strip() for item in args.models.split(",") if item.strip()]
              if args.models else list(DEFAULT_MODELS))
    wandb = parse_bool(args.wandb, "--wandb")
    checkpoint_all = parse_bool(args.checkpoint_all, "--checkpoint-all")
    total = 0
    total += emit_subset("v7_variants_clean", models,
                         wandb=wandb, checkpoint_all=checkpoint_all,
                         force=args.force, dry_run=args.dry_run)
    total += emit_subset("v7_cv5_experts", models,
                         wandb=wandb, checkpoint_all=checkpoint_all,
                         force=args.force, dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
