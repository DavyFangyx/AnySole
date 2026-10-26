#!/usr/bin/env python3
"""仅 U 训练套餐生成器（2026-10-26 用户裁定：训练套餐与实验解耦）。

训练内容：只训 U（统一模型，[0.5,0.25,0.25]），结构筛选阶段用——
E1 的专家行与 E2/E5 的对照评估不在此列。

DEFAULT_MODELS：U = V7_Cv5。换 U 时改 DEFAULT_MODELS 或用 --models 覆盖。

底层条目：v7_variants_clean（--models 限定 U）。

用法：
  python configs/z_gen/gen_u_only.py
  python configs/z_gen/gen_u_only.py --models V7_Cv5
"""

from __future__ import annotations

import argparse

from common import add_gen_args, emit_subset, parse_bool

DEFAULT_MODELS = ["V7_Cv5"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gen_args(parser)
    args = parser.parse_args()
    models = ([item.strip() for item in args.models.split(",") if item.strip()]
              if args.models else list(DEFAULT_MODELS))
    wandb = parse_bool(args.wandb, "--wandb")
    checkpoint_all = parse_bool(args.checkpoint_all, "--checkpoint-all")
    total = emit_subset("v7_variants_clean", models,
                        wandb=wandb, checkpoint_all=checkpoint_all,
                        force=args.force, dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
