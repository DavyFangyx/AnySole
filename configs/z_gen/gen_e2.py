#!/usr/bin/env python3
"""E2 主性能评估生成器（测试层）：训练 cell 自带 val+test 正式统计
（_statistics 后缀已作废）；--ckpt best → v7_statefix_best_eval。

用法：
  python configs/z_gen/configs/z_gen/gen_e2.py [--models a,b] [--concat-method f6_soft] [--seed 1,2,3] [--force] [--dry-run]
"""
from __future__ import annotations

import argparse

from common import (add_gen_args, ckpt_models, concat_models,
                    emit_subset, parse_bool, parse_seeds, seed_models)

DEFAULT_MODELS = ['V7_Cv5']


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gen_args(parser)
    parser.add_argument("--ckpt", choices=("last", "best"), default="last",
                        help="ckpt 口径（best → *_evalbest* 条目）")
    args = parser.parse_args()
    models = ([item.strip() for item in args.models.split(",") if item.strip()]
              if args.models else list(DEFAULT_MODELS))
    models = concat_models(models, args.concat_method)
    seeds = parse_seeds(args.seed)
    if seeds:
        models = seed_models(models, seeds)
    wandb = parse_bool(args.wandb, "--wandb")
    checkpoint_all = parse_bool(args.checkpoint_all, "--checkpoint-all")
    if args.ckpt == "best":
        models = ckpt_models(models, "best")
        total = emit_subset("v7_statefix_best_eval", models, wandb=wandb,
                            checkpoint_all=checkpoint_all, force=args.force,
                            dry_run=args.dry_run)
    else:
        total = emit_subset("v7_train", models, wandb=wandb,
                            checkpoint_all=checkpoint_all, force=args.force,
                            dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
