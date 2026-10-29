#!/usr/bin/env python3
"""E4 消融组生成器：训练 + 统计。

用法：
  python configs/z_gen/configs/z_gen/gen_e4.py [--models a,b] [--concat-method f6_soft] [--seed 1,2,3] [--force] [--dry-run]
"""
from __future__ import annotations

import argparse

from common import (add_gen_args, concat_models, emit_subset, parse_bool,
                    parse_seeds, seed_models)

DEFAULT_MODELS = ["V7_Cv%d" % i for i in (2, 4, 5, 6, 7)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gen_args(parser)
    args = parser.parse_args()
    models = ([item.strip() for item in args.models.split(",") if item.strip()]
              if args.models else list(DEFAULT_MODELS))
    models = concat_models(models, args.concat_method)
    seeds = parse_seeds(args.seed)
    if seeds:
        models = seed_models(models, seeds)
    wandb = parse_bool(args.wandb, "--wandb")
    checkpoint_all = parse_bool(args.checkpoint_all, "--checkpoint-all")
    total = emit_subset("v7_train", models, wandb=wandb, checkpoint_all=checkpoint_all,
                        force=args.force, dry_run=args.dry_run)
    total += emit_subset("v7_eval_clean", models, wandb=wandb,
                         checkpoint_all=checkpoint_all, force=args.force,
                         dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
