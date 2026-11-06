#!/usr/bin/env python3
"""仅 U 训练套餐生成器（训练层）。

用法：
  python configs/z_gen/configs/z_gen/gen_u_only.py [--models a,b] [--concat-method f6_soft] [--seed 1,2,3] [--force] [--dry-run]
"""
from __future__ import annotations

import argparse

from common import (add_gen_args, concat_models, emit_subset, parse_bool,
                    parse_seeds, seed_models)

DEFAULT_MODELS = ['V7_Cv5']


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gen_args(parser)
    parser.add_argument("--resume", action="store_true",
                        help="写 RESUME=true：从既有 ckpt_last 断点续训（须同时 --force）")
    args = parser.parse_args()
    if args.resume and not args.force:
        raise SystemExit("--resume 需要同时 --force（runner 只在 FORCE=true 时进入训练分支）")
    models = ([item.strip() for item in args.models.split(",") if item.strip()]
              if args.models else list(DEFAULT_MODELS))
    models = concat_models(models, args.concat_method)
    seeds = parse_seeds(args.seed)
    if seeds:
        models = seed_models(models, seeds)
    monitor = args.monitor
    checkpoint_all = parse_bool(args.checkpoint_all, "--checkpoint-all")
    total = emit_subset("v7_train", models, monitor=monitor, checkpoint_all=checkpoint_all,
                        force=args.force, resume=args.resume, dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
