#!/usr/bin/env python3
"""Current-source seven-model package; experts require explicit historical IDs.

Defaults: V7_Cv1..7, validation only, fixed last.
Use --dry-run to preview without enqueueing any training.
"""

# Default entries now use * and validation only. Old IDs remain explicit historical options.
from __future__ import annotations

import argparse

from common import (add_gen_args, concat_models, emit_subset, parse_bool,
                    parse_seeds, seed_models)

# 当前套餐只包含七个统一模型；旧专家仅支持显式历史ID。
DEFAULT_MODELS = ["V7_Cv%d" % i for i in range(1, 8)]


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
    total += emit_subset("v7_state_corrected_train", models, wandb=wandb,
                         checkpoint_all=checkpoint_all, force=args.force, dry_run=args.dry_run)
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
