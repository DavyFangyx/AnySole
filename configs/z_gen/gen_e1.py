#!/usr/bin/env python3
"""E1 current-source unified V/T/VT panel; no default expert training.

Default evaluates the same Cv5 last checkpoint on validation data.
Historical expert identifiers are supported only when explicitly supplied.
"""

# Default entries now use *_statefix and validation only. Old IDs remain explicit historical options.
from __future__ import annotations

import argparse

from common import (parse_seeds, seed_models,
add_gen_args, concat_models, emit_subset, parse_bool)

DEFAULT_MODELS = ["V7_Cv5_statistics"]


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
    total = emit_subset("v7_eval_clean", models, wandb=wandb, checkpoint_all=checkpoint_all,
                        force=args.force, dry_run=args.dry_run)
    total += emit_subset("v7_state_corrected_train", models, wandb=wandb,
                         checkpoint_all=checkpoint_all, force=args.force, dry_run=args.dry_run)
    total += emit_subset("v7_cv5_experts", models,
                        wandb=wandb, checkpoint_all=checkpoint_all,
                        force=args.force, dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
