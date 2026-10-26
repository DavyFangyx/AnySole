#!/usr/bin/env python3
"""E5 遮挡退化实验生成器（z_note/实验计划.md）。

内容：occ 五档（000/010/020/030/050，occ000 = cam3 基准）交叉评估，固定
ckpt_last（表 3）；判据 = U@VT 与 U@V 的差距随遮挡等级增大。

前置：U 已训（gen_full_package.py 或 gen_u_only.py）。

底层条目（eval-only 零重训）：
- v7_e5_screen：候选 A/B/C × occ 四档筛选（val 口径，早期选型用）；
- v7_ckpt_caliber + v7_r2_eval：occ 交叉评估格（与 E2 共用同批 conf，
  E2 读 cam3 格、E5 读 occ 格；cell 已在队列任一状态或结果 complete 即跳过）。

用法：
  python configs/z_gen/gen_e5.py
  python configs/z_gen/gen_e5.py --models V7_Av1_evalocc010 --dry-run
"""

from __future__ import annotations

import argparse

from common import add_gen_args, emit_subset, parse_bool

DEFAULT_MODELS: list[str] = []


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gen_args(parser)
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()] \
        if args.models else DEFAULT_MODELS
    wandb = parse_bool(args.wandb, "--wandb")
    checkpoint_all = parse_bool(args.checkpoint_all, "--checkpoint-all")
    total = 0
    total += emit_subset("v7_e5_screen", models,
                         wandb=wandb, checkpoint_all=checkpoint_all,
                         force=args.force, dry_run=args.dry_run)
    total += emit_subset("v7_ckpt_caliber", models,
                         wandb=wandb, checkpoint_all=checkpoint_all,
                         force=args.force, dry_run=args.dry_run)
    total += emit_subset("v7_r2_eval", models,
                         wandb=wandb, checkpoint_all=checkpoint_all,
                         force=args.force, dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
