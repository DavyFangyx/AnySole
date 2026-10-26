#!/usr/bin/env python3
"""E2 主性能实验生成器（z_note/实验计划.md）。

内容：三块（V 块 / T 块 / VT 块）十指标 vs 外部基线（表 2），一个 ckpt
在 U@V / U@T / U@VT 直接推理、零重训；口径并列 ckpt_last（表 2a）与
ckpt_best（表 2b）。

前置：U 已训（gen_full_package.py 或 gen_u_only.py）；外部基线行来自
`z_note/E_Test/ABC对照/E2_外部基线主结果.md`，无 conf。

底层条目（eval-only 零重训，与 E5 共用）：
- v7_r2_eval：固定四门选点（*_sel）cam3 三块评估；
- v7_ckpt_caliber：best/last 双口径对照（cam3 格 + occ 格，occ 格供 E5 读）。
共用条目先跑先生成，后跑自动跳过（cell 已在队列任一状态或结果 complete）。

用法：
  python configs/z_gen/gen_e2.py
  python configs/z_gen/gen_e2.py --models V7_Cv1_sel --dry-run
"""

from __future__ import annotations

import argparse

from common import add_gen_args, emit_subset, parse_bool

# E2 读这两组评估条目的 cam3 格；occ 格是 E5 的（共用同批 conf）。
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
    total += emit_subset("v7_r2_eval", models,
                         wandb=wandb, checkpoint_all=checkpoint_all,
                         force=args.force, dry_run=args.dry_run)
    total += emit_subset("v7_ckpt_caliber", models,
                         wandb=wandb, checkpoint_all=checkpoint_all,
                         force=args.force, dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
