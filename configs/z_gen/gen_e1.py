#!/usr/bin/env python3
"""E1 同架构专家对比生成器（z_note/实验计划.md）。

内容：U@T vs S_T@T、U@V vs S_V@V、U@VT（附录表 A 五行口径）。判据：
T 设定看 traj_ATE/GlobalMPJPE（V→T 正迁移）、V 设定看足部指标（T→V 正迁移）。

前置：U 已训（gen_full_package.py 或 gen_u_only.py）；本脚本只补专家行
S_V/S_T 的训练+正式评估。U 与专家同架构（v7_c2 双门骨架）、各自固定
ckpt_last、专家不参与 U 的选点。

DEFAULT_MODELS：S_V/S_T = V7_Cv5_SV1 / V7_Cv5_ST1。

底层条目：v7_cv5_experts。

用法：
  python configs/z_gen/gen_e1.py
  python configs/z_gen/gen_e1.py --models V7_Cv5_SV1,V7_Cv5_ST1
"""

from __future__ import annotations

import argparse

from common import add_gen_args, emit_subset, parse_bool

DEFAULT_MODELS = ["V7_Cv5_SV1", "V7_Cv5_ST1"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gen_args(parser)
    args = parser.parse_args()
    models = ([item.strip() for item in args.models.split(",") if item.strip()]
              if args.models else list(DEFAULT_MODELS))
    wandb = parse_bool(args.wandb, "--wandb")
    checkpoint_all = parse_bool(args.checkpoint_all, "--checkpoint-all")
    total = emit_subset("v7_cv5_experts", models,
                        wandb=wandb, checkpoint_all=checkpoint_all,
                        force=args.force, dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
