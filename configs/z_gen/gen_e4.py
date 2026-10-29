#!/usr/bin/env python3
"""E4 组件消融组生成器（z_note/multimodal_fusion/V7_方案/V7_C方案.md §7）。

消融组五模型（Vx = V7_Cvx 简称；主模型 V5 = 双门 + 枚举三条件训练）：
- V5 主模型 / V4（单门+枚举）/ V2（双门+随机 dropout）——已训满 740 ep，
  E2/E5 数字见 z_note/E_Test/C版本对照/；
- Cv6（双门+始终 VT，无模态缺失训练）/ Cv7（无门+枚举）——训练在队列
  （00139/00140），统计评估（无波次无闸门，两级 skip + ckpt 守卫兜底）。

本脚本 = 消融组的训练与统计入口（缺什么补什么，已训已评的自动 skip）：
- 训练：v7_variants_clean（--models 覆盖成员）
- 统计：v7_variants_stats（Cv1–7 普通发射，已评 skip）
- TTA 多视角线（独立驱动，不进 queue；V7 已接入）：
  python anysole_tta/run_tta.py --model V7_Cv5 --split val --lr 0.01 --device cuda:7

用法：
  python configs/z_gen/gen_e4.py
  python configs/z_gen/gen_e4.py --models V7_Cv6,V7_Cv7
  python configs/z_gen/gen_e4.py --dry-run
"""

from __future__ import annotations

import argparse

from common import (parse_seeds, seed_models,
add_gen_args, concat_models, emit_subset,
                    emit_waved, parse_bool)

# 消融组成员（换模型时改这里或 --models）
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
    total = 0
    total += emit_subset("v7_train", models, wandb=wandb,
                         checkpoint_all=checkpoint_all, force=args.force, dry_run=args.dry_run)
    total += emit_subset("v7_eval_clean", [m + "_statistics" for m in models],
                         wandb=wandb, checkpoint_all=checkpoint_all,
                         force=args.force, dry_run=args.dry_run)
    # 训练：已训/在队者 skip，缺口者补发
    total += emit_subset("v7_variants_clean", models,
                         wandb=wandb, checkpoint_all=checkpoint_all,
                         force=args.force, dry_run=args.dry_run)
    # 统计：普通发射（已评 skip）
    if any(not model.endswith("_smpl24") for model in models):
        total += emit_waved("v7_variants_stats", None, None,
                            wandb=wandb, checkpoint_all=checkpoint_all,
                            force=args.force, dry_run=args.dry_run)
    print("generated %d queue task(s)%s" % (total, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
