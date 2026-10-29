#!/usr/bin/env python3
"""统一实验生成器（2026-10-26 重构：每实验一脚本 → 实验条目 + 参数）。

取代此前的 configs/z_gen/<实验>.py 一族——EXPERIMENTS 注册表（common.py）
是唯一真源，实验条目即参数：

  python configs/z_gen/gen.py                        # 无参数 = 列出全部实验条目
  python configs/z_gen/gen.py <条目>                 # 按该条目 MODEL_IDS 全量生成
  python configs/z_gen/gen.py <条目> --models a,b    # 只发指定模型（子集校验）
  python configs/z_gen/gen.py <条目> --wave 2        # 波次条目（wo05 / tw_scan / per_action_folds）
  python configs/z_gen/gen.py <条目> --wandb true --checkpoint-all false
  python configs/z_gen/gen.py <条目> --force         # 写 FORCE=true（跳过 ckpt 复用强制重训）
  python configs/z_gen/gen.py <条目> --dry-run       # 只打印将写入的 conf，不落盘

--wandb / --checkpoint-all / --force：覆盖 EXPERIMENTS 级默认，直写 conf。
波次实验（WAVES/WAVE_GATES/WAVE_EVAL 数据在 common.py）：--wave 指定波次，
闸门 glob 匹配不到 done/ 即退出不发射；缺省波次 = DEFAULT_WAVE（wo05 为
auto：波 1 + 闸门满足的后续波）。

cell 命名与 skip（2026-10-26 用户裁定「conf 与结果一一对应」）：
conf = {seq}__{model_id}.conf（模型×条件即命名，实验只在文件内容）；
结果 = results/cells/{model_id}/（跨实验共享，先到先评）。生成默认跳过
队列任一状态已有同名 conf、或结果目录已有 complete 产物的 cell——
先做 E5 再做 E2 时，E2 的共用格不重复生成；--force 绕过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import CONCAT_VERSIONS, EXPERIMENTS, concat_models, emit_waved, parse_bool


def _list_experiments() -> None:
    print("experiments (%d):" % len(EXPERIMENTS))
    for experiment_id in sorted(EXPERIMENTS):
        raw = EXPERIMENTS[experiment_id]
        n = len(raw["MODEL_IDS"])
        waves = raw.get("WAVES")
        wave_desc = ""
        if waves:
            wave_desc = " waves=%s" % ",".join(str(k) for k in sorted(waves, key=str))
            if "WAVE_EVAL" in raw:
                wave_desc += "+eval"
        print("  %-24s %d model(s)  wandb=%s  checkpoint_all=%s%s"
              % (experiment_id, n,
                 str(raw.get("WANDB", False)).lower(),
                 str(raw.get("CHECKPOINT_ALL", False)).lower(),
                 wave_desc))


def _emit(experiment_id: str, wave: str | None, models: list[str],
          wandb: bool | None, checkpoint_all: bool | None,
          force: bool, dry_run: bool) -> int:
    """委托 common.emit_waved（波次感知发射，与 gen_*.py 共用同一引擎）。"""
    return emit_waved(experiment_id, wave, models,
                      wandb=wandb, checkpoint_all=checkpoint_all,
                      force=force, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", default="",
                        help="EXPERIMENTS 条目名（缺省 = 列出全部条目）")
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs（必须是该条目 MODEL_IDS 子集）")
    parser.add_argument("--concat-method", choices=list(CONCAT_VERSIONS),
                        default="f6_soft",
                        help="接触标签版本（默认 f6_soft；tafix/statefix 映射到 *_tafix/*_statefix 条目）")
    parser.add_argument("--wave", default=None,
                        help="波次键（wo05 / tw_scan）")
    parser.add_argument("--wandb", default=None, help="true/false，覆盖实验级默认")
    parser.add_argument("--checkpoint-all", default=None, help="true/false，覆盖实验级默认")
    parser.add_argument("--force", action="store_true",
                        help="写 FORCE=true（跳过 ckpt 复用检查强制重训）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将写入的 conf，不落盘")
    args = parser.parse_args()

    if not args.experiment:
        _list_experiments()
        return 0
    if args.experiment not in EXPERIMENTS:
        raise SystemExit("unknown experiment: %s（gen.py 无参数列出全部条目）"
                         % args.experiment)

    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    if selected:
        selected = concat_models(selected, args.concat_method)
        allowed = set(str(item) for item in EXPERIMENTS[args.experiment]["MODEL_IDS"])
        if args.wave and "WAVES" in EXPERIMENTS[args.experiment]:
            allowed = set(str(item) for item in
                          EXPERIMENTS[args.experiment]["WAVES"].get(args.wave, allowed))
        unknown = [item for item in selected if item not in allowed]
        if unknown:
            raise SystemExit("unknown model IDs for %s: %s" %
                             (args.experiment, ",".join(unknown)))

    wandb = parse_bool(args.wandb, "--wandb")
    checkpoint_all = parse_bool(args.checkpoint_all, "--checkpoint-all")
    n = _emit(args.experiment, args.wave, selected, wandb, checkpoint_all,
              args.force, args.dry_run)
    print("generated %d queue task(s)%s" % (n, " (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
