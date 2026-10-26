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

from common import CONFIG_DIR, EXPERIMENTS, emit_experiment, parse_bool

DONE_DIR = CONFIG_DIR / "done"


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


def _wave_key(raw: dict[str, Any], key: str, table: str) -> Any:
    """按 str 匹配 WAVES/WAVE_GATES 的键（兼容 int 键 1/2 与 str 键 "12"/"3"）。"""
    for candidate in raw.get(table, {}):
        if str(candidate) == str(key):
            return candidate
    raise SystemExit("unknown wave %r for %s (waves: %s)"
                     % (key, raw.get("_experiment_id"),
                        ",".join(str(k) for k in sorted(raw.get("WAVES", {}), key=str))))


def _check_gates(raw: dict[str, Any], wave: Any, hard: bool) -> bool:
    gates = raw.get("WAVE_GATES", {})
    patterns = [gates[k] for k in gates if str(k) == str(wave)]
    patterns = [p for group in patterns for p in group]
    missing = [pattern for pattern in patterns if not list(DONE_DIR.glob(pattern))]
    if not missing:
        return True
    if hard:
        raise SystemExit("波 %s 闸门未满足：%s 无 conf 进 done/"
                         % (wave, "、".join(missing)))
    for pattern in missing:
        print("[gate] wave %s skipped: %s 未进 done/（完成后用 --wave %s 补发）"
              % (wave, pattern, wave))
    return False


def _emit_wave(raw: dict[str, Any], wave: Any, selected: list[str],
              wandb: bool | None, checkpoint_all: bool | None,
              force: bool, dry_run: bool) -> int:
    """发射一个训练波：闸门（硬）→ --models 子集 → 跳过已入队模型 → emit。"""
    _check_gates(raw, wave, hard=True)
    selected_set = set(selected)
    wave_list = raw["WAVES"][_wave_key(raw, wave, "WAVES")]
    models = [str(item) for item in wave_list
              if not selected_set or str(item) in selected_set]
    if not models:
        print("[skip] wave %s: no selected model in this wave" % wave)
        return 0
    return emit_experiment(str(raw["_experiment_id"]), models,
                           wandb=wandb, checkpoint_all=checkpoint_all,
                           force=force, dry_run=dry_run)


def _emit_eval_wave(raw: dict[str, Any], hard: bool,
                    wandb: bool | None, checkpoint_all: bool | None,
                    force: bool, dry_run: bool) -> int:
    """发射 WAVE_EVAL 波（wo05 波 3）：按家族闸门筛 ready 家族后发 eval 条目。"""
    eval_spec = raw["WAVE_EVAL"]
    eval_id = str(eval_spec["experiment"])
    ready: list[str] = []
    for fam, base in eval_spec["families"].items():
        if list(DONE_DIR.glob("*__%s.conf" % base)):
            ready.extend(str(item) for item in EXPERIMENTS[eval_id]["MODEL_IDS"]
                         if str(item).startswith(fam))
        else:
            print("[gate] wave %s skipped: %s 未进 done/" % (eval_spec["wave"], base))
    if not ready:
        if hard:
            raise SystemExit("波 %s 闸门未满足：没有任何家族 _drop conf 进 done/"
                             % eval_spec["wave"])
        return 0
    return emit_experiment(eval_id, ready,
                           wandb=wandb, checkpoint_all=checkpoint_all,
                           force=force, dry_run=dry_run)


def _emit(experiment_id: str, wave: str | None, models: list[str],
          wandb: bool | None, checkpoint_all: bool | None,
          force: bool, dry_run: bool) -> int:
    raw = dict(EXPERIMENTS[experiment_id])
    raw["_experiment_id"] = experiment_id
    waves = raw.get("WAVES")
    wave_eval = raw.get("WAVE_EVAL")

    if waves:
        if wave is None:
            wave = str(raw.get("DEFAULT_WAVE", sorted(waves, key=str)[0]))
        if wave == "auto":
            total = 0
            for w in sorted(waves, key=str):
                if _check_gates(raw, w, hard=False):
                    total += _emit_wave(raw, w, models, wandb, checkpoint_all,
                                        force, dry_run)
            if wave_eval:
                total += _emit_eval_wave(raw, hard=False, wandb=wandb,
                                         checkpoint_all=checkpoint_all,
                                         force=force, dry_run=dry_run)
            return total
        if str(wave) in {str(k) for k in waves}:
            total = _emit_wave(raw, wave, models, wandb, checkpoint_all,
                               force, dry_run)
            if wave_eval and str(wave_eval.get("wave")) == str(wave):
                total += _emit_eval_wave(raw, hard=True, wandb=wandb,
                                         checkpoint_all=checkpoint_all,
                                         force=force, dry_run=dry_run)
            return total
        if wave_eval and str(wave_eval.get("wave")) == str(wave):
            return _emit_eval_wave(raw, hard=True, wandb=wandb,
                                   checkpoint_all=checkpoint_all,
                                   force=force, dry_run=dry_run)
        raise SystemExit("unknown wave %r for %s (waves: %s)"
                         % (wave, experiment_id,
                            ",".join(str(k) for k in sorted(waves, key=str))))

    selected = models or None
    return emit_experiment(experiment_id, selected,
                           wandb=wandb, checkpoint_all=checkpoint_all,
                           force=force, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", default="",
                        help="EXPERIMENTS 条目名（缺省 = 列出全部条目）")
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs（必须是该条目 MODEL_IDS 子集）")
    parser.add_argument("--wave", default=None,
                        help="波次键（wo05 / tw_scan / per_action_folds）")
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
