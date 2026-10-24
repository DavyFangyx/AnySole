#!/usr/bin/env python3
"""Generate the wo05 (去绕圈跑) experiment group queue tasks (2026-10-24).

canonical split（train/val/test = 92/12/36）去掉动作 05（绕圈跑）全部 4 条
train 会话（S11051/S11053/S7053/S8053）→ train 88，val/test 不变
（AnysoleWorkspace/protocol/splits/default/splits_wo05.csv）。3 家族 × E1
四变体 = 12 conf；训练规格与 v3off E1 条目逐项一致，config = v1_wo05.yaml
（与 v1.yaml 唯一差异 = split_csv）。

两波 + 波 3（对齐 model_registry 同名块）：
  波 1（8 conf）：V3_Ev1_wo05 + V5_Av4_wo05 各四变体——V5_Av4 保持
    carrier_from="scratch"（anysole/models/v5_1.py 的 scratch 分支在
    torch.load 之前返回，不装载任何 E ckpt → 无 E 依赖，随波 1 并行）；
  波 2（4 conf）：V4_Av1_wo05 四变体——carrier_from 指向波 1 的
    V3_Ev1_wo05 ckpt_best（v4_1 entry 装载）。闸门 = V3_Ev1_wo05 的 conf
    进 done/（训练+评估完成）；ckpt_best.pt 存在不作数——训练中途首个
    val 改善即写，会让 V4_Av1 冻结到未收敛的早中期载体上。
  波 3（12 conf，wo05_eval 实验组）：3 家族 _drop 统一模型 × occ010/020/
    030/050 遮挡交叉评估（E5 遮挡退化：U@V/U@VT 各 4 档 + U@T 1 组 =
    一次 eval 产出三设定行）。eval-only 零重训，闸门 = 对应家族 _drop
    conf 进 done/。
MODEL_IDS 顺序即 conf 顺序：波 1 排最前（worker 按序号取件）。

用法：
  python configs/z_gen/wo05.py            # 波 1；波 2/3 仅在闸门满足时随发
  python configs/z_gen/wo05.py --wave 2   # 仅波 2（闸门硬断言：未满足即退出）
  python configs/z_gen/wo05.py --wave 3   # 仅波 3（occ 交叉评估，按家族闸门）

幂等：queue/ 中已存在同名 conf 的模型跳过（不重复写、不占新序号）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import emit_experiment

EXPERIMENT_ID = "wo05"
_QUEUE_DIR = Path(__file__).resolve().parents[1] / "queue"
_DONE_DIR = Path(__file__).resolve().parents[1] / "done"

WAVES = {
    1: ["V3_Ev1_wo05", "V3_Ev1_wo05_drop", "V3_Ev1_wo05_vonly",
        "V3_Ev1_wo05_tonly",
        "V5_Av4_wo05", "V5_Av4_wo05_drop", "V5_Av4_wo05_vonly",
        "V5_Av4_wo05_tonly"],
    2: ["V4_Av1_wo05", "V4_Av1_wo05_drop", "V4_Av1_wo05_vonly",
        "V4_Av1_wo05_tonly"],
}


EVAL_EXPERIMENT_ID = "wo05_eval"
_EVAL_FAMILIES = {
    "V3_Ev1": ["V3_Ev1_wo05_drop_evalocc010", "V3_Ev1_wo05_drop_evalocc020",
               "V3_Ev1_wo05_drop_evalocc030", "V3_Ev1_wo05_drop_evalocc050"],
    "V4_Av1": ["V4_Av1_wo05_drop_evalocc010", "V4_Av1_wo05_drop_evalocc020",
               "V4_Av1_wo05_drop_evalocc030", "V4_Av1_wo05_drop_evalocc050"],
    "V5_Av4": ["V5_Av4_wo05_drop_evalocc010", "V5_Av4_wo05_drop_evalocc020",
               "V5_Av4_wo05_drop_evalocc030", "V5_Av4_wo05_drop_evalocc050"],
}


def _eval_ready_models() -> list[str]:
    """波 3 闸门：家族 _drop 统一模型的 conf 进 done/ 后，该家族的 4 档
    occ 交叉评估（E5 遮挡退化，eval-only 零重训）才可入队。"""
    ready: list[str] = []
    for fam, ids in _EVAL_FAMILIES.items():
        if list(_DONE_DIR.glob(f"*__wo05__{fam}_wo05_drop.conf")):
            ready.extend(ids)
        else:
            print(f"[gate] wave 3 skipped: {fam}_wo05_drop 未进 done/")
    return ready


def _wave2_ready() -> bool:
    """波 2 闸门：V3_Ev1_wo05 的 conf 已进 done/（训练+评估完成）。"""
    return bool(list(_DONE_DIR.glob("*__wo05__V3_Ev1_wo05.conf")))


def _assert_wave2_ready() -> None:
    if not _wave2_ready():
        raise SystemExit(
            "波 2 闸门未满足：V3_Ev1_wo05 conf 未进 done/（先等波 1 完成）")


def _queued_models(experiment_id: str = EXPERIMENT_ID) -> set[str]:
    """queue/running/done/failed 任一状态已有该实验 conf 的模型名（幂等：
    跳过不重复写——只看 queue/ 会在 done 后重发导致重复评估，2026-10-24
    教训：E occ 评估被波 3 重发三遍）。"""
    names: set[str] = set()
    for state in ("queue", "running", "done", "failed"):
        for path in (_QUEUE_DIR.parent / state).glob("*__%s__*.conf" % experiment_id):
            names.add(path.name.split("__", 2)[-1][: -len(".conf")])
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, default=None,
                        choices=sorted(WAVES) + [3],
                        help="只发该波次（默认：波 1 + 闸门满足时的波 2/3）")
    args = parser.parse_args()
    eval_models = _eval_ready_models() if args.wave in (None, 3) else []
    if args.wave == 3:
        if not eval_models:
            raise SystemExit("波 3 闸门未满足：没有任何家族 _drop conf 进 done/")
        waves: list[int] = []
    elif args.wave == 2:
        _assert_wave2_ready()
        waves = [2]
    elif args.wave == 1:
        waves = [1]
    else:
        waves = [1]
        if _wave2_ready():
            waves.append(2)
        else:
            print("[gate] wave 2 skipped: V3_Ev1_wo05 未进 done/（完成后用"
                  " --wave 2 补发）")
    queued = _queued_models()
    total = 0
    for wave in waves:
        models = [model_id for model_id in WAVES[wave]
                  if model_id not in queued]
        if not models:
            print("[skip] wave %d: all confs already queued" % wave)
            continue
        total += emit_experiment(EXPERIMENT_ID, models)
    if eval_models:
        queued_eval = _queued_models(EVAL_EXPERIMENT_ID)
        eval_models = [m for m in eval_models if m not in queued_eval]
        if eval_models:
            total += emit_experiment(EVAL_EXPERIMENT_ID, eval_models)
    print("generated %d queue task(s)" % total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
