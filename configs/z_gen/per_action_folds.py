#!/usr/bin/env python3
"""Generate the 5-fold per-action coverage experiment group (2026-10-22).

受试者轮换 leave-subjects-out：5 个 fold 各一个 split 文件
（canonical fS13S14 + 4 轮换，AnysoleWorkspace/protocol/splits/default/
splits_test_*.csv），test 覆盖全部 11 动作。五模型（V3_Ev1 载体线 +
V3_Bv1/V5_Av4/V5_Av1/V4_Av1 评估线）× 5 fold = 25 conf。

波次闸门（对齐 model_registry 同名块）：
  波 1+2（15 conf，无依赖）：V3_Ev1 ×5 + V3_Bv1/V5_Av4 ×5；
  波 3（10 conf）：V5_Av1/V4_Av1 ×5——carrier_from 指向该 fold 的
  V3_Ev1 ckpt_best，必须在波 1 完成后生成。

用法：
  python configs/z_gen/per_action_folds.py            # 波 1+2
  python configs/z_gen/per_action_folds.py --stage 3  # 波 3（E1 完成后）
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import emit_experiment

EXPERIMENT_ID = "per_action_folds"
_DONE_DIR = Path(__file__).resolve().parents[1] / "done"

_FOLD_PREFIXES = ("fS13S14", "fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")

STAGE_MODELS = {
    "12": ["V3_Ev1_%s" % f for f in _FOLD_PREFIXES]
    + ["V3_Bv1_%s" % f for f in _FOLD_PREFIXES]
    + ["V5_Av4_%s" % f for f in _FOLD_PREFIXES],
    "3": ["V5_Av1_%s" % f for f in _FOLD_PREFIXES]
    + ["V4_Av1_%s" % f for f in _FOLD_PREFIXES]
    # occ050 档（2026-10-23 裁定「加」；载体闸门同波 3）
    + ["V5_Av1_occ050_%s" % f for f in _FOLD_PREFIXES]
    + ["V4_Av1_occ050_%s" % f for f in _FOLD_PREFIXES],
}


def _assert_stage3_ready() -> None:
    """波 3 闸门：波 1 的 5 个 E1 fold conf 必须全部进 done/（训练+评估
    完成）。ckpt_best.pt 文件存在不作数——训练中途首个 val 改善即写。"""
    missing = [f for f in _FOLD_PREFIXES
               if not list(_DONE_DIR.glob("*per_action_folds__V3_Ev1_%s.conf" % f))]
    if missing:
        raise SystemExit(
            "波 3 闸门未满足：E1 fold conf 未进 done/: %s（先等波 1 完成）"
            % ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="12", choices=("12", "3"),
                        help="conf 波次：12 = 波 1+2（默认）；3 = 波 3")
    args = parser.parse_args()
    if args.stage == "3":
        _assert_stage3_ready()
    return emit_experiment(EXPERIMENT_ID, STAGE_MODELS[args.stage])


if __name__ == "__main__":
    raise SystemExit(main())
