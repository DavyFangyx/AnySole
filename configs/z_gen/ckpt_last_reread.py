#!/usr/bin/env python3
"""Generate the ckpt_last re-read queue tasks (2026-10-24).

V4_Av1_wo05_drop 的 ckpt_last 跑 cam3 + occ010/020/030/050 五轴
（E5 选择假象检查）；V5_Av1_fS13S14 的 ckpt_last 跑 cam3（塌缩 fold
选择假象检查）；V4_Av1/V5_Av1 其余 fold 格（5-fold 按动作表修正口径）。
eval-only 零重训——条目定义见 model_registry 同名块。

用法：
  python configs/z_gen/ckpt_last_reread.py [--models a,b,...]

--models：只发射指定条目（其余跳过，不发 conf 不占序号）——补发
新增条目时使用，避免把已消费的历史 conf 重新发一遍。

幂等：queue/ 中已存在同名 conf 的模型跳过（不重复写、不占新序号）。
"""

from __future__ import annotations

import argparse

from common import emit_experiment

EXPERIMENT_ID = "ckpt_last_reread"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs to subset the group")
    args = parser.parse_args()
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    n = emit_experiment(EXPERIMENT_ID, selected or None)
    print(f"generated {n} queue task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
