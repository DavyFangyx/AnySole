#!/usr/bin/env python3
"""Generate the B query-pressure-head queue tasks (2026-10-24).

V3_Bv2（Q-NN 非负 + 训练集 cell 均值先验）→ V3_Bv3（Q-FACT total×allocation
因式化）依次训练：B 的既有配方（740ep/seed1/三条件采样），唯一变化 = 压力
头结构。条目定义见 model_registry 同名块；版本节登记见 V3_B方案.md §6。

用法：
  python configs/z_gen/bv_query.py

幂等：queue/ 中已存在同名 conf 的模型跳过（不重复写、不占新序号）。
"""

from __future__ import annotations

from common import emit_experiment

EXPERIMENT_ID = "bv_query"

if __name__ == "__main__":
    raise SystemExit(emit_experiment(EXPERIMENT_ID))
