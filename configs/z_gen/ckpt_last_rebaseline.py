#!/usr/bin/env python3
"""Generate the ckpt_last full re-baseline queue tasks (2026-10-24).

三批 eval-only 零重训（条目定义见 model_registry 同名块）：
  批 1：E1 生产轴 V5_Av1/V4_Av1 四变体 8 格（T2M 选择污染）；
  批 2：tw_scan 28 格（「零兑现 tw 不变式」选择假象检查）；
  批 3：λ 扫描 18 格（绝对值 last 复核）。
MODEL_IDS 顺序即 conf 顺序。

用法：
  python configs/z_gen/ckpt_last_rebaseline.py

幂等：queue/ 中已存在同名 conf 的模型跳过（不重复写、不占新序号）。
"""

from __future__ import annotations

from common import emit_experiment

EXPERIMENT_ID = "ckpt_last_rebaseline"

if __name__ == "__main__":
    raise SystemExit(emit_experiment(EXPERIMENT_ID))
