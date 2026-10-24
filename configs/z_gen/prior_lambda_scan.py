#!/usr/bin/env python3
"""Generate the λ-scan (prior lambda) queue tasks (2026-10-24).

V5_Av1 / V5_Av4 / V4_Av1 × λ_prior_pose {1,2,3,4} × λ_prior_traj {0,1}
共 24 conf；cam3 训练 + 评估、splits_wo05.csv（v1_wo05.yaml）。纯扫描、
无调参目标——每格产出三条件面板（V2M/VT2M/T2M），终表由各格
results/experiments/prior_lambda_scan/<model>/metrics/test.json 汇总。
条目定义见 model_registry 同名块（基座 = v3off 生产轴条目；载体 = 各
基座现役 carrier，不依赖 wo05-E 因 OOM 未产出的 ckpt）。

MODEL_IDS 顺序即 conf 顺序：V5_Av1 8 格 → V5_Av4 8 格 → V4_Av1 8 格。

用法：
  python configs/z_gen/prior_lambda_scan.py

幂等：queue/ 中已存在同名 conf 的模型跳过（不重复写、不占新序号）。
"""

from __future__ import annotations

from common import emit_experiment

EXPERIMENT_ID = "prior_lambda_scan"

if __name__ == "__main__":
    raise SystemExit(emit_experiment(EXPERIMENT_ID))
