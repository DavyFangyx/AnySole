#!/usr/bin/env python3
"""Generate the V7_Cv1 first-round tuning grid (2026-10-26).

lr {5e-5,1e-4,2e-4} × λ_contact {0.1,0.3,1.0} × lr_schedule
{constant,cosine} = 18 conf；λ_pose 3 / λ_traj 1 / λ_air 2 固定（基座）。
条目定义见 model_registry 同名块。

用法：
  python configs/z_gen/v7c_tune18.py

幂等：queue/ 中已存在同名 conf 的模型跳过（不重复写、不占新序号）。
"""

from __future__ import annotations

from common import emit_experiment

EXPERIMENT_ID = "v7c_tune18"

if __name__ == "__main__":
    raise SystemExit(emit_experiment(EXPERIMENT_ID))
