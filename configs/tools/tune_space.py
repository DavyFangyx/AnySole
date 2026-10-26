"""Optuna 调参空间定义（2026-10-25，唯一编辑点）。

改初值 / 范围 / 调参方式只动这个文件；驱动（configs/tools/tune.py）不存
任何参数定义。每个模型一个条目，参数各自独立——不同模型的参数集合、
初值、范围、调参方式都可以不同。

参数规格：
  {"kind": "categorical", "values": [...], "default": v}   枚举调参
  {"kind": "float", "low": a, "high": b, "log": True/False,
   "default": v}                                           连续均匀（log=True = 对数域均匀）
  {"kind": "int", "low": a, "high": b, "step": s,
   "default": v}                                           整数步进

default = 该参数的默认值（--defaults-only 只跑默认点的基线对照）。
未在空间里列出的参数一律用模型注册表的基座规格（configs/z_gen/
model_registry.py 的 train_args），不被调参。
"""

# V7 三候选的公共空间（结构差异由模型各自的注册表 STRUCTURE 开关承载，
# 不在调参空间内——B 的足-地一致性、C 的路由门是结构件不是 λ）。
_V7_COMMON = {
    "lr": {"kind": "categorical",
           "values": [5e-5, 1e-4, 2e-4], "default": 1e-4},
    "lambda_pose": {"kind": "categorical",
                    "values": [1.0, 3.0, 5.0], "default": 3.0},
    "lambda_traj": {"kind": "categorical",
                    "values": [0.5, 1.0, 2.0], "default": 1.0},
    "lambda_contact": {"kind": "categorical",
                       "values": [0.1, 0.3, 1.0], "default": 1.0},
    "lr_schedule": {"kind": "categorical",
                    "values": ["constant", "cosine"], "default": "constant"},
    "lambda_air": {"kind": "categorical",
                   "values": [1.0, 2.0, 4.0], "default": 2.0},
}

# V5 家族与 V4（λ 扫描的三模型，wo05 平条目）：各自的空间不同——
# V5 两载体共 λ_prior 对 + lr；V4 另有 lambda_pos（修正头权重）。
_V5_COMMON = {
    "lr": {"kind": "categorical",
           "values": [5e-5, 1e-4, 2e-4], "default": 1e-4},
    "lambda_prior_pose": {"kind": "categorical",
                          "values": [1.0, 2.0, 3.0, 4.0], "default": 1.0},
    "lambda_prior_traj": {"kind": "categorical",
                          "values": [0.0, 1.0], "default": 0.0},
}
_V4_COMMON = dict(_V5_COMMON)
_V4_COMMON.update({
    "lambda_pos": {"kind": "categorical",
                   "values": [1.0, 3.0, 5.0], "default": 3.0},
})

# V7_Cv1 第一轮（2026-10-26 用户裁定）：固定 λ_pose 3 / λ_traj 1 /
# λ_air 2（不采样、走注册表基座），只搜 lr × λ_contact × lr_schedule。
# λ_contact 保留 1.0 作基线，不预设越小越好。
_V7C_ROUND1 = {
    "lr": {"kind": "categorical",
           "values": [5e-5, 1e-4, 2e-4], "default": 1e-4},
    "lambda_contact": {"kind": "categorical",
                       "values": [0.1, 0.3, 1.0], "default": 1.0},
    "lr_schedule": {"kind": "categorical",
                    "values": ["constant", "cosine"], "default": "constant"},
}

MODEL_SPACE = {
    "V7_Av1": _V7_COMMON,
    "V7_Bv1": _V7_COMMON,
    "V7_Cv1": _V7C_ROUND1,
    "V5_Av1_wo05": _V5_COMMON,
    "V5_Av4_wo05": _V5_COMMON,
    "V4_Av1_wo05": _V4_COMMON,
}
