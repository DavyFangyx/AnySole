"""V8_Cv5（2026-11-05）：V8 系列基座。

结构逐参数 = V7_Cv5SD2（``v7_c2sd2``）：双独立 T 门 + 三条件枚举 +
根头速率/方向解耦（``v7_root_sd`` + B′ 恒等式配平）+ 平移锚头（必选）。
V8 命名空间单独存在，便于在基座上重建门控/条件消融；结构本身不复制。
"""

from anysole.models.v7_c2sd2 import CONFIG_EXTRA, STRUCTURE, build  # noqa: F401
