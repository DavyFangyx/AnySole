"""V7_Cv5AR2MP（2026-11-03，继承路线 Prefix）：结构同 V7_Cv5AR2M，配方差异 = t_prefix_p。

结构完全沿用 AR2M（ar2=True、ar2_fb=False、因果掩码、无反馈支路）；差异只在训练配方：
enum 的 T 档里一部分有观测历史的样本改成「历史窗口 V 真实、当前窗 V 换 null、T 全程真实」
（train.py --t-prefix-p，见 z_note/multimodal_fusion/V7_方案/V7_继承路线实验定义.md）。
独立入口只为独立的结果目录与身份；不改 model_v7.py。
"""

from anysole.models.v7_c2ar2m import CONFIG_EXTRA, STRUCTURE, build  # noqa: F401
