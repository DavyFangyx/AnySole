"""SAM3.1 人形 mask 的采用与派生 mask。

采用外部索引 ``shared/frontends/human_masks/sam31/v1``（不运行新分割模型）；
派生 pressure_tookit 的 depth mask：human ∩ finite_depth ∩ [0.4m, 5m]。
"""
