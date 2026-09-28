"""单人 CLIFF（HR48）视觉初始化生产（PoseTransOpt + pressure_tookit 共用）。

生产者 ``run_cliff.py``：逐 canonical 帧写 ``CLIFF_results.npz``（一人一行，
bbox 由 SAM3.1 单受试者 mask 重算，mask 行按 visual_time_s ≤20ms 最近邻连接），
默认写入 ``shared/frontends/cliff_hr48/v1``；``export_toolkit_init.py`` 再由它
派生 pressure_tookit 的 ``<session>_cliff_hr48.npz`` 初始化契约。
"""
