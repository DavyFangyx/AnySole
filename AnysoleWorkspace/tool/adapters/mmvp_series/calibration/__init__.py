"""calibration 口径转换：``protocol/calibration/<date>.json`` → 模型原生契约。

逐日期 cam3 真实 fx（D6 裁定，已执行）：pressure_tookit 的
``calibration.npy``（color_Intr/depth_Intr/d2c）、PoseTransOpt 的
``join_manifest.json`` camera 块都从同一份日期标定导出。
"""
