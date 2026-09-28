"""DepthPro 深度生产（pressure_tookit 模型输入）+ PoseTransOpt 模板场景。

- ``rgb2depth.py``：shared facts RGB → 16 位毫米刻度深度 PNG，默认写入
  ``shared/frontends/depthpro/v1``（pressure_tookit 的 depth / depth_mask /
  行走地面观测源）；
- ``make_template_scene.py``：首有效帧 RGB-D 模板（PoseTransOpt 静态观测）。
"""
