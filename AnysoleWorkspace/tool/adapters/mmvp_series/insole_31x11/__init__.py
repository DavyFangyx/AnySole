"""31×11 鞋垫生产（FPP-Net + pressure_tookit 共用）。

公共表示唯一：``shared/representations/tactile/mmvp_31x11/v1``（由
``tool/build_shared.py`` 从 raw 48 通道压力生成）。本包只做上游契约包装
（``insole.py``：``{'insole': [L, R], 'frame_id'}``，数值零变换），不重建
私有 31×11 树。
"""
