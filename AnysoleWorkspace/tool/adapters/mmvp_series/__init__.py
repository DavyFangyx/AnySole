"""MMVP 方法族（FPP-Net / PoseTransOpt / pressure_tookit）数据端合并包。

T1-B 裁定（11 号文档 §5）：三模型同属 MMVP/VP-MoCap 方法族，硬件契约相同
（31×11 鞋垫、同一相机），数据端一次生产、三处消费。目录分工：

    common/          raw 读取、manifest/split、provenance 公共件
    keypoints/       RTMPose HALPE-26（三模型共用）
    cliff/           单人 CLIFF（PoseTransOpt + pressure_tookit 共用）
    insole_31x11/    31×11 生产（FPP-Net + pressure_tookit 共用）
    masks/           SAM3.1 采用 + 派生 mask（PoseTransOpt bbox / pressure_tookit depth mask）
    calibration/     D6 协议 json → 各模型口径（逐日期 cam3 fx）
    depth/           DepthPro 深度 + PoseTransOpt 模板场景
    fpp/             FPP-Net 输入组装 + 训练数据 + 推理导出
    posetransopt/    PoseTransOpt 输入组装 + 优化调度
    pressure_tookit/ pressure_tookit 输入组装 + fitting 调度

消费端私有产物的组装（tensor 布局、scheduler 调度）留在各模型自己的目录；
跨模型共享的中间产物走 ``shared/frontends/``（provenance 在公共件共享）。
"""
