"""RTMPose HALPE-26 关键点生产（三模型共用前端）。

生产者 ``run_rtmpose.py``：逐 canonical 帧写 ``<frame_id:06d>.npy`` 侧车
（keypoints (26,2) / keypoint_scores (26,) / frame_id / source_image / model），
默认写入 ``shared/frontends/rtmpose_halpe26/v1``，并刷新 FPP-Net 与
PoseTransOpt 模型树的关键点目录软链接。
"""
