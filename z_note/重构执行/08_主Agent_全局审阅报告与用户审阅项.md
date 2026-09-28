# 主 Agent 全局审阅报告与用户审阅项

审阅日期：2026-09-28  
状态：**partial**

本报告只记录当前代码、Workspace 产物、文档和可复现命令的审阅结果。本次不修改模型代码、数据产物、任务书正文，不删除或恢复历史文件。

## 1. 工程第一性

- AnySole 是主模型，目标是尽可能好的效果。
- MotionPRO、Step2Motion、FPP-Net、PoseTransOpt、pressure_toolkit 是基线，保留原生模型、算法和专属前处理。
- 公共层统一 session、subject、split、frame、valid/fake、provenance、正式结果出口和公共评估。
- 模型私有层保留各自输入表示、转换、归一化、标签、训练和优化流程。
- Step2Motion 使用原生 BVH-23；MotionPRO 使用自身 `gen_bbox → gen_image_feature → gen_kps`，公共 CSV bbox 不替换 MotionPRO pipeline。

## 2. 全局模型矩阵

| 模型/方法 | 原生输入与私有管线 | 输出/评估协议 | 状态 |
| --- | --- | --- | --- |
| AnySole | 自有 RGB/压力/SMPL 特征、标签和训练链 | SMPL-24；当前不正式评估 GT-beta PVE 和 contact | partial |
| MotionPRO | YOLOX bbox、图像特征、关键点、dual-sole raster、私有 soft-f6 | SMPL-24；contact 仅私有 loss/诊断 | partial |
| Step2Motion | BVH-23、Y-X-Z、root translation、48→16、原生 IMU/force/CoP | BVH-23；不评估 MPJAE/PVE/shape/contact | partial |
| FPP-Net | temporal-5、31×11 MMVP、RTMPose、原生 pressure/contact head | V2T 九项；连续 contact_smpl 指标待正式重导出核对 | partial |
| PoseTransOpt / VP-MoCap | FPP prediction、CLIFF、keypoints、scene、原生 fitting | SMPL-24 | partial |
| pressure_toolkit | DepthPro、floor、depth mask、HALPE、insole、SMPL fitting | SMPL-24；地面契约严重失效 | blocked |

## 3. 公共数据基座

当前事实：manifest 144 行，严格公共交集 140 个 session，split 为 train/val/test = 104/36/36；缺少 SMPL 的 session 为 S12072、S12081、S14101、S6101。当前 split 不再按固定 92/12/36 验收，所有项目应读取用户生成的 canonical split CSV。

已发现风险：

1. `build_shared.py` 在 RGB 帧数不一致时会截断或重复最后一帧：`selected_images = images[:n] + [images[-1]] * max(0, n - len(images))`。本报告只记录，不补帧、不截断、不伪造。
2. eligible manifest 总帧数为 52612，human-mask frontend 当前 index 仅 51329 行。
3. shared facts 根 artifact 当前记录 `parameters.session_count=1`，与实际 140 个 session 不一致。
4. 公共 MMVP builder 当前读取 `Baselines/VP-MoCap/FPP-Net/essentials/insole2cont`，公共表示存在对 FPP-Net 私有资产的依赖。
5. 公共压力事实当前使用 normalized `t_us` 重采样，valid/fake 使用最近邻；该语义待用户调研确认。

## 4. bbox 与前处理

当前 human-mask frontend：session_count=140，index rows=51329，time_error_s 全为 null，valid=0 为 774，skipped_existing 为 24387，zero bbox 为 24387，重复 session/frame 为 0。

- CSV bbox 按用户说明视为可信来源。
- 零 bbox 视为导入不完整，不能自动重算、补齐或替代。
- FPP-Net、PoseTransOpt、pressure_toolkit 是否消费该 frontend，分别按各自原生管线处理。
- MotionPRO 继续使用自身 YOLOX bbox pipeline。

MotionPRO 权重已存在：`AnysoleWorkspace/assets/third_party/MotionPRO/mmdetection/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth`。但当前 MotionPRO model input 实际只有 1 个 session，不能以 pipeline/权重存在代替全量产物完成。

## 5. 评估协议

以 `z_note/metrics_指标改动说明.md` 为当前指标来源。代码层面已具备 SMPL-24、BVH-23、native protocol、protocol-aware root/trajectory/temporal/foot sliding、V2T 压力指标和缺失能力 `—`。common19 不作为正式数值协议。

仍存在文档漂移：新 metrics 文档启用 pressure_toolkit PVE，旧评估任务书仍禁用；旧 Agent C 报告仍写 AnySole pose-only PVE；AnySole `infer.py` 仍导出 `betas_source=ground_truth_session`；FPP 旧 sidecar contact GT 口径与当前 f6_soft 连续口径需要正式重导出核对。

## 6. 路径、README 与上下游

当前可执行代码仍存在 `workspace://derived`、`model-input://VP-MoCap`、`model_inputs/VP-MoCap`、`model-input://pressure_toolkit/fitting` 及失效入口 `build_fpp_metadata.py`、`check_splits.py`、`process_gait.py`。

当前实际情况：`model_inputs/VP-MoCap exists False`，`work/VP-MoCap exists True`，`model_inputs/FPP-Net exists True`，`model_inputs/PoseTransOpt exists True`。

README 失效命令的原始输出：`python3: can't open file '.../AnysoleWorkspace/tool/build_fpp_metadata.py': [Errno 2] No such file or directory`，退出码 2；`check_splits.py` 同样退出码 2。

## 7. 实际产物与正式结果

Workspace 产物：MotionPRO 4 files/1 session；Step2Motion 12 files；FPP-Net 52618 files；PoseTransOpt 154488 files/140 sessions；pressure_toolkit 3519 files/约 8 sessions。

正式结果：FPP-Net 1 个 npz；MotionPRO 36 个 npz；Step2Motion 37 个 npz、39 个 bvh；VP-MoCap 1 个 npz；pressure_toolkit 0 个正式 npz。全模型正式联合评估尚未完成。

## 8. 测试与验收证据

当前基础检查通过：`shared ok: 140 sessions; quality flags disjoint; boundaries clean`；`workspace ok: canonical tree, raw links and required artifacts are present`。

当前默认 Python 执行基线测试失败于环境依赖收集：`ModuleNotFoundError: No module named 'torch'`、`ModuleNotFoundError: No module named 'cv2'`、`5 errors during collection`、`pytest_exit=2`。

历史报告引用、但当前已删除的测试文件：`tests/test_baseline_utils.py`、`tests/test_protocol_consumer.py`、`tests/test_public_metrics.py`。历史测试结果不能直接视为当前工作区可复现证据。

当前 diff 检查：`configs/Z_README.md:246: new blank line at EOF.`，退出码 2。

## 9. 用户审阅项

以下项目只记录，不自动修复、不自动选择；当前状态均为 `blocked_by_user_review`。

| 编号 | 审阅事项 | 当前证据 | 用户需要决定 |
| --- | --- | --- | --- |
| U1 | pressure_toolkit 地面契约 | 棋盘格平面与真实行走地面偏差约 1.293–1.445m | 保留、舍弃、替换，或不纳入正式比较 |
| U2 | CSV bbox 零值与缺帧 | 24387 zero bbox；frontend 少于 manifest 总帧数 | 导入修复范围、有效帧定义和最终来源 |
| U3 | RGB/视觉帧数不一致 | builder 当前会补最后一帧或截断 | 提供调研报告和最终处理方案 |
| U4 | fake/valid、padding、截断 | 不同模型私有 pipeline 规则不同 | 哪些属于公共事实，哪些保留在私有 pipeline |
| U5 | 压力时间重采样 | shared facts 使用 normalized `t_us` | 是否作为公共固定口径 |
| U6 | MMVP 映射资产归属 | public builder 读取 FPP-Net essentials | 是否迁为公共固定资产 |
| U7 | pressure_toolkit PVE | 新 metrics 文档启用，旧任务书禁用 | 最终采用哪一版指标口径 |
| U8 | 历史删除与恢复 | 多个旧文件/测试已删除，工作区有大量未提交变更 | 哪些恢复、保留或进入删除清单 |
| U9 | 正式结果完整性 | FPP/VP-MoCap/pressure_toolkit 正式结果不齐 | 是否允许进入正式联合评估 |
| U10 | 任务书更新范围 | 仍含固定 split、mask 重算、旧路径等过期内容 | 是否统一修订 00–07 任务书 |

## 10. 总结

```text
评估协议基本成型；
公共数据和各模型 adapter 已部分实现；
但生产链、实际产物、路径检查、任务书和交付报告尚未形成一致闭环。
```

在用户审阅项未裁决前，不执行地面替换、bbox 补齐、帧数补齐/截断、公共资产迁移、历史文件清理或正式全模型联合验收。
