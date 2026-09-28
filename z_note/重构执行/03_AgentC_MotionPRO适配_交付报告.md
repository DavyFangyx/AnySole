# Agent C 交付报告：MotionPRO-InsoleAdapted 复现适配

任务：`z_note/重构执行/03_AgentC_MotionPRO适配.md`
状态：**partial（C1 完成，等待用户确认后进入 C2）**

## 修改文件

| 文件 | 内容 |
| --- | --- |
| `Baselines/MotionPRO/app/train_frappe.py` | M1 修复（joint [10,11] / contact 列 [6,7] 分离）；原生 IoU 标记 diagnostic_only；删除死函数 `contact_iou` |
| `Baselines/MotionPRO/lib/dataset/image_pressure.py` | 改读 `model_inputs/MotionPRO/<ver>/cam3`；fake/valid 读 shared facts frames.npz；M5 窗口索引（train/eval 只索引完整且全 valid 窗口；test 保留尾部补齐仅用于推理导出）；删除 pressure_96 缓存分支与 contact_method 分支 |
| `Baselines/MotionPRO/lib/util/workspace.py` | canonical URI 前缀（model-input/protocol/shared/asset/work）；`sequence_root` 默认指向 model_inputs；legacy 前缀迁到 assets/third_party；raw bvh 解析迁 raw/bvh |
| `Baselines/MotionPRO/lib/eval/metrics.py` | 对齐评估整改后的 canonical API（`foot_sliding_joints` 关节口径为公共 foot_sliding_mm；列集冻结为历史 14 列）——修复预存 ImportError |
| `Baselines/MotionPRO/data_prep/make_splits.py` | M4：重写为只读验证命令（val==test、列内去重、subject 隔离、shared facts 存在性；写 0 文件） |
| `Baselines/MotionPRO/config/task/imagepressure2smpl.yaml` | 新 URI（model-input://MotionPRO/adapter_v1/cam3、protocol://、asset://）；移除 contact_method |
| `Baselines/MotionPRO/README.md`、`data_prep/README.md` | 新输入链、C1/C2 门控、删除自行划分 split 引导 |
| `Baselines/MotionPRO/lib/util/{gen_bbox,gen_kps,gen_image_feature,FRAPPE}.py` | dependencies/ → assets/third_party/（C2 生成脚本路径修复） |
| `AnysoleWorkspace/tool/adapters/MotionPRO/adapter.py` | **新增**：私有 adapter（冻结 rasterize_feet + 私有四档 soft-f6 + 落盘 + C1 审计） |
| `Baselines/MotionPRO/tests/` | **新增** 8 个测试脚本（见下） |
| `z_note/重构执行/03_AgentC_MotionPRO适配_交付报告.md` | 本报告 |

删除（用户指示：有问题的旧代码直接移除）：

- `Baselines/MotionPRO/data_prep/prepare_sequences.py` —— 旧管线本体：绝对 t_us 插值（审计报告记录的 -11 帧错位来源）+ `>100` 二值接触 + 写 derived/ 旧树，全部被 shared facts + 新 adapter 取代
- `Baselines/MotionPRO/lib/util/gen_contact.py` —— 上游压力地毯邻近接触生成（T,4 口径），与私有 soft-f6 (T,10) 契约冲突

## 新增产物

```text
AnysoleWorkspace/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/
├── pressure.npz      # dual-sole raster (543,160,120)，冻结固定左右块
├── contact.npy       # 私有四档 soft-f6 脚损失权重 (543,10)，仅列 6/7 非零
├── frame_id.npy      # shared frame id（逐帧一致）
└── artifact.json     # contact 语义声明（见下）
AnysoleWorkspace/reports/MotionPRO_insole_adapted/C1_audit/
├── S5091_raw_4x12.png / S5091_raster.png
├── S5091_soft_f6_series.png / S5091_soft_f6_hist.png
└── S5091_audit.json  # 四档计数、每脚时序统计、窗口统计
```

artifact.json 的 contact 声明（任务书 §5 逐字）：

```json
{
  "physical_file": "contact.npy",
  "semantic_role": "motionpro_private_soft_f6_foot_loss_weight",
  "value_set": [0.05, 0.30, 0.70, 0.95],
  "formal_metric_input": false,
  "native_diagnostic_only": true
}
```

## 保留的上游语义

- FRAPPE 模型结构、loss 权重体系、`lamda_foot=5`、十列 contact 官方对齐 `[1,2,4,5,7,8,10,11,20,21]`（保留为文档常量）。
- 冻结转换口径：`rasterize_feet`（clip 1023 → /1023*255 → 4×12 → 20×4 块 → 左框 rows[40:120]/cols[6:54]、右框 rows[40:120]/cols[66:114]）+ bilinear 96×96 align_corners=False，数据集载入时 /255。
- F6 判据（fix_plan_v2.md F6a）全局常数，未重设计、未逐 session 拟合。

## 修复的本地适配问题

- **M1**：`footContactIds=[6,7]` 原被同时当作 SMPL joint id 使用（`pred['joint'][:, self.footContactIds]` 取的是关节 6/7）。现已分离：权重取 contact 列 [6,7]，预测/GT 关节取 SMPL [10,11]。测试证明扰动关节 6/7 不改变 loss_foot、扰动 10/11 改变。
- **M2**：作废 `>100` 二值 contact；adapter 自算私有四档软值，不读 AnySole 已二值化 npy。
- **M3**：adapter 直读 shared frame id 与 pressure_48；不再解释 t_us/visual offset/40Hz 网格（源码断言测试）。
- **M4**：make_splits.py 不再写 split，只读验证。
- **M5**：train/eval 只索引完整且全 valid 窗口；尾部不足窗长帧不进训练。
- 顺带修复评估整改遗留的 `lib/eval/metrics.py` ImportError（`foot_sliding` 更名）。

## 未修复的上游缺陷

无新增。发现并上报（非 MotionPRO 域，未改）：

1. **canonical split 实为 104/36/36（val=test），与 Agent A 冻结声明 `CANONICAL_SPLIT_COUNTS={92,12,36}` 不符**。validator 按结构不变量校验并以 WARNING 报告计数漂移，请 Agent A 核对声明或 CSV。
2. C2 依赖项：`smpl.npy` 的生成逻辑随 prepare_sequences.py 删除而空缺（可从 git 历史/上游备份恢复 `motion_to_smpl`，或改由 raw smpl npz 按 mocap_time_s 插值）；`color/` 帧图需改读 shared facts `rgb/` 软链接；gen_bbox/gen_image_feature/gen_kps 的目录遍历已指向新树，但未在 C1 运行。

## 执行的测试（8 个脚本全过，touch_gait 直接运行）

| 测试 | 覆盖的任务书 §7 要求 |
| --- | --- |
| `tests/test_contact_foot_loss_indices.py` | contact 进 loss_foot；[6,7] 列与 [10,11] 关节分离；删除 IoU 不影响 loss；scheduler/best 只依赖 loss_eval['loss'] |
| `tests/test_adapter_soft_contact.py` | 软值只来自 {0.05,0.30,0.70,0.95}；其余 8 列为 0；frame_id 一致；artifact 声明 |
| `tests/test_raster_frozen_conversion.py` | 冻结常数与 D_Test4 显示层逐值一致；round-trip parity（含 clip）；固定框与背景；bilinear 96 契约 |
| `tests/test_window_indexing_m5.py` | M5 窗口索引；fake 剔除；尾部剔除；test 模式保留补齐 |
| `tests/test_make_splits_validation.py` | M4 只读验证、损坏 CSV 拒绝、零写入 |
| `tests/test_public_evaluator_no_motionpro_contact.py` | 公共 Baselines/utils 不读 MotionPRO contact |
| `tests/test_shared_frame_alignment.py` | M3：不重解释时间轴；frame 一致性；**互相关中位时移 L=1 / R=2 帧（旧口径中位 -11 帧，已消除）** |
| `tests/test_f6_criterion_parity.py` | 与 anysole F6 实现逐位一致（motion_f6/pressure_f6/四档软值）——"复用 F6 中间软判据"证据 |

旧测试 `app/test_eval_metrics.py`、`app/test_wandb_cli.py` 仍通过。

## 未执行的重型测试及原因

- train one-batch / val one-batch / test one-session / 公共 prediction validator：**C1 门禁**——任务书禁止确认前训练；且 model_inputs 尚无 feature_hrnet/smpl/keypoints/bbox（C2 全量生成范围）。
- 全量 140 session 生成：C2 门禁同上。

## C1 审计摘要（代表 session S5091，train，543 帧，30 fake）

- dual-sole raster：值域 0–255；round-trip 与冻结逆变换 maxdiff < 1e-3（float32 精度内）。
- 私有 soft-f6 四档计数（每脚总和=543）：左脚 {0.95:214, 0.70:149, 0.30:58, 0.05:122}；右脚 {0.95:222, 0.70:141, 0.30:44, 0.05:136}。四档全部出现。
- 接触率（motion_f6）：左/右均 0.6685；承重率（pressure_f6）：左 0.501 / 右 0.490。
- 窗口统计：27 个完整窗口（20 帧）→ 保留 25、因 fake 剔除 2、尾部不足剔除 3 帧。
- 压力↔足高互相关：密集窗口（步长 10、长 100）中位时移 左 1 帧 / 右 2 帧，无旧口径的系统 -11 帧错位。
- 转换 parity：新旧源 CSV 的 t_us 同为传感器相对钟，旧管线按绝对 t_us 插值到视觉网格是错位根源（审计报告 z_note/评估/基线语义审计_总报告_20260927.md 亦记录）；新 raster = 冻结函数作用于 shared 校正后格子，转换口径未重定义。

## 开始前已存在改动 / 本轮未触碰

- 开始时仓库已有大量用户改动（AnysoleWorkspace/tool、anysole/**、README 等，见会话开头 git status）。本轮只写任务书允许范围。
- 未触碰：`anysole/**`、其他 baseline、公共指标公式、`Baselines/utils/**`、Agent A 的 shared facts/protocol/schema、canonical splits.csv、D_Test4 显示层。

## 建议主 Agent 重点复核

1. C1 审计产物（PNG + audit.json + model_inputs 小样本）是否确认通过 → 放行 C2。
2. canonical split 104/36/36 vs 冻结声明 92/12/36 的归属裁定（Agent A）。
3. prepare_sequences.py 删除后 C2 的 smpl.npy 生成方案。
4. `lib/eval/metrics.py` 的 foot_sliding_mm 已切关节口径（canonical），MotionPRO 历史表的该列口径随之变化，旧结果不受影响。
