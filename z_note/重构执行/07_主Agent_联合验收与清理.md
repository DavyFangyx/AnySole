# 主 Agent 联合验收与清理任务书

> 前置：Agent A–F 已提交交付报告。  
> 职责：只以代码、数据、测试和 provenance 证据验收；不以“能跑”代替语义验收。  
> 删除：仅本阶段在新链路全部通过后有资格执行。

## 1. 验收输入

必须收集：

- Agent A–F 六份完整交付报告；
- 当前 `git status --short` 和 `git diff --stat`；
- 每个 Agent 的修改文件清单；
- 全部自动测试输出；
- manifest/split/shared validator 报告；
- 六个模型的小样本 smoke 结果；
- 旧→新路径映射和候选删除清单。

任一项缺失时，不进入删除阶段。

## 2. 第一层：文件所有权与用户改动

1. 核对每个 Agent 是否只修改分配路径。
2. 识别本轮前已存在的用户改动，确认未被覆盖。
3. 检查共享文件是否只由 Agent A 收口。
4. 如发现跨 Agent 文件冲突，退回文件所有者解决，不直接选一份覆盖。

## 3. 第二层：目录与依赖结构

必须满足：

```text
raw → protocol → shared/{facts,representations,frontends} → model_inputs/<model> → work → results → results_display
```

检查项：

- manifest builder 不读 `model_inputs` 或旧 `derived/MotionPRO`；
- `shared/facts` 不含 contact、模型特征、模型专用压力表示、地面估计或人体 mask；
- `shared/representations` 只含已批准、版本化、确定性的多模型共用表示；
- `model_inputs/A` 不读 `model_inputs/B`；
- workspace tool 不 import `results_display`；
- work 不写入 results；只有统一 exporter 写入正式 predictions/metrics；
- 新代码无 `derived/MotionPRO` 默认路径；
- 旧 URI 只能显式兼容读，不能成为新产物默认写入点。

## 4. 第三层：公共协议

### Manifest/split

- manifest 能在不存在任何 model input 时重建；
- split 为 92/12/36；
- subject-disjoint；
- val/test 无重复；
- manifest 重建不会静默覆盖 split。

### Frame

- frame id 从 0 开始、唯一、单调；
- visual/mocap time 符合冻结公式；
- pressure/RGB/valid/fake 长度一致；
- fake 与 valid 互斥；
- 所有 frontend/model adapter 可追溯回 shared frame id。

### 唯一受试者 mask/frontend

- canonical 140 session 全部存在外部 `masks_index.csv`；
- 52,612 个 shared cam3 frame 全部能按 `visual_time_s` 在 20ms 内匹配非空 mask；
- mask 仅包含受试者，外部目录保持只读；
- bbox 从 mask 重算，不使用 `skipped_existing` 记录中的零 bbox；
- 缺失/空 mask/超容差时标 invalid，无多人或全景回退。

### 触觉源与 D_Test4

- 每个 session 的 shared pressure 可追溯到左/右 PressureWasher `pressure_*.csv` 的数值列 `1..48`；
- 原始物理布局统一记录为每脚 4×12，两脚 `(2,4,12)`；
- MotionPRO/Step2Motion 从 shared facts pressure 生成私有产物；FPP-Net/pressure_toolkit 共用单一公共 MMVP 31×11 representation；
- AnySole 直接使用 shared 48 格及其私有 token；
- D_Test4 已经用户人工验收的转换几何/池化/排列未被重新设计；
- D_Test4 现为只读审计，缺少模型输入时明确报错，不自动生产训练/优化数据；
- D_Test4 报告显示 raw CSV hash、shared pressure hash、adapter/version、模型私有落点；
- FPP-Net 与 pressure_toolkit 只读同一份 31×11 公共 representation，无重复私有树。

## 5. 第四层：模型语义保真

### AnySole

- 仍为 SMPL-24，模型/loss 未被 workspace 重构改变；
- contact method 和 visual cache 在 AnySole 私有目录；
- 不向基线输出统一 GT。
- 正式配置显式为 `f6_soft`，配置缺失时无隐式 fallback；AnySole 二值 f6 NPY 未被 MotionPRO 任务改写。

### MotionPRO-InsoleAdapted

- 当前固定左右块已明确记为 dual-sole raster，未冒充 pressure carpet；
- `contact.npy` 仍用于 `loss_foot`，且 artifact 标记为 MotionPRO 私有四档 soft-f6 权重；
- 公共 evaluator 不读它；
- 原生 Contact IoU 不进正式表；
- soft-f6 仅在 MotionPRO 私有产物恢复四档值，没有改写 AnySole 二值 NPY；
- contact 列 `[6,7]` 加权 SMPL joint `[10,11]`，M1 本地索引回归已修复；
- C1 审计经用户确认后才执行 C2 全量生成/训练；
- 无 fake/padding 帧进入训练 loss。

### Step2Motion

- 数据是 BVH-23，正式配置的压力/IMU/translation 契约未改；
- `.pt` 严格使用 canonical split；
- normalizer 只由 train 生成；
- 公共腿/脚指标按关节名称计算。

### FPP-Net / PoseTransOpt

- FPP GT 为原生 press2Cont 顶点级接触，不读 f6；
- toe/heel 四区有独立差异；
- RTMPose/FPP/CLIFF 的 frame id 契约已验证；CLIFF 每个 canonical frame 恰好一行唯一受试者结果；
- 上游 augmentation/loss 未被未命名修改。

### pressure_toolkit

- 地面是上游深度估计语义，不是棋盘格平面；
- depth mask 为 `SAM3.1 human mask ∩ finite depth ∩ 0.4–5m`，地面/背景不再主导观测；
- D_Test6 已完成四日期地面可视化、用户确认和约 1.45m 错位修复；
- 正式配置无适配层 stride=2；
- insole 是模型输入，fake 帧不进优化；
- NaN、前帧接触和 HALPE 处理符合任务书边界。

## 6. 第五层：数值和端到端验收

必须执行：

1. 公共指标数值测试；SMPL-24/BVH-23 pred=GT 归零。
2. split/session/frame validator。
3. 触觉 provenance validator：原始左/右 CSV → shared facts 48 格 → 私有或公共 representation。
4. human-mask validator：140 session/52,612 frame 覆盖、20ms 容差、非空 mask、bbox 重算。
5. D_Test4 代表 session 的新旧转换数值 parity 与只读行为测试。
6. 每个模型的固定 session dataset smoke。
7. 学习模型 one-batch forward/loss；优化模型 one-frame/one-session smoke。
8. 六类标准 prediction 的 exporter/validator。
9. 公共 evaluator 对每类 prediction 至少一个 session 成功计算。
10. 正式 contact 能力门控：Motion 模型没有显式 contact prediction 时必须为不适用。

## 7. 旧路径和旧结果扫描

扫描：

- 所有代码中的 `derived/MotionPRO/sequences`；
- 模型 A 对模型 B 私有目录的引用；
- `results_display` 作为训练数据生产者的引用；
- 旧 gait.pt、坏 CLIFF NPZ、f6 软链接、旧 pressure_toolkit fitting；
- 由旧协议生成的 checkpoint、prediction 和 metrics。

形成逐项删除 manifest：绝对路径、类型、文件数、字节数、为何可删、新替代物路径。

## 8. 删除门

用户已裁定旧错误结果可直接删除，但仍必须满足：

1. 对应新数据/预测已通过 validator；
2. 删除目标不是 raw、protocol、calibration 或静态 asset；
3. 删除清单不使用未解析变量或宽泛 glob；
4. 每批删除后重跑 doctor、validator 和一个消费端 smoke；
5. 记录释放空间、删除时间和不可恢复性。

删除范围默认包含：

- 旧 `derived/` 中已被新 shared/model_inputs/work 完整替代的产物；
- 使用错误 split/时间轴/contact/地面/mask 产生的 checkpoint、prediction、metric；
- 失效的兼容软链接和临时中间态。

## 9. 返工规则

| 问题 | 返工对象 |
| --- | --- |
| manifest/split/shared schema | Agent A |
| AnySole 标签/cache/路径 | Agent B |
| MotionPRO contact/时间轴/窗口 | Agent C |
| Step2Motion split/normalizer/BVH | Agent D |
| FPP contact/CLIFF/frame join | Agent E |
| pressure_toolkit floor/mask/fitting | Agent F |
| 共享文档与 doctor | Agent A 收口阶段 |

## 10. 最终报告模板

```text
Agent A：complete / partial / blocked
Agent B：complete / partial / blocked
Agent C：complete / partial / blocked
Agent D：complete / partial / blocked
Agent E：complete / partial / blocked
Agent F：complete / partial / blocked

用户原有改动保护：
公共底座：
split/frame 契约：
跨层依赖：

AnySole 语义保真：
MotionPRO 语义保真：
Step2Motion 语义保真：
FPP-Net/PoseTransOpt 语义保真：
pressure_toolkit 语义保真：

数值测试：
模型 smoke：
标准导出/评估：
旧路径残留：

已删除内容：
释放空间：
不可恢复内容：

返工项：
未解决阻断：
最终裁定：pass / conditional pass / fail
```
