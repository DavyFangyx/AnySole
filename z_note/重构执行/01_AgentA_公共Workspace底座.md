# Agent A 任务书：公共 Workspace 底座

> 角色：公共数据契约和目录所有者。  
> 执行时机：第一个执行；B–F 不得在本任务的阶段验收前切换路径。  
> 重要：本任务只建新底座和兼容读路径，不删除旧树。

## 1. 现状问题

1. `session_manifest` 目前扫描 `derived/MotionPRO/sequences` 才能生成，存在公共协议反向依赖 MotionPRO 的问题。
2. `derived/MotionPRO/sequences` 混合了公共 RGB/时间轴、MotionPRO 输入、AnySole 标签和其他基线的数据源。
3. `sources/PressureWasher/outputs` 是生成产物，却与外部只读源混在 `sources/`。
4. `dependencies/` 混放第三方源码、权重、人体模型、上游数据和本项目训练产物。
5. 当前工具和配置大量硬编码 `derived/MotionPRO`。

## 2. 目标结构

建立以下目录，名称大小写固定：

```text
AnysoleWorkspace/
├── raw/
├── protocol/
├── shared/
│   ├── facts/
│   ├── representations/
│   └── frontends/
├── model_inputs/
├── work/
├── assets/
├── reports/
└── tool/
```

### `raw/`

只允许外部只读源或指向它们的软链接：

```text
raw/rgb
raw/pressure
raw/bvh
raw/smpl
raw/calibration
```

不得在 `raw/` 中写入清洗、插值、标签或特征。

### `protocol/`

```text
protocol/manifests/session_manifest.csv
protocol/manifests/session_manifest.jsonl
protocol/splits/default/splits.csv
protocol/schemas/
protocol/calibration/
```

`splits.csv` 已冻结为 92/12/36。manifest 重建不得默认重写 split。如需从零创建 split，必须使用不同的显式子命令和输出路径。

### `shared/facts/sessions`

每个 session：

```text
shared/facts/sessions/cam3/<date>/<subject>/<session>/
├── session.json
├── frames.npz
├── rgb/
└── pressure_48.npz
```

`frames.npz` 至少包含：

- `frame_id: int64[T]`；
- `visual_time_s: float64[T]`；
- `mocap_time_s: float64[T]`；
- `valid: uint8[T]`；
- `fake: uint8[T]`。

`pressure_48.npz` 至少包含：

- `frame_id: int64[T]`；
- `left48: float32[T,48]`；
- `right48: float32[T,48]`；
- 量程/单位/源文件记录。

原始数据来源必须在 `session.json`/`artifact.json` 逐脚写清：

- `left_source`：PressureWasher 最终 fake_marked 的 `pressure_left.csv`；
- `right_source`：PressureWasher 最终 fake_marked 的 `pressure_right.csv`；
- 数值列：`1..48`；
- 原始布局：每脚 `4×12`，每帧两脚合计 96 格；
- 时间列：`t_us`，并记录 shared frame time 的对齐方法；
- 质量列：`fake`/`valid_mask` 的来源；
- 左右 CSV 和 shared NPZ 的 hash。

不得将旧 MotionPRO `pressure.npz` 登记为 raw/shared 触觉源。该文件是 MotionPRO 私有 raster 产物，不是原始采集。

`session.json` 记录 session/date/subject/camera/fps/offset、原始 RGB/BVH/SMPL/压力路径、帧数、quality、源文件 hash 和 schema version。

`shared/facts` 不得保存：

- `contact*.npy`；
- `feature_hrnet.pth`、bbox、模型 keypoints；
- MotionPRO 160×120/96×96 压力图；
- Step2Motion 16/50 通道；
- MMVP 31×11 insole（它属于 `shared/representations`）；
- 地面估计、人体 mask 或模型预测。

### `shared/frontends`

允许多模型共用的确定前端产物，但必须带版本、frame id 和 provenance：

```text
shared/frontends/depthpro/<version>/...
shared/frontends/rtmpose_halpe26/<version>/...
shared/frontends/human_masks/sam31/...
shared/frontends/cliff_hr48/<version>/...
```

Agent A 新建 human-mask frontend 索引，但不复制外部 PNG：

- 源根：`/data/lizhe/projects/Tactile/3_Result/processed/rgb_human_masks`；
- 只纳入 canonical split 的 140 session 和 camera=3；
- 读取 `masks_index.csv`/`segmentation_manifest.json`/`failures.csv`；
- 按 `visual_time_s` 与 shared frames 最近邻匹配，上限 20ms；
- 记录 frame_id、visual_time_s、source image、mask path、time error、mask area、重算 bbox、source manifest/hash；
- `status=skipped_existing` 不代表失败；只要 mask 存在且非空就重算 bbox；
- 不读 CSV 中可能为零的 bbox 作为正式 bbox。

CLIFF frontend 的输入 bbox 必须来自该 human-mask frontend，不再保留官方 YOLOv3 的多人行作为 canonical NPZ。

### `shared/representations/tactile/mmvp_31x11`

Agent A 负责把已经 D_Test4 人工验收的 4×12→31×11 映射收口为一份版本化公共表示：

```text
shared/representations/tactile/mmvp_31x11/<version>/<date>/<subject>/<session>/
├── frame_id.npy
├── insole/<frame_id>.npy
└── artifact.json
```

pressure_toolkit 和 FPP-Net 只读这一份数据。转换器不得位于 `results_display`，D_Test4 也不再触发生产。

## 3. 实施要求

### 3.1 路径解析

新增唯一路径解析实现，支持：

- `raw://`
- `protocol://`
- `shared://`
- `model-input://<model>/`
- `work://`
- `asset://`
- `results://`

旧 `workspace://derived/...` 只允许在迁移期作为只读兼容入口，不得继续成为新代码默认值。

### 3.2 Manifest 与 split

- manifest 从 `raw/` 、原始 session 索引和 PressureWasher 最终有效记录构建。
- 禁止扫描 `model_inputs/` 或旧 `derived/MotionPRO` 来决定 session 是否存在。
- manifest builder 默认只更新 manifest，不更新 split。
- split validator 必须检查重复 session、subject 泄漏、manifest 缺失和 92/12/36 计数。

### 3.3 PressureWasher 分层

- inspect/reconstruct/mark-fake 过程文件进入 `work/data_pipeline/pressure_washer/`。
- 只有通过验收的 frame/time/pressure/valid/fake 被提升到 `shared/facts/sessions`。
- 不得将 PressureWasher 过程目录称为 raw source。

### 3.4 Artifact manifest

每个生成根写 `artifact.json`，至少含：

```json
{
  "schema_version": "...",
  "producer": "...",
  "producer_commit": "...",
  "parameters": {},
  "source_artifacts": [],
  "source_hashes": {},
  "frame_id_min": 0,
  "frame_id_max": 0,
  "frame_count": 0,
  "consumers": []
}
```

## 4. 写入范围

允许修改：

- `AnysoleWorkspace/README.md`；
- `AnysoleWorkspace/tool/workspace.py`；
- manifest/split/shared-session 相关工具；
- 新的公共 path/schema/validator 模块；
- 公共结构测试；
- `Baselines/command_manual.md`，但只在 B–F 全部交付后的收口阶段修改。

`AnysoleWorkspace/dependencies/` 本轮不做第三方源码的物理迁移。Agent A 只增加清单/类型/provenance 分类，不把 CLIFF、mmpose 等移到新 `third_party/`，避免破坏环境。

禁止修改：

- 任何模型结构、loss 或 augmentation；
- B–F 所属的模型代码；
- 公共指标公式；
- 正式 results 数值。

## 5. 兼容与迁移

1. 先建新树并生成报告，不移除旧树。
2. 新老数据并存时，新工具只写新树。
3. 不创建可写的新老双向软链接。
4. 为 B–F 提供只读的 path API 和一个小 session fixture。
5. 交付一份旧路径候选删除清单，但不执行删除。

## 6. 必做测试

- path URI 正常解析和路径逃逸拒绝。
- manifest 在隐藏旧 MotionPRO sequence root 后仍能构建。
- split validator 确认 92/12/36、subject-disjoint、val/test 无重复。
- shared facts session 的 frame id 唯一且单调，所有数组等长。
- MMVP 31×11 公共表示只有一份，FPP-Net/pressure_toolkit 无重复数据树。
- human-mask frontend 覆盖 canonical 140 session，52,612 帧全部在 20ms 容差内有非空 mask；
- 每帧 bbox 由 mask 重算，不依赖 CSV 的零 bbox；
- CLIFF canonical frontend 每帧恰好一行，frame id 唯一。
- pressure 左右脚各 48 格，fake 与 valid 互斥。
- 结构 lint 拒绝 `shared` 对 `model_inputs` 的依赖。
- 扫描新增代码，不得新增 `derived/MotionPRO` 硬编码。

## 7. 阶段验收输出

必须交付：

- 新目录树及用途说明；
- schema 和 path API 使用示例；
- 一个完整 session 的 shared fixture；
- manifest/split/shared validator 结果；
- 旧→新路径映射表；
- 触觉 provenance 表：每个 session 的左/右原始 CSV → shared `pressure_48.npz`；
- 候选删除清单（仅清单）；
- 给 B–F 的已冻结接口说明。

## 8. 收口阶段

B–F 完成后，再根据它们的实际 CLI 和路径统一更新 Workspace README 与 Baselines command manual。不得在第一阶段预写尚未实现的命令。

同时更新 `results_display/README.md` 的 D_Test4：

- 写明单一物理采集源是左/右 48 格 PressureWasher CSV；
- 写明每个模型私有产物或公共 representation 的 source → adapter → output 路径；
- 写明转换已经用户肉眼验收通过，转换口径冻结；
- 将 D_Test4 改为只读审计，移除“缺产物则自动生产模型输入”的责任。
