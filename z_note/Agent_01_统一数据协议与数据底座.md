# Agent 01：统一数据协议与数据底座

## 任务目标

为 AnySole、MotionPRO、Step2Motion 和 pressure toolkit 冻结一份可复用的数据底座，保证四个项目在相同 session、相同时间轴、相同动捕 GT、相同压力原始数据和相同数据划分上运行。模型内部可保留各自的特征工程（例如 48→16 池化或压力栅格化），但不得改变观测数据的来源、时间索引和有效帧定义。

## 工作范围与约束

- 统一工作区为 `AnysoleWorkspace/`；AnySole 主干代码位于仓库根目录的 `anysole/`、`configs/`、`data/`、`outputs/`。
- 默认视频相机为 `cam3`，目标频率为 40 Hz，关节协议为 Skeleton3 的 23 个关节及其 parent hierarchy。
- 只负责数据索引、对齐、质量检查和适配测试，不修改任何模型结构、loss 或训练逻辑。
- 不覆盖或删除原始数据；所有生成物写入 `AnysoleWorkspace/derived/`、`AnysoleWorkspace/manifests/` 或临时目录。

## 输入

1. `AnysoleWorkspace/` 下已有的统一序列（视频、压力、`align_meta.json`、`keypoints.npy` 或 BVH）。
2. 原始 BVH、压力 CSV/NPZ 及视频目录（若统一序列缺失，以 manifest 中的 source 字段指向它们）。
3. 现有划分文件（如 `splits.csv`）和各项目的数据读取脚本，仅用于核对，不直接作为最终协议。

## 输出

### 1. 冻结 manifest

在 `AnysoleWorkspace/manifests/` 生成版本化的 session manifest（CSV 或 JSONL 均可，字段名必须稳定），至少包含：

`session_id, subject_id, action, trial, camera, video_path, pressure_path, bvh_path, target_fps, n_frames, visual_start_s, mocap_start_s, offset_s, fake_frame_indices, valid_frame_indices, quality, split_iid, split_ood`。

路径可使用相对仓库根目录的形式；`valid_frame_indices` 与 `fake_frame_indices` 必须能被其他 agent 直接解析。若原始数据缺字段，保留空值并在质量报告中标注阻塞原因。

### 2. 协议说明与质量报告

- `AnysoleWorkspace/manifests/统一数据协议.md`：说明坐标系、单位、帧率、时间轴、23 关节顺序、压力布局、归一化及 mask 规则。
- `AnysoleWorkspace/manifests/data_quality_report.(md|csv)`：逐 session 报告文件存在性、帧数、时间戳、压力范围、BVH 关节和 fake 帧统计，并汇总失败项。

### 3. 压力布局测试

将测试代码/脚本放在 `tests/` 或 `AnysoleWorkspace/tool/`（不改模型代码），覆盖 48 点原始压力到各适配表示的确定性映射。

## 执行步骤

### A. 盘点现有数据与读取接口

1. 使用 `rg --files AnysoleWorkspace Baselines anysole configs data` 找到 session、BVH、压力、视频和已有 split。
2. 阅读 `align_meta.json`、序列生成脚本及三个 baseline 的 dataset loader，记录各自使用的路径、帧率、首帧和 fake-mask 逻辑。
3. 列出冲突（同一 session 的帧数、offset、相机或关节顺序不同），但先不修改下游代码。

### B. 构建统一时间轴

1. 以 `align_meta.json` 中的视觉起点和 mocap offset 为准，生成 40 Hz 帧索引 `frame_index=0..T-1`。
2. 对视频、压力和 BVH 做边界检查；禁止负索引、重复帧和静默截断。若必须截短，记录明确的 `n_frames` 和原因。
3. 只允许中央 GT 处理一次 BVH 重采样：记录原始 FPS、Euler 顺序、角度单位、root translation 单位及旋转插值方法，供下游读取缓存。

### C. 冻结 23 关节协议

1. 从 Skeleton3/BVH 提取 23 个 joint names、parent indices、offsets；检查所有 session 名称和拓扑完全一致。
2. 统一世界坐标单位为米；保留 root translation 和每关节世界位置的定义。
3. 为每个 session 生成可校验的关节顺序 checksum（例如 names+parents 的 SHA256），写入 manifest。

### D. 冻结压力协议

1. 确认左右脚各 48 点、4×12 行列方向、左右脚顺序、toe/heel 方向及原始量程（通常 0–1023）。
2. 规定原始压力只在适配层归一化；记录 `/1023`、`/255` 等缩放发生的位置，避免重复缩放。
3. 明确缺失值、全零帧和 fake-pressure 的编码；`fake_frame_indices` 必须由统一来源生成。

### E. 冻结数据划分

1. 生成 IID split（subject+action/trial 规则与现有实验保持兼容）和 OOD-subject split（整 subject 留出测试）。
2. 检查 train/val/test 无 session 重叠；输出各 split 的 subject、session、有效帧和窗口统计。
3. 下游 agent 只能读取此 manifest，不得重新生成 split。

### F. 编写并运行布局单元测试

测试至少包含：单点左脚、单点右脚、全常数压力、toe 集中、heel 集中、左右脚交换、量程边界（0/1023）。验证映射后的空间位置、左右隔离、总力近似守恒及归一化范围。

## 测试命令

在仓库根目录执行（按实际脚本名称替换）：

```bash
python AnysoleWorkspace/tool/build_manifest.py --fps 40 --camera cam3
python AnysoleWorkspace/tool/check_data_quality.py --manifest AnysoleWorkspace/manifests/session_manifest.jsonl
python AnysoleWorkspace/tool/pressure_washer/test_pressure_layout.py
python AnysoleWorkspace/tool/check_splits.py --manifest AnysoleWorkspace/manifests/session_manifest.jsonl
```

若脚本尚不存在，先实现最小可运行版本或提供等价命令；命令必须返回非零状态表示失败。所有测试不得修改原始输入。

## 验收标准

- manifest 可被四个项目读取，每个 session 的视频、压力、BVH 均有明确路径和 `n_frames`。
- 所有纳入 session 的 40 Hz 时间轴单调、无重复，视频/压力/BVH 帧索引一致；offset 和截短原因可追溯。
- 23 个 joint names、parents、单位和关节 checksum 全部一致。
- fake/invalid 帧在 manifest 中显式列出；IID 与 OOD split 无重叠且统计完整。
- 压力布局测试全部通过：左右不串位、toe/heel 方向正确、常数输入不产生异常缩放、量程范围明确。
- 质量报告列出所有失败 session；任何未解决阻塞必须在回传中说明，不能静默跳过。

## 禁止事项

- 不得修改 AnySole、MotionPRO、Step2Motion 或 pressure toolkit 的网络结构和损失函数。
- 不得把 3D GT 关键点投影后冒充视频/RTM-pose 观测，也不得用预测结果生成 GT。
- 不得在各 baseline 内重新定义时间轴、split、fake-mask 或 23 关节顺序。
- 不得删除、覆盖或移动原始视频、压力、BVH 和标定文件。
- 不得把 Step2Motion 的 16 通道池化结果当作统一原始 48 点压力协议。

## 回传格式（发给主 agent）

```text
状态：完成 / 部分完成 / 阻塞
改动文件：逐项列出路径及用途
Manifest：路径、版本、session 数、各 split 数量
协议摘要：FPS、相机、23 关节 checksum、压力布局、mask 规则
运行命令：逐条列出实际执行命令
测试结果：通过/失败、关键统计和日志路径
未解决问题：按优先级列出，说明复现影响和建议
与其他 agent 的接口：下游应读取的字段、文件和示例
```

