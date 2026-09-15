# 数据集文件树说明

## 1. `Toy_Sequence`

实际目录：
`/data/lizhe/projects/Tactile/tmp/Toy_Sequence`

```text
Toy_Sequence/
└── 2024-07-29-09-58-38/
    ├── 0.mp4
    ├── 1.mp4
    ├── 2.mp4
    ├── 3.mp4
    ├── smpl.npy
    └── pressure.npz
```

文件说明：

| 文件 | 类型 | 内容 |
|---|---|---|
| `0.mp4` ~ `3.mp4` | MP4 视频 | 4 视角同步视频，用于可视化/对齐辅助 |
| `smpl.npy` | NumPy NPY | 动捕数据；实际为一个字典，包含 `betas`、`global_orient`、`body_pose`、`transl` |
| `pressure.npz` | NumPy NPZ | 触觉数据；键为 `pressure`，示例 shape 为 `(18036, 160, 120)`，可按帧与视频/动捕对齐，属于“触觉数据 + 对应时间顺序” |

备注：
- 本目录下未见单独的时间戳文件。
- 时间信息主要由序列目录名 `2024-07-29-09-58-38` 和帧序对齐关系隐式提供。

## 2. `MotionPRO_1/nasdata/shenghao/Pressure_release_new`

实际目录：
`/data/lizhe/projects/Tactile/tmp/MotionPRO_1/nasdata/shenghao/Pressure_release_new`

整体结构按“日期 / 被试 / 序列”组织：

```text
MotionPRO_1/
└── nasdata/
    └── shenghao/
        └── Pressure_release_new/
            ├── 0729/
            │   ├── csy/   (12 段)
            │   └── qnx/   (16 段)
            ├── 0730/
            │   ├── xty/   (13 段)
            │   └── zxy/   (12 段)
            ├── 0731/
            │   ├── sdc/   (15 段)
            │   └── yxy/   (12 段)
            └── 0801/
                ├── hcc/   (9 段)
                └── lh/    (15 段)
```

每个序列目录内部结构一致，例如：

```text
Pressure_release_new/
└── 0729/
    └── csy/
        └── 2024-07-29-09-58-38/
            ├── 0.mp4
            ├── 1.mp4
            ├── 2.mp4
            ├── 3.mp4
            ├── smpl.npy
            └── pressure.npz
```

文件说明：

| 文件 | 类型 | 内容 |
|---|---|---|
| `0.mp4` ~ `3.mp4` | MP4 视频 | 4 视角同步视频，用于观察动作与触觉变化 |
| `smpl.npy` | NumPy NPY | 动捕数据；实际为一个字典，包含 `betas`、`global_orient`、`body_pose`、`transl` |
| `pressure.npz` | NumPy NPZ | 触觉数据；键为 `pressure`，示例 shape 为 `(18036, 160, 120)`，可按帧与视频/动捕对齐，属于“触觉数据 + 对应时间顺序” |

备注：
- 该数据集是大量序列的集合，`Toy_Sequence/2024-07-29-09-58-38` 对应的是其中一个同结构样例。
- 本目录同样未见单独的时间戳文件，时间信息主要由序列目录名和帧对齐关系体现。


## 3. 帧序对齐关系

这里的 `MP4`、`smpl.npy`、`pressure.npz` 之间，**在数据语义上是有对应关系的**；但这种对应不是通过单独的时间戳文件显式保存，而是通过“**同一序列目录 + 相同帧序 index**”来隐式表达。

也就是说，可以理解为：

```text
同一序列目录 2024-07-29-09-58-38/
├── 视频帧 t=0,1,2,...        <- 来自 MP4
├── pressure[t, ...]          <- pressure.npz
└── smpl 的 pose/trans[t, ...] <- smpl.npy
```

结论：

- 这些数据**可以被当作已经清洗成统一时间轴上的干净数据**来使用。
- 但更准确地说，它们是“**已经预先对齐并按统一帧序保存**”，而不是“目录里额外提供了显式时间戳文件”。
- 因此这里的对齐方式应称为：**隐式帧序对齐**。

### 3.1 `Baselines/MotionPRO` 的实际对应关系

`Baselines/MotionPRO/lib/dataset/image_pressure.py` 的读取器并不会读取 `mp4` 时间戳，也不会做二次时序匹配；它直接假设同一目录下的各模态已经一一对齐：

```text
同一序列目录/
├── feature_hrnet.pth   <- 由视频帧提取出的视觉特征
├── pressure.npz        <- 触觉数据
├── smpl.npy            <- 动捕/SMPL 参数
├── keypoints.npy       <- 由 smpl.npy 再生出的 3D 关节点
└── contact.npy         <- 由 pressure.npz + smpl.npy 计算出的接触标签
```

读取器里的对应方式是：

- 用 `feature_hrnet.pth` 作为样本入口。
- 通过同目录替换文件名，自动找到：
  - `pressure.npz`
  - `smpl.npy`
  - `keypoints.npy`
  - `contact.npy`
- 之后对所有模态使用**同一个窗口切片**：
  - `pressure[window_left:window_right]`
  - `feature[window_left:window_right]`
  - `keypoints[window_left:window_right]`
  - `smpl['body_pose'][window_left:window_right]`
  - `smpl['global_orient'][window_left:window_right]`
  - `smpl['transl'][window_left:window_right]`

因此在 `MotionPRO` 中，真正的“帧序对齐关系”是：

```text
第 i 帧视觉特征
<-> 第 i 帧 pressure
<-> 第 i 帧 SMPL pose / orient / transl
<-> 第 i 帧 keypoints
<-> 第 i 帧 contact
```

换句话说，`MotionPRO` 假设：

- **同一序列目录中的所有模态长度已经基本一致或至少可按同一 index 截取。**
- **第 i 个元素就是同一时刻的多模态观测。**

### 3.2 视频在这个 baseline 里的位置

需要注意：`MotionPRO` 的训练读取器**不直接读取 `0.mp4~3.mp4`**。

它的实际视频 pipeline 是：

1. 原始视频先被拆成图像帧，放到 `color/` 目录。
2. `gen_bbox.py` 对 `color/*.jpg|png` 逐帧做人框检测，生成 `bbox.npy`。
3. `gen_image_feature.py` 读取 `color/` 和 `bbox.npy`，逐帧提取人体视觉特征，生成 `feature_hrnet.pth`。
4. 训练时数据集类读取的视觉模态不是原始 MP4，而是 `feature_hrnet.pth`。

所以对 `MotionPRO` 而言，最终参与对齐的视觉模态并不是“mp4 文件本身”，而是：

```text
MP4 -> color/*.jpg(or png) -> bbox.npy -> feature_hrnet.pth
```

### 3.3 动捕与触觉的派生流程

除视觉模态外，代码里还有两个派生步骤：

- `gen_kps.py`
  - 输入：`smpl.npy`
  - 输出：`keypoints.npy`
  - 含义：由 SMPL 参数恢复出逐帧 3D 关节点

- `gen_contact.py`
  - 输入：`pressure.npz` + `smpl.npy`
  - 输出：`contact.npy`
  - 含义：根据脚踝/脚部在地毯坐标中的位置，从压力图中计算逐帧接触标签

因此，`MotionPRO` 使用的数据不是最原始三件套本身，而是建立在以下默认前提上：

```text
原始/已对齐序列
├── 视频
├── pressure.npz
└── smpl.npy

再经过预处理得到
├── feature_hrnet.pth
├── keypoints.npy
└── contact.npy
```

### 3.4 一个更准确的表述

如果要严谨描述这批数据，建议写成：

- 这些数据**不是“显式时间戳对齐”**；
- 而是**“预处理后按统一帧序保存的隐式对齐多模态序列”**；
- `MotionPRO` baseline 在训练时**完全依赖这种 index-level 一一对应关系**。
