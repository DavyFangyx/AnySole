# MocapVideoAligner_0811 输出说明

`MocapVideoAligner_0811` 是对齐器。它读取采集后的相机数据和 BVH 动捕数据，在 GUI 中完成自动对齐、人工微调、裁剪标记和导出。

## 1. 当前实际会输出什么

这个项目主要有两类输出：

### 1.1 导出当前对齐结果

默认输出到：

```text
sync/output/<session_id>/
```

例如：

```text
sync/output/S301_1/
```

目录内当前通常会有：

```text
sync/output/S301_1/
├── S301_1_position_aligned.bvh
├── S301_1_order_aligned.bvh
├── S301_1_alignment.json
├── S301_1_aligned_curves.csv
└── S301_1_calibration.png
```

如果该试次有更多 BVH，比如 `skeleton2`、`skeleton3`，还会继续输出：

```text
S301_1_skeleton2_aligned.bvh
S301_1_skeleton3_aligned.bvh
```

### 1.2 导出裁剪记录

默认建议保存到：

```text
sync/output/results_csv/
```

例如：

```text
sync/output/results_csv/S3011.csv
```

这是一个单独的 CSV，用于登记这次手工记录的起点和终点，不是完整的对齐结果包。

## 2. 各输出文件说明

## 2.1 `*_aligned.bvh`

类型：BVH 动捕文件

内容：

- 这是“按当前偏移量 `delta_t` 裁切后”的 BVH。
- 不会重采样，也不会改写骨架层级。
- 本质上是从原始 BVH 的某个起始帧开始，截取到文件末尾。

文件名中的角色名可能包括：

| 文件名片段 | 含义 |
| --- | --- |
| `position` | 通常对应 `Skeleton0` 或 position 类 BVH |
| `order` | 通常对应 `Skeleton1` 或 motion/order 类 BVH |
| `skeleton2`、`skeleton3` | 额外 BVH 文件 |

文件内部结构仍是标准 BVH：

```text
HIERARCHY
...
MOTION
Frames: N
Frame Time: ...
<frame 0 channel values>
<frame 1 channel values>
...
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `HIERARCHY` 段 | 骨架结构、关节层级、偏移、通道定义 |
| `Frames` | 导出后剩余帧数 |
| `Frame Time` | 单帧时长，沿用原 BVH |
| 每一帧的数字行 | 各通道在该帧的值，通道顺序沿用原 BVH |

也就是说，对齐器导出的 BVH 主要改了两件事：

- 起始帧被裁掉一部分。
- `Frames` 总数随之变化。

## 2.2 `*_alignment.json`

类型：JSON

作用：

- 记录这次导出时使用的对齐参数。
- 说明每路相机、每个 BVH 是怎样被对齐和裁切的。

当前主要字段：

| 字段 | 含义 |
| --- | --- |
| `session_id` | 会话编号，如 `S301_1` |
| `source_mode` | 视觉数据来源模式，`frames`、`mp4` 或 `none` |
| `delta_t` | 当前采用的对齐偏移量，单位秒 |
| `camera_start_frames` | 当 `delta_t < 0` 时，各相机需要跳过的起始帧 |
| `reference_visual_fps` | 用于统一时间轴的参考视觉帧率 |
| `axis_preset` | 骨架显示坐标预设，如 `zup` 或 `raw` |
| `bvh` | 各 BVH 导出信息 |
| `visual_cameras` | 参与加载的相机标签列表，如 `cam1`、`cam2` |
| `visual_camera_fps` | 每路相机实际 FPS |
| `visual_camera_frame_count` | 每路相机总帧数 |
| `display_bvh_role` | GUI 当前用于显示的 BVH 角色 |
| `alignment_bvh_role` | 当前用于对齐计算的 BVH 角色 |

其中 `bvh` 是一个对象，按角色名分组，例如：

```json
{
  "position": {
    "file": "S301_1_position_aligned.bvh",
    "start_frame": 23,
    "raw_fps": 120.0,
    "source_file": "Skeleton0.bvh"
  }
}
```

`bvh.<role>` 下各字段定义：

| 字段 | 含义 |
| --- | --- |
| `file` | 导出的对齐后 BVH 文件名 |
| `start_frame` | 从原始 BVH 第几帧开始导出 |
| `raw_fps` | 原始 BVH 帧率 |
| `source_file` | 原始 BVH 文件名 |

## 2.3 `*_aligned_curves.csv`

类型：CSV

作用：

- 保存导出时那条“已经对齐后的统一时间轴曲线”。
- 方便后续用 Excel、Python、MATLAB 重新画图或检查偏移。

当前表头形式：

```csv
time_s,cam1,cam2,cam3,cam4,bvh
```

但实际列数取决于当前加载了几路相机：

- 至少有 `time_s`
- 有哪些相机，就会有哪些 `camX`
- 有 BVH 时，最后会有 `bvh`

字段定义：

| 字段 | 含义 |
| --- | --- |
| `time_s` | 对齐后的统一参考时间轴，单位秒 |
| `cam1`/`cam2`/... | 对应相机的能量曲线值 |
| `bvh` | 当前用于对齐的 BVH 能量曲线值 |

补充说明：

- 这些值不是原始像素，也不是原始关节角度。
- 它们是对齐算法内部使用的“运动能量曲线”。
- 曲线已经被重采样到统一参考 FPS 上。

如果当前没有可写入的曲线数据，文件至少会有：

```csv
time_s
```

## 2.5 `results_csv/*.csv` 裁剪记录

类型：CSV

作用：

- 记录你在界面里点击“记录起点”“记录终点”时保存的两个标记。
- 用于人工整理试次边界。

当前表头由固定字段和动态相机路径字段组成：

```csv
标记,视觉时间(s),视觉帧,动捕时间(s),动捕帧,偏移量(s),cam1_路径,cam2_路径,...,mocap_avi路径,bvh_position路径,bvh_order路径,bvh_全部路径,会话编号,导出时间
```

文件里通常有 3 行：

1. 表头
2. `start`
3. `end`

字段定义：

| 字段 | 含义 |
| --- | --- |
| `标记` | `start` 或 `end` |
| `视觉时间(s)` | 当前标记点在视觉统一时间轴上的时间，单位秒 |
| `视觉帧` | 当前标记点对应的视觉帧号 |
| `动捕时间(s)` | 当前标记点对应的动捕时间，单位秒 |
| `动捕帧` | 当前标记点对应的动捕帧号 |
| `偏移量(s)` | 当前使用的 `delta_t`，单位秒 |
| `camX_路径` | 对应相机数据源路径 |
| `mocap_avi路径` | 若动捕目录下存在 `.avi`，记录第一个 `.avi` 路径 |
| `bvh_position路径` | position BVH 原始路径 |
| `bvh_order路径` | order BVH 原始路径 |
| `bvh_全部路径` | 所有已加载 BVH 路径，分号拼接 |
| `会话编号` | 试次编号 |
| `导出时间` | 导出这个 CSV 的时间 |

如果你只记录了起点没有记录终点，或反过来，也是允许导出的；空缺位置会留空。

## 3. `delta_t` 怎么理解

这是整个对齐导出的核心字段。

可以简单理解为：

- `delta_t > 0`：BVH 需要从更后面的帧开始，去追上视觉。
- `delta_t < 0`：视觉需要从更后面的帧开始，去追上 BVH。

因此：

- 在 `alignment.json` 里，它决定 `camera_start_frames` 和各 BVH 的 `start_frame`。
- 在 `*_aligned.bvh` 里，它最终体现为“从原始 BVH 的哪一帧开始导出”。
- 在 `*_aligned_curves.csv` 里，它体现为多条曲线已经被平移到同一时间轴后再保存。

## 4. 和采集器输出的关系

这个项目本身不采集原始数据，它处理的是采集器产物和动捕文件：

- 视觉侧可读取采集器输出的逐帧目录 `1/2/3/4`，也可读取 `1_*.mp4` 到 `4_*.mp4`
- 动捕侧读取 `.bvh`
- 然后再导出新的“对齐结果包”

所以它的输出不是“原始数据”，而是：

- 对齐后的 BVH
- 对齐元数据
- 对齐曲线
- 曲线截图
- 人工裁剪记录

