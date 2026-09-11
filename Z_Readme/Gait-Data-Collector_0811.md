# Gait-Data-Collector_0811 输出说明

`Gait-Data-Collector_0811` 是采集器。它负责把四路相机数据和双脚压力传感器数据保存成一个“会话目录”。

## 1. 当前实际会输出什么

### 1.1 录制模式：相机 + 压力

每次录制会创建一个目录：

```text
<save_dir>/recYYYYMMDD_HHMMSS_<subject>_<counter>/
```

例如：

```text
D:\Data\rec20260812_153000_张三_1/
```

目录内当前实际会输出：

```text
rec.../
├── 1_YYYYMMDD_HHMMSS.mp4
├── 2_YYYYMMDD_HHMMSS.mp4
├── 3_YYYYMMDD_HHMMSS.mp4
├── 4_YYYYMMDD_HHMMSS.mp4
├── 1/
│   ├── 1_HHMMSS.mmm.jpg
│   └── ...
├── 2/
├── 3/
├── 4/
├── pressure_left.csv
├── pressure_right.csv
└── meta.json
```

说明：

- `1/2/3/4` 是四个机位的输出编号，不是随机数字。
- `1_*.mp4` 到 `4_*.mp4` 是对应机位的视频。
- `1/2/3/4/` 目录里是同一次录制导出的逐帧图像。
- `pressure_left.csv`、`pressure_right.csv` 是左右脚压力数据。
- `meta.json` 是本次录制的说明文件。


## 2. 各输出文件说明

## 2.1 `*.mp4`

类型：MP4 视频文件

内容：

- 单机位录像。
- 文件名中的前缀 `1/2/3/4` 对应机位编号。
- 视频由缓存中的逐帧图像编码得到。

补充：

- 当前默认码率来自 `movie_rate_kbps`。
- 当前帧率来自两种来源：
- 开启软件触发时，使用 `software_trigger_fps`。
- 未开启软件触发时，根据实际采集时间估算 FPS。

## 2.2 `1/2/3/4/` 里的逐帧图像

类型：图像文件，扩展名由 `frame_type` 决定，默认常见为 `jpg`

内容：

- 与该机位 MP4 对应的逐帧导出结果。
- 每张图是一帧原始采集画面。

文件名格式：

```text
<output_id>_HHMMSS.mmm.<frame_type>
```

例如：

```text
2_153000.127.jpg
```

这里的时间字符串来自相机帧的 `captured_at`，是 PC 在接收该帧时记录的本地时钟时间。

## 2.3 `pressure_left.csv` / `pressure_right.csv`

类型：CSV

内容：

- 一行代表一帧压力数据。
- 左右脚分别单独存一个文件。
- 当前表头固定为：

```csv
frame_idx,t_us,1,2,3,...,48
```

字段定义：

| 字段 | 含义 |
| --- | --- |
| `frame_idx` | 该脚在本次录制文件内的本地帧序号，从 `0` 开始 |
| `t_us` | 相对录制起点的时间，单位微秒；统计脚本会按 `t_us` 和视频帧率换算成时间槽 |
| `1` ~ `48` | 48 个阵点的原始压力值，类型可视为 `uint16` 整数 |

### `t_us` 的定义与生成 pipeline

当前代码里的 `t_us` 不是传感器硬件自带时间戳，而是 PC 侧单调时钟时间差：

1. 串口线程读到一批字节后，立即记录 `arrival_monotonic_us = time.monotonic_ns() // 1000`。
2. 这批字节被切成完整的 99 字节压力帧。
3. 每帧解析后，把这个 `arrival_monotonic_us` 写进 `PressureFrame.monotonic_us`。
4. 开始录制时，再记录一次本次会话起点 `epoch_monotonic_us = time.monotonic_ns() // 1000`。
5. 保存 CSV 时，计算：

```text
t_us = frame.monotonic_us - session.epoch_monotonic_us
```

所以：

- `t_us` 是“该压力帧到达 PC 时刻”相对“开始录制时刻”的微秒差。
- 它是相对时间，不是 Unix 时间戳。
- 它使用单调时钟，适合做同一台机器内的相对对齐。

### `1` 到 `48` 阵点是什么意思

当前协议每帧有 48 个测量点，每点 2 字节，共 96 字节数据区。

保存前，程序会先做一次 `4 x 12` 行顺序翻转，把硬件原始输出顺序改成更接近物理鞋垫布局的顺序，然后按一维写入 CSV。

因此：

- `1` 到 `48` 表示保存后的 48 个有效阵点编号。
- 它们按 `4 x 12` 的网格展开写入。
- 可理解为：

```text
1  - 12
13 - 24
25 - 36
37 - 48
```

- 每组 12 个点是一排。
- 真实空间位置请以：
- `foot_sensor_layout/leftfoot_dots.csv`
- `foot_sensor_layout/rightfoot_dots.csv`
为准。

这两个文件的字段定义：

| 字段 | 含义 |
| --- | --- |
| `dot_index` | 阵点编号，对应压力 CSV 中的 `1` 到 `48` |
| `norm_x` | 该点在鞋垫图中的归一化横坐标 |
| `norm_y` | 该点在鞋垫图中的归一化纵坐标 |

也就是说，如果你想知道压力 CSV 中第 `22` 列到底在脚底哪个位置，应去查对应脚的 `*_dots.csv` 里 `dot_index=22` 的那一行。

### 压力帧底层二进制格式

单帧长度 99 字节：

| 字节范围 | 含义 |
| --- | --- |
| `B1` | 帧头，固定 `0x40` |
| `B2` | 左右脚编号，左脚 `0x31`，右脚 `0x32` |
| `B3-B98` | 48 个阵点，每点 2 字节，大端 `uint16` |
| `B99` | 校验和，等于前 98 字节求和后取低 8 位 |

## 2.4 `meta.json`

类型：JSON

作用：

- 记录本次会话的整体说明。
- 说明这次录制有哪些相机、哪些压力文件、时间基准是什么。

### 相机 + 压力录制时的 `meta.json`

主要字段：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 元数据版本，当前为 `2` |
| `started_at_iso` | 本地开始录制时间，ISO 格式 |
| `epoch_wall_utc_iso` | 开始录制时的 UTC 时间，ISO 格式 |
| `epoch_monotonic_us` | 本次录制起点的单调时钟微秒值 |
| `subject` | 对象名 |
| `counter` | 第几次 |
| `software_trigger_enabled` | 是否使用软件触发采集 |
| `software_trigger_fps` | 软件触发目标帧率 |
| `movie_rate_kbps` | MP4 编码码率 |
| `timebase` | 时间基准的文字说明 |
| `cameras` | 相机输出列表 |
| `pressure` | 压力输出列表 |

`cameras` 数组中每个元素包含：

| 字段 | 含义 |
| --- | --- |
| `slot` | 内部机位名，如 `top/left/front/right` |
| `file` | 对应 MP4 文件名 |
| `frames_dir` | 对应逐帧图像目录名 |
| `output_id` | 对外输出编号，通常是 `1/2/3/4` |
| `device_serial` | 相机序列号 |
| `fps` | 保存时采用的帧率 |
| `frame_count` | 该机位导出的总帧数 |
| `resolution` | 分辨率，格式如 `1920x1080` |
| `exposure_time_us` | 曝光时间，单位微秒 |
| `gain_db` | 增益 |
| `gamma_value` | Gamma |
| `black_level` | 黑电平 |

`pressure` 数组中每个元素包含：

| 字段 | 含义 |
| --- | --- |
| `sensor_id` | 压力传感器 ID，如 `left_foot` |
| `file` | 对应 CSV 文件名 |
| `frame_count` | 该传感器导出的总帧数 |

### 仅触觉模式下的 `meta.json`

与上面类似，但更简单，主要多一个：

| 字段 | 含义 |
| --- | --- |
| `mode` | 固定为 `pressure_only` |

并且 `pressure` 数组里每项还会记录：

| 字段 | 含义 |
| --- | --- |
| `enabled` | 该传感器是否启用 |
| `port` | 串口号 |

## 3. 一个你给出的表头例子怎么理解

例如：

```csv
frame_idx,t_us,1,2,3,...,48
```

可以直接理解为：

- 第 1 列 `frame_idx`：第几帧脚压。
- 第 2 列 `t_us`：这帧脚压距离“开始录制”的微秒数。
- 第 3 到第 50 列：48 个阵点的原始值。

如果某一行是：

```csv
15,402133,120,98,...,305
```

表示：

- 这是该脚文件里的第 15 帧。
- 这帧出现在录制开始后 `402133 us`，也就是约 `0.402133 s`。
- 后面 48 个数是这 48 个阵点在这一时刻的原始压力值。

## 4. 一个容易混淆但需要说明的点

代码里存在 `_save_camera_times()`，设计上会生成类似：

```csv
frame_idx,t_us
```

的相机时间 CSV，但当前主保存流程并没有调用它。

因此，按“当前实际运行结果”来说：

- 压力时间 CSV 会产出。
- 相机 `*_times.csv` 当前不会自动产出。
