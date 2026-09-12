# Reconstructed Tactile Dataset Format

本文说明 `AnysoleWorkspace/sources/PressureWasher/outputs/reconstructed/` 下重建后的触觉数据组织方式。

说明范围：
- 不同重建参数下，文件组织方式相同。
- `-40hz`、`--snap-contact-transition-interpolation` 只影响文件内容，不影响目录结构。

`--snap-contact-transition-interpolation` 的作用：
- 仅影响插值点的数值生成方式。
- 当插值点左右两侧分别是“接触”和“不接触”时，不做线性插值，直接取最近一帧。
- 不影响原始有效帧，也不影响文件命名和目录组织。

## 1. 总体目录

```text
AnysoleWorkspace/sources/PressureWasher/outputs/reconstructed/
└── reconstruction_<timestamp>/
    ├── <date>/
    │   ├── <Sx>/
    │   │   ├── <rec...>/
    │   │   │   ├── pressure_left.csv
    │   │   │   ├── pressure_right.csv
    │   │   │   └── reconstruction_manifest.csv
    │   │   └── ...
    │   └── ...
    └── ...
```

层级含义：
- `reconstruction_<timestamp>`：一次重建任务的输出。
- `<date>`：采集日期，如 `20260810`。
- `<Sx>`：被试组，如 `S14`。
- `<rec...>`：单次原始会话目录。

## 2. 会话级输出

每个 `rec...` 目录只输出 3 个文件：

- `pressure_left.csv`
- `pressure_right.csv`
- `reconstruction_manifest.csv`

其中：
- 左右脚各一个连续 CSV，不再按 `t0/t1/t2` 拆分。
- 时间断点处的补帧仍按同样的重建机制生成。
- 跨长时间缺失区间的 bridge 补帧在 CSV 的 `valid_mask=0` 中标记。

## 3. 触觉 CSV 字段

```csv
frame_idx,t_us,valid_mask,source_frame_idx,source_t_us,1,2,3,...,48
```

字段含义：
- `frame_idx`：会话内连续帧编号，从 `0` 开始。
- `t_us`：重建时间轴，单位微秒。
- `valid_mask`：`1` 表示正常连续片段时间轴上的帧，包含原始帧、片段内部的小缺口补帧和正常 40Hz 重采样帧；`0` 表示跨长缺失区间的 bridge 帧。
- `source_frame_idx` / `source_t_us`：仅原始帧有值；正常补帧和 bridge 帧为空。
- `1..48`：48 个压力通道。

## 4. `reconstruction_manifest.csv`

manifest 不存压力值，只记录重建结构。

字段：

```csv
block_type,side,name,prev_name,next_name,frame_idx_start,frame_idx_end,t_us_start,t_us_end,source_rows,inserted_rows,dt_hat_us,source_t_us_start,source_t_us_end,contact_transition_mode
```

说明：
- `block_type=segment`：原始数据切出的连续段。
- `block_type=bridge`：段与段之间补出来的重建桥接帧。
- `block_type=resample`：`-40hz` / `--resample-40hz` 触发的整段重采样记录。
- `side`：`left` 或 `right`。
- `name`：如 `t0`、`t1`、`t0->t1`。
- `inserted_rows`：该块里插入的重建帧数。
- `contact_transition_mode`：`interpolate` 或 `snap`，表示桥接补帧时是否启用最近帧拷贝。

## 5. 读取方式

推荐顺序：

1. 先读 `reconstruction_manifest.csv`
2. 再读 `pressure_left.csv` 和 `pressure_right.csv`
3. 用 `valid_mask` 判断哪些行是重建帧
4. 用 manifest 里的 `segment` / `bridge` 行定位分段和补帧区间

## 6. 要点

- 现在不再输出 `pressure_left_t*.csv` / `pressure_right_t*.csv`。
- 所有连续性由单个左右文件承载。
- 跨长断点的 bridge 补帧只在 `valid_mask=0` 的行里体现。
- manifest 负责说明这些补帧属于哪个 `segment` 或 `bridge` 区间。
