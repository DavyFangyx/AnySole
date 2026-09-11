# Reconstructed Tactile Dataset Format

本文说明 `PressureWasher/reconstructed_tactile_dataset/` 下重建后的触觉数据组织方式。

说明范围：
- 不同重建参数下，文件组织方式相同。
- `-40hz`、`--snap-contact-transition-interpolation` 只影响文件内容，不影响目录结构。

## 1. 总体目录

```text
PressureWasher/reconstructed_tactile_dataset/
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
- 左右脚各一个连续 CSV。
- 时间断点处的补帧仍按同样的重建机制生成。
- 所有重建帧都在 CSV 的 `valid_mask=0` 中标记。

## 3. 触觉 CSV 字段

```csv
frame_idx,t_us,valid_mask,source_frame_idx,source_t_us,1,2,3,...,48
```

字段含义：
- `frame_idx`：会话内连续帧编号，从 `0` 开始。
- `t_us`：重建时间轴，单位微秒。
- `valid_mask`：`1` 为原始有效帧，`0` 为补帧/插值/重采样帧。
- `source_frame_idx` / `source_t_us`：仅对 `valid_mask=1` 有值。
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
- `contact_transition_mode`：接触/不接触跨界时，是否采用 `snap`。

## 5. 读取方式

推荐顺序：

1. 先读 `reconstruction_manifest.csv`
2. 再读 `pressure_left.csv` 和 `pressure_right.csv`
3. 用 `valid_mask` 判断哪些行是重建帧
4. 用 manifest 里的 `segment` / `bridge` 行定位分段和补帧区间

## 6. 要点

- 现在不再输出 `pressure_left_t*.csv` / `pressure_right_t*.csv`。
- 所有连续性由单个左右文件承载。
- 断点补帧只是在 `valid_mask=0` 的行里体现。
- manifest 负责说明这些补帧属于哪个 `segment` 或 `bridge` 区间。
