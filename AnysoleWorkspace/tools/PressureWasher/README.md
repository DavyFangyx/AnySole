# PressureWasher

## 1. Raw Data

当前原始数据默认根目录：
`/data/lizhe/projects/Tactile/1_Data`

当前数据按“日期 -> 被试组 -> session”组织，统计脚本和清洗脚本都按这个结构递归扫描：

```text
/data/lizhe/projects/Tactile/1_Data/
├── 20260419/
│   ├── S1/
│   │   ├── rec20260419_162851_万山_S101_1/
│   │   │   ├── Pressure_left_20260419_162851.csv
│   │   │   ├── Pressure_right_20260419_162851.csv
│   │   │   ├── 1/                # 参考相机 JPG
│   │   │   ├── 2/
│   │   │   ├── 3/
│   │   │   ├── 4/
│   │   │   └── *.mp4
│   │   └── ...
│   ├── mocap_csv/
│   └── ...
├── 20260422/
│   ├── S2/
│   ├── S3/
│   ├── S4/
│   └── ...
├── 20260804/
│   ├── S5/
│   ├── S6/
│   └── S7/
├── 20260808/
│   ├── S9/
│   ├── S10/
│   ├── S11/
│   └── S12/
└── 20260810/
    ├── S13/
    └── S14/
```

对 AI 或脚本来说，真正需要关注的是每个 `rec...` session 目录里的内容：

```text
recYYYYMMDD_HHMMSS_<name>_Sxyz_n/
├── Pressure_left_*.csv
├── Pressure_right_*.csv
├── 1/                  # 用于时间参考的 JPG 序列
├── 2/
├── 3/
├── 4/
└── *.mp4
```

其中压力 CSV 当前表头格式是：

```csv
frame_idx,t_us,1,2,3,...,48
```

## 2. Project Layout

脚本位于 `AnysoleWorkspace/tools/PressureWasher/`；运行产物位于
`AnysoleWorkspace/sources/PressureWasher/outputs/`，不会写回工具目录。

```text
PressureWasher/
├── analyze_pressure_csv_stats.py
├── reconstruct_pressure_dataset.py
├── mark_fake_frames_in_reconstruction.py
├── encode_fake_marked_reconstruction.py
├── configs/
│   └── fake_frames/
├── docs/
│   ├── pipeline_overview.md
│   ├── refactor_plan.md
│   └── reconstruction_format.md
├── outputs/
│   ├── stats/
│   ├── reconstructed/
│   ├── fake_marked/
│   └── encoded/
└── pwlib/
```

## 3. Pipeline Stages

### 3.1 `inspect`

脚本：
[analyze_pressure_csv_stats.py](analyze_pressure_csv_stats.py)

输入：原始 `Pressure_left/right.csv`。  
输出：按会话组织的质量统计、每秒帧数、时间间隔直方图和 gap 分析。

运行示例：

```bash
/data/fangyuxuan/miniconda3/envs/trident/bin/python PressureWasher/analyze_pressure_csv_stats.py
```

### 3.2 `reconstruct`

脚本：
[reconstruct_pressure_dataset.py](reconstruct_pressure_dataset.py)

输入：原始 session。  
输出：统一时间轴后的 `pressure_left.csv`、`pressure_right.csv`、`reconstruction_manifest.csv`。

要点：
- 小缺口补齐后仍记为 `valid_mask=1`
- 时间断点的 bridge 补帧记为 `valid_mask=0`
- `-40hz` 可再重采样到固定 40Hz
- `--snap-contact-transition-interpolation` 可在接触状态切换时改用最近帧

运行示例：

```bash
/data/fangyuxuan/miniconda3/envs/trident/bin/python PressureWasher/reconstruct_pressure_dataset.py
/data/fangyuxuan/miniconda3/envs/trident/bin/python PressureWasher/reconstruct_pressure_dataset.py -40hz
/data/fangyuxuan/miniconda3/envs/trident/bin/python PressureWasher/reconstruct_pressure_dataset.py -40hz --snap-contact-transition-interpolation
```

### 3.3 `mark_fake`

脚本：
[mark_fake_frames_in_reconstruction.py](mark_fake_frames_in_reconstruction.py)

输入：`reconstruct` 的输出。  
输出：在每个压力 CSV 中加入 `fake` 列，并移除 `source_frame_idx/source_t_us`。

运行示例：

```bash
/data/fangyuxuan/miniconda3/envs/trident/bin/python PressureWasher/mark_fake_frames_in_reconstruction.py -input AnysoleWorkspace/sources/PressureWasher/outputs/reconstructed/reconstruction_20260817_161459
```

### 3.4 `encode`

脚本：
[encode_fake_marked_reconstruction.py](encode_fake_marked_reconstruction.py)

输入：`mark_fake` 的输出。  
输出：每个 side 对应的 `fake_mask_*.npy`。

运行示例：

```bash
/data/fangyuxuan/miniconda3/envs/trident/bin/python PressureWasher/encode_fake_marked_reconstruction.py -input AnysoleWorkspace/sources/PressureWasher/outputs/fake_marked/reconstruction_20260817_161459_fake_marked
```

## 4. Output Layout

### 4.1 Statistics

```text
AnysoleWorkspace/sources/PressureWasher/outputs/stats/pressure_stats_<timestamp>/
├── overall/
│   ├── frame_per_second.csv
│   └── frame_per_second.png
└── <date>/<Si>/<rec...>/
    ├── frame_per_second.csv
    ├── frame_per_second.png
    ├── gap_list.csv
    ├── gap_positions.png
    ├── t_us_diff_histogram.csv
    └── t_us_diff_histogram.png
```

### 4.2 Reconstruction

```text
AnysoleWorkspace/sources/PressureWasher/outputs/reconstructed/reconstruction_<timestamp>/
└── <date>/<Si>/<rec...>/
    ├── pressure_left.csv
    ├── pressure_right.csv
    └── reconstruction_manifest.csv
```

### 4.3 Fake Marking / Encoding

```text
AnysoleWorkspace/sources/PressureWasher/outputs/fake_marked/reconstruction_<timestamp>_fake_marked/
└── <date>/<Si>/<rec...>/
    ├── pressure_left.csv
    ├── pressure_right.csv
    └── reconstruction_manifest.csv

AnysoleWorkspace/sources/PressureWasher/outputs/encoded/reconstruction_<timestamp>_fake_marked_encoded/
└── <date>/<Si>/<rec...>/
    ├── fake_mask_left.npy
    ├── fake_mask_right.npy
    └── reconstruction_manifest.csv
```

## 5. Recommended Order

1. 先跑 `inspect`，确认原始 CSV 没有明显结构异常
2. 再跑 `reconstruct`，生成统一时间轴数据
3. 然后跑 `mark_fake`，补齐 fake 标记
4. 最后跑 `encode`，输出训练所需 mask

如果只是排查原始数据质量，直接运行 `inspect` 即可。
