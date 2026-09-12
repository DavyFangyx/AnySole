# AnysoleWorkspace

工作区路径和数据说明。

## 目录索引

`sources/PressureWasher` 只保存触觉清洗、审核及运行产物；`tools/PressureWasher` 仅保存脚本、配置和文档。

```text
AnysoleWorkspace/
├── sources/raw/                         # 原始触觉 + 视频 + 采集文件
├── sources/PressureWasher/              # PressureWasher 数据
│   └── outputs/                         # stats/reconstructed/fake_marked/encoded
├── sources/published/                   # 发布数据
├── sources/calibration_artifacts/       # 标定原始文件
├── tools/PressureWasher/                # 工具代码、配置和文档
├── derived/MotionPRO/sequences/cam3/    # MotionPRO 序列
├── derived/AnySole/hrnet_cache/cam3/    # HRNet + bbox 特征
├── derived/Step2Motion/gait/            # Step2Motion 数据
├── dependencies/                        # 模型权重和依赖
├── calibration/                         # 标定摘要
└── splits/default/                      # 数据划分
```

`sources/raw` 是原始采集数据软链接，包含触觉、视频和原始姿态。按记录时间展开如下：

```text
sources/raw/
└── {记录日期序列}20260808/
    ├── mocap_ori_bvh/   # 原始 BVH 姿态
    ├── mocap_ori_c3d/   # 原始 C3D
    ├── mocap_ori_cmr/   # 原始 CMR
    ├── mocap_ori_trc/   # 原始 TRC
    ├── S9/              # 记录者/被试组
    └── S11/             # 记录者/被试组
        └── rec.../      # 触觉 CSV、相机 JPG、视频
```
`sources/raw` 当前指向 `/data/lizhe/projects/Tactile/1_Data`，不复制原始数据。

## 触觉数据

```text
sources/PressureWasher/outputs/fake_marked/<run>/<date>/<subject>/<rec...>/
├── pressure_left.csv
├── pressure_right.csv
└── reconstruction_manifest.csv
```

CSV 的列 `1` 到 `48` 是单脚压力通道，每行是一帧；双脚单帧为 `(2,48)`，AnySole 拼接成 `T_raw` `(T,96)`。`encode` 只生成 `fake_mask_left/right.npy`，不保存压力值。`derived/MotionPRO/.../pressure.npz` 是 `(T,160,120)` 栅格图。

## 流程

```bash
python AnysoleWorkspace/script/prepare_pressure_data.py inspect
python AnysoleWorkspace/script/prepare_pressure_data.py reconstruct
python AnysoleWorkspace/script/prepare_pressure_data.py mark-fake
python AnysoleWorkspace/script/prepare_pressure_data.py encode
```

输出依次位于 `sources/PressureWasher/outputs/stats`、`reconstructed`、`fake_marked`、`encoded`；可用 `--input PATH` 指定输入、`--overwrite` 覆盖输出。

## 检查

```bash
python AnysoleWorkspace/script/build_workspace.py init
python AnysoleWorkspace/script/build_workspace.py doctor
python AnysoleWorkspace/script/build_workspace.py relink
```
