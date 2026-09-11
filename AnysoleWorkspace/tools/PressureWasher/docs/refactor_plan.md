# PressureWasher 重构方案

## 1. 目标

把当前“统计 -> 重建 -> fake 标记 -> 编码”的流程，整理成一个职责清晰、目录稳定、命名统一的流水线。

目标是：
- 代码按阶段分层，不再按单脚本堆叠
- 原始数据、处理中间产物、最终产物分离
- 每个阶段只有一个明确入口
- 任何输出都能从目录名直接看出它属于哪一步

## 2. 当前问题

- 脚本都放在根目录，职责边界不够明显
- 统计、重建、标记、编码的输出目录风格不统一
- 旧的结果目录曾混在一起，现在已迁移到 `outputs/` 和 `configs/`
- 目前依赖“记住脚本顺序”而不是“看目录就知道怎么跑”

## 3. 建议目录树

```text
PressureWasher/
├── README.md
├── docs/
│   ├── pipeline_overview.md
│   └── refactor_plan.md
├── configs/
│   └── default.yaml
├── src/
│   └── pressurewasher/
│       ├── __init__.py
│       ├── cli/
│       │   ├── inspect.py
│       │   ├── reconstruct.py
│       │   ├── mark_fake.py
│       │   └── encode.py
│       ├── stages/
│       │   ├── raw_inspection.py
│       │   ├── reconstruction.py
│       │   ├── fake_marking.py
│       │   └── encoding.py
│       ├── io/
│       │   ├── dataset_paths.py
│       │   ├── pressure_csv.py
│       │   └── manifest.py
│       └── utils/
│           ├── logging.py
│           └── time_axis.py
├── scripts/
│   ├── pressurewash-inspect
│   ├── pressurewash-reconstruct
│   ├── pressurewash-mark-fake
│   └── pressurewash-encode
└── outputs/
    ├── pressure_stats/
    ├── reconstructed/
    ├── fake_marked/
    └── encoded/
```

## 4. 阶段职责

### 4.1 `inspect`

输入：原始压力 CSV 和原始 session 目录。  
输出：每个 session 的质量统计、缺口统计、每秒帧数图表。

保留内容：
- `frame_per_second.csv/png`
- `gap_list.csv`
- `gap_positions.png`
- `t_us_diff_histogram.csv/png`

### 4.2 `reconstruct`

输入：原始 session。  
输出：重建后的 `pressure_left.csv`、`pressure_right.csv`、`reconstruction_manifest.csv`。

保留内容：
- `valid_mask`
- `source_frame_idx`
- `source_t_us`
- `manifest`

### 4.3 `mark_fake`

输入：重建结果 + fake frame 配置。  
输出：在重建 CSV 中加入 `fake` 列，并移除源字段。

### 4.4 `encode`

输入：fake-marked CSV。  
输出：每个 side 的 `npy mask` 或训练所需编码文件。

## 5. 命名规则

- 原始数据：`raw`
- 检查输出：`stats`
- 重建输出：`reconstructed`
- fake 标记输出：`fake_marked`
- 编码输出：`encoded`

建议统一成：

```text
<stage>/<date>/<subject>/<session>/
```

## 6. 迁移原则

1. 先不改算法，只改组织结构
2. 先把公共逻辑抽到 `src/pressurewasher/`
3. 再把现有根目录脚本改成薄封装
4. 最后再考虑合并 CLI、补配置、补测试

## 7. 推荐落地顺序

1. 建 `src/pressurewasher/io/`，统一 session 发现、路径拼接、CSV 读写
2. 建 `src/pressurewasher/stages/`，把四个阶段的核心逻辑移动进去
3. 建 `src/pressurewasher/cli/`，每个阶段一个入口
4. 保留旧脚本作为兼容包装
5. 统一输出目录和 README
6. 清理/忽略大体积生成目录

## 8. 预期结果

- 读目录就能看懂流程
- 读代码就能找到对应阶段
- 输出结构稳定，后续扩展新阶段更容易
- `PressureWasher` 会从“脚本集合”变成“可维护流水线”
