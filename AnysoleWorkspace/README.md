# AnysoleWorkspace

## 文件结构

```text
AnysoleWorkspace/
├── script/          # 工作区和依赖准备脚本
├── tools/           # 仍在使用的工具集
├── sources/         # 原始数据软链接
├── dependencies/    # 模型和第三方依赖
├── derived/         # 生成的训练数据和特征
├── calibration/     # 标定文件
├── splits/          # 数据划分
└── workspace.yaml   # 工作区路径记录
```

## 准备脚本

### 构建工作区

```bash
python AnysoleWorkspace/script/build_workspace.py init
```

参数：`init` 初始化，`migrate` 迁移旧目录，`relink` 修复链接，`doctor` 检查工作区；可加 `--dry-run` 预览操作。

### 生成 HRNet 特征

```bash
conda activate touch_gait
python AnysoleWorkspace/script/generate_hrnet_cache.py --cam-id 3
```

参数：`--cam-id` 相机编号；`--session` 指定 session；`--batch-size` batch 大小；`--device` 使用 `cuda` 或 `cpu`；`--overwrite` 覆盖已有结果；`--limit-sessions` 限制 session 数量。

### 准备压力数据

```bash
python AnysoleWorkspace/script/prepare_pressure_data.py inspect
python AnysoleWorkspace/script/prepare_pressure_data.py reconstruct
python AnysoleWorkspace/script/prepare_pressure_data.py mark-fake
python AnysoleWorkspace/script/prepare_pressure_data.py encode
```

参数：`inspect` 统计原始数据；`reconstruct` 重建统一时间轴；`mark-fake` 标记补帧；`encode` 生成 fake mask。`mark-fake` 和 `encode` 默认读取上一步最新输出，也可用 `--input PATH` 指定输入；`--overwrite` 覆盖已有输出。实际处理脚本位于 `AnysoleWorkspace/tools/PressureWasher/`。

推荐顺序：`inspect → reconstruct → mark-fake → encode`。

## AnySole 运行

python AnysoleWorkspace/script/generate_hrnet_cache.py \
    --cam-id 3 \
    --device cuda

```bash
cd AnySole
PYTHONPATH=. python -m anysole.train --config configs/v1.yaml
PYTHONPATH=. python -m anysole.eval --config configs/v1.yaml --ckpt outputs/v1/ckpt_last.pt
PYTHONPATH=. python -m anysole.infer --ckpt outputs/v1/ckpt_last.pt --session S12021 --config-id 0
```

HRNet 特征必须提前生成；训练和推理不会现场提取。视觉 cache 的 shape 为 `(T, 2051)`：HRNet `(2048)` 加 CLIFF 归一化 `bbox_info` `(3)`。
