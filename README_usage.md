# AnySole 使用说明

AnySole 使用统一工作区 `AnysoleWorkspace/`，输入为 40 Hz 对齐后的 RGB、压力和 Skeleton3 BVH 数据。

## 1. 环境

```bash
conda activate touch_gait
cd /data/fangyuxuan/projects/gait
```

检查工作区：

```bash
python AnysoleWorkspace/script/build_workspace.py doctor
```

## 2. 准备数据

### 2.1 准备压力数据

如果 `PressureWasher/outputs/fake_marked/` 已存在，可跳过此步骤。

```bash
python AnysoleWorkspace/script/prepare_pressure_data.py inspect
python AnysoleWorkspace/script/prepare_pressure_data.py reconstruct
python AnysoleWorkspace/script/prepare_pressure_data.py mark-fake
python AnysoleWorkspace/script/prepare_pressure_data.py encode
```

推荐顺序：`inspect → reconstruct → mark-fake → encode`。

### 2.2 生成 MotionPRO 序列

```bash
cd Baselines/MotionPRO
python data_prep/prepare_sequences.py --cam-id 3 --overwrite
python data_prep/make_splits.py --cam-id 3
```

生成 bbox、关键点和 MotionPRO 特征：

```bash
conda activate bbox_scan
python -m lib.util.gen_bbox --cam-id 3

conda activate touch_gait
python -m lib.util.gen_image_feature --cam-id 3
python -m lib.util.gen_kps --cam-id 3
```

输出目录：

```text
AnysoleWorkspace/derived/MotionPRO/sequences/cam3/<date>/<subject>/<session>/
```

每个 session 至少应包含：

```text
color/
align_meta.json
bbox.npy
pressure.npz
smpl.npy
keypoints.npy
contact.npy
fake_mask.npy
```

## 3. 生成 AnySole 视觉 cache

AnySole 不直接使用 MotionPRO 的 `feature_hrnet.pth`，需要使用 AnySole 的 crop + CLIFF-HRNet 流程重新生成。

视觉特征格式为：

```text
V_feat = HRNet 2048 维 + bbox_info 3 维 = 2051 维
bbox_info = [cx, cy, b]
```

生成全部 session：

```bash
cd /data/fangyuxuan/projects/gait
conda activate touch_gait

CUDA_VISIBLE_DEVICES=4 \
python AnysoleWorkspace/script/generate_hrnet_cache.py \
  --cam-id 3 \
  --device cuda \
  --batch-size 32
```

只生成一个 session：

```bash
python AnysoleWorkspace/script/generate_hrnet_cache.py \
  --cam-id 3 \
  --session S12021 \
  --device cuda
```

输出：

```text
AnysoleWorkspace/derived/AnySole/hrnet_cache/cam3/<session>.pt
```

例如 `S12021.pt` 的 shape 应为 `(339, 2051)`。

## 4. 训练

```bash
cd AnySole
PYTHONPATH=. python -m anysole.train \
  --config configs/v1.yaml \
  --device cuda
```

训练输出：

```text
AnySole/outputs/v1/ckpt_last.pt
```

常用参数：

```bash
PYTHONPATH=. python -m anysole.train \
  --config configs/v1.yaml \
  --epochs 1 \
  --batch-size 32 \
  --limit-sessions 2 \
  --device cuda
```

训练会随机使用三种条件配置：

| 配置 | 输入 |
|---|---|
| `VT` | RGB + 压力 |
| `V` | RGB |
| `T` | 压力 |

## 5. 测试与评估

使用验证集评估三种配置：

```bash
PYTHONPATH=. python -m anysole.eval \
  --config configs/v1.yaml \
  --ckpt outputs/v1/ckpt_last.pt \
  --device cuda
```

输出指标包括：

```text
MPJPE
PA-MPJPE
MPJRE
traj_ATE
contact_acc
```

同时导出验证集 BVH：

```bash
PYTHONPATH=. python -m anysole.eval \
  --config configs/v1.yaml \
  --ckpt outputs/v1/ckpt_last.pt \
  --write-bvh outputs/v1/eval_bvh \
  --device cuda
```

## 6. 单 session 推理

```bash
PYTHONPATH=. python -m anysole.infer \
  --ckpt outputs/v1/ckpt_last.pt \
  --session S12021 \
  --config configs/v1.yaml \
  --config-id 0 \
  --device cuda
```

`config-id`：

```text
0 = VT
1 = V-only
2 = T-only
```

默认输出：

```text
AnySole/outputs/v1/S12021_VT.bvh
```

也可以指定输出路径：

```bash
PYTHONPATH=. python -m anysole.infer \
  --ckpt outputs/v1/ckpt_last.pt \
  --session S12021 \
  --config configs/v1.yaml \
  --config-id 0 \
  --out outputs/v1/S12021_VT.bvh \
  --device cuda
```

## 7. 常见问题

- `Missing AnySole HRNet+bbox cache`：先运行第 3 节的 cache 生成命令。
- `Missing CLIFF checkpoint`：确认权重位于 `AnysoleWorkspace/dependencies/MotionPRO/cliff_ckpt/`。
- CUDA 不可用：检查 `CUDA_VISIBLE_DEVICES`、`nvidia-smi` 和当前 conda 环境中的 PyTorch CUDA 版本。
- BVH 路径失效：AnySole 会优先读取 `align_meta.json`，失败后从 `workspace://sources/raw/<date>/mocap_ori_bvh/<session>/` 查找。
- cache 已存在时默认跳过；需要重算时添加 `--overwrite`。
