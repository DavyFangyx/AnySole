# 统一使用说明

本项目以 `AnysoleWorkspace/` 为统一数据工作区，固定使用 cam3、40 Hz、同一份
session split 和 Skeleton3 23 关节协议。主模型为 AnySole，基线模型为
MotionPRO、Step2Motion 和 pressure toolkit。

## 1. 环境与路径

```bash
cd /data/fangyuxuan/projects/gait
conda activate touch_gait
python AnysoleWorkspace/script/build_workspace.py doctor
```

默认路径：

```text
数据与依赖：AnysoleWorkspace/
主模型输出：outputs/
基线模型输出：results/、resultsdisplay/
```

可用 `ANYSOLE_WORKSPACE`、`ANYSOLE_RESULTS` 和 `ANYSOLE_RESULTSDISPLAY` 覆盖这些
根目录。

## 2. 统一数据准备

先选择一张 GPU，后续所有 GPU 密集型图像步骤统一使用这个变量。`CUDA_VISIBLE_DEVICES`
是进程级 GPU 宏观开关；变量值 `0` 表示物理 GPU 0，`4` 表示物理 GPU 4。

```bash
export ANYSOLE_GPU=4
export CUDA_VISIBLE_DEVICES="$ANYSOLE_GPU"
```

并非本节所有步骤都需要 GPU：压力清洗、序列整理、split、Step2Motion 转换和
normalizer 主要使用 CPU；bbox、MotionPRO 图像特征、AnySole HRNet cache 和
Depth Pro 深度图是 GPU 密集型步骤。`gen_kps` 当前实现固定使用 CPU。

本节所有会写入准备数据的 CLI 都支持一组互斥的通用参数：

- `--skip-existing`：已有目标产物就跳过，适合首次生成和中断后续跑。
- `--force`：忽略已有产物，全部重新计算并覆盖。

两者不能同时使用。下面以安全的 `--skip-existing` 为默认示例；确认需要重算时，将其替换为 `--force`。

### 2.1 压力数据（所有模型共用）

如果 `PressureWasher/outputs/fake_marked/` 已准备好，可跳过：

```bash
python AnysoleWorkspace/script/prepare_pressure_data.py inspect --skip-existing
python AnysoleWorkspace/script/prepare_pressure_data.py reconstruct --skip-existing
python AnysoleWorkspace/script/prepare_pressure_data.py mark-fake --skip-existing
python AnysoleWorkspace/script/prepare_pressure_data.py encode --skip-existing
```

执行顺序为 `inspect → reconstruct → mark-fake → encode`。

### 2.2 MotionPRO 序列、特征与 AnySole cache

MotionPRO 序列是 AnySole、MotionPRO 和 Step2Motion 的共同输入。命令在
`Baselines/MotionPRO/` 下执行：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
python data_prep/prepare_sequences.py --cam-id 3 --skip-existing
# 序列生成后立即冻结所有模型共用的 train/val/test 划分
python data_prep/make_splits.py --skip-existing
```

参数说明：
- `prepare_sequences.py --cam-id 3`：生成统一的 cam3 序列；相机选择在数据转换阶段完成。
- `make_splits.py`：读取已生成的统一 cam3 序列，不需要再次传入相机编号。
- `--ood S11,S10`：将指定 subject 整体放入 OOD 的 val/test。
- `--exclude S5,S6`：从当前 split 中排除指定 subject。

默认只生成一份中央 split：
`AnysoleWorkspace/splits/default/splits.csv`。完成 split 后，再运行下面的
bbox、图像特征、关键点和 AnySole cache 生成命令；后续模型不得自行重新划分数据。

生成 MotionPRO 输入：

```bash
conda activate bbox_scan
CUDA_VISIBLE_DEVICES="$ANYSOLE_GPU" python -m lib.util.gen_bbox --cam-id 3 --skip-existing

conda activate touch_gait
# 图像特征使用 GPU；关键点脚本当前固定使用 CPU
CUDA_VISIBLE_DEVICES="$ANYSOLE_GPU" python -m lib.util.gen_image_feature --cam-id 3 --skip-existing
python -m lib.util.gen_kps --cam-id 3 --skip-existing
```

然后生成 AnySole 专用视觉 cache（命令从仓库根目录执行）：

```bash
cd /data/fangyuxuan/projects/gait
conda activate touch_gait
CUDA_VISIBLE_DEVICES="$ANYSOLE_GPU" python AnysoleWorkspace/script/generate_hrnet_cache.py \
  --cam-id 3 --device cuda --batch-size 32 --skip-existing
```

主要输出：

```text
AnysoleWorkspace/derived/MotionPRO/sequences/cam3/<date>/<subject>/<session>/
AnysoleWorkspace/derived/AnySole/hrnet_cache/cam3/<session>.pt
AnysoleWorkspace/splits/default/splits.csv
```

### 2.3 Step2Motion 数据

这一步是数据预处理，不是训练：它读取第 2.2 节生成的 cam3 序列、压力数据和
中央 split，转换并写出 Step2Motion 使用的 `gait_{train,val,test}.pt` 数据集。
`--self-test` 是脚本内置的快速自测选项，只检查压力池化、COP、clip 切分、四元数/IMU
等核心函数的形状和数值约束；执行后立即退出，不读取完整数据，也不会生成 `.pt` 文件。
正式转换前建议先运行一次：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/Step2Motion
conda activate touch_gait
# 只运行脚本内置的快速自测，检查压力池化、COP、clip 切分、四元数和 IMU 合成等逻辑是否正常。它不会读取完整数据，也不会生成 .pt 文件。
python src/process_gait.py --self-test
# 生成对应训练测试 pt 文件
python src/process_gait.py --skip-existing
python src/normalizer.py gait \
  workspace://derived/Step2Motion/gait/gait_train.pt --skip-existing
```
这里原来的 `../../` 是相对路径。

输出为 `AnysoleWorkspace/derived/Step2Motion/gait/gait_{train,val,test}.pt`。

### 2.4 pressure toolkit 数据

pressure toolkit 需要额外准备真实的 essential、标定、floor 参数和 RTM-pose
输入。适配数据写入 `AnysoleWorkspace/derived/pressure_tookit/`，不得使用 3D GT
关键点代替 RTM-pose 观测。

这一步是为 pressure toolkit 生成 RGB-D 输入中的深度图，默认一次处理全部 session：
1. 读取 color/ 目录中的 RGB 图片。
2. 使用本地 Depth Pro 模型估计每张图的米制深度。
3. 把深度保存到 depth/ 目录，格式是 uint16 PNG，单位为毫米。
4. 同时生成 meta.json，记录焦距、尺寸、深度范围等信息。

```bash
cd /data/fangyuxuan/projects/gait
conda activate depthpro
#  pressure toolkit 生成深度图
CUDA_VISIBLE_DEVICES="$ANYSOLE_GPU" python Baselines/pressure_tookit/data_prep/rgb2depth.py \
  --device cuda \
  --skip-existing
```

脚本默认递归扫描 `derived/MotionPRO/sequences/cam3/<date>/<subject>/<session>/color/`，并在每个 session
旁边创建对应的 `depth/` 目录；终端会显示 session 和帧两层处理进度。调试单个 session
时，仍可显式传入 `SRC DST` 两个目录参数，也可用
`--session <session>` 或 `--limit-sessions N` 限制全量扫描范围。若要扫描其他目录，使用
`--images-root workspace://...` 覆盖默认根目录。

## 3. 训练

### 3.1 主模型 AnySole

```bash
# 主模型
PYTHONPATH=. python -m anysole.train \
  --config configs/v1.yaml \
  --modal anysolev1 \
  --device cuda

# 漂移补偿消融模型
PYTHONPATH=. python -m anysole.train \
  --config configs/v1.yaml \
  --modal anysolev1_insole_drift \
  --device cuda
```

输出：`outputs/v1/ckpt_last.pt`。训练包含 `VT`（RGB+压力）、`V`（RGB）和
`T`（压力）三种输入配置。

### 3.2 三个基线

#### MotionPRO

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
CUDA_VISIBLE_DEVICES="$ANYSOLE_GPU" python -m app.train_frappe
```

输出：`results/MotionPRO/checkpoints/`

#### Step2Motion

```bash
cd /data/fangyuxuan/projects/gait/Baselines/Step2Motion
python src/train.py --config configs/config_gait.json
```

输出：`results/Step2Motion/checkpoints/gait_model/`

#### pressure toolkit

按第 4.4 节的 `init_shape → init_pose → tracking` 顺序运行：

```bash
cd /data/fangyuxuan/projects/gait
python Baselines/pressure_tookit/main_singleview.py \
  --config Baselines/pressure_tookit/configs/fit_smpl_rgbd.yaml
```

输出：`results/pressure_tookit/`

## 4. 评估、推理与可视化

### 4.1 AnySole

评估验证集并导出 BVH：

```bash
cd /data/fangyuxuan/projects/gait
PYTHONPATH=. python -m anysole.eval \
  --config configs/v1.yaml --ckpt outputs/v1/ckpt_last.pt --device cuda \
  --write-bvh outputs/v1/eval_bvh
```

单 session 推理：

```bash
PYTHONPATH=. python -m anysole.infer \
  --ckpt outputs/v1/ckpt_last.pt --session S12021 \
  --config configs/v1.yaml --config-id 0 --device cuda
```

`config-id` 为 `0=VT`、`1=V-only`、`2=T-only`；默认输出
`outputs/v1/S12021_VT.bvh`。

### 4.2 MotionPRO

训练结束后会自动评估 test split；需要单独重测或可视化时：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
python -m app.test_frappe
python ../../resultsdisplay/script/visualize_motionpro.py
```

指标写入 `results/MotionPRO/metrics/`，可视化写入 `resultsdisplay/MotionPRO/`。

### 4.3 Step2Motion

```bash
cd /data/fangyuxuan/projects/gait/Baselines/Step2Motion
python src/test_model.py \
  results://Step2Motion/checkpoints/gait_model \
  ../../AnysoleWorkspace/derived/Step2Motion/gait/gait_test.pt \
  --only_test
```

单条 clip：

```bash
python src/test.py <MODEL_DIR> \
  --dataset ../../AnysoleWorkspace/derived/Step2Motion/gait/gait_test.pt \
  --clip 0
```

预测 BVH 位于模型目录的 `predictions/`。

### 4.4 pressure toolkit

修改 `Baselines/pressure_tookit/configs/fit_smpl_rgbd.yaml` 中的
`dataset`、`sub_ids`、`seq_name`、`fitting_stage`、`start_idx` 和
`end_idx`，然后执行：

```bash
cd /data/fangyuxuan/projects/gait
python Baselines/pressure_tookit/main_singleview.py \
  --config Baselines/pressure_tookit/configs/fit_smpl_rgbd.yaml
```

按 `init_shape → init_pose → tracking` 执行；结果写入
`results/pressure_tookit/`。缺少真实 essential、标定或 floor 文件时停止执行。

### 4.5 统一评估格式

四个模型进入统一评估前，应导出：

```text
session_id
frame_index
valid_mask
joint_xyz_world (23, 3)
root_xyz_world (3)
```

fake/invalid 帧不参与指标计算；不得使用 `keypoints.npy` 的 3D GT 投影替代
RTM-pose 观测。

## 5. 常见问题

- `Missing AnySole HRNet+bbox cache`：先执行第 2.2 节的 AnySole cache 命令。
- `No module named 'lib'`：MotionPRO 的 `python -m lib.util.*` 必须在
  `Baselines/MotionPRO/` 下执行。
- 缺少 pressure toolkit 的 essential、标定或 floor 文件：停止拟合，不下载或伪造文件。
- CUDA 不可用：检查 `CUDA_VISIBLE_DEVICES`、`nvidia-smi` 和当前环境的 PyTorch CUDA 版本。
