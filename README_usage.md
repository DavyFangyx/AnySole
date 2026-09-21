# 统一使用说明

本项目以 `AnysoleWorkspace/` 为统一数据工作区，固定使用 cam3、40 Hz、同一份
session split。主模型 AnySole 使用标准 SMPL-24 关节协议（BVH-23 仅 Step2Motion
基线使用）。基线模型为 MotionPRO、Step2Motion 和 pressure toolkit。

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
基线模型输出：results/、results_display/
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

`prepare_sequences.py` 同时为各 session 生成 `contact.npy`，口径为「48 格压力和 > 100」
（即 Test5 的 `tactile_abs` 方案）。如需用 Test5 的其他接触检测方案训练/评估
（见 3.1 与 4.1 的 `--contact-method`），先用 `results_display/script/contact_methods.py`
额外生成 `contact_<method>.npy`，详见 `results_display/README.md` 的「## Test5 接触检测」。

默认只生成一份中央 split：
`AnysoleWorkspace/splits/default/splits.csv`。完成 split 后，再运行下面的
bbox、图像特征、关键点和 AnySole cache 生成命令；后续模型不得自行重新划分数据。

生成 MotionPRO 输入：

```bash
conda activate bbox_scan
CUDA_VISIBLE_DEVICES=N python -m lib.util.gen_bbox --cam-id 3 --skip-existing

conda activate touch_gait
# 图像特征
CUDA_VISIBLE_DEVICES=N python -m lib.util.gen_image_feature --cam-id 3 --skip-existing
python -m lib.util.gen_kps --cam-id 3 --skip-existing
```

然后生成 AnySole 专用视觉 cache（命令从仓库根目录执行）：

```bash
cd /data/fangyuxuan/projects/gait
conda activate touch_gait
CUDA_VISIBLE_DEVICES=N python AnysoleWorkspace/script/generate_hrnet_cache.py \
  --cam-id 3 --batch-size 32 --skip-existing
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
CUDA_VISIBLE_DEVICES=7 python Baselines/pressure_tookit/data_prep/rgb2depth.py \
  --skip-existing
```

脚本默认递归扫描 `derived/MotionPRO/sequences/cam3/<date>/<subject>/<session>/color/`，并在每个 session
旁边创建对应的 `depth/` 目录；终端会显示 session 和帧两层处理进度。调试单个 session
时，仍可显式传入 `SRC DST` 两个目录参数，也可用
`--session <session>` 或 `--limit-sessions N` 限制全量扫描范围。若要扫描其他目录，使用
`--images-root workspace://...` 覆盖默认根目录。

## 3. 训练

### 3.1 主模型 AnySole

AnySole 的训练入口需要从仓库根目录执行。`conda activate touch_gait` 只切换
Python 解释器和依赖环境；它不会自动设置仓库的模块搜索路径。不过从仓库根目录
运行 `python -m anysole.train` 时，当前目录已经在 Python 的搜索路径中，因此
从仓库根目录执行时不需要额外设置 `PYTHONPATH`。

`--device` 默认 `auto`：有 CUDA 时自动使用 GPU，无需显式传入。要选择具体的物理
GPU，使用 `CUDA_VISIBLE_DEVICES`；例如前文的 `$ANYSOLE_GPU=4` 表示物理 GPU 4。
设置后，程序内部的 `cuda` 会指向这张可见卡（即`cuda:0`）。如果前面第 2 节已经执行了
`export CUDA_VISIBLE_DEVICES=...`，这里可以省略命令前的重复设置。需要覆盖自动选择时，
可显式传 `--device cuda`、`--device cuda:N` 或 `--device cpu`。

漂移补偿消融模型还需要按 subject 构建鞋垫模板库。首次训练该模型前执行：

```bash
conda activate touch_gait
python -m anysole.ablations.insole_drift.build_templates \
  --manifest AnysoleWorkspace/manifests/session_manifest.csv \
  --out AnysoleWorkspace/calibration/insole_templates.json
```

该命令读取压力数据并写出 `configs/v1.yaml` 中 `template_path` 指向的文件；普通
`anysolev1` 主模型不依赖此文件。

```bash
cd /data/fangyuxuan/projects/gait
conda activate touch_gait

# 主模型（anysolev2 回归主线；当前基座与各变体的完整命令见
# anysole/model_fix_note/command_manual.md）
CUDA_VISIBLE_DEVICES=4 python -m anysole.train \
  --modal anysolev2 \
  --contact-method joint_and \
  --grad-clip 5.0 \
  --wandb_mode online \
  --wandb_project Anysole --wandb_experiment_tag <tag> --wandb_eval_interval 10

# 漂移补偿消融模型（anysolev1 系 legacy 变体）
# python -m anysole.ablations.insole_drift.build_templates ...（见上文）
# 接触标签方案由 --contact-method 指定：tactile_abs / bvh_h / bvh_soft /
#   joint_or / joint_and / motion_f6 / pressure_f6 / f6_soft
```

### 3.2 三个基线

#### MotionPRO

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
conda activate touch_gait

CUDA_VISIBLE_DEVICES=3 python -m app.train_frappe task.contact_method=bvh_h
```

MotionPRO 的 test 指标（MPJPE/PVE/WBCE）不使用接触标签，只有训练期损失与 IoU 受影响。

输出：`results/MotionPRO/checkpoints/`

#### Step2Motion

```bash
cd /data/fangyuxuan/projects/gait/Baselines/Step2Motion
conda activate touch_gait

python src/train.py --config configs/config_gait.json --no-imu
--config configs/config_gait.json
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

训练结束后会自动评估 `test` split，并写入对应模型的 `metrics/test.json`。
不带参数直接运行即自动扫描 `results/AnySole` 下所有 `<modal>_<contact_method>`
模型目录（跳过缺少 checkpoint 的目录），逐个完成全部评估：BVH/轨迹 npz 与指标
分别写入各模型自己的 `predictions/eval_bvh` 与 `metrics/test.json`。
扫描时 `--write-bvh <目录>` 会按模型名建子目录，避免不同模型互相覆盖。

```bash
cd /data/fangyuxuan/projects/gait
conda activate touch_gait

python -m anysole.eval
```

需要单独重测某个模型时：

```bash
python -m anysole.eval \
  --modal anysolev1 \
  --contact-method joint_and

# 漂移补偿消融模型：只换 --modal
  --modal anysolev1_insole_drift \

python -m anysole.eval \
  --ckpt results/AnySole/E4_normdiff/checkpoints/ckpt_last.pt
```
由于 results/AnySole 下的模型存储位置与命名具有明确规则：
例如：anysolev1_{insole_drift}_{tactile_abs}

模型由 `--modal` 与 `--contact-method` 两个参数唯一确定：自动读取
`results/AnySole/<modal>_<contact_method>/checkpoints/ckpt_last.pt`，BVH 与轨迹 npz
默认写入同目录 `predictions/eval_bvh`，指标写入 `metrics/test.json`。其余参数均有默认值：
`--config` 默认 `configs/v1.yaml`、`--device` 默认 `auto`（有 CUDA 自动用 GPU）、
`--split` 默认 `test`、`--config-id` 默认 `VT2M,V2M,T2M`。需要评估其他 checkpoint 或
自定义导出目录时，用 `--ckpt <路径>`、`--write-bvh <目录>` 覆盖；`--no-write-bvh`
关闭默认的 BVH 导出（训练结束时的自动评估即用此开关，只出指标）。

评估指标会自动写入同一模型的 `metrics/test.json`（含 `contact_method` 字段记录标签口径）。
每个模式行除 MPJPE/PA-MPJPE/MPJRE/traj_ATE/contact_acc 外，还输出触觉重建指标
`T_mae`/`T_rmse`/`T_corr`（96 格归一化压力）；其中 **V2M 行即 V2T（仅视觉生成触觉）
质量**，JSON 顶层 `v2t` 字段为该行摘要。可视化与逐格误差分析见
`results_display/README.md` 的「## Test10 触觉生成（V2T）」。

单 session 推理：

```bash
python -m anysole.infer \
  --modal anysolev1 \
  --contact-method tactile_abs \
  --session S12021 --config-id VT2M
```

漂移补偿消融同样只把 `--modal` 换成 `anysolev1_insole_drift`；输出 BVH 默认写入
对应模型的 `predictions/`。

`--config-id` 使用模式名：`VT2M`（视觉+压力）、`V2M`（仅视觉）、`T2M`（仅压力）。
一次选择多个模式时用逗号连接，例如 `--config-id VT2M,V2M,T2M`；程序会在同一次运行中
分别输出 `S12021_VT2M.bvh`、`S12021_V2M.bvh` 和 `S12021_T2M.bvh`。省略该参数时，
程序仍会根据可用输入自动选择一种模式。

### 4.2 MotionPRO

训练结束后会自动评估 test split；需要单独重测或可视化时：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
python -m app.test_frappe
python ../../results_display/script/visualize_motionpro.py
```

指标写入 `results/MotionPRO/metrics/`，可视化写入 `results_display/Test1_visualization/MotionPRO/`。

### 4.3 Step2Motion

```bash
cd /data/fangyuxuan/projects/gait/Baselines/Step2Motion
conda activate touch_gait

python src/test_model.py \
  results://Step2Motion/checkpoints/gait_model_noimu \
  ../../AnysoleWorkspace/derived/Step2Motion/gait_noimu/gait_test.pt \
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
- 缺少 `contact_<method>.npy`（指定 `--contact-method` 后报错）：先运行
  `python results_display/script/contact_methods.py --methods <method>` 生成对应方案标签。
- 缺少 pressure toolkit 的 essential、标定或 floor 文件：停止拟合，不下载或伪造文件。
- CUDA 不可用：检查 `CUDA_VISIBLE_DEVICES`、`nvidia-smi` 和当前环境的 PyTorch CUDA 版本。

## 6. 后台训练队列

`offline_train/` 提供退出终端后仍继续运行的串行队列。先生成任务（已有任务文件不会覆盖）：

```bash
cd /data/fangyuxuan/projects/gait
python offline_train/create_queue.py \
  --models anysole,anysole_insole_drift,motionpro,step2motion,pressure_toolkit \
  --gpu-list 0,1,2,3
```

每张 GPU 启动一个 scheduler；同卡串行，不同卡并行：

```bash
CUDA_VISIBLE_DEVICES=0 bash offline_train/bg.sh
CUDA_VISIBLE_DEVICES=1 bash offline_train/bg.sh
```

查看队列状态：

```bash
python offline_train/status.py
```

任务日志位于 `offline_train/queues/GPU<N>/logs/`，成功和失败任务分别移动到
`done/`、`failed/`。每次运行的 checkpoint 和配置快照位于
`results/offline/<run_name>/`；pressure toolkit 的三个阶段固定在同一张 GPU 上按顺序执行。
