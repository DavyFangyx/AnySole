# AnysoleWorkspace

工作区路径、数据协议与**全部中间数据准备** CLI 手册。基线训练/评估命令见
`../Baselines/command_manual.md`；主模型命令见 `../anysole/model_fix_note/command_manual.md`。

## 0. workspace 初始化/体检

```bash
python AnysoleWorkspace/tool/workspace.py init
python AnysoleWorkspace/tool/workspace.py doctor
python AnysoleWorkspace/tool/workspace.py relink
```

- `init`：建立 sources 软链、本地目录与 calibration 软链。
- `doctor`：全量体检——软链有效性、标定 JSON、canonical splits、必需权重与结果路径。
- `relink`：把 MotionPRO 序列内的 color 软链重指向 `sources/raw`（外部路径变更后用）。

## 目录索引

```text
AnysoleWorkspace/
├── sources/                        # 软链到 /data/lizhe/projects/Tactile/（多项目共用，只读勿改）
│   ├── raw/  smpl/  published/  calibration_artifacts/
│   └── PressureWasher/outputs/     # 本项目清洗产物 stats/reconstructed/fake_marked/encoded/AlignReviews_csv
├── derived/                        # 本项目私有中间产物
│   ├── AnySole/hrnet_cache/cam3/        # AnySole 专用
│   ├── MotionPRO/sequences/cam3/        # 共用数据源：AnySole + MotionPRO + Step2Motion + pressure_toolkit 都从这里读
│   ├── MotionPRO/pressure_96/           # 三基线触觉落盘（步骤 3.6）
│   ├── Step2Motion/gait/                # Step2Motion 正式数据（gait_noimu/ 探针、pressure_16ch/ 展示口径）
│   ├── pressure_toolkit/                # pressure_toolkit + VP-MoCap 共用（canonical 拼写；源码目录保留历史名 tookit）
│   │   └── images/  annotations/  input/  initialization/  fitting/
│   ├── VP-MoCap/                        # FPP-Net/PoseTransOpt 适配数据 <date>/<subject>/<session>/
│   └── baseline_tactile/                # 三基线触觉转换产物 <sid>/meta.json（D_Test4 前置）
├── dependencies/                        # 全体共用：CLIFF/ mmpose/ MotionPRO/ pressure_toolkit/(depthpro+essential)/ rtmpose/ smpl/ Step2Motion/ VP-MoCap/
├── calibration/                         # 标定摘要（软链）+ insole_templates.json
├── manifests/                           # session_manifest.csv|jsonl、统一数据协议.md、data_quality_report.md
├── splits/default/splits.csv            # 训练划分（全体唯一来源，只生成一次）
└── tool/                                # 数据构建代码（唯一入口，见下方 CLI 手册）
```

多项目/多模型共用说明：

- `sources/` 四个软链指向 `/data/lizhe/projects/Tactile/` 下的外部共享数据（采集方与其他项目共用），只读；本项目的一切写入只发生在 `sources/PressureWasher/` 与 `derived/`。
- `splits/`、`manifests/`、`dependencies/`、`calibration/` 为本项目全部模型共用。
- `derived/MotionPRO/sequences/` 是主模型与基线的共同消费点：AnySole 从这里读
  `contact_<method>.npy` 与 `fake_mask.npy`（`anysole/types.py`、`anysole/data/dataset.py`），
  Step2Motion 读序列（3.2）、pressure_toolkit 读 RGB（3.5 深度图）。
- `derived/pressure_toolkit/` 被 pressure_toolkit 与 VP-MoCap（FPP-Net/PoseTransOpt）共同消费。

## 数据协议（摘要）

- AnySole 只读 SMPL：默认 `/data/lizhe/projects/Tactile/Mocap/{0804,0807,0808,0810}/`
  下的 `motion_neutral_smpl.npz`（即 `sources/smpl/`）；不读取或导出 BVH，原始 BVH
  仅供 Step2Motion。环境变量 `ANYSOLE_SMPL_ROOTS` 可覆盖根目录。
- 触觉单帧 = 双脚 `(2,48)`（CSV 列 `1..48`），AnySole 拼接为 `T_raw (T,96)`；
  `encode` 只生成 `fake_mask_left/right.npy`，不保存压力值。
- splits train/val/test = 92/12/36，只生成一次，任何模型不得自行重新划分。
- 完整协议（时间轴、SMPL-24/BVH-23 口径、压力 4×12 布局、fake mask、eligibility）
  的唯一来源是 `manifests/统一数据协议.md`，本 README 不重复。

## 依赖文件总表

按各模型代码实际读取的路径统计（读取位置见各步正文）：

| 模型 | 依赖的中间数据 | 生成步骤 |
| --- | --- | --- |
| **AnySole**（主模型） | splits；触觉清洗产物（fake_marked、fake_mask）；序列内 contact_\<method\>.npy；hrnet_cache；消融另需 insole_templates.json | 1.1 / 1.3 / 1.4 / 1.5 / 2.1 / 2.2 |
| **MotionPRO** | 序列与输入特征；contact.npy；触觉清洗产物（fake_marked/stats/AlignReviews） | 1.3 / 1.4 / 1.5 / 3.1 |
| **Step2Motion** | 序列；触觉清洗产物（fake_marked/stats）；gait pt | 1.3 / 1.4 / 3.2 |
| **pressure_toolkit** | 序列 RGB；images/{depth,depth_mask}；initialization/ | 1.4 / 3.5 |
| **VP-MoCap**（FPP-Net+PoseTransOpt） | 序列 RGB；annotations、input/keypoints；CLIFF_results；template_scene | 1.4 / 3.3 / 3.5 |
| 展示端（D_Test4） | baseline_tactile/<sid>/meta.json 等 4 类落盘产物 | 3.6（←1.1/1.4） |

各模型需要的代码库与权重**均已放在 `dependencies/`**（`doctor` 体检覆盖必需项），
本手册正文不再重复下载命令；清单与下载来源见文末附录，`doctor` 报缺失时按附录补齐。
顺序依赖：1.1 → 1.2（质检读 manifest）；1.4 依赖 1.3（prepare_sequences 读
fake_marked/stats）；1.5 依赖 1.4；2.1/2.2 互不依赖；3.2 依赖 1.4；3.3 依赖
1.4 与 3.5 深度；3.5 CLIFF 依赖 3.3；3.4 依赖 3.3+3.5；3.6 依赖 1.1+1.4。
所有写入准备数据的 CLI 都支持互斥的 `--skip-existing`（已有产物就跳过，适合
首次生成和中断后续跑）与 `--force`（忽略已有产物全部重算）；下面以
`--skip-existing` 为默认示例。

## 1. 共用底座（全体模型）

### 1.1 session manifest + splits

```bash
python AnysoleWorkspace/tool/build_manifest.py --fps 40 --camera cam3
```

产物：`manifests/session_manifest.csv|jsonl`、`splits/default/splits.csv`
（train/val/test = 92/12/36，统一中间数据与推理准备集合为三者并集的 140 个
session；manifest 中 4 个 `unassigned` session 不处理）。split 只生成一次，
后续模型不得自行重新划分。

### 1.2 数据质检

```bash
python AnysoleWorkspace/tool/check_data_quality.py --manifest AnysoleWorkspace/manifests/session_manifest.jsonl
python AnysoleWorkspace/tool/check_splits.py --manifest AnysoleWorkspace/manifests/session_manifest.jsonl
```

产物：`manifests/data_quality_report.md`。

### 1.3 触觉清洗（PressureWasher，4 stage 顺序执行）

```bash
python AnysoleWorkspace/tool/pressure_washer/run.py inspect --skip-existing
python AnysoleWorkspace/tool/pressure_washer/run.py reconstruct --skip-existing
python AnysoleWorkspace/tool/pressure_washer/run.py mark-fake --skip-existing
python AnysoleWorkspace/tool/pressure_washer/run.py encode --skip-existing
```

产物：`sources/PressureWasher/outputs/{stats,reconstructed,fake_marked,encoded}`。
消费方：MotionPRO `prepare_sequences.py`、Step2Motion `process_gait.py` 直接读
fake_marked/stats；AnySole 读序列内 `fake_mask.npy`；pressure_toolkit/VP-MoCap
经 1.4 序列间接消费。

### 1.4 序列划分

命令在 `Baselines/MotionPRO/` 下执行：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
python data_prep/prepare_sequences.py --cam-id 3 --skip-existing
# 序列生成后立即冻结所有模型共用的 train/val/test 划分
python data_prep/make_splits.py --skip-existing
```

产物：`derived/MotionPRO/sequences/cam3/<date>/<subject>/<session>/`，其中
`prepare_sequences.py` 同时生成 `contact.npy`（口径「48 格压力和 > 100」，
即 D_Test3 的 `tactile_abs`）。这是主模型与基线的共同消费点：AnySole 从这里读
`contact_<method>.npy` 与 `fake_mask.npy`（`anysole/types.py` 的 SEQ_ROOT、
`anysole/data/dataset.py`），Step2Motion 读序列（3.2），pressure_toolkit 读
RGB（3.5 深度图）。可选 `--ood S11,S10`（指定 subject 整体进 OOD 的
val/test）与 `--exclude S5,S6`（排除 subject）。

### 1.5 接触 npz（AnySole、MotionPRO 消费）

```bash
python AnysoleWorkspace/tool/contact_labels.py
```

产物：序列目录下 `contact_<method>.npy`（方案：tactile_abs / bvh_h / bvh_soft /
joint_or / joint_and / motion_f6 / pressure_f6 / f6_soft）。这是**训练接触 GT**：
AnySole（`anysole/data/dataset.py`，按 `--contact-method` 读取）与 MotionPRO
（`lib/dataset/image_pressure.py`）消费；其他模型不读。

### 1.6 只读预检

```bash
cd /data/fangyuxuan/projects/gait
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
  AnysoleWorkspace/tool/smoke_baseline_integration.py --session S13013 --motionpro
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
  AnysoleWorkspace/tool/check_baseline_readiness.py --split all
```

通过条件：split=92/12/36、S13013 两个 insole 目录各 339 帧、每脚 shape=31×11、
FPP metadata=92/12/36（全 fake session 可为 0 个 temporal-5 window）、
MotionPRO pressure shape=(20,96,96)。

## 2. 主模型（AnySole）

### 2.1 HRNet 特征缓存

```bash
CUDA_VISIBLE_DEVICES=0 python AnysoleWorkspace/tool/generate_hrnet_cache.py --cam-id 3 --skip-existing
```

产物：`derived/AnySole/hrnet_cache/cam3/*.pt`。消费方：AnySole（`anysole/types.py`
HRNET_CACHE_ROOT、`data/dataset.py` 读取）与展示端 `results_display/script/r_test4_v2t.py`。

### 2.2 鞋垫模板（漂移补偿消融）

```bash
python -m anysole.ablations.insole_drift.build_templates \
  --manifest AnysoleWorkspace/manifests/session_manifest.csv \
  --out AnysoleWorkspace/calibration/insole_templates.json
```

产物：`calibration/insole_templates.json`（普通 `anysolev1` 主模型不依赖，
仅供漂移补偿消融）。

## 3. 基线模型

### 3.1 MotionPRO（学习型基线 V+T→M）

序列与接触标签见 1.4/1.5；本节只生成 MotionPRO 训练专用的输入特征
（bbox 与图像特征为 GPU 密集步骤，`gen_kps` 固定 CPU）：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO

conda activate bbox_scan
CUDA_VISIBLE_DEVICES=N python -m lib.util.gen_bbox --cam-id 3 --skip-existing

conda activate touch_gait
CUDA_VISIBLE_DEVICES=N python -m lib.util.gen_image_feature --cam-id 3 --skip-existing
python -m lib.util.gen_kps --cam-id 3 --skip-existing
```

产物：序列目录下 MotionPRO 训练读取的 bbox/图像特征/kps 文件
（`lib/dataset/image_pressure.py`）。

### 3.2 Step2Motion（学习型基线 T→M）

读取 1.4 的 cam3 序列、压力数据和中央 split，转换出 `gait_{train,val,test}.pt`
（首次转换前可先 `--self-test` 自测核心函数，不读完整数据、不生成 `.pt`）：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/Step2Motion
conda activate touch_gait
python src/process_gait.py --skip-existing
python src/normalizer.py gait \
  workspace://derived/Step2Motion/gait/gait_train.pt --skip-existing
```

产物：`derived/Step2Motion/gait/gait_{train,val,test}.pt`。

### 3.3 FPP-Net（学习型基线 V→T）

```bash
# 生成 temporal-5 split 和各日期 sub_info；不触碰 insole
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
  AnysoleWorkspace/tool/build_fpp_metadata.py --force

# 链接真实 RGB，生成 calibration.npy/floor_*.npy，建立两个 keypoint 目录
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
  AnysoleWorkspace/tool/prepare_mmvp_observations.py --split all

# 执行 RTMPose 关键点提取（--model 缺省即 dependencies/rtmpose 下的
/data/fangyuxuan/miniconda3/envs/bbox_scan/bin/python \
  AnysoleWorkspace/tool/run_rtmpose_halpe26.py \
  --split all \
  --config AnysoleWorkspace/dependencies/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py \
  --device cuda:7
```

验收文件：

```text
AnysoleWorkspace/derived/pressure_toolkit/input/<subject>/<session>/keypoints/*.npy
AnysoleWorkspace/derived/VP-MoCap/<date>/<subject>/<session>/keypoints/*.npy
```

每帧必须含 `keypoints=(26,2)`、`keypoint_scores=(26,)`；不得使用 3D GT 投影。

### 3.4 PoseTransOpt（FPP-Net → PoseTransOpt 串行视觉动捕 V → T̂ → M）

无独立数据准备步骤，输入全部来自其他小节：`pred_contact_smpl`（3.3 的 FPP-Net
推理产物，见 `../Baselines/command_manual.md` §4）、`template_scene_rgbd.npy` 与
深度 mask（3.5 深度图）、`CLIFF_results.npz`（3.5 CLIFF 初始化）。

### 3.5 pressure_toolkit（优化型多模态基线 V+T→M）

**深度图（Depth Pro）**——读取 1.4 序列的 RGB、估计米制深度并写 uint16 PNG
（毫米）+ `meta.json`（显存不足时可分片并行，见脚本 `--help`）：

```bash
cd /data/fangyuxuan/projects/gait
/data/fangyuxuan/miniconda3/envs/depthpro/bin/python \
  Baselines/pressure_tookit/data_prep/rgb2depth.py \
  --images-root workspace://derived/pressure_toolkit/images \
  --batch-size 4 --skip-existing

# 对已有 depth PNG 生成 0.4–5.0m mask 和 PoseTransOpt scene 文件
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
  AnysoleWorkspace/tool/prepare_mmvp_observations.py \
  --split all --make-depth-mask --make-template-scene
```

验收文件：

```text
AnysoleWorkspace/derived/pressure_toolkit/images/<date>/<subject>/<session>/depth/*.png
AnysoleWorkspace/derived/pressure_toolkit/images/<date>/<subject>/<session>/depth_mask/*.png
AnysoleWorkspace/derived/VP-MoCap/<date>/<subject>/<session>/template_scene_rgbd.npy
```

其中 `template_scene_rgbd.npy` 供 3.4 PoseTransOpt 使用。

**CLIFF 初始化**——CLIFF 代码与权重（含 YOLOv3 检测器、SMPL 模型、hr48 ckpt）
已在 `dependencies/CLIFF/`，无需重新下载。运行原生 YOLOv3 detector：

```bash
CUDA_VISIBLE_DEVICES=7 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
  AnysoleWorkspace/tool/run_cliff_mmvp.py \
  --split all \
  --ckpt AnysoleWorkspace/dependencies/MotionPRO/cliff_ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt \
  --backbone hr48 --batch-size 32 \
  --detector official_yolov3
```

只做接入 smoke 时才用替代 detector（`--detector motionpro_yolox`）。输出：

```text
AnysoleWorkspace/derived/VP-MoCap/<date>/<subject>/<session>/CLIFF_results.npz
AnysoleWorkspace/derived/pressure_toolkit/initialization/<date>/<subject>/<session>/<session>_cliff_hr48.npz
```

其中 `CLIFF_results.npz` 同时供 3.4 PoseTransOpt 使用；适配 smoke 的 NPZ 中
`bbox_source`/`reproduction_status` 保留偏离记录；CLIFF 初始化不再写入
`results/`。smoke 输出与 canonical 路径隔离。

### 3.6 三基线触觉转换（仅展示端 D_Test4，不参与训练）

把原始触觉逐帧转换成三个基线的触觉输入口径（MotionPRO 96×96 压力图、
Step2Motion 16 通道/脚、MMVP 两模型共用 31×11 insole）。与 1.5 接触 npz 不同：
1.5 是 AnySole/MotionPRO 训练读取的接触 GT，本步产物只供 D_Test4 跨模型触觉
可视化对比，任何模型训练都不读取：

```bash
python AnysoleWorkspace/tool/generate_baseline_tactile.py --split test
```

产物：

```text
AnysoleWorkspace/derived/MotionPRO/pressure_96/<sid>.npz
AnysoleWorkspace/derived/Step2Motion/pressure_16ch/<sid>.npz
AnysoleWorkspace/derived/pressure_toolkit/images/<date>/<subject>/<sid>/insole/*.npy
AnysoleWorkspace/derived/VP-MoCap/<date>/<subject>/<sid>/insole/*.npy
AnysoleWorkspace/derived/baseline_tactile/<sid>/meta.json
```

可视化前端 `results_display/script/d_test4_baseline_tactile.py`（产物齐全自动跳过
生成，`--force` 强制重生成）。

## 附录：权重清单与下载链接
