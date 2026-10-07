# AnysoleWorkspace

这是项目唯一的数据与中间产物工作区。旧版 `sources/`、`derived/`、`dependencies/`、`calibration/`、`manifests/`、`splits/` 已删除；所有新代码只允许使用下面的目录和 URI。

```text
AnysoleWorkspace/
├── raw/
│   ├── rgb/ pressure/ bvh/ smpl/                 # 模态链接农场（relink_raw.py 生成）
│   └── calibration/ human_masks/                 # 外部只读源软链接
├── protocol/
│   ├── manifests/session_manifest.{csv,jsonl}
│   ├── splits/default/splits.csv                # 由参数生成，默认 val=test
│   ├── schemas/
│   └── calibration/
├── shared/
│   ├── facts/sessions/cam3/<date>/<subject>/<session>/
│   ├── representations/tactile/mmvp_31x11/v1/<date>/<subject>/<session>/
│   └── frontends/<name>/<version>/
├── model_inputs/<model>/                        # 模型私有产物
├── work/data_pipeline/pressure_washer/
├── assets/                                      # 依赖、布局和固定资产
├── reports/
└── tool/
```

## 路径规则

公共解析入口是 `tool/workspace.py`，只支持：

`raw://`、`protocol://`、`shared://`、`model-input://<model>/`、`work://`、`asset://`、`results://`。

旧 `workspace://` 已移除，路径逃逸和 `raw://` 写入都会直接报错。

## 数据基座构建

```bash
# 初始化 canonical Workspace 目录、raw 外部只读软链接，并构建 raw 模态链接农场
python3 AnysoleWorkspace/tool/workspace.py init
# init 创建的外部只读软链接：
# raw/calibration  -> /data/lizhe/projects/Tactile/0_Calibration
# raw/human_masks  -> /data/lizhe/projects/Tactile/3_Result/processed/rgb_human_masks
# init 构建的模态链接农场（raw/rgb raw/pressure raw/bvh raw/smpl，路径形状与 manifest URI 逐段一致）：
#   raw/rgb/<date>/<S#>/<rec>/{meta.json, <cam>/}          相机目录软链（默认 --cam 3）
#   raw/pressure/<date>/<S#>/<rec>/pressure_{left,right}.csv
#   raw/bvh/<date>/mocap_ori_bvh/<sid>/*.bvh (+ 0804 *.avi, calibration_Skeleton0.bvh)
#   raw/smpl/<MMDD>/<MMDD>smpl/mocap_ori_c3d/<sid>/motion_neutral_smpl.npz
# 上游新增日期/改名 rec 目录后，重建并校验农场（幂等，原子替换）：
python3 AnysoleWorkspace/tool/relink_raw.py build --dates 20260804,20260807,20260808,20260810 --cam 3
python3 AnysoleWorkspace/tool/relink_raw.py verify
# 方案与回滚快照：reports/raw_relink_plan_20261001.md、reports/raw_relink_rollback_20261002.txt

# 从现有 protocol manifest 建立 raw session 索引
python3 AnysoleWorkspace/tool/build_raw_index.py \
  --from-manifest AnysoleWorkspace/protocol/manifests/session_manifest.jsonl

# 生成带 valid/fake 标记的公共触觉源
python3 AnysoleWorkspace/tool/pressure_washer/run.py inspect --skip-existing
python3 AnysoleWorkspace/tool/pressure_washer/run.py reconstruct --skip-existing
python3 AnysoleWorkspace/tool/pressure_washer/run.py mark-fake --skip-existing
# 右脚故障格（1-based cell 35）清洗：D7 判据全树扫描，命中的 recording 整条右足通道置零
# （103/313，含 3 个 CV 刀刃 override：S13072/S14033/S6101）；就地改写 final_fake_marked 的
# pressure_*.csv 并落 fault_cell_manifest.csv 审计清单；幂等（重跑零命中），无需 --force；
# 必须在 encode 之前。清洗前原树存档于 work/data_pipeline/pressure_washer/archive/（确认后可删）
python3 AnysoleWorkspace/tool/pressure_washer/run.py clean-fault-cell
# encode 读 final_fake_marked 平铺树根（--force = 覆盖 encoded/ 同名输出）
python3 AnysoleWorkspace/tool/pressure_washer/run.py encode --force

# 从 raw、session 索引和 PressureWasher 最终记录构建公共 manifest
python3 AnysoleWorkspace/tool/build_manifest.py --fps 40 --camera cam3
# 生成 split，默认 val=test（当前冻结文件为 train/val/test = 104/36/36）
python3 AnysoleWorkspace/tool/build_manifest.py split \
  --manifest AnysoleWorkspace/protocol/manifests/session_manifest.jsonl \
  --test S13,S14

# 构建公共 shared facts
python3 AnysoleWorkspace/tool/build_shared.py --facts-only --force
# 校验 shared facts 和目录依赖边界
python3 AnysoleWorkspace/tool/validate_shared.py \
  --facts AnysoleWorkspace/shared/facts/sessions

# 检查 canonical Workspace、raw 链接和必需 artifact
python3 AnysoleWorkspace/tool/workspace.py doctor
```

split 由 `build_manifest.py split` 生成，默认 `val=test`；manifest 只登记 session 元数据，模型直接读取 `protocol/splits/default/splits.csv`。
触觉质量标注（A/B/C/D）登记在 manifest 的 `pressure_quality_class` / `pressure_quality_reason` 两列，作为排除依据
（C/D 一律不进入管线、S9 整组排除等），详见 `protocol/splits/default/README.md`。

## 各模型过程产物与生成脚本

公共 facts 建好后，各模型过程文件由对应脚本生成到自己的 `model_inputs/<model>/` 或 `work/`，不能写回 shared facts。

### AnySole

```bash
# 生成 AnySole HRNet 视觉特征缓存
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/generate_hrnet_cache.py --cam-id 3 --skip-existing

# 生成 AnySole 接触标签（10 方法 × 140 session；读 facts 的 pressure_48.npz + BVH；
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.data.contact_adapter --split test --methods all --force
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/adapters/AnySole/write_label_artifacts.py
# 查看 AnySole HMR/HRNet adapter 参数
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python anysole/data/extract_hmr.py --help
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python anysole/data/extract_hrnet.py --help
```

输出位于 `model_inputs/AnySole/`。

### MotionPRO

```bash
# 生成 MotionPRO 私有人体框（mmdet 只在 bbox_scan）
/data/fangyuxuan/miniconda3/envs/bbox_scan/bin/python -m lib.util.gen_bbox --cam-id 3 --skip-existing
# 生成 MotionPRO 私有图像特征
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m lib.util.gen_image_feature --cam-id 3 --skip-existing
# 生成 MotionPRO 私有关键点缓存
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m lib.util.gen_kps --cam-id 3 --skip-existing
# 推荐：一条命令跑完三步链（bbox 子进程自动切 bbox_scan 解释器）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/adapters/MotionPRO/run_visual_chain.py --cuda-device 6

# 触觉依赖产物：虚拟毯 pressure.npz + soft-f6 contact.npy
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/adapters/MotionPRO/adapter.py --split train --force
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/adapters/MotionPRO/adapter.py --split val --force
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/adapters/MotionPRO/adapter.py --split test --force
```

输出位于 `model_inputs/MotionPRO/`；dual-sole raster 和 soft-f6 也只能从 `shared/facts/pressure_48.npz` 生成（视觉链三条与压力无关，压力改动后无需重跑）。

### Step2Motion

```bash
# 生成 Step2Motion 原生 BVH/16 通道输入 + normalizer
# facts（pressure_48.npz）重建后必须重跑（输入含 16 通道压力条件）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/adapters/Step2Motion/build_gait.py
```

输出位于 `model_inputs/Step2Motion/adapter_v1/{gait, gait_noimu}/`（含 train/val/test `.pt`、train-only normalizer、`split_sessions.json`）。

### MMVP 方法族（FPP-Net / PoseTransOpt / pressure_toolkit）

数据端已合并为 `AnysoleWorkspace/tool/adapters/mmvp_series/`（一次生产、三处消费）。

```bash
# 生成 pressure_toolkit/FPP-Net 共用的 MMVP 31×11 过程表示
python3 AnysoleWorkspace/tool/build_shared.py --mmvp-only --force
# 提取 HALPE-26 关键点（写 shared/frontends/rtmpose_halpe26/v1，刷新两模型树软链接）
/data/fangyuxuan/miniconda3/envs/bbox_scan/bin/python AnysoleWorkspace/tool/adapters/mmvp_series/keypoints/run_rtmpose.py --split all
# 生成 CLIFF 视觉初始化（写 shared/frontends/cliff_hr48/v1 + pressure_tookit 初值）
/data/fangyuxuan/miniconda3/envs/mmvp/bin/python AnysoleWorkspace/tool/adapters/mmvp_series/cliff/run_cliff.py --split all --backbone hr48

# 用 DepthPro 生成深度图（写 shared/frontends/depthpro/v1）
CUDA_VISIBLE_DEVICES=3 /data/fangyuxuan/miniconda3/envs/depthpro/bin/python -m AnysoleWorkspace.tool.adapters.mmvp_series.depth.rgb2depth --batch-size 4 --skip-existing

# 生成 FPP-Net 模型输入树（RGB 链接 + adapter manifest）
python3 AnysoleWorkspace/tool/adapters/mmvp_series/fpp/build_inputs.py --split all
# 生成 FPP-Net temporal-5 split 和 subject metadata
python3 AnysoleWorkspace/tool/adapters/mmvp_series/fpp/build_metadata.py --force

# 生成 PoseTransOpt 模型输入树（join manifest 为对齐权威）
python3 AnysoleWorkspace/tool/adapters/mmvp_series/posetransopt/build_inputs.py --split all

# 生成 pressure_toolkit 上游原生目录契约
conda activate mmvp
python3 AnysoleWorkspace/tool/adapters/mmvp_series/pressure_tookit/build_inputs.py --split all

# 行走地面 ROI 审计（输出 results_display/DataTest/D6Test_floor/）
python3 AnysoleWorkspace/tool/adapters/mmvp_series/pressure_tookit/floor_audit.py
'''
1. 每个候选 ROI 拟合一张平面，算残差（点离平面 RMS mm）——小 = 该区域是一张平地面；
2. 对比旧口径：算棋盘格平面与拟合地面的偏移（1.29–1.45m）与法向夹角（140–145°）——证明旧适配器把棋盘格当地面是错的。
3. 9 个候选是脚本里预设的,不是自动搜出来的
'''
# 依赖深度文件 + 底部中央带 ROI 生成行走地面（floor）；
# 四日期审计最优 ROI 为 lower_narrow（868,1178,568,1055），显式指定：
python3 AnysoleWorkspace/tool/adapters/mmvp_series/pressure_tookit/build_inputs.py --split all --floor-roi 868,1178,568,1055
--sessions S5011
# 质量在 session_reports/<date>/<subject>/<session>.json 的 floor.residual_mm看
```

输出分别位于 `model_inputs/FPP-Net/adapter_v1/`、`model_inputs/PoseTransOpt/adapter_v1/`、`model_inputs/pressure_toolkit/v1/` 和 `shared/frontends/`（`rtmpose_halpe26/v1`、`cliff_hr48/v1`、`depthpro/v1`、`human_masks/sam31/v1`）；31×11 压力统一读取 `shared/representations/tactile/mmvp_31x11/v1/`。

MMVP 31×11 是 pressure_toolkit/FPP-Net 共用的过程表示，不被 AnySole 或 Step2Motion 读取。

### D_Test4

D_Test4 读取 shared representation 和各模型已经生成的 `model_inputs`，查看触觉适配到异构传感器上。

### Human-mask bbox frontend（独立产物）

```bash
# 将外部 SAM3.1 masks_index.csv 以只读链接方式登记，并生成独立 bbox artifact
python3 AnysoleWorkspace/tool/build_human_mask_frontend.py --force
# 校验 frontend session 数量、bbox shape 和 artifact
python3 AnysoleWorkspace/tool/validate_shared.py \
  --facts AnysoleWorkspace/shared/facts/sessions \
  --mmvp AnysoleWorkspace/shared/representations/tactile/mmvp_31x11/v1 \
  --human-masks AnysoleWorkspace/shared/frontends/human_masks/sam31/v1
```

该 frontend 位于 `shared/frontends/human_masks/sam31/v1/`，不复制外部 PNG，直接链接每个 session 的 `masks_index.csv` 并保留 CSV bbox。当前不自动接入旧模型消费者；旧模型指定的 bbox 路径保持独立。

## 公共契约

- `shared/facts` 只保存 RGB、时间轴、BVH/SMPL 追溯和双脚 48 格压力，不保存模型特征、contact、bbox、keypoints、mask 或模型预测。
- `frames.npz`：`frame_id:int64`、`visual_time_s:float64`、`mocap_time_s:float64`、`valid:uint8`、`fake:uint8`。
- `pressure_48.npz`：`left48/right48:float32[T,48]`，时间轴为**绝对 wall-clock 最近邻映射**（视频帧 = 首 cam3 jpeg 名 + i/fps；压力行 = `started_at_iso` + `raw_t_us` 列；`alignment_method=nearest_values_absolute_t_us`；越界钳位帧数记入 `session.json` 的 `out_of_span_frames_{left,right}`），并记录 `t_us`、`valid_mask`、`fake`、左右源 CSV URI 与 hash。
- MMVP 表示每帧为 `float32(2,31,11)`，只生成一份，供 pressure_toolkit/FPP-Net 共同只读。
- 每个生成根和 session/representation 都写 `artifact.json`。
- 正式 checkpoint、prediction、metric 只能写 `results/`。

阶段报告和逐 session provenance 位于 `reports/`；`assets/third_party/` 只保存依赖和固定资产，不承担数据事实或模型输出职责。
