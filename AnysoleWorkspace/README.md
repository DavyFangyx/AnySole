# AnysoleWorkspace

这是项目唯一的数据与中间产物工作区。旧版 `sources/`、`derived/`、`dependencies/`、`calibration/`、`manifests/`、`splits/` 已删除；所有新代码只允许使用下面的目录和 URI。

```text
AnysoleWorkspace/
├── raw/
│   ├── rgb/ pressure/ bvh/ smpl/ calibration/   # 外部只读源软链接
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
# 初始化 canonical Workspace 目录和 raw 外部只读链接
python3 AnysoleWorkspace/tool/workspace.py init
# init 创建的 raw 只读软链接：
# raw/rgb        -> /data/lizhe/projects/Tactile/1_Data
# raw/pressure   -> /data/lizhe/projects/Tactile/1_Data
# raw/bvh        -> /data/lizhe/projects/Tactile/1_Data
# raw/smpl       -> /data/lizhe/projects/Tactile/Mocap
# raw/calibration -> /data/lizhe/projects/Tactile/0_Calibration

# 从现有 protocol manifest 建立 raw session 索引
python3 AnysoleWorkspace/tool/build_raw_index.py \
  --from-manifest AnysoleWorkspace/protocol/manifests/session_manifest.jsonl

# 生成带 valid/fake 标记的公共触觉源
python3 AnysoleWorkspace/tool/pressure_washer/run.py inspect --skip-existing
python3 AnysoleWorkspace/tool/pressure_washer/run.py reconstruct --skip-existing
python3 AnysoleWorkspace/tool/pressure_washer/run.py mark-fake --skip-existing
python3 AnysoleWorkspace/tool/pressure_washer/run.py encode --skip-existing

# 从 raw、session 索引和 PressureWasher 最终记录构建公共 manifest
python3 AnysoleWorkspace/tool/build_manifest.py --fps 40 --camera cam3
# 生成 split，默认 val=test
python3 AnysoleWorkspace/tool/build_manifest.py split \
  --manifest AnysoleWorkspace/protocol/manifests/session_manifest.jsonl \
  --exclude S4,S5 --test S13,S14
# 将新 split 写回 manifest 的 split 字段
python3 AnysoleWorkspace/tool/build_manifest.py --fps 40 --camera cam3

# 构建公共 shared facts
python3 AnysoleWorkspace/tool/build_shared.py --facts-only
# 校验 shared facts 和目录依赖边界
python3 AnysoleWorkspace/tool/validate_shared.py \
  --facts AnysoleWorkspace/shared/facts/sessions
# 检查 canonical Workspace、raw 链接和必需 artifact
python3 AnysoleWorkspace/tool/workspace.py doctor
```

split 由 `build_manifest.py split` 生成，默认 `val=test`；manifest 只登记 session 元数据，模型直接读取 `protocol/splits/default/splits.csv`。

## 各模型过程产物与生成脚本

公共 facts 建好后，各模型过程文件由对应脚本生成到自己的 `model_inputs/<model>/` 或 `work/`，不能写回 shared facts。

### AnySole

```bash
# 生成 AnySole HRNet 视觉特征缓存
python3 AnysoleWorkspace/tool/generate_hrnet_cache.py --cam-id 3 --skip-existing
# 查看 AnySole HMR/HRNet adapter 参数
python3 anysole/data/extract_hmr.py --help
python3 anysole/data/extract_hrnet.py --help
```

输出位于 `model_inputs/AnySole/`。

### MotionPRO

```bash
# 生成 MotionPRO 私有人体框
python3 -m lib.util.gen_bbox --cam-id 3 --skip-existing
# 生成 MotionPRO 私有图像特征
python3 -m lib.util.gen_image_feature --cam-id 3 --skip-existing
# 生成 MotionPRO 私有关键点缓存
python3 -m lib.util.gen_kps --cam-id 3 --skip-existing
```

输出位于 `model_inputs/MotionPRO/`；dual-sole raster 和 soft-f6 也只能从 `shared/facts/pressure_48.npz` 生成。

### Step2Motion

```bash
# 生成 Step2Motion 原生 BVH/16 通道输入 + normalizer（touch_gait 环境）
python3 AnysoleWorkspace/tool/adapters/Step2Motion/build_gait.py
```

输出位于 `model_inputs/Step2Motion/adapter_v1/{gait, gait_noimu}/`（含 train/val/test `.pt`、train-only normalizer、`split_sessions.json`）。

### pressure_toolkit / FPP-Net / PoseTransOpt

```bash
# 生成 pressure_toolkit/FPP-Net 共用的 MMVP 31×11 过程表示
python3 AnysoleWorkspace/tool/build_shared.py --mmvp-only
# 生成 FPP-Net 模型输入树（RGB 链接 + adapter manifest）
python3 AnysoleWorkspace/tool/adapters/FPP-Net/build_inputs.py --split all
# 提取 FPP-Net HALPE-26 关键点
python3 AnysoleWorkspace/tool/adapters/FPP-Net/run_rtmpose.py --split all
# 生成 FPP-Net temporal-5 split 和 subject metadata
python3 AnysoleWorkspace/tool/adapters/FPP-Net/build_metadata.py --force
# 生成 PoseTransOpt 模型输入树（join manifest 为对齐权威）
python3 AnysoleWorkspace/tool/adapters/PoseTransOpt/build_inputs.py --split all
# 生成 CLIFF 单人视觉初始化
python3 AnysoleWorkspace/tool/adapters/PoseTransOpt/run_cliff.py --split all --backbone hr48
# 生成 pressure_toolkit 上游原生目录契约
python3 AnysoleWorkspace/tool/adapters/pressure_toolkit/build_inputs.py --split all
# 用 DepthPro 生成深度图（直接写入 pressure_toolkit v1 树）
python3 Baselines/pressure_tookit/data_prep/rgb2depth.py \
  --images-root model-input://pressure_toolkit/v1/images --batch-size 4 --skip-existing
```

输出分别位于 `model_inputs/FPP-Net/adapter_v1/`、`model_inputs/PoseTransOpt/adapter_v1/`、`model_inputs/pressure_toolkit/v1/` 和 `shared/frontends/`；31×11 压力统一读取 `shared/representations/tactile/mmvp_31x11/v1/`。

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
- `pressure_48.npz`：`left48/right48:float32[T,48]`，记录 `t_us`、`valid_mask`、`fake`、左右源 CSV URI 与 hash。
- MMVP 表示每帧为 `float32(2,31,11)`，只生成一份，供 pressure_toolkit/FPP-Net 共同只读。
- 每个生成根和 session/representation 都写 `artifact.json`。
- 正式 checkpoint、prediction、metric 只能写 `results/`。

阶段报告和逐 session provenance 位于 `reports/`；`assets/third_party/` 只保存依赖和固定资产，不承担数据事实或模型输出职责。
