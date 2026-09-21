# AnysoleWorkspace

工作区路径、数据协议与数据准备 CLI 手册。

## 目录索引

`sources/PressureWasher/` 只保存触觉清洗、审核及运行产物；`tool/` 保存全部数据构建代码。

```text
AnysoleWorkspace/
├── sources/raw/                         # 原始触觉 + 视频 + 采集文件（软链）
├── sources/smpl/                        # SMPL-NPZ 外部归档（软链，不复制原文件）
├── sources/published/                   # 发布数据（软链）
├── sources/calibration_artifacts/       # 标定原始文件（软链）
├── sources/PressureWasher/outputs/      # 触觉清洗产物 stats/reconstructed/fake_marked/encoded
├── derived/MotionPRO/sequences/cam3/    # MotionPRO 序列（训练读这里）
├── derived/AnySole/hrnet_cache/cam3/    # HRNet + bbox 特征
├── derived/Step2Motion/gait/            # Step2Motion 数据
├── dependencies/                        # 模型权重和依赖
├── calibration/                         # 标定摘要（软链） + insole_templates.json
├── manifests/                           # session_manifest + 数据协议 + 质检报告
├── splits/default/                      # splits.csv（训练划分）
└── tool/                                # 数据构建代码（唯一入口，见下方 CLI 手册）
```

`sources/raw` 按记录时间展开：`{日期序列}/mocap_ori_{bvh,c3d,cmr,trc}/` + `S{受试者}/rec.../`（触觉 CSV、相机 JPG、视频）。

## 数据协议要点

- AnySole 只读取 SMPL：默认读 `/data/lizhe/projects/Tactile/Mocap/{0804,0807,0808,0810}/`
  下的 `motion_neutral_smpl.npz`（即 `sources/smpl/`）；不读取或导出 BVH，原始 BVH
  仅供 Step2Motion 基线。环境变量 `ANYSOLE_SMPL_ROOTS` 可覆盖根目录。
- 触觉单帧 = 双脚 `(2,48)`（CSV 列 `1..48`），AnySole 拼接为 `T_raw (T,96)`；
  `encode` 只生成 `fake_mask_left/right.npy`，不保存压力值。
- `derived/MotionPRO/.../pressure.npz` 是 `(T,160,120)` 栅格图。
- `manifests/session_manifest.csv` 的 `eligible_anysole=1` = 同时具备左右触觉 CSV、
  视频、SMPL 与 BVH；`splits.csv` 只由这些 session 生成。

## 数据准备 CLI 手册

一条中间数据一条命令。**先 init/doctor 后按需执行**；常用参数见各脚本 `--help`。

| # | 中间数据 | 命令 | 产物 |
| --- | --- | --- | --- |
| 0 | workspace 初始化/体检 | `python AnysoleWorkspace/tool/workspace.py init` / `doctor` / `relink` | sources 软链、本地目录、calibration 软链 |
| 1 | session manifest + splits | `python AnysoleWorkspace/tool/build_manifest.py --fps 40 --camera cam3` | `manifests/session_manifest.csv|jsonl`、`splits/default/splits.csv` |
| 2 | 数据质检 | `python AnysoleWorkspace/tool/check_data_quality.py --manifest manifests/session_manifest.jsonl`；`check_splits.py` 同参 | `manifests/data_quality_report.md` |
| 3 | 触觉清洗（4 stage） | `python AnysoleWorkspace/tool/pressure_washer/run.py {inspect,reconstruct,mark-fake,encode} [--skip-existing\|--force]` | `sources/PressureWasher/outputs/{stats,reconstructed,fake_marked,encoded}` |
| 4 | HRNet 特征缓存 | `CUDA_VISIBLE_DEVICES=N python AnysoleWorkspace/tool/generate_hrnet_cache.py --cam-id 3 [--session S...]` | `derived/AnySole/hrnet_cache/cam3/*.pt` |
| 5 | 接触 npz | `python AnysoleWorkspace/tool/contact_labels.py [--methods bvh_h,joint_and,...] [--session S...]` | 序列目录下 `contact_<method>.npy` |
| 6 | 鞋垫模板 | `python -m anysole.ablations.insole_drift.build_templates --manifest manifests/session_manifest.csv --out calibration/insole_templates.json` | `calibration/insole_templates.json` |
| 7 | MotionPRO 序列 | 历史数据，生产脚本随 Baselines 归档（`Baselines_backup/`） | `derived/MotionPRO/sequences/cam3/` |
| 8 | Step2Motion gait 数据集 | 迁移自 Baselines（见 `tool/workspace.py` MIGRATIONS） | `derived/Step2Motion/gait/*.pt` |
| 9 | pressure_tookit 深度适配 | 待写（Agent_06 未合入） | `derived/pressure_tookit/` |

顺序依赖：0 → 1 → 2（质检读 manifest）；3/4/5/6 互不依赖，均以 0 的前置为准；
训练侧（anysole）消费 1/4/5/6 的产物。

## 检查

```bash
python AnysoleWorkspace/tool/workspace.py doctor      # 软链/标定/splits/必需权重的全量体检
python AnysoleWorkspace/tool/check_splits.py --manifest AnysoleWorkspace/manifests/session_manifest.jsonl
```
