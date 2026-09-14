# Results Display

集中存放「结果可视化（Test1）」「参数对照（Test2）」「轨迹可视化（Test3）」「鞋垫漂移（Test4）」「接触检测（Test5）」等实验的输出。所有源码脚本位于 `script/`（唯一被 git 跟踪的目录），生成物全部落在对应实验目录下，不再散落顶层。

可用环境变量 `ANYSOLE_RESULTSDISPLAY` 覆盖本根目录；`ANYSOLE_RESULTS` 覆盖 `results/` 数据根；`ANYSOLE_WORKSPACE` 覆盖 `AnysoleWorkspace/`。

## 统一 CLI 约定

所有脚本共享同一套参数语义（由 `script/cli_common.py` 统一注册）：

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--session S1,S2` | 会话过滤（逗号分隔）；空 = 取 split 列全部会话 | 空（全量） |
| `--split-csv PATH` | split 文件 | `AnysoleWorkspace/splits/default/splits.csv` |
| `--split` | 读取的 split 列 | `test` |
| `--modal m1,m2` | AnySole 模型（anysole/traj/evaluate_compare） | `anysolev1,anysolev1_insole_drift` |
| `--config-id A,B` | 生成配置（AnySole 系） | `VT2M,V2M,T2M` |
| `--gen {gif,mp4}` | **输出动画格式（单选）** | `gif` |
| `--fps F` | 输出动画 FPS | `40.0`；gt_bvh/step2motion 为 `0.0` = 自动（按源 BVH 帧率，保持动捕动画播放速度） |
| `--stride N` | 每隔 N 帧采样渲染 | `2`；gt_bvh 为 `4`（原始 BVH 帧数大，纯渲染成本选择） |
| `--max-frames N` | 最多渲染帧数 | `0` = 全部帧 |
| `--force` | 重建覆盖已有产物（默认存在且非空则跳过） | 关 |
| `--out-dir PATH` | 输出目录 | 各实验目录 |

- 路径统一支持 `results://`、`workspace://`、`display://` scheme。
- `--gen` 只控制动画格式；静态图/表格（Test3 png、Test5 csv/png 等）始终生成。traj 脚本可用 `--no-png` 关闭静态图。
- 默认不再同时产出 gif+mp4：`--gen gif`（默认）只出 gif，`--gen mp4` 只出 mp4，需要两种就分别跑一次。

## 目录结构

动画产物一律位于 `{gif,mp4}/` 子目录；静态图/表格/日志平铺于实验目录根：

```
results_display/
├── script/                      # 源码（git 跟踪）
│   ├── cli_common.py            # 共享 CLI 约定 / 路径解析 / 媒体写入
│   ├── bvh_aligner_pose.py      # 共享 BVH 解析库
│   ├── visualize_motionpro.py   # Test1：MotionPRO 动画
│   ├── visualize_step2motion.py # Test1：Step2Motion 动画
│   ├── visualize_anysole.py     # Test1：AnySole 动画（主模型 + 消融）
│   ├── visualize_gt_bvh.py      # Test1：GT 原始 BVH 渲染
│   ├── visualize_anysole_traj.py# Test3：AnySole 轨迹对比动画 + 静态图
│   ├── test4_insole_drift.py    # Test4：鞋垫漂移补偿器测试
│   ├── test5_contact.py         # Test5：接触标签动画 + 阈值分析
│   └── evaluate_compare.py      # Test2：跨模型/参数对照评估
├── Test1_visualization/         # 实验①输出
│   ├── MotionPRO/<ckpt_tag>/{gif,mp4}/<session>_compare.{gif,mp4}
│   ├── Step2Motion/gait_model/{gif,mp4}/<session>_compare.{gif,mp4}
│   ├── AnySole/<modal>/<config>/{gif,mp4}/<session>_<config>_compare.{gif,mp4}
│   └── gt/<session>/<bvh_stem>/
│       ├── skeleton_zup.npz             # 数据文件（位置不变，不受 --gen 影响）
│       └── {gif,mp4}/skeleton_zup.{gif,mp4}
├── Test2_comparison/            # 实验②输出（只出指标，不做动画）
│   ├── comparison_per_session.csv   # 逐会话明细
│   ├── comparison_summary.csv       # 精简表：模型 × 指标
│   ├── comparison_summary.png       # 精简表的表格图
│   └── evaluation.log               # 运行信息 + 各模型 checkpoint
├── Test3_trajectory/            # 实验③输出（轨迹可视化）
│   └── AnySole/<modal>/<config>/{gif,mp4,png}/
├── Test4_insole_drift/          # 实验④输出
│   ├── {gif,mp4}/<session>_drift_compare.{gif,mp4}   # 左=原始触觉 右=补偿后
│   └── theta_summary.csv                           # 逐会话漂移统计（根目录）
└── Test5_contact/               # 实验⑤输出：接触标签动画 + 阈值分析
    ├── {gif,mp4}/<session>_contact.{gif,mp4}   # 触觉热力图 + GT BVH + 接触指示动画
    ├── contact_summary.csv                     # 逐会话接触率/压力和统计（根目录）
    └── threshold_analysis.png                  # 压力和直方图 + 接触率-阈值曲线（根目录）
```

## Test1 结果可视化

| 脚本 | 读取方式 | 输出 |
| --- | --- | --- |
| `visualize_motionpro.py` | `--checkpoint` 指定模型（支持 `results://`），空 = 自动扫描 `results/MotionPRO/checkpoints/` | `Test1_visualization/MotionPRO/<ckpt_tag>/` |
| `visualize_step2motion.py` | 位置参数给定生成 BVH（支持 `results://`），空 = 自动扫描 test 集导出 | `Test1_visualization/Step2Motion/gait_model/` |
| `visualize_anysole.py` | 自动扫描 `results/AnySole/<modal>/predictions/eval_bvh/`（不重新推理） | `Test1_visualization/AnySole/<modal>/<config>/` |
| `visualize_gt_bvh.py` | 默认全部 test 会话的原始动捕 BVH；`--bvh` 指定单个文件 | `Test1_visualization/gt/` |

```bash
conda activate touch_gait

# 默认：全量 test split（无参 = 全量扫描）
python results_display/script/visualize_motionpro.py
python results_display/script/visualize_step2motion.py
python results_display/script/visualize_anysole.py
python results_display/script/visualize_gt_bvh.py

# 单会话 / 指定模型 / 指定格式（所有脚本通用）
python results_display/script/visualize_motionpro.py --session S14103 \
  --checkpoint results://MotionPRO/checkpoints/imagepressure2smpl/init/5e-05/imagepressure2smpl_best.pth --gen mp4
python results_display/script/visualize_anysole.py --modal anysolev1 --config-id VT2M --session S7013
python results_display/script/visualize_gt_bvh.py --bvh <单个.bvh> --gen gif
```

## Test2 参数对照

`evaluate_compare.py` 对 AnySole（modal × config 组合）、MotionPRO、Step2Motion 统一计算 MPJPE / PA-MPJPE / W-MPJPE / WAMPJPE / RTE / Accel / Jitter 指标（纯指标，无动画，无 `--gen`）。

```bash
python results_display/script/evaluate_compare.py

# 或完全自动扫描 results/ 下的自包含模型目录
python results_display/script/evaluate_compare.py --auto-scan
```

模型清单默认由脚本扫描 `results/` 自动构建；也可用 `--models-config <yaml|json>` 指定固定清单。

## Test3 轨迹可视化

`visualize_anysole_traj.py` 可视化 AnySole 生成的根轨迹（真实轨迹 vs 预测轨迹）：
单面板 3D 空间动画（gif/mp4）+ 每 session 一张静态图（png，3D 斜视图 / 俯视 / 高度曲线）。

**数据来源**：`anysole.eval --write-bvh` 现在会在 BVH 旁同时写出
`<session>_<config>_traj.npz`（`pred_trans_world` / `gt_trans_world`，与 `traj_ATE` 指标严格同源）。
Test3 只读这些 npz，不重新推理；请先重跑 eval 刷新产物：

```bash
conda activate touch_gait

# 重新 eval（同时刷新 BVH 与 traj npz，两者与 metrics 保持一致）
python -m anysole.eval --ckpt results/AnySole/anysolev1/checkpoints/ckpt_last.pt \
  --modal anysolev1 --config-id VT2M,V2M,T2M --split test \
  --write-bvh results/AnySole/anysolev1/predictions/eval_bvh \
  --metrics-out results/AnySole/anysolev1/metrics/test.json
python -m anysole.eval --ckpt results/AnySole/anysolev1_insole_drift/checkpoints/ckpt_last.pt \
  --modal anysolev1_insole_drift --config-id VT2M,V2M,T2M --split test \
  --write-bvh results/AnySole/anysolev1_insole_drift/predictions/eval_bvh \
  --metrics-out results/AnySole/anysolev1_insole_drift/metrics/test.json

# Test3 渲染（默认 主模型 + 消融 × 全部 config × test split）
python results_display/script/visualize_anysole_traj.py
# 单 session / 单 config / 只要动画
python results_display/script/visualize_anysole_traj.py --session S7013 --config-id VT2M --no-png
```

输出：`Test3_trajectory/AnySole/<modal>/<config>/{gif,mp4,png}/<session>_<config>_traj.{gif,mp4,png}`。

> 注意：eval 生成的 BVH 与 traj npz 必须与 checkpoint 配套。若重新训练了模型，
> 旧的 `eval_bvh/` 产物会与新 metrics 不一致（轨迹头更新后曾出现整窗漂移的过期 BVH），
> 需重跑上述 eval 命令再渲染 Test1/Test3。

## Test4 鞋垫偏移补偿模块

```bash
conda activate touch_gait
# 全部 test split
python results_display/script/test4_insole_drift.py
# 单 session
python results_display/script/test4_insole_drift.py --session S7013
```

## Test5 接触检测

`test5_contact.py` 渲染 GT BVH 骨架 + 触觉鞋垫热力图，并叠加接触指示
（**红色=接触、绿色=无接触**）。接触标签来自 `contact.npy` 第 6/7 列（左/右足），
其口径为「48 格鞋垫 CSV 逐帧压力和 > 100」（`prepare_sequences.py:contact_from_insoles`，阈值 `CONTACT_SUM_THRESH`）。
候选阈值分析基于 48 格 CSV 压力和计算。
产出：`{gif,mp4}/<session>_contact.{gif,mp4}`、`contact_summary.csv`、`threshold_analysis.png`。

```bash
conda activate touch_gait

python results_display/script/test5_contact.py                              # 全部 test split
python results_display/script/test5_contact.py --session S10103 --max-frames 100   # 单 session 冒烟
python results_display/script/test5_contact.py --force                      # 重建动画与分析
```

## 历史迁移说明

- 动画统一进入各 Test 目录的 `{gif,mp4}/` 子目录；对照实验进入 `Test2_comparison/`；过时配置删除（默认自动扫描 results/）。
- 旧平铺产物（gt/Test4/Test5 早期直接放在实验目录根的 gif/mp4）**保留原地不迁移**；新渲染写入新目录。首次按新规则全量重跑时，skip 检测读不到旧路径，会重新渲染一次。
- 旧参数 `--no-gif` / `--no-mp4` / `--output-dir` 已移除，统一使用 `--gen {gif,mp4}` 与 `--out-dir`。
