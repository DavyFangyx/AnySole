# Results Display

集中存放实验产物。实验分两大部分，互不耦合：

- **第一部分：数据检验（D_TestN）** —— 处理**原始数据**（对应 AnysoleWorkspace）：
  原始 GT 的可视化、自检、标签判定与协议探针。
- **第二部分：结果检验（R_TestN）** —— 处理**实验结果**（对应 results）：
  模型输出的可视化、参数对照、轨迹与专项消融。

实验代码位于 `script/`（唯一被 git 跟踪的目录），生成物落在对应的实验目录。
本文已按 D/R 编号组织；**脚本与产物目录的改名在 P1-4 落盘**，当前磁盘上脚本
仍为旧名（命令照抄可用），新旧对应见下方对照表。

可用环境变量 `ANYSOLE_RESULTSDISPLAY` 覆盖本根目录。

## 编号对照（旧 TestN ↔ 新编号）

| 新编号 | 旧 | 脚本（现名 → P1-4 新名） |
| --- | --- | --- |
| D_Test1 原始数据可视化 | Test1 的 GT/输入部分 | `visualize_gt_bvh.py` → `d_test1_data_viz.py`（BVH/SMPL GT 渲染） |
| D_Test2 数据集自检 | Test6 | `test6_dataset_check.py` → `d_test2_dataset_check.py` |
| D_Test3 接触检测 | Test5 | `contact_methods.py` → `AnysoleWorkspace/tool/contact_labels.py`；`test5_contact.py` → `d_test3_contact.py` |
| D_Test4 SMPL 协议预检 | F4 预检 | `probe_f4_smpl24_preflight.py` + `probe_smpl_*` → `d_test4_smpl_protocol.py` |
| R_Test1 模型可视化 | Test1 的模型部分 | `visualize_{motionpro,step2motion,anysole}.py` → `r_test1_visualize.py` |
| R_Test2 参数对照 | Test2 | `evaluate_compare.py` → `r_test2_compare.py` |
| R_Test3 轨迹可视化 | Test3 | `visualize_anysole_traj.py` → `r_test3_traj.py` |
| R_Test4 鞋垫偏移补偿 | Test4 | `test4_insole_drift.py` → `r_test4_insole_drift.py` |
| R_Test5 均值姿态推理 | Test7 | `test7_mean_pose_infer.py` → `r_test5_mean_pose.py` |
| R_Test6 输入-输出消融 | Test8 | `test8_input_ablation.py` → `r_test6_input_ablation.py` |
| R_Test7 legacy 扩散诊断 | Test9 / Test9.1 | `test9_overfit.py` + `test9_1_sampler_checks.py` → `r_test7_legacy_diffusion.py` |
| R_Test8 触觉生成 | Test10 | `test10_tgen.py` → `r_test8_v2t.py` |
| R_Test9 τ 训练区间 | Test11 | `test11_tau_regime.py` → `r_test9_tau_regime.py` |
| R_Test10 pose head 系列 | Test12 | `test12_*.py` ×6 → `r_test10_pose_head.py` |

## 目录结构

```text
results_display/                        # 纯产物目录（实验代码在 script/）
├── script/                             # 源码（git 跟踪；P1-4 迁 experiments/）
│   ├── bvh_aligner_pose.py             # 共享 BVH 解析库
│   ├── cli_common.py                   # 共享 CLI 公共件
│   ├── motion_io.py                    # 统一动作读取器（格式自动检测）
│   ├── render_common.py                # 共享渲染公共件
│   ├── ── 数据检验（D_TestN）──
│   │   ├── visualize_gt_bvh.py         # D_Test1：GT 原始渲染
│   │   ├── test6_dataset_check.py      # D_Test2：数据集自检
│   │   ├── contact_methods.py          # D_Test3：多方案接触标签生成（→ tool/contact_labels.py）
│   │   ├── test5_contact.py            # D_Test3：接触标签动画 + 阈值分析
│   │   ├── probe_f4_smpl24_preflight.py# D_Test4：真实 SMPL-24 batch/warm-start/loss/backward
│   │   └── probe_smpl_*.py             # D_Test4：SMPL 写出/读回、轴系、协议契约探针
│   └── ── 结果检验（R_TestN）──
│       ├── visualize_motionpro.py      # R_Test1：MotionPRO 动画
│       ├── visualize_step2motion.py    # R_Test1：Step2Motion 动画
│       ├── visualize_anysole.py        # R_Test1：AnySole 动画（自动识别 SMPL NPZ/BVH）
│       ├── evaluate_compare.py         # R_Test2：跨模型/参数对照评估
│       ├── visualize_anysole_traj.py   # R_Test3：轨迹对比动画 + 静态图
│       ├── test4_insole_drift.py       # R_Test4：鞋垫漂移补偿器测试
│       ├── test7_mean_pose_infer.py    # R_Test5：均值/GT 姿态推理
│       ├── test8_input_ablation.py     # R_Test6：输入-输出相关性消融
│       ├── test9_overfit.py            # R_Test7：legacy diffusion 过拟合
│       ├── test9_1_sampler_checks.py   # R_Test7：τ0 vs DDIM 五项定位
│       ├── test10_tgen.py              # R_Test8：V2T 触觉生成
│       ├── test11_tau_regime.py        # R_Test9：τ 训练区间消融
│       └── test12_*.py                 # R_Test10：pose head 系列探针
├── ── 数据检验产物（P1-4 起 data/d_testN/；现为 TestN_* 目录）──
│   ├── Test5_contact/                  # D_Test3 输出（按方案分目录）
│   └── Test6_dataset_check/            # D_Test2 输出（自检 JSON + 均值姿态 npz）
└── ── 结果检验产物（P1-4 起 result/r_testN/；现为 TestN_* 目录）──
    ├── Test1_visualization/            # R_Test1 输出
    ├── Test2_comparison/               # R_Test2 输出（只出指标，不做动画）
    ├── Test3_trajectory/               # R_Test3 输出
    ├── Test4_insole_drift/             # R_Test4 输出
    ├── Test7_mean_pose/                # R_Test5 输出
    ├── Test8_input_ablation/           # R_Test6 输出
    ├── Test9_overfit/                  # R_Test7 输出（legacy）
    ├── Test9_1_sampler/                # R_Test7 输出（legacy）
    ├── Test10_tgen/                    # R_Test8 输出
    └── Test11_tau_regime/              # R_Test9 输出
```

## 时间对齐约定（动捕 ↔ 触觉/视频）

视频、触觉与动捕之间存在时间偏差与漂移。偏差来自各设备独立起录：每个 session 的
人工复核表 `AlignReviews_csv/<session>.csv` 记录常量偏移 `偏移量(s)`（= 视觉时间 − 动捕时间）；
漂移则由触觉/视频时间轴按逐帧时钟重建（触觉 `t_us` 时间戳、视频文件名时钟）吸收。
所有动捕与触觉同帧对比的脚本，GT motion 都按同一补偿方式
重采样到 40 Hz 会话网格：`t_mocap = t_grid − offset_s`，其中 `t_grid = visual_start_s + n/40`

## 模型协议约定（SMPL / BVH 双协议）

展示与对照层（D_Test1 与 R_Test1/2/3）按文件格式自动检测协议，无协议 flag：

| 协议 | 模型 | 产物约定 | 检测规则 |
| --- | --- | --- | --- |
| SMPL-24 | AnySole | `predictions/eval_motion/<session>_<config>.npz`（原生 SMPL-24，内嵌 `pred_pelvis_trans/gt_pelvis_trans`） | npz keys 含 `poses/root_orient/pose_body` → SMPL |
| BVH-23 | Step2Motion 等 baseline | `predictions/<run>/<session>_gen.bvh`（Skeleton3） | `.bvh` 后缀 → bvh23 |

- GT 读取统一走 `motion_io.load_session_gt`：优先会话 SMPL，缺失时回退 BVH（仅限展示层）。
- R_Test2 的跨协议指标统一映射到 common19 语义关节（`evaluate_compare.py` 的 `protocol_gt`）。
- 归档的 `*_backup*` 目录不参与任何自动扫描。

# 第一部分：数据检验（D_TestN）

## D_Test1 原始数据可视化（BVH / SMPL）

直接渲染 AnysoleWorkspace 的原始 GT 动捕数据，不涉及任何模型输出：

| 脚本 | 作用 |
| --- | --- |
| `visualize_gt_bvh.py` | 原始 GT 动捕 BVH 的骨架渲染（纯 BVH 可视化） |
| （SMPL GT 渲染，P1-4 并入 `d_test1_data_viz.py`） | 原始 SMPL-24 GT 的骨架渲染 |

```bash
conda activate touch_gait

# BVH 原始数据可视化
python results_display/script/visualize_gt_bvh.py
python results_display/script/visualize_gt_bvh.py --bvh <单个.bvh>
```

## D_Test2 数据集自检（GT 自洽 + 均值姿态基线）

`test6_dataset_check.py` 只读数据集、不加载任何模型，回答两个问题：

- **A. GT 自洽**：FK(GT 6D, GT offsets) 必须能逐关节还原数据集内的 `kp_gt`。
  A1 numpy FK（数据集构建路径）/ A2 torch FK（train/eval 路径）交叉验证、A3 单位与几何量程、A4 SMPL axis-angle↔6D 回环、
  A5 骨骼模板一致性诊断（按受试者分组比较跨动作/采样的 rest offsets；会话 ID 为
  `S<人><动作><采样>`）。源 MoSh 文件是逐 session 独立拟合，因而 A5 不是 loader
  正确性的硬断言，而是检查"同一受试者能否视作同一 shape"的数据集设计前提。
  本管线当前使用每个 SMPL 文件自己的 betas，通过 neutral SMPL
  shapedirs/J-regressor 生成 shape-dependent rest joints 与 offsets。全量检查中 A5 实际
  FAIL：同一受试者最大 offset 分量跨度为 18.02–38.29mm。这意味着现任务是"给定每段
  GT shape 的 motion prediction"，不是同时生成 shape；不能把 A5 失败误报为 loader
  错误，也不能在未重建所有目标与基线前擅自改用平均 betas。A1–A4 仍是 FK、单位和
  旋转协议的硬验收项。
- **B. 均值姿态基线**：拿训练集均值姿态当预测算 MPJPE。B1 = 均值姿态 + GT 根轨迹（只差姿态）；
  B2 = 完全静态均值姿态。若模型 MPJPE ≈ B1 → 条件被无视；若 B1 明显低于模型 MPJPE →
  模型比"啥也不干"还差，更像 bug（脚本会自动与 `anysolev1_joint_and/metrics/test.json` 对比）。

产出（`Test6_dataset_check/`）：`fk_selfcheck.json`、`mean_baseline.json`、
`mean_pose.npz`、`input_means.npz`（后两者被 R_Test5/R_Test6 复用为"均值输入"）。

```bash
conda activate touch_gait
python results_display/script/test6_dataset_check.py
# 单 session 冒烟
python results_display/script/test6_dataset_check.py --session S10103 --limit-sessions 1 --max-windows 16
```

## D_Test3 接触检测

`test5_contact.py` 渲染自动识别的 GT motion 骨架 + 触觉鞋垫热力图，并叠加接触指示
（**红色=接触、绿色=无接触**）：鞋垫描边与徽章、骨架足部关节着色、底部整段接触时间轴。
每帧同步显示 48 格压力和值。

背景：原 `contact.npy` 标签口径为「48 格鞋垫 CSV 逐帧压力和 > 100」
（`prepare_sequences.py:contact_from_insoles`，阈值 `CONTACT_SUM_THRESH`），但大量鞋垫在脚
离地后压力不归零（如 S11023 左脚摆动相压力和最低 653），曾导致大量运动学离地帧被错标为接触。
为对比候选修复方案，Test5 改为**按方案分子目录**：`contact_methods.py` 为每个方案生成
`contact_<method>.npy`（与 `contact.npy` 同目录、同 10 列格式），`test5_contact.py --methods ...`
渲染判定动画。标签生成器属数据构建代码，P1-3 起迁 `AnysoleWorkspace/tool/contact_labels.py`。

| 方法 | 判据 |
| --- | --- |
| `bvh_soft` | 直接导出 BVH 运动学，与 `losses.soft_contact_from_keypoints` 同构（高<5cm ∧ 速<0.2m/s） |
| `bvh_h` | 直接导出 BVH 脚部/ToeBase 相对地面高度 h<5cm 判接触，4/6cm 滞回 |
| `tactile_abs` | 触觉绝对阈值：48 格压力和 > 100（原 contact.npy 口径） |
| `tactile_rel` | 触觉相对阈值：min + 0.25·(max−min) |
| `tactile_gmm` | 触觉自适应：log1p(压力和) 双峰 GMM 谷值阈值，单峰判全接触 |
| `pat_offset` | 患者级偏移补偿：摆动相压力和的中位数（BVH 自动选模版）+ 100 |
| `joint_or` | air union / contact intersection（任一来源判离地即离地） |
| `joint_and` | air intersection / contact union（两来源都判离地才离地） |

```bash
conda activate touch_gait
# 1) 生成全部方案的标签（全部 session）+ 对比报告
python results_display/script/contact_methods.py
# 2) 渲染动画（默认全部方案 × test split；标签缺失时自动补齐）
python results_display/script/test5_contact.py --methods bvh_soft --gen mp4 --session S11113
# 单方案 / 单 session 冒烟
python results_display/script/test5_contact.py --methods bvh_h --session S11023
```

产出（`Test5_contact/`）：每个方案一个子目录（`<method>/`）：
`gif|mp4/<session>_contact.gif/.mp4`（触觉热力图 + GT motion + 该方案接触指示动画）、
`contact_summary.csv`（逐会话接触率/压力和统计）、`diff_vs_bvh_h.csv`
（该方案 vs 直接导出 BVH 高度参考的差异区间清单：会话/脚/起止时间s/方向/区间内高度·速度·压力和）、
`diff_by_subject.png` / `diff_by_sequence.png`（差异 × 用户/动作序列分布）。
另有与方案无关的 `threshold_analysis.png`（压力和直方图 + 接触率-阈值曲线）、
`threshold_analysis.csv`（逐帧左右脚 48 格压力和）、`threshold_analysis_hist.csv`
（64 分箱分布）与 `comparison.{csv,png}`（各方案 vs bvh_h 参考的一致率/假接触率/假离地率）。

## D_Test4 SMPL-24 协议预检

F4/AnySoleV2 请使用真实训练路径探针；它会检查 24×6D、SMPL parent tree、shape-dependent
offset、`joint_and` sidecar、旧 23/138 warm-start 隔离、各 loss 分量、梯度裁剪以及固定 batch
优化。默认 `lambda_con=0`，所以 contact 在当前命令中仅参与评估，不参与反向传播。

```bash
PYTHONPATH=. /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
  results_display/script/probe_f4_smpl24_preflight.py \
  --device cpu --steps 20 --grad-clip 5.0 --lr 1e-4

PYTHONPATH=. /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
  results_display/script/probe_smpl_export_roundtrip.py
```

# 第二部分：结果检验（R_TestN）

## R_Test1 模型可视化

| 脚本 | 作用 | 默认输出 |
| --- | --- | --- |
| `visualize_motionpro.py` | MotionPRO 触觉输入/预测/GT 三栏对比动画 | `Test1_visualization/MotionPRO/<checkpoint tag>/` |
| `visualize_step2motion.py` | Step2Motion 足底压力/生成 BVH/GT 对比动画 | `Test1_visualization/Step2Motion/gait_model/` |
| `visualize_anysole.py` | AnySole 主模型与消融（足底压力/预测动作/SMPL GT；自动检测格式） | `Test1_visualization/AnySole/<modal>/<config>/` |

```bash
conda activate touch_gait

# motionpro 可视化
python results_display/script/visualize_motionpro.py
--session S14103
--checkpoint results://MotionPRO/checkpoints/imagepressure2smpl/init/5e-05/imagepressure2smpl_best.pth

# step2motion 可视化
python results_display/script/visualize_step2motion.py
--self-test

# 默认 主模型 + 消融，全部 config，全部concat
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal anysolev1 --gen gif --contact-method joint_and
--modal anysolev1,anysolev1_insole_drift
--contact-method bvh_h,bvh_soft,tactile_abs,pat_offset,joint_and
--config-id VT2M

--modal E3 --contact-method normdiff
```

## R_Test2 参数对照

`evaluate_compare.py` 对 AnySole（modal × config 组合）、MotionPRO、Step2Motion 统一计算 MPJPE / PA-MPJPE / W-MPJPE / WAMPJPE / RTE / Accel / Jitter 指标。
产出（`Test2_comparison/`）：
`comparison_per_session.csv`（逐会话明细）、`comparison_summary.csv`（精简表：仅模型 × 指标）、
`comparison_summary.png`（精简表的表格图）、`evaluation.log`（运行信息 + 各模型 checkpoint）。

```bash
python results_display/script/evaluate_compare.py \
  --manifest AnysoleWorkspace/manifests/session_manifest.csv \
  --modal anysolev1,anysolev1_insole_drift \
  --config-id VT2M,V2M,T2M \
  --split test

# 或者完全自动扫描 results/ 下的自包含模型目录
python results_display/script/evaluate_compare.py --auto-scan
```

模型清单默认由脚本扫描 `results/` 自动构建；也可用 `--models-config <yaml|json>` 指定固定清单。

## R_Test3 轨迹可视化

`visualize_anysole_traj.py` 可视化预测根轨迹 vs 真实轨迹（**SMPL/BVH 协议自动检测**）：
单面板 3D 空间动画（gif/mp4）+ 每 session 一张静态图（png，3D 斜视图 / 俯视 / 高度曲线）。
动画中 GT 整段显示（橙）、预测轨迹随帧生长（蓝），窗口边界用小点标记（每窗口在 GT 锚点处重新锚定），footer 实时显示当前帧 ATE 与整段 ATE。

**数据来源（双协议）**：

- SMPL 协议模型（AnySole）：`anysole.eval` 导出的原生 SMPL motion NPZ
  （`predictions/eval_motion/<session>_<config>.npz`）内嵌轨迹字段
  `pred_pelvis_trans` / `gt_pelvis_trans`（与 `traj_ATE` 指标严格同源）。
- BVH 协议模型（Step2Motion 等）：`predictions/<run>/<session>_gen.bvh`，
  预测轨迹取 BVH 根关节路径，GT 用 `motion_io.load_session_gt`
  （SMPL 优先、BVH 回退）的根关节。

R_Test3 只读这些文件，不重新推理；请先重跑 eval 刷新产物：

```bash
conda activate touch_gait

# 重新 eval（自动推导 ckpt/metrics 路径，同时刷新标准 SMPL motion NPZ）
python -m anysole.eval \
  --modal anysolev1 \
  --contact-method tactile_abs

# R_Test3 渲染（默认 主模型 + 消融 × 全部 config × test split）
python results_display/script/visualize_anysole_traj.py --modal anysolev1 --gen gif --contact-method joint_and
# 单 session / 单 config
python results_display/script/visualize_anysole_traj.py --session S7013 --config-id VT2M
# baseline 模型（扫描 results/ 下自包含模型目录，排除 *_backup*）
python results_display/script/visualize_anysole_traj.py --auto
```

输出：`Test3_trajectory/AnySole/<modal>/<config>/{gif,mp4,png}/<session>_<config>_traj.{gif,mp4,png}`；
baseline 模型输出在 `Test3_trajectory/<model>/gen/`。

> 注意：eval 生成的 motion NPZ 必须与 checkpoint 配套。若重新训练了模型，
> 旧的 `eval_motion/` 产物会与新 metrics 不一致，
> 需重跑上述 eval 命令再渲染 R_Test1/R_Test3。

## R_Test4 鞋垫偏移补偿模块

`test4_insole_drift.py` 验证 insole-drift 补偿器（`anysole.ablations.insole_drift`）
在 `anysolev1_insole_drift` 模态下的行为。

```bash
conda activate touch_gait
# 全部 test split
python results_display/script/test4_insole_drift.py
# 单 session
python results_display/script/test4_insole_drift.py --session S7013
```

## R_Test5 均值姿态推理（模型输出 vs 输入姿态）

`test7_mean_pose_infer.py` 用 `results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt`（`--ckpt` 可换）
把不同输入姿态送进模型（条件固定为真实 VT），导出原生 SMPL-24 NPZ：

| arm | 输入 | 说明 |
| --- | --- | --- |
| `tau0_mean` | 均值姿态，tau=0 | 干净输入直通重建 |
| `tau0_gt` | GT 姿态，tau=0 | 干净 GT 重建（应→0，否则欠训/条件弱） |
| `ddim_noise` | 纯噪声 | 标准 DDIM（与 eval VT2M 同路径，复现性检查） |
| `ddim_mean500` | 均值姿态加噪到 tau=500 再 DDIM | 均值姿态"真正进入"模型做完整推理 |
| `ddim_gt500` | GT 姿态加噪到 tau=500 再 DDIM | 推理初值上界 |

看两件事：`tau0_mean` vs `tau0_gt` 输出距离（输出对输入姿态的敏感度）；`ddim_mean500` vs `ddim_gt500` 的
MPJPE 差（初值对最终输出的影响）。NPZ 写进 `Test7_mean_pose/<session>/`，可用
`visualize_anysole.py`/统一 `motion_io.py` 自动识别并查看骨架动画。

```bash
conda activate touch_gait
# 全部 test，前 4 个 session 出 BVH
python results_display/script/test7_mean_pose_infer.py
python results_display/script/test7_mean_pose_infer.py --session S10103 --export-sessions 1
```

## R_Test6 输入-输出相关性消融（输出是否与输入无关）

`test8_input_ablation.py` 固定模型与初始噪声，置换条件输入（`real` / `zero` / `mean` / `shuffle`，
shuffle = batch 内错位配对，真实输入、错误窗口），各跑一遍 DDIM 与 tau=0 GT 重建，
并对第一个 batch 做梯度检查（∂L/∂V、∂L/∂T vs ∂L/∂x）：

- shuffle MPJPE ≈ real MPJPE → 输入被无视；
- 四种条件输出两两距离 ≈ 0 → 输出与输入无关；
- 输入梯度范数比 ≈ 0 → 条件路径梯度死区。

产出（`Test8_input_ablation/`）：`<session>/<session>_<variant>_<arm>.npz`（原生 SMPL-24 motion）、
`test8_report.json`（各条件 MPJPE + 输出两两距离 + 梯度范数比）。

```bash
conda activate touch_gait
python results_display/script/test8_input_ablation.py
python results_display/script/test8_input_ablation.py --session S10103 --export-sessions 1
```

## R_Test7 Legacy 扩散诊断（diffusion V1 历史诊断，不用于 F4 验收）

`test9_overfit.py` 保留用于旧 diffusion V1 诊断。它不是 AnySoleV2/F4 的可执行验收入口；
当前 F4 必须使用 D_Test4 的 `probe_f4_smpl24_preflight.py`。
训到 loss≈0 → 目标/管线正常，泛化差是欠训/条件弱；
训不下去 → 目标/管线有问题。

产出（`Test9_overfit/`）：`overfit_log.csv`、`overfit_curves.png`（loss 与 MPJPE 曲线）、
`overfit_report.json`（PASS/FAIL 判定），`--save-ckpt` 时额外存 `overfit_ckpt.pt`。

```bash
conda activate touch_gait
python results_display/script/test9_overfit.py
```

Test9 只留 L_pose，λ_con = λ_kp = λ_traj = λ_T = λ_V = 0，batch 降到 4–8 个窗口，lr 扫 {1e-3, 3e-4, 1e-4, 3e-5}，3000 步。CLI指令：
```bash
# 默认：batch=8 窗 × lr{1e-3,3e-4,1e-4,3e-5} × 3000 步
CUDA_VISIBLE_DEVICES=4 python results_display/script/test9_overfit.py

# 自定义扫描（batch 4 窗）
python results_display/script/test9_overfit.py --batch-size 4 --lrs 1e-3,3e-4,1e-4,3e-5 --steps 3000

# 保存每个 lr 的过拟合 checkpoint
python results_display/script/test9_overfit.py --save-ckpt
```

### R_Test7b τ0 vs DDIM 差距定位（五项检查）

Test9 同一 batch 上 τ0 ≈ 58mm 而 DDIM ≈ 220mm，且 DDIM 在训练中单调变差、τ0 同期变好——两个指标反向走，说明除了"模型只输出 g(F)"之外还存在第二个独立故障。`test9_1_sampler_checks.py` 按序跑五项检查：

1. **假模型**：DDIM 每步用 GT x0 替换模型输出，终点必须 ≈0mm（隔离采样器本身）；
2. **同批**：确认 τ0 与 DDIM 读同一 batch 张量，并测 held-out batch 的 τ0 量化"8 窗不泛化"的代价；
3. **τ 网格**：打印实际采样 τ、首末步 ᾱ、每步 ‖x_τ‖ 与训练 q_sample 包络对比；
4. **末步 x0_hat vs τ0 输出**逐元素比 + 每步模型输出轨迹（检验 g(F) 假设与 τ=0 对 x_penultimate 的敏感性）；
5. **6D→SO(3)**：两条评测路径 FK/正交化一致性 + torch/numpy 转换一致性。

检查 1–3 不需要模型（`--skip-train` 秒级）；检查 2b/4/5 默认重训一个 lr（3e-5 × 3000 步，复现 Test9）或 `--ckpt` 加载已存过拟合模型。

产出（`Test9_1_sampler/`）：`train_log.csv`、`check3_tau_grid.csv`、`check4_trace.csv`、
`test91_report.json`（五项检查数值 + PASS/FAIL）、`overfit_ckpt_<lr>.pt`（仅 `--save-ckpt` 时）。

```bash
conda activate touch_gait
# 完整：重训 + 五项检查
CUDA_VISIBLE_DEVICES=5 python results_display/script/test9_1_sampler_checks.py --save-ckpt

# 只跑模型无关的检查 1-3
python results_display/script/test9_1_sampler_checks.py --skip-train

# 复用上次的过拟合 checkpoint，跳过训练
python results_display/script/test9_1_sampler_checks.py --ckpt results_display/Test9_1_sampler/overfit_ckpt_3e-05.pt
```

## R_Test8 触觉生成（V2T）

`test10_tgen.py` 用 V-only 条件（触觉输入置零）跑模型，让辅助头 `pressure_hat` 变成
**视觉→触觉（V2T）生成器**：模型仅凭 HRNet 视觉特征输出 96 格足底压力。触觉头不经过
扩散采样，每个窗口一次 tau=0 前向即可得到确定性的生成触觉，无需 DDIM。

三种条件（与 eval.py 的 `--config-id` 同名）：

| 条件 | 输入 | 意义 |
| --- | --- | --- |
| `VT2M` | 真 V + 真 T | 重建 sanity（上界参考） |
| `V2M` | 真 V + 零 T | **V2T 生成本身（主指标）** |
| `T2M` | 零 V + 真 T | 触觉自重建（输入端 sanity） |

产出（`Test10_tgen/`）：`tgen_summary.csv`（逐会话 × 条件 MAE/RMSE/相关系数）、
`tgen_report.json`（跨会话聚合，`v2t` 字段 = V2M 行汇总）、每个 session 的 96 格逐格 MAE
（`cells/*.npz` + 静态误差图 `*.png`），以及前 `--export-sessions`（默认 4）个
session 的 GT | 生成 | |GT-Gen| 三栏热力图动画（`gif|mp4/<session>_<mode>_tgen.{gif,mp4}`）。

与 `anysole.eval` 的关系：eval 在 `metrics/test.json` 每个模式行输出 `T_mae/T_rmse/T_corr`
（V2M 行即 V2T，JSON 顶层 `v2t` 字段），R_Test8 是该指标的逐格/逐帧可视化解剖。

```bash
conda activate touch_gait
python results_display/script/test10_tgen.py                     # 全部 test split
python results_display/script/test10_tgen.py --session S10103    # 单 session 冒烟
python results_display/script/test10_tgen.py --config-id VT2M,V2M --export-sessions 2
```

> 注意：首版仅支持主模型 `anysolev1`；`anysolev1_insole_drift` 对零触觉输入先过漂移补偿器，
> 生成口径不同，暂不支持。

## R_Test9 τ 训练区间消融（τ≡0 恒等映射 / 低噪声带训练）

`test11_tau_regime.py` 在训练集固定一个 batch（同 Test9 设置：关模态 dropout、VT-only、关模型 dropout、
pose-only 损失，lr 扫 {1e-3, 3e-4, 1e-4, 3e-5} × 3000 步），把训练时的 τ 采样分布切成三种区间，
定位「τ0 MPJPE 压不下去」的根因：

| arm | τ 分布 | 问题 |
| --- | --- | --- |
| `tau0` | τ ≡ 0（x_tau = x0 干净输入） | 任务退化为恒等映射 x0_hat = x_τ。L_pose 收敛不到 ~0 → x_tau → 输出**没有可用带宽**（分区 Linear 投影丢信息 / cross-attn 把残差流冲掉），门控不是主因 |
| `tau0-nocond` | τ ≡ 0 且 V/T 置零（`--no-cond` 追加） | 隔离纯 x_tau → 输出路径，排除条件 F 的补偿，进一步定位瓶颈 |
| `lowband` | τ ~ U(0, 100)（`--tau-band-max` 可调） | 低噪声区训练。τ0 MPJPE 压到**个位数 mm** → 门控 + SNR 是主因，结构改法对症；压不下去 → 同上（结构瓶颈） |
| `full` | τ ~ U(0, 1000) | 全区间对照（与 Test9 同设置），是「压下去」的参照 |

判定：`tau0` PASS = L_pose < 1e-3 且 tau0 MPJPE < 5 mm（任一 lr 达到即可）；`lowband` PASS = tau0
MPJPE < 10 mm。tau0 MPJPE = τ=0 干净 GT 姿态直通重建（GT 轨迹、SMPL-24 FK，mm），与训练侧看板
`val/tau0_mpjpe` 同口径。DDIM 仅在 `full` 臂采样——τ≡0 / 低噪声带训练的模型没见过高噪声区，
DDIM 无意义。

```bash
conda activate touch_gait
python results_display/script/test11_tau_regime.py                     # tau0 + lowband + full × 4 lr × 3000 步
python results_display/script/test11_tau_regime.py --arms tau0 --lrs 1e-3 --steps 1000   # 单臂冒烟
python results_display/script/test11_tau_regime.py --no-cond --save-ckpt                # 追加 tau0-nocond 臂 + 存 ckpt
```

产出（`Test11_tau_regime/`）：`tau_regime_log.csv`、`tau_regime_curves.png`、
`tau_regime_report.json`（各 arm × lr 最终数值 + PASS/FAIL 判定 + 诊断结论）、
`tau_regime_ckpt_<arm>_<lr>.pt`（仅 `--save-ckpt` 时）。

**全数据集重训**：`python -m anysole.train` 新增两个旋钮（写入 checkpoint 配置），用于在完整训练
分布下验证 R_Test9 的结论（如 `--tau-max 100` 重训后看 `val/tau0_mpjpe` 能否到个位数）：

```bash
conda activate touch_gait
# 低噪声带全量重训
python -m anysole.train --modal anysolev1 --contact-method joint_and --tau-max 100
# τ≡0 恒等映射全量重训
python -m anysole.train --modal anysolev1 --contact-method joint_and --tau-fixed 0
```

## 附录：训练侧 dropout 开关（两个独立旋钮）

`python -m anysole.train` 现在有两个独立 dropout 参数，都会写入 checkpoint 配置（eval/infer 自动读取）：

- `--dropout 0.1`：模型内部 nn.Dropout（fusion/pose/traj transformer）。`0.0` 关闭。
- `--dropoutVT DROP_V_PCT DROP_T_PCT`（或逗号写法 `--dropoutVT 20,30`）：**模态 dropout**，
  V 丢弃 DROP_V_PCT%、T 丢弃 DROP_T_PCT%（映射为 config_probs `[1-v-t, t, v]`，即 VT/V-only/T-only）。
  `--dropoutVT 0,0` 关闭模态 dropout。

重训实验示例（关全部 dropout + 更多轮次）：

```bash
conda activate touch_gait
python -m anysole.train --modal anysolev1 --contact-method joint_and \
  --dropoutVT 0,0 --epochs 400 \
  --out-dir results/AnySole/anysolev1_joint_and/checkpoints
```

> 注意：重训后必须重跑 `python -m anysole.eval --modal ... --contact-method ...` 刷新 BVH/npz/metrics（见 R_Test3 说明），
> 再重跑 R_Test5/R_Test6 才有意义（D_Test2 与模型无关，只需跑一次）。

## 历史迁移说明

动画统一进入 `Test1_visualization/`，对照实验进入 `Test2_comparison/`，
过时配置删除（默认自动扫描 results/）。

> **2026-09-20 SMPL 转向归档**：BVH-23 时代的 AnySole 实验目录已归档至
> `results/AnySole_BVH_backup/`，baseline 试跑归档至 `results/Baselines_BVH_backup/`；
> 展示层工具不再消费这些目录（auto 扫描会跳过 `*_backup*`）。当前 AnySole
> 只产出 SMPL-24 NPZ（`predictions/eval_motion/`）；BVH 格式支持仅服务于
> MotionPRO / Step2Motion 等 baseline 的展示与对照。
>
> **2026-09-21 D/R 重组**：实验按「数据检验 D_TestN / 结果检验 R_TestN」两大部分
> 重排（对照表见文首）；脚本与产物目录的改名在 P1-4 执行，届时本文命令同步更新。
