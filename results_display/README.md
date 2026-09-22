# Results Display

集中存放实验产物。实验分两大部分，互不耦合：

- **第一部分：数据检验（D_TestN）** —— 处理**原始数据**（对应 AnysoleWorkspace）：
  原始 GT 的可视化、自检、标签判定与协议探针。
- **第二部分：结果检验（R_TestN）** —— 处理**实验结果**（对应 results）：
  模型输出的可视化、参数对照、轨迹与专项消融。

实验代码位于 `script/`（唯一被 git 跟踪的目录），生成物落在对应的实验目录。
本文已按 D/R 编号组织；P1-4 的脚本平铺改名已落盘（命令照抄可用），
新旧对应见下方对照表。

可用环境变量 `ANYSOLE_RESULTSDISPLAY` 覆盖本根目录。

## 编号对照（2026-09-22 第二次重排）

| 编号 | 来源 | 脚本 |
| --- | --- | --- |
| D_Test1 原始数据可视化 | Test1 的 GT/输入部分 | `d_test1_data_viz.py`（BVH/SMPL GT 渲染） |
| D_Test2 数据集自检 | Test6 | `d_test2_dataset_check.py` |
| D_Test3 接触检测 | Test5 | `AnysoleWorkspace/tool/contact_labels.py`；`d_test3_contact.py` |
| D_Test4 基线触觉审计 | 09-21 的 D_Test5 | `d_test4_baseline_tactile.py` → `AnysoleWorkspace/tool/generate_baseline_tactile.py` |
| D_Test5 鞋垫偏移补偿（**待适配**） | 09-21 的 R_Test4 | `d_test5_insole_drift.py` |
| R_Test1 模型可视化 | Test1 的模型部分 | `r_test1_visualize_{anysole,mmvp,motionpro,step2motion}.py` |
| R_Test2 参数对照 | Test2 | `r_test2_compare.py` |
| R_Test3 轨迹可视化 | Test3 | `r_test3_traj.py` |
| R_Test4 触觉生成 V2T | 09-21 的 R_Test8 | `r_test4_v2t.py` |
| R_Test5 ρ 网格 | 任务书实验 1 | `anysole/rho_grid.py`（生成器）+ `r_test5_rho_grid.py`（前端） |
| R_Test6 互补分析 | 任务书实验 2 | `r_test6_complement.py` |
| R_Test7 dropout 消融 | 任务书实验 3 | `r_test7_dropout_ablation.py` |
| R_Test8 T2M 上半身 | 任务书实验 4 | `r_test8_t2m_upper.py` |
| R_Test9 信任堆叠条 | 任务书实验 5 | `r_test9_trust.py` |
| R_Test10 单模态训练对比 | 任务书实验 6 | `r_test10_singlemodal_compare.py` |

任务书 = `anysole/model_fix_note/missing_rate_experiment_plan.md`（实验 1–6 ↔ R_Test5–10）。
已归档（编号撤销，见文末历史迁移说明）：09-21 的 R_Test5 均值姿态推理、R_Test6 输入消融、
R_Test7 legacy 扩散诊断、R_Test9 τ 区间、R_Test10 pose head 系列 → `script/legacy/`。

## 目录结构

```text
results_display/                        # 纯产物目录（实验代码在 script/）
├── script/                             # 源码（唯一被 git 跟踪的目录）
│   ├── utils/                          # 共享公共件
│   │   ├── bvh_aligner_pose.py / cli_common.py / motion_io.py / render_common.py / smpl_mesh.py
│   │   └── ridge_probe.py              # ridge-on-F 读出天花板公共件（R_Test6/7/8 复用）
│   ├── legacy/                         # 已归档实验（09-21 编号撤销，保留供查阅）
│   │   ├── r_test5_mean_pose.py、r_test6_input_ablation.py
│   │   ├── r_test7_legacy_{overfit,sampler_checks}.py、r_test9_tau_regime.py
│   │   └── r_test10_*.py ×6
│   ├── ── 数据检验（D_TestN）──
│   │   ├── d_test1_data_viz.py         # D_Test1：GT 原始渲染（SMPL-24/BVH-23 双协议）
│   │   ├── d_test2_dataset_check.py    # D_Test2：数据集自检
│   │   │   （D_Test3 标签生成器在 AnysoleWorkspace/tool/contact_labels.py，不在本目录）
│   │   ├── d_test3_contact.py          # D_Test3：接触标签动画 + 阈值分析
│   │   ├── d_test4_baseline_tactile.py # D_Test4：四基线触觉格式审计（生成器前置，产物落盘）
│   │   └── d_test5_insole_drift.py     # D_Test5：鞋垫漂移补偿器测试（待适配）
│   └── ── 结果检验（R_TestN）──
│       ├── r_test1_visualize_{anysole,mmvp,motionpro,step2motion}.py   # R_Test1：模型动画
│       ├── r_test2_compare.py          # R_Test2：跨模型/参数对照评估（只出指标）
│       ├── r_test3_traj.py             # R_Test3：轨迹对比动画 + 静态图
│       ├── r_test4_v2t.py              # R_Test4：V2T 触觉生成
│       └── （r_test5..10_*.py          # R_Test5–10：缺失率任务书实验，规划中）
├── ── 数据检验产物（data/d_testN/）──
│   ├── d_test1_data_viz/               # D_Test1 输出（<session>/<smp24|bvh23>/）
│   ├── d_test2_dataset_check/          # D_Test2 输出（自检 JSON + 均值姿态 npz）
│   ├── d_test3_contact/                # D_Test3 输出（按方案分目录）
│   ├── d_test4_baseline_tactile/       # D_Test4 输出（每 session 一个 <session>_adapted_tactile.gif/.mp4）
│   └── d_test5_insole_drift/           # D_Test5 输出（补偿前后对比动画 + theta_summary.csv）
└── ── 结果检验产物（result/r_testN/）──
    ├── r_test1_visualize/              # R_Test1 输出（AnySole/MMVP/MotionPRO/Step2Motion 分栏）
    ├── r_test2_compare/                # R_Test2 输出（只出指标，不做动画）
    ├── r_test3_traj/                   # R_Test3 输出
    ├── r_test4_v2t/                    # R_Test4 输出（tgen 汇总 + 热力图动画）
    └── （r_test5..10 预留）
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
- R_Test2 的跨协议指标统一映射到 common19 语义关节（`r_test2_compare.py` 的 `protocol_gt`）。
- 归档的 `*_backup*` 目录不参与任何自动扫描。

# 第一部分：数据检验（D_TestN）

## D_Test1 原始数据可视化（BVH / SMPL）

直接渲染 AnysoleWorkspace 的原始 GT 动捕数据，不涉及任何模型输出：

| 脚本 | 作用 |
| --- | --- |
| `d_test1_data_viz.py` | 原始 GT 骨架渲染，**双协议**：默认 auto（有 SMPL-24 用 SMPL，缺失回退 BVH-23）；`--protocol smp24/bvh23` 强制单协议；`--bvh <路径>` 单文件 BVH |

```bash
conda activate touch_gait

# 原始数据可视化（auto：SMPL-24 优先，BVH-23 回退）
python results_display/script/d_test1_data_viz.py
python results_display/script/d_test1_data_viz.py --protocol bvh23   # 强制 BVH-23
python results_display/script/d_test1_data_viz.py --bvh <单个.bvh>    # 单文件 BVH
```

## D_Test2 数据集自检（GT 自洽 + 均值姿态基线）

`d_test2_dataset_check.py` 只读数据集、不加载任何模型，回答两个问题：

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
python results_display/script/d_test2_dataset_check.py
# 单 session 冒烟
python results_display/script/d_test2_dataset_check.py --session S10103 --limit-sessions 1 --max-windows 16
```

## D_Test3 接触检测

`d_test3_contact.py` 渲染自动识别的 GT motion 骨架 + 触觉鞋垫热力图，并叠加接触指示
（**红色=接触、绿色=无接触**）：鞋垫描边与徽章、骨架足部关节着色、底部整段接触时间轴。
每帧同步显示 48 格压力和值。

背景：原 `contact.npy` 标签口径为「48 格鞋垫 CSV 逐帧压力和 > 100」
（`prepare_sequences.py:contact_from_insoles`，阈值 `CONTACT_SUM_THRESH`），但大量鞋垫在脚
离地后压力不归零（如 S11023 左脚摆动相压力和最低 653），曾导致大量运动学离地帧被错标为接触。
为对比候选修复方案，Test5 改为**按方案分子目录**：`contact_labels.py` 为每个方案生成
`contact_<method>.npy`（与 `contact.npy` 同目录、同 10 列格式），`d_test3_contact.py --methods ...`
渲染判定动画。标签生成器属数据构建代码，位于 `AnysoleWorkspace/tool/contact_labels.py`。

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
python AnysoleWorkspace/tool/contact_labels.py
# 2) 渲染动画（默认全部方案 × test split；标签缺失时自动补齐）
python results_display/script/d_test3_contact.py --methods bvh_soft --gen mp4 --session S11113
# 单方案 / 单 session 冒烟
python results_display/script/d_test3_contact.py --methods bvh_h --session S11023
```

产出（`Test5_contact/`）：每个方案一个子目录（`<method>/`）：
`gif|mp4/<session>_contact.gif/.mp4`（触觉热力图 + GT motion + 该方案接触指示动画）、
`contact_summary.csv`（逐会话接触率/压力和统计）、`diff_vs_bvh_h.csv`
（该方案 vs 直接导出 BVH 高度参考的差异区间清单：会话/脚/起止时间s/方向/区间内高度·速度·压力和）、
`diff_by_subject.png` / `diff_by_sequence.png`（差异 × 用户/动作序列分布）。
另有与方案无关的 `threshold_analysis.png`（压力和直方图 + 接触率-阈值曲线）、
`threshold_analysis.csv`（逐帧左右脚 48 格压力和）、`threshold_analysis_hist.csv`
（64 分箱分布）与 `comparison.{csv,png}`（各方案 vs bvh_h 参考的一致率/假接触率/假离地率）。

## D_Test4 基线触觉审计（Agent_06 前置）

`d_test4_baseline_tactile.py` 是纯可视化前端：先确保转换产物落盘（产物齐全则跳过，
缺则 subprocess 调用 `AnysoleWorkspace/tool/generate_baseline_tactile.py`；`--force`
强制重生成）。

| 工作 | 触觉输入 | 落盘位置 |
| --- | --- | --- |
| AnySole（主方法） | 原始 pressure.npz → 原生 4×12 / 脚 | `AnysoleWorkspace/derived/MotionPRO/sequences/cam3/<date>/<sub>/<sid>/pressure.npz` |
| MotionPRO | pressure.npz → bilinear 96×96 /255（FRAPPE 口径；可视化按 L/R 脚区等尺度显示） | `AnysoleWorkspace/derived/MotionPRO/pressure_96/<sid>.npz` |
| MMVP（pressure_tookit / VP-MoCap） | 共享同一份 insole `{'insole': [L(31,11), R(31,11)]}`；两棵目录逐帧一致 | `AnysoleWorkspace/derived/pressure_tookit/.../insole/%03d.npy` + `AnysoleWorkspace/derived/VP-MoCap/.../insole/%03d.npy` |
| Step2Motion | 16 通道/脚（process_gait 冻结池化，展示/审计产物） | `AnysoleWorkspace/derived/Step2Motion/pressure_16ch/<sid>.npz` |

   方法                                  实际数据                            当前显示
  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   AnySole              4×12 = 48 格/脚，96 格/帧                      48 格/脚，正确
  ─────────────  ─────────────────────────────────  ──────────────────────────────────
   MotionPRO      输入是 96×96 图像，即 9216 个像    当前没有显式网格，只显示连续热图
                    素；底层来源仍是 48 格/脚，共
                                    96 个传感器格
  ─────────────  ─────────────────────────────────  ──────────────────────────────────
   MMVP               31×11 = 341 个位置/脚；有效                       31×11 网格/脚
                            mask 约 242/241 个/脚
  ─────────────  ─────────────────────────────────  ──────────────────────────────────
   Step2Motion             16 通道/脚，32 通道/帧     当前显示 48 格/脚，但只有 16 个
                                                              唯一值，每个值重复 3 格

关键实测（`AnysoleWorkspace/derived/baseline_tactile/<sid>/meta.json`）：MMVP mask 242/241 像素/脚；
最近格距离 L 均值 9.2mm / R 8.7mm；FPP weight = 静立帧总压；布局文件未标注内外侧
方向，默认与模板 x 同向（`--mirror-x` 翻转，透传生成器）。

```bash
conda activate touch_gait
# 默认全部 splits session（train ∪ val ∪ test，缺产物自动生成）
python results_display/script/d_test4_baseline_tactile.py
python results_display/script/d_test4_baseline_tactile.py --split test      # 只跑 test
python results_display/script/d_test4_baseline_tactile.py --session S12072 --force
python results_display/script/d_test4_baseline_tactile.py --session S12072 --gen mp4
python AnysoleWorkspace/tool/generate_baseline_tactile.py --split test    # 只落盘不渲染
```

## D_Test5 鞋垫偏移补偿（待适配）

`d_test5_insole_drift.py` 验证 insole-drift 补偿器（`anysole.ablations.insole_drift`，
tactile-in → tactile-out 纯数据模块）在真实 session 上的行为：加载训练好的补偿器权重，
渲染补偿前后触觉对比动画 + 逐 session 漂移统计（`theta_summary.csv`）。

> **待适配**：默认 checkpoint 指向 `results/AnySole/anysolev1_insole_drift_<contact>/checkpoints/ckpt_last.pt`，
> 该训练产物已不在盘上（补偿器需要训练过的权重，模板 bank 只是输入）。
> 恢复产物（重训/备份）之前本实验不可运行。

```bash
conda activate touch_gait
python results_display/script/d_test5_insole_drift.py
python results_display/script/d_test5_insole_drift.py --session S7013
```

# 第二部分：结果检验（R_TestN）

## R_Test1 模型可视化

| 脚本 | 作用 | 默认输出 |
| --- | --- | --- |
| `r_test1_visualize_motionpro.py` | MotionPRO 触觉输入/预测/GT 三栏对比动画 | `Test1_visualization/MotionPRO/<checkpoint tag>/` |
| `r_test1_visualize_step2motion.py` | Step2Motion 足底压力/生成 BVH/GT 对比动画 | `Test1_visualization/Step2Motion/gait_model/` |
| `r_test1_visualize_anysole.py` | AnySole 主模型与消融（足底压力/预测动作/SMPL GT；自动检测格式） | `Test1_visualization/AnySole/<modal>/<config>/` |
| `r_test1_visualize_mmvp.py` | MMVP 两条方法线的轻量动画（消费 eval_motion npz，缺预测显式报告） | `result/r_test1_visualize/` |

```bash
conda activate touch_gait

# motionpro 可视化
python results_display/script/r_test1_visualize_motionpro.py
--session S14103
--checkpoint results://MotionPRO/checkpoints/imagepressure2smpl/init/5e-05/imagepressure2smpl_best.pth

# step2motion 可视化
python results_display/script/r_test1_visualize_step2motion.py
--self-test

# 默认 主模型 + 消融，全部 config，全部concat
CUDA_VISIBLE_DEVICES=4 python results_display/script/r_test1_visualize_anysole.py --modal V4B --contact-method joint_and --gen gif --mesh --session S12042
--modal V4B
--contact-method bvh_h,bvh_soft,tactile_abs,pat_offset,joint_and
--config-id VT2M

--modal F0b --contact-method joint_and   # 历史 run 对照
```

## R_Test2 参数对照

`r_test2_compare.py` 对 AnySole（modal × config 组合）、MotionPRO、Step2Motion 统一计算 MPJPE / PA-MPJPE / W-MPJPE / WAMPJPE / RTE / Accel / Jitter 指标。
产出（`Test2_comparison/`）：
`comparison_per_session.csv`（逐会话明细）、`comparison_summary.csv`（精简表：仅模型 × 指标）、
`comparison_summary.png`（精简表的表格图）、`evaluation.log`（运行信息 + 各模型 checkpoint）。

```bash
python results_display/script/r_test2_compare.py \
  --manifest AnysoleWorkspace/manifests/session_manifest.csv \
  --modal anysolev1,anysolev1_insole_drift \
  --config-id VT2M,V2M,T2M \
  --split test

# 或者完全自动扫描 results/ 下的自包含模型目录
python results_display/script/r_test2_compare.py --auto-scan
```

模型清单默认由脚本扫描 `results/` 自动构建；也可用 `--models-config <yaml|json>` 指定固定清单。

## R_Test3 轨迹可视化

`r_test3_traj.py` 可视化预测根轨迹 vs 真实轨迹（**SMPL/BVH 协议自动检测**）：
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
python results_display/script/r_test3_traj.py --modal V4B --gen gif --contact-method joint_and
# 单 session / 单 config
python results_display/script/r_test3_traj.py --session S7013 --config-id VT2M
# baseline 模型（扫描 results/ 下自包含模型目录，排除 *_backup*）
python results_display/script/r_test3_traj.py --auto
```

输出：`Test3_trajectory/AnySole/<modal>/<config>/{gif,mp4,png}/<session>_<config>_traj.{gif,mp4,png}`；
baseline 模型输出在 `Test3_trajectory/<model>/gen/`。

> 注意：eval 生成的 motion NPZ 必须与 checkpoint 配套。若重新训练了模型，
> 旧的 `eval_motion/` 产物会与新 metrics 不一致，
> 需重跑上述 eval 命令再渲染 R_Test1/R_Test3。

## R_Test4 触觉生成（V2T）

`r_test4_v2t.py` 用 V-only 条件（触觉输入置零）跑模型，让辅助头 `pressure_hat` 变成
**视觉→触觉（V2T）生成器**：模型仅凭 HRNet 视觉特征输出 96 格足底压力。触觉头不经过
扩散采样，每个窗口一次 tau=0 前向即可得到确定性的生成触觉，无需 DDIM。

三种条件（与 eval.py 的 `--config-id` 同名）：

| 条件 | 输入 | 意义 |
| --- | --- | --- |
| `VT2M` | 真 V + 真 T | 重建 sanity（上界参考） |
| `V2M` | 真 V + 零 T | **V2T 生成本身（主指标）** |
| `T2M` | 零 V + 真 T | 触觉自重建（输入端 sanity） |

产出（`result/r_test4_v2t/`）：`tgen_summary.csv`（逐会话 × 条件 MAE/RMSE/相关系数）、
`tgen_report.json`（跨会话聚合，`v2t` 字段 = V2M 行汇总）、每个 session 的 96 格逐格 MAE
（`cells/*.npz` + 静态误差图 `*.png`），以及前 `--export-sessions`（默认 4）个
session 的 GT | 生成 | |GT-Gen| 三栏热力图动画（`gif|mp4/<session>_<mode>_tgen.{gif,mp4}`）。

与 `anysole.eval` 的关系：eval 在 `metrics/test.json` 每个模式行输出 `T_mae/T_rmse/T_corr`
（V2M 行即 V2T，JSON 顶层 `v2t` 字段），R_Test4 是该指标的逐格/逐帧可视化解剖。

```bash
conda activate touch_gait
python results_display/script/r_test4_v2t.py                     # 全部 test split
python results_display/script/r_test4_v2t.py --session S10103    # 单 session 冒烟
python results_display/script/r_test4_v2t.py --config-id VT2M,V2M --export-sessions 2
```

> 注意：脚本按 anysolev1 扩散口径编写（默认 ckpt 与 tau=0 调用）；主线 anysolev2 为回归式
> forward（无 tau），`pressure_hat` 头仍在（`model_v2.py`），对 V4B 使用前需适配 v2 调用路径。

## R_Test5–10 缺失率实验组（任务书，规划中）

编号按任务书 `anysole/model_fix_note/missing_rate_experiment_plan.md` 实验 1–6 顺序占用，
脚本随各阶段落地：

| 编号 | 任务书实验 | 回答的问题 | 脚本 |
| --- | --- | --- | --- |
| R_Test5 ρ 网格 | 实验 1 | 模态按任意比例缺失时性能怎么变 | `anysole/rho_grid.py`（生成器）+ `r_test5_rho_grid.py`（前端）✅ |
| R_Test6 互补分析 | 实验 2 | 融合是互补还是拼贴；F 是不是"完整状态" | `r_test6_complement.py`（2a/2b/2c 全落地）✅ |
| R_Test7 dropout 消融 | 实验 3 | 训练时的模态 dropout 是不是必需的 | `r_test7_dropout_ablation.py`（C1 对比表 + C2 单流探针）✅ |
| R_Test8 T2M 上半身 | 实验 4 | 触觉没有上肢信号，合理上半身从哪来 | `r_test8_t2m_upper.py`（t_tok 无先验 ridge 基线）✅ |
| R_Test9 信任画像 | 实验 5 | 每个身体部位"信谁" | `r_test9_trust.py`（E1 信任堆叠条 + E2 注意力）✅ |
| R_Test10 单模态训练对比 | 实验 6 | V-only / T-only 各自天花板 | `r_test10_singlemodal_compare.py`（对比表）✅ |

### R_Test5 ρ 网格（已实装）

机制：帧级 null-token mask（`ModalEncoders.forward` 的 `mask_v/mask_t`，默认 None 零重训兼容；
已实测与 config 级 null 逐字节一致）。生成器按 (session, cell, seed) 确定性推理 36 格，
指标复用 `eval_protocol._session_metrics`（与 fseries 同源），每格落 eval_motion 同格式 npz
+ F/t_tok/v_tok 表征；末尾 corner_check 自检三角 vs fseries（必须 ≈0）。

```bash
conda activate touch_gait
# 生成（val 3 种子估方差；test 单种子；--reuse 断点续跑）
python -m anysole.rho_grid --ckpt results/AnySole/V4B_joint_and/checkpoints/ckpt_last.pt \
    --split val --seeds 0,1,2 --reuse
python -m anysole.rho_grid --ckpt ... --split test --seeds 0 --reuse

# 前端（热力图/切片/自检表）
python results_display/script/r_test5_rho_grid.py
python results_display/script/r_test5_rho_grid.py --metric MPJPE
```

产出（`result/r_test5_rho_grid/`）：`grid_metrics.json`（全格协议指标 + corner_check）、
`npz/<session>_rhoV<rV>_rhoT<rT>[_s<seed>].npz`（接 R_Test1 动画）、
`repr/<session>_..._repr.npz`（R_Test6 2c / R_Test9 消费）、前端 `heatmap.png` / `slices.png` / `corner_check.csv`。

**V4B val 已跑完（3 种子，自检 diff 全 0）**：整图由 V 主导（同 V 列 T 几乎不动 PA），
T 仅在 V 缺失时兜底（上限 69.6）；纯先验 77.0 PA / 171.9 MPJPE；全表最低 35.3 在 V80%/T20%。
完整数值与首读结论见任务书实验 1。

```bash
# 2b 探针表（Ft/Fv/融合 F 三列 ridge 读出 + 上界判定；fit 需 train repr）
python -m anysole.rho_grid --ckpt ... --split train --seeds 0 --reuse    # 先落 train repr
python results_display/script/r_test6_complement.py --probe

# 2c t-SNE（三配置 F 散点，读 ρ 网格 repr dump）
python results_display/script/r_test6_complement.py --tsne --sessions 8

# 3 C1 对比表（主线 vs nodrop fseries）/ C2 单流探针（v_tok/t_tok ridge 解码）
python results_display/script/r_test7_dropout_ablation.py --c1 --fs-nodrop <nodrop_fseries>
python results_display/script/r_test7_dropout_ablation.py --c2 --split val --sessions 6

# 4 D1 T2M 上半身（t_tok 无先验基线 vs 模型 T2M 输出，gap = 先验贡献）
python results_display/script/r_test8_t2m_upper.py --split val --sessions 6

# 5 E1 信任画像（读 grid_metrics 三角）/ E2 注意力（decoder 钩子，需 GPU）
python results_display/script/r_test9_trust.py --e1
python results_display/script/r_test9_trust.py --e2 --ckpt results/AnySole/V4B_joint_and/checkpoints/ckpt_last.pt --sessions 3

# 6 单模态训练对比表（主线 vs vonly/tonly/nodrop fseries）
python results_display/script/r_test10_singlemodal_compare.py \
    --fs-vonly <vonly_fseries> --fs-tonly <tonly_fseries>
```

### R_Test6 2a 互补 bar（已落地）

纯读 `metrics/<split>_fseries.json`（零重训），出三张图 + 汇总表：每部位三配置
PA-MPJPE 柱图、发散 Δ 判据图（蓝=融合更优/红=更差/灰=持平）、辅助指标小倍图，
并给双向判据（V4B val 实测：**V 向 6/6 全增益，T 向 gain 2 / loss 5 / flat 3
→ 判定「拼贴，仅 V 向增益」**；总体 Δ(VT−V2M)=+5.0 / Δ(VT−T2M)=−29.2 mm）。

```bash
conda activate touch_gait
python results_display/script/r_test6_complement.py --bar                     # val
python results_display/script/r_test6_complement.py --bar --split val \
    --model-dir results/AnySole/V4B_joint_and
```

产出（`result/r_test6_complement/`）：`complement_bar.png`、`complement_delta.png`、
`complement_aux.png`、`complement_summary.{json,csv}`（表视图，Δ 统一"负=融合更好"口径）。

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

> 注意：重训后必须重跑 `python -m anysole.eval --modal ... --contact-method ...` 刷新 BVH/npz/metrics（见 R_Test3 说明）。
>
> 上述 dropout 旋钮仅适用于 anysolev1 扩散线；主线 anysolev2 为回归式模型，忽略这些参数
> （train.py 会提示 knobs are ignored by modal anysolev2）。

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
>
> **2026-09-22 第二次重排（缺失率实验组前置，主线 anysolev2/V4B）**：
> D_Test5 基线触觉审计 → D_Test4；R_Test4 鞋垫偏移 → D_Test5（待适配）；R_Test8 V2T → R_Test4。
> 09-21 编号的 R_Test5 均值姿态推理 / R_Test6 输入消融 / R_Test7 legacy 扩散诊断 /
> R_Test9 τ 区间 / R_Test10 pose head 系列已归档至 `script/legacy/`
> （诊断对象为 anysolev1 扩散线，主线回归式已不适用）。R_Test5–10 编号由缺失率
> 任务书实验 1–6 复用。原 D_Test4（SMPL-24 协议预检）维持撤销状态：探针在
> `z_note/probes/`（probe_f4_smpl24_preflight.py 等，不占 D/R 编号）。
