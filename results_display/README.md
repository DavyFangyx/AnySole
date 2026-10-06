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

**各模型对应的评价指标（能力矩阵、指标唯一实现位置、`—` 规则、V2T 层级）见
`README_metrics.md`。**

## 两组数据生产实验与分析入口

数据生产实验统一由 `configs/` 调度（正式入口见 `configs/Z_README.md`）。当前只有两组：
`singlemodal_eval` 和 `rho_grid_eval`，主线模型均为 `V3_3B`。任务由
`configs/tools/runner.py` 负责完成 checkpoint 准备、val/test 正式评估和结果归档；
分析脚本不进入队列，只读取这两组结果：

```text
results/experiments/singlemodal_eval/<model>/            # V3_3B、V3_3B_vonly、V3_3B_tonly
results/experiments/rho_grid_eval/<model>/               # 全网格 grid_metrics + npz/ + repr/
results_display/script/<analysis>.py
results_display/{ATest,BTest}/<编号Test_分析名>/<model>/
```

```bash
conda activate touch_gait
# 数据生产（生成队列 conf + 后台 worker）见 configs/Z_README.md
# 分析（--split val|test 默认 test；--model 可重复限定模型）
python results_display/script/complement.py
python results_display/script/v2t_upper.py
```

A/B 系列分析入口为语义化脚本（`complement.py`、`dropout_ablation.py`、
`rho_grid_analysis.py` 等），编号与产物目录的对应见下表与「目录结构」。

## 编号对照

| 编号 | 说明 | 脚本 |
| --- | --- | --- |
| D_Test1 原始数据可视化 | GT/输入渲染（BVH/SMPL） | `d_test1_data_viz.py` |
| D_Test2 数据集自检 | 数据集自检 | `d_test2_dataset_check.py` |
| D_Test3 接触检测 | 接触标签判定 | `AnysoleWorkspace/tool/contact_labels.py`；`d_test3_contact.py` |
| D_Test4 基线触觉审计 | 四基线触觉格式审计（审计只读） | `d_test4_baseline_tactile.py`（数据源 = `build_shared.py` 产物） |
| D_Test5 鞋垫偏移补偿（**待适配**） | 鞋垫漂移补偿器测试 | `d_test5_insole_drift.py` |
| D_Test6 地面估计与坐标系审计 | 行走地面 ROI 审计（`AnysoleWorkspace/tool/.../floor_audit.py`） + 生产 floor 查看 | `floor_audit.py`；`d_test6_floor_view.py` |
| R_Test1 模型可视化 | 姿态动画（选中 + 生成什么统一入口） | `r_test1_visualize.py`（缺省单独生成；`--compare` 同 mode 1×N 横排，复用单独生成中间帧） |
| R_Test2 参数对照 | 跨模型/参数对照评估 | `r_test2_compare.py`（`--by-mode` 模式分块；原生协议 + 能力门控） |
| R_Test3 轨迹可视化 | 轨迹对比动画 | `r_test3_traj.py`（`--compare` 并排轨迹；配对协议 GT） |
| R_Test4 触觉生成 V2T | V2T 触觉生成 | `r_test4_v2t.py`（四层层级 brief 7 + 叶 5） |
| A0 | 专才分工 | `singlemodal_analysis.py`（→ `utils/component_analysis.py --a0`） |
| A1 | 完整输入融合 | `complement.py`（→ `utils/component_analysis.py --a1`） |
| A2 | 缺失分支分工 | `dropout_ablation.py`（→ `--a2`） |
| B1 | V 分支的 T 相关能力 | `v2t_upper.py`（→ `--b1`） |
| B2 | T 分支的 V 相关能力 | `trust.py`（→ `--b2`） |
| B3 | 连续缺失鲁棒性 | `rho_grid_analysis.py`（→ `utils/rho_grid_display.py`） |

任务书 = `anysole/model_fix_note/missing_rate_experiment_plan.md`（实验编号 A0–B3）。

## 目录结构

```text
results_display/                        # 纯产物目录（实验代码在 script/）
├── script/                             # 源码（唯一被 git 跟踪的目录）
│   ├── utils/                          # 共享公共件
│   │   ├── bvh_aligner_pose.py / cli_common.py / motion_io.py / render_common.py / smpl_mesh.py
│   │   └── ridge_probe.py              # ridge-on-F 读出天花板公共件
│   ├── legacy/                         # 已归档旧实验（保留供查阅，不占编号）
│   ├── ── 数据检验（D_TestN）──
│   │   ├── d_test1_data_viz.py         # D_Test1：GT 原始渲染（SMPL-24/BVH-23 双协议）
│   │   ├── d_test2_dataset_check.py    # D_Test2：数据集自检
│   │   │   （D_Test3 标签生成器在 AnysoleWorkspace/tool/contact_labels.py，不在本目录）
│   │   ├── d_test3_contact.py          # D_Test3：接触标签动画 + 阈值分析
│   │   ├── d_test4_baseline_tactile.py # D_Test4：四基线触觉格式审计（审计只读，读 build_shared 产物）
│   │   ├── d_test5_insole_drift.py     # D_Test5：鞋垫漂移补偿器测试（待适配）
│   │   └── d_test6_floor_view.py       # D_Test6：生产 floor 查看（floor 系点云/RGB+ROI/高度直方图；审计工具在 AnysoleWorkspace）
│   ├── ── 结果检验（R_TestN）──
│   │   ├── r_test1_visualize.py         # R_Test1：姿态动画（选中 + 生成什么；--compare 同 mode 横排）
│   │   ├── r_test2_compare.py          # R_Test2：跨模型/参数对照评估（只出指标）
│   │   ├── r_test3_traj.py             # R_Test3：轨迹对比动画 + 静态图
│   │   └── r_test4_v2t.py              # R_Test4：V2T 触觉生成
│   └── ── 任务书分析（A/B 系列，入口脚本 + utils/component_analysis.py）──
│       ├── singlemodal_analysis.py / complement.py / dropout_ablation.py
│       ├── v2t_upper.py / trust.py / rho_grid_analysis.py
│       └── utils/mpl_fonts.py          # CJK 字体公共设置（中文标注不豆腐块）
├── DataTest/                           # D 组：数据检验产物（D_TestN）
│   ├── D1Test_data_viz/                # D_Test1 输出（<session>/<smp24|bvh23>/）
│   ├── D2Test_dataset_check/           # D_Test2 输出（自检 JSON + 均值姿态 npz）
│   ├── D3Test_contact/                 # D_Test3 输出（按方案分目录）
│   ├── D4Test_baseline_tactile/        # D_Test4 输出（每 session 一个 <session>_adapted_tactile.gif/.mp4）
│   ├── D5Test_insole_drift/            # D_Test5 输出（补偿前后对比动画 + theta_summary.csv）
│   └── D6Test_floor/                   # D_Test6 输出（<date>_floor_audit.png + floor_candidates.csv + <date>_<subject>_floor_view.png）
├── ResultTest/                         # R 组：结果检验产物（R_TestN）
│   ├── R1Test_visualize/               # R_Test1 输出（AnySole/MMVP/MotionPRO/Step2Motion 分栏 + compare/<mode> 并排）
│   ├── R2Test_compare/                 # R_Test2 输出（只出指标，不做动画 + by_mode/ 模式分块）
│   ├── R3Test_traj/                    # R_Test3 输出（+ compare/<mode> 并排轨迹）
│   └── R4Test_v2t/                     # R_Test4 输出（tgen 汇总 + 热力图动画）
├── ATest/                              # A 组：V 与 T 是否互补（A0–A2，各含 <model>/ 子目录）
│   ├── A0Test_specialist/              # A0：专才分工（a0_* 产物）
│   ├── A1Test_complement/              # A1：完整输入融合（a1_* 产物）
│   └── A2Test_dropout_ablation/        # A2：缺失分支分工（a2_* 产物）
└── BTest/                              # B 组：缺失条件下能否工作（B1–B3，各含 <model>/ 子目录）
    ├── B1Test_v2t_upper/               # B1：从 F_V 读出 T 负责分量（b1_* 产物）
    ├── B2Test_trust/                   # B2：从 F_T 读出 V 负责分量（b2_* 产物）
    └── B3Test_rho_grid/                # B3：ρ 网格连续缺失鲁棒性（heatmap/slices）
```

## 时间对齐约定（动捕 ↔ 触觉/视频）

视频、触觉与动捕之间存在时间偏差与漂移。偏差来自各设备独立起录：每个 session 的
人工复核表 `AlignReviews_csv/<session>.csv` 记录常量偏移 `偏移量(s)`（= 视觉时间 − 动捕时间）；
漂移则由触觉/视频时间轴按逐帧时钟重建（触觉 `t_us` 时间戳、视频文件名时钟）吸收。
所有动捕与触觉同帧对比的脚本，GT motion 都按同一补偿方式
重采样到 40 Hz 会话网格：`t_mocap = t_grid − offset_s`，其中 `t_grid = visual_start_s + n/40`

## 模型协议约定（原生 SMPL-24 / BVH-23 双协议，无 common19）

展示与对照层（D_Test1 与 R_Test1/2/3）按文件格式自动检测协议，无协议 flag。
**2026-09-26 评估整改起 common19 已删除**：数值评测与并排可视化都不再裁剪公共
关节集——每个模型按其原生骨架对**同协议配对 GT** 评测（SMPL-24 对 SMPL GT、
BVH-23 对 BVH GT），协议标识写为 `smpl24-native` / `bvh23-native`：

| 协议 | 模型 | 产物约定 | 检测规则 |
| --- | --- | --- | --- |
| SMPL-24 | AnySole / MotionPRO / MMVP pressure_toolkit / MMVP_VP-MoCap | `predictions/eval_motion/<session>[_<config>].npz`，统一包含 `joint_xyz_world`、`joint_names`、`valid_mask`；AnySole 仍保留原生 SMPL 字段 | `joint_xyz_world` + 24 个 `joint_names` → SMPL-24 |
| BVH-23 | Step2Motion 等 baseline | `predictions/<run>/<session>_gen.bvh`（Skeleton3） | `.bvh` 后缀 → bvh23 |

- GT 读取按模型协议走 `compare_core.protocol_gt`：SMPL 协议读 manifest 的
  `smpl_path`，BVH 协议读 manifest 的 `bvh_path`（同一 session 网格重采样）。
- 指标适用性由注册表 `models_modes.yaml` 的 `protocol`/`capabilities`/`sources`
  门控：模型没有该真实输出、或输出来源是 GT 回填/模板重建/拟合 shape/可视化
  占位时，指标为 `not_applicable`，CSV 留空、表格显示 `—`（绝不写 0）。预测文件
  不存在是 `missing`；声明能力与文件矛盾是 `invalid`。
- 正式 contact：motion 模型没有训练并显式导出的 contact prediction，正式
  `contact_f1` 一律 `—`。FPP-Net 接触头显式生成连续 SMPL 接触图（评估
  `contact_smpl_mse/bce`），GT = **press2Cont 顶点级二值 th=0.5**（E3-A
  裁定，训练与评估同源；archive 携带 `contact_gt_source` /
  `contact_gt_threshold` / `pixel_weight_revision` provenance）。`contact_f1`
  经 E3-Q5 裁定保留在诊断（AnySole 无接触头，正式键将只有 FPP 一行有值）。
  任何运动学/压力阈值推导的接触只用于诊断。
- pressure_toolkit 的拟合表面属其 pipeline 原生生成（生成清单裁定 2026-09-27），
  PVE 照常计算；其真实 pose/translation/rotation 支持的关节、轨迹、朝向、时序
  指标照常计算。AnySole 不生成 shape/表面，PVE 一律 `—`。
- Step2Motion 的 MPJAE/PVE/shape 为 `—`；其 BVH Hips 通道携带真实 root
  translation/rotation，轨迹与 yaw 指标正常补算。
- V2T 正式指标为四层层级（网格/力/CoP/接触）：表列只含 brief 7 键
  （`T_corr`/`T_rmse` + force rmse/r2 + CoP 左右 + `contact_smpl_mse`），叶 5 键（`T_mse`/
  `T_mae`/`pressure_force_mae`/`pressure_cop_error_mean`/`contact_smpl_bce`）保留在逐会话明细；
  R_Test2 与 R_Test4 调用同一个 pressure 求解器（`compare_core.v2t_metrics`）。
  接触级：`contact_smpl_mse` [brief]、`contact_smpl_bce` [叶]，pred 来自 FPP-Net
  接触头、gt 来自 f6_soft 软标签；`contact_f1` 待处理（缺二值接触 GT）；
  AnySole 无接触输出，两键为 `—`。
- 归档的 `*_backup*` 目录不参与任何自动扫描；Step2Motion no-IMU 探针不注册、
  不进入主排名。

历史说明：2026-09-26 前的 `common19` 语义关节映射与 `mmvp_common_metrics_v1`
comparison schema 已废弃（旧协议，仅历史审计记录保留）。

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
# 强制 BVH-23
python results_display/script/d_test1_data_viz.py --protocol bvh23
# 单文件 BVH
python results_display/script/d_test1_data_viz.py --bvh <单个.bvh>
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
  模型比"啥也不干"还差，更像 bug（脚本会自动与 `anysolev1_joint_and/metrics/test.json` 对比；该文件现在是全量明细口径，
简略口径在同目录 `test_brief.json`）。

产出（`DataTest/D2Test_dataset_check/`）：`fk_selfcheck.json`、`mean_baseline.json`、
`mean_pose.npz`、`input_means.npz`（后两者作"均值输入"基线）。

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
为对比候选修复方案，D_Test3 改为**按方案分子目录**：`contact_labels.py` 为每个方案生成
`contact_<method>.npy`（与 `contact.npy` 同目录、同 10 列格式），`d_test3_contact.py --methods ...`
渲染判定动画。标签生成器属数据构建代码，位于 `AnysoleWorkspace/tool/contact_labels.py`。
F6 三件套（`motion_f6`/`pressure_f6`/`f6_soft`，2026-09-18 定稿，设计见 fix_plan_v2.md F6a）
是最终标签设计：`bvh_h` 在台阶承重与低净空两类反例上失效后降级为对比参考；
`f6_soft` 中间阶段按 pressure_f6 一致度计算 4 档软值，导出的 npy 已二值化
（== `motion_f6`，与其他方案统一 {0, 1}；软值不进 npy）。

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
| `motion_f6` | F6 运动学状态机：接触→离地 v>0.6m/s 连续 2 帧（压力永不触发离地）；离地→接触 (v<0.3 且 \|az\|<3 连续 2 帧) 或 (压力承重 且 v<0.6)；h≥20cm 且无压力时拒绝接触 |
| `pressure_f6` | F6 承重：逐格基线 = 离地帧值 90 分位（自举自 motion_f6），corrected 和 + 迟滞阈值（80 或 0.2×stance 中位，1.5× 滞回） |
| `f6_soft（目前作为GT）` | F6 软标签：中间按 pressure_f6 一致度取 4 档软值 {0.95 触地承重 / 0.70 触地未承重 / 0.05 离地 / 0.30 离地残余压}，导出二值化 == motion_f6（与其他方案统一 {0, 1}） |

```bash
conda activate touch_gait
# 1) 生成全部方案的标签（全部 session）+ 对比报告
python AnysoleWorkspace/tool/contact_labels.py
# 按 split 三组限定范围（train/val/test，--session 显式列表优先）
python AnysoleWorkspace/tool/contact_labels.py --split test
# 2) 渲染动画（默认全部方案 × test split；标签缺失时自动补齐）
python results_display/script/d_test3_contact.py --methods f6_soft
```

产出（`DataTest/D3Test_contact/`）：每个方案一个子目录（`<method>/`）：
`gif|mp4/<session>_contact.gif/.mp4`（触觉热力图 + GT motion + 该方案接触指示动画）、
`contact_summary.csv`（逐会话接触率/压力和统计）、`diff_vs_bvh_h.csv`
（该方案 vs 直接导出 BVH 高度参考的差异区间清单：会话/脚/起止时间s/方向/区间内高度·速度·压力和）、
`diff_by_subject.png` / `diff_by_sequence.png`（差异 × 用户/动作序列分布）。
另有与方案无关的 `threshold_analysis.png`（压力和直方图 + 接触率-阈值曲线）、
`threshold_analysis.csv`（逐帧左右脚 48 格压力和）、`threshold_analysis_hist.csv`
（64 分箱分布）与 `comparison.{csv,png}`（各方案 vs bvh_h 参考的一致率/假接触率/假离地率）。

## D_Test4 基线触觉审计（审计只读）

`d_test4_baseline_tactile.py` 是纯可视化前端，**不做任何生成**：四个面板从
`build_shared.py` 与各 adapter 的落盘产物直接读取，并逐 session 审计
（mapping 字段 / 源 hash / frame_id parity / 帧数一致），审计失败即报错。

| 工作 | 触觉输入 | 数据位置 |
| --- | --- | --- |
| AnySole（主方法） | 原始 pressure_48 → 原生 4×12 / 脚（原始传感器单位） | `AnysoleWorkspace/shared/facts/sessions/cam3/<date>/<sub>/<sid>/pressure_48.npz` |
| MotionPRO | 虚拟毯 (T,320,120)（1.25cm/px 世界系；整幅显示 + painted 像素计数） | `AnysoleWorkspace/model_inputs/MotionPRO/adapter_v1/cam3/<date>/<sub>/<sid>/pressure.npz` |
| MMVP（pressure_toolkit / VP-MoCap） | 共享唯一公共表示，每帧 (2,31,11) float32 裸数组 | `AnysoleWorkspace/shared/representations/tactile/mmvp_31x11/v1/<date>/<sub>/<sid>/insole/%06d.npy` |
| Step2Motion | 16 通道/脚：运行时 `build_gait.pool_48_to_16` 复算（无落盘 16ch 产物） | 源头 = `AnysoleWorkspace/shared/facts/.../pressure_48.npz` |

   方法                                  实际数据                            当前显示
  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   AnySole              4×12 = 48 格/脚，96 格/帧                      48 格/脚，正确
  ─────────────  ─────────────────────────────────  ──────────────────────────────────
   MotionPRO            虚拟毯 320×120（世界系锚定，                       整幅热图 + painted
                         R1 锚点偏移，大量帧毯外）                            像素计数标注
  ─────────────  ─────────────────────────────────  ──────────────────────────────────
   MMVP               31×11 = 341 个位置/脚；有效                       31×11 网格/脚
                            mask 约 242/241 个/脚
  ─────────────  ─────────────────────────────────  ──────────────────────────────────
   Step2Motion             16 通道/脚，32 通道/帧     当前显示 48 格/脚，但只有 16 个
                                                              唯一值，每个值重复 3 格

映射口径已冻结在 `build_shared.py`（`audited_4x12_to_31x11_nearest_cell`：每个 31×11
格取最近 48 格的值，脚形外置 0）；D_Test4 只审计 parity，不再持有任何映射/生成
逻辑。历史审计数字（映射冻结时实测）：MMVP mask 242/241 格/脚、最近格距离 L 均值
9.2mm / R 8.7mm。FPP weight = 静立帧总压（build_metadata 计算）。

```bash
conda activate touch_gait
# 默认全部 splits session（train ∪ val ∪ test），只读审计 + 渲染
python results_display/script/d_test4_baseline_tactile.py
# 只跑 test
python results_display/script/d_test4_baseline_tactile.py --split test
python results_display/script/d_test4_baseline_tactile.py --session S12072
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

## D_Test6 地面估计与坐标系审计（pressure_toolkit）

```bash
conda activate touch_gait
# 四日期代表 session 审计：每日期 3 帧 × 3 ROI = 9 候选，选出最优 ROI
python AnysoleWorkspace/tool/adapters/mmvp_series/pressure_tookit/floor_audit.py
# 生产 floor 查看器：读 floor_<subject>.npy + 生产深度帧
python results_display/script/d_test6_floor_view.py --date 20260808 --subject S12
```

产出（`DataTest/D6Test_floor/`）：`<date>_floor_audit.png`（四联审计图）、
`floor_candidates.csv`（36 候选残差/棋盘格偏移/法向）、`<date>_<subject>_floor_view.png`
（生产地面查看图：floor 系点云 / 侧视 / RGB+ROI / 高度直方图）。

# 第二部分：结果检验（R_TestN）

## R_Test1 模型可视化

统一入口 `r_test1_visualize.py`（2026-10-06 用户裁定：**选中 + 生成什么**）。
`--models` 选中谁出谁（`anysole` 一次出四个系列 VT2M/V2M/T2M/V2T，基线各出自己
注册的那个 mode；缺省 = 全部）；`--compare` 同 mode 模型 1×N 横排——**输入只放
一次**（VT2M/T2M 触觉热力图，V2M/V2T 视频帧），中间是各模型的预测骨架（各自
原生协议 + 逐帧 MPJPE），右侧每个出现过的协议一栏 GT。

| 产物 | 位置 |
| --- | --- |
| 单独生成（输入\|预测\|GT 三栏动画，仅 gif/mp4） | `ResultTest/R1Test_visualize/<model>/<mode>/{gif,mp4}/<session>.{ext}` |
| 对比（同 mode 1×N 横排） | `ResultTest/R1Test_visualize/compare/<mode>/` |

```bash
# 单独生成：选中谁出谁；缺省全部模型
python results_display/script/r_test1_visualize.py --models anysole,motionpro --session S13011
python results_display/script/r_test1_visualize.py --session S13011

# 对比：同 mode 横排
python results_display/script/r_test1_visualize.py --compare --session S13011
```

`--variant` 缺省自动选 run（只有一个直接取、多个取最新并 log 点名）；`--variant all`
显式处理全部历史 run。缺预测的模型显示灰色占位并写明路径（不回退、不报错）。
`--split` 缺省 `val`（迭代集）；正式 36-session 集加 `--split test`。

四个分模型脚本（`r_test1_visualize_{anysole,mmvp,motionpro,step2motion}.py`）与旧
`r_test1_compare.py` 已并入本入口，移至 `script/legacy/`。

模式/协议/能力的唯一声明点：`results_display/models_modes.yaml`（AnySole 的模式 =
config-id，无需声明；基线必须声明，解析不出模式的行不进比较）。跨模式行永不共屏。

## R_Test2 参数对照

`r_test2_compare.py` 是 R_Test 的统一数值评估入口（不是模型推理入口）。它对 AnySole、MotionPRO、Step2Motion 和 MMVP 按 VT2M/V2M/T2M/V2T 分块评估。Motion 模式按**各模型原生协议**计算完整骨架指标（SMPL-24 对 SMPL GT、BVH-23 对 BVH GT），
指标适用性由注册表能力门控；V2T 单独评估压力/接触重建指标（表列 brief 7 键，叶 5 键留在明细，见 `README_metrics.md` §4）。
产出（`ResultTest/R2Test_compare/`）：
`comparison_per_session.csv`（逐会话明细，含 `protocol`/`gt_source`/`metric_reasons` 列）、
`comparison_summary.csv`（精简表：模型 × 指标，含 `protocol` 列）、
`comparison_summary.png`（精简表的表格图，缺失项显示 `—`）、`evaluation.log`
（split/manifest hash + 每模型 protocol/ok/missing/invalid 统计）。

```bash
# 默认全模型 × val（--models 选中谁评估谁，anysole 一次四个 config）
python results_display/script/r_test2_compare.py --by-mode
python results_display/script/r_test2_compare.py --models motionpro,step2motion --by-mode
# 正式 36-session 集
python results_display/script/r_test2_compare.py --split test --by-mode --write-model-metrics --surface-metrics --force
```

基线模型清单来自 `models_modes.yaml`（协议/能力唯一声明点），AnySole 行按
model/contact-method/run（`--variant`）/config 展开；无法解析协议或能力的行被
拒绝并记入 `evaluation.log`（绝不猜测）。`--models-config <yaml|json>` 可指定外部
固定清单（条目必须自带 `protocol`/`capabilities`）。

### 状态与缺失值约定

- `missing`：预测文件不存在；`invalid`：声明协议与文件不符 / 缺统一契约 /
  帧数不符 / 声明为真但必要字段缺失；`not_applicable`（指标级）：模型没有该
  能力或来源为 GT 回填/模板重建。
- 不可用指标 CSV 留空、表格显示 `—`；逐会话 `metric_reasons` 列记录每项缺失原因。
- `--surface-metrics`：仅对声明 surface 能力的模型（MotionPRO / MMVP_VP-MoCap）
  计算 PVE；AnySole 按任务 01 已批准的 SMPL pose-only 协议（GT beta）例外启用；
  pressure_toolkit 与 Step2Motion 的 PVE 恒为 `—`。不传该 flag 时 PVE 列留空并
  记录 `not_computed` 原因。

### R_Test2 模式分块（--by-mode）

按 `results_display/models_modes.yaml` 把汇总表拆成 VT2M/V2M/T2M/V2T 四块，每块只含同模式行
（V2T 块表列严格 6 个 brief 键）；无法解析模式的行不进任何块并记入 `evaluation.log`（绝不猜测）。
基线未导出统一 predictions 时行状态为 missing（显式占位行）。没有 checkpoints 的
优化型基线也会正常注册；`checkpoints/` 只对训练型模型有意义。没有有效帧的 session
会记录为 `excluded_no_valid_frames`，不计入模型聚合。

`--write-model-metrics` 会把同一套指标写入各模型的
`results/<Model>/metrics/<split>_comparison.json`（schema
`mmvp_native_metrics_v2`，含 evaluation_protocol/capabilities/GT source/split 与
manifest hash/metric 缺失原因）；旧 `mmvp_common_metrics_v1`（common19）文件
不会被当作新结果读取，重跑时被 v2 取代。原生模型日志保留不改。

```bash
python results_display/script/r_test2_compare.py \
  --model-name V3_4b --contact-method joint_and \
  --config-id VT2M,V2M,T2M,V2T --split test --by-mode --force
```

产出（`ResultTest/R2Test_compare/by_mode/`）：`mode_overview.png`（四块堆叠总览）、
`<mode>/comparison_summary.{csv,png}`。

## R_Test3 轨迹可视化

`r_test3_traj.py` 可视化预测根轨迹 vs 真实轨迹（**SMPL/BVH 协议自动检测**）：
单面板 3D 空间动画（gif/mp4）+ 每 session 一张静态图（png，3D 斜视图 / 俯视 / 高度曲线）。
动画中 GT 整段显示（橙）、预测轨迹随帧生长（蓝），窗口边界用小点标记（每窗口在 GT 锚点处重新锚定），footer 实时显示当前帧 ATE 与整段 ATE。

**数据来源（双协议，配对 GT）**：

- SMPL 协议模型（AnySole）：`anysole.eval` 导出的原生 SMPL motion NPZ
  （`predictions/eval_motion/<session>_<config>.npz`）内嵌轨迹字段
  `pred_pelvis_trans` / `gt_pelvis_trans`（与 `root_ate_mm` 指标严格同源）。
- BVH 协议模型（Step2Motion 等）：`predictions/<run>/<session>_gen.bvh`，
  预测轨迹取 BVH Hips 路径，GT 取该 session 的 BVH GT Hips 路径
  （两者都重采样到会话网格）——不再对 BVH 模型共用 SMPL GT。
- 统一 `eval_motion` NPZ（基线）：`joint_xyz_world` 根关节配 SMPL GT pelvis。

R_Test3 只读这些文件，不重新推理；请先重跑 eval 刷新产物：

```bash
conda activate touch_gait

# 重新 eval（自动推导 ckpt/metrics 路径，同时刷新标准 SMPL motion NPZ）
python -m anysole.eval \
  --model-name V4A \
  --contact-method joint_and

# R_Test3 渲染（默认 主模型 + 消融 × 全部 config × test split）
python results_display/script/r_test3_traj.py --model-name V4B --gen gif --contact-method joint_and
# 单 session / 单 config
python results_display/script/r_test3_traj.py --session S7013 --config-id VT2M
# baseline 模型（扫描 results/ 下自包含模型目录，排除 *_backup*）
python results_display/script/r_test3_traj.py --auto
```

输出：`ResultTest/R3Test_traj/AnySole/<model-name>/<config>/{gif,mp4,png}/<session>_<config>_traj.{gif,mp4,png}`；
baseline 模型输出在 `ResultTest/R3Test_traj/<model>/gen/`。

> 注意：eval 生成的 motion NPZ 必须与 checkpoint 配套。若重新训练了模型，
> 旧的 `eval_motion/` 产物会与新 metrics 不一致，
> 需重跑上述 eval 命令再渲染 R_Test1/R_Test3。

### R_Test3 并排轨迹对比（--compare，配对协议 GT）

每模式一行轨迹面板：所有面板共享同一 `TrajProjector`（union 全部预测 + GT 定相机），
可直接目测各模型的轨迹偏差；主模型蓝 / GT 橙 / 基线注册表配色，footer 显示各模型
整段 ATE。**每个模型只画自己配对协议的 GT**（SMPL 模型配 SMPL root GT，BVH 模型配
BVH/Hips root GT），ATE 对各自配对 GT 计算；面板标题与静态图图例标协议。
只纳入 `root_translation=true` 且轨迹来源不是 GT 回填/模板重建的正式模型；被拒模型
记录原因到日志。缺预测的模型在行内跳过并记 warning（行内其余模型照常渲染）。
静态图按协议分组：SMPL 模型和 BVH 模型各画在自协议 GT 的坐标轴上。

```bash
# 冒烟
python results_display/script/r_test3_traj.py --compare --session S13013 --mode VT2M
# 三模式全出（默认 val，--split test 换正式 36 集）
python results_display/script/r_test3_traj.py --compare --models anysole,motionpro
```

产出（`ResultTest/R3Test_traj/compare/`）：`<mode>/{gif,png}/<session>_<mode>_traj_compare.{gif,png}`。

## R_Test4 触觉生成（V2T）

`r_test4_v2t.py` 默认只读取已经导出的标准 V2T archive，不重新推理模型。它展示
**视觉→触觉（V2T）生成器**的结果；旧的即时推理诊断保留为显式 `--source infer`。

V2T 指标统一到每脚 31×11 网格：AnySole 的 4×12 网格只在评估层双线性重采样，FPP-Net 的
31×11 网格直接使用。正式 V2T 指标为**四层层级（brief 7 + 叶 5，见
`README_metrics.md` §4）**：表列只含 `T_corr`、`T_rmse`、`pressure_force_rmse`、
`pressure_force_r2`、`pressure_cop_error_left`、`pressure_cop_error_right`；
叶键 `T_mse`、`T_mae`、`pressure_force_mae`、`pressure_cop_error_mean` 只进
逐会话明细。contact 不再是正式 V2T 指标
（archive 中的 contact 字段仅为诊断保留，不输出、不消费）。

三种条件（与 eval.py 的 `--config-id` 同名）：

| 条件 | 输入 | 意义 |
| --- | --- | --- |
| `VT2M` | 真 V + 真 T | 重建 sanity（上界参考） |
| `V2M` | 真 V + 零 T | **V2T 生成本身（主指标）** |
| `T2M` | 零 V + 真 T | 触觉自重建（输入端 sanity） |

产出（`ResultTest/R4Test_v2t/`）：`tgen_summary.csv`、`tgen_report.json`、31×11 统一网格的
逐格误差 `cells/*.npz`/`*.png`，以及热力图动画（布局 = D4Test_baseline_tactile 口径：
1×3 面板 GT | Generated | |GT-Gen|，每面板 L/R 鞋垫块 + 分离式双行标题，0.75 缩放）。

与 `anysole.eval` 的关系：`anysole.eval` 和 FPP-Net 导出器共同写标准 V2T archive，R_Test4
只消费这些 archive；R_Test2 与 R_Test4 使用同一套 V2T 指标定义。

```bash
conda activate touch_gait
# 默认 anysole(V3_3B) + FPP-Net，全部 test 会话；--models 选中谁评估谁
python results_display/script/r_test4_v2t.py --source archives --split test
python results_display/script/r_test4_v2t.py --source archives --session S10103
# 仅运行旧的模型即时推理诊断：
python results_display/script/r_test4_v2t.py --source infer --config-id VT2M,V2M --export-sessions 2
```

> `--source infer` 是兼容性的诊断入口，不是最终跨模型评估入口。

## A/B 组缺失率实验（A0–B3，注册式全链路）

正式入口见 `configs/Z_README.md`。当前由 `configs` 生成两组数据任务
（主线模型 `V3_3B`），再由 `results_display/` 的分析脚本读取结果并出图；
A0–B3 不再生成调度 conf。

| 编号 | 任务书实验 | 回答的问题 | 脚本 |
| --- | --- | --- | --- |
| A0 | 专才分工 | V、T 原本是否各有所长 | `singlemodal_analysis.py` |
| A1 | 完整输入融合 | VT 是否吸收两路优势 | `complement.py` |
| A2 | 缺失分支分工 | 主线 V-only 与 T-only 谁负责什么 | `dropout_ablation.py` |
| B1 | V 分支的 T 相关能力 | 从 (F_V) 能否读出 T 负责分量 | `v2t_upper.py` |
| B2 | T 分支的 V 相关能力 | 从 (F_T) 能否读出 V 负责分量 | `trust.py` |
| B3 | 连续缺失鲁棒性 | ρ 网格下任意缺失组合是否稳定 | `rho_grid_analysis.py` |

### B3 ρ 网格（已实装）

在 (ρV, ρT) 保留率网格上逐一评估：每格对每一帧独立掷硬币，把该模态 token 换成 null
token（与 config 级 null 逐字节一致；四角 = VT2M/V2M/T2M/纯先验，corner_check 自检 ≈0）。

**生产**（148 个任务 = train 2×2 + val 6×6×3 种子 + test 6×6，`--reuse` 断点续跑）：

```bash
python configs/z_gen/rho_grid_eval.py --models V3_3B
CUDA_VISIBLE_DEVICES=4 bash configs/bg.sh
```

产出（`results/experiments/rho_grid_eval/V3_3B/`）：
- `grid_metrics[_<split>].json` —— 全格指标 + corner_check
- `npz/<session>_rhoV<rV>_rhoT<rT>[_s<seed>].npz` —— eval_motion 同格式（接 R_Test1 动画）
- `repr/<session>_..._repr.npz` —— F/t_tok/v_tok 表征

**出图**（热力图 + 切片 + 自检表）：

```bash
python results_display/script/rho_grid_analysis.py --split test
```

前端输出在 `results_display/BTest/B3Test_rho_grid/V3_3B/`（`heatmap.png` / `slices.png` /
`corner_check.csv` / `b3_criteria.json`）。

### A0–B2 分量分析

三份明细文件（主线、V 专才、T 专才）由 `singlemodal_eval` 队列任务落盘到
`results/experiments/singlemodal_eval/{V3_3B,V3_3B_vonly,V3_3B_tonly}/metrics/<split>.json`，
入口脚本按 registry 自动定位，无需手工传路径；B1/B2 另读
`results/experiments/rho_grid_eval/V3_3B/grid_metrics[_<split>].json` 作空输入先验。
实现收敛在 `utils/component_analysis.py`，仍可单独按 `--a0/--a1/--a2/--b1/--b2` 调用：

```bash
conda activate touch_gait
# A0
python results_display/script/singlemodal_analysis.py --split val
# A1
python results_display/script/complement.py --split val
# A2
python results_display/script/dropout_ablation.py --split val
# B1（需 rho_grid 已跑完）
python results_display/script/v2t_upper.py --split val
# B2（需 rho_grid 已跑完）
python results_display/script/trust.py --split val
# --split 默认 test；--model V3_3B 限定模型
```

输出：`results_display/{ATest,BTest}/<编号Test_分析名>/V3_3B/`（分析名 = A0Test_specialist /
A1Test_complement / A2Test_dropout_ablation / B1Test_v2t_upper / B2Test_trust）。

## 附录：训练侧 dropout 开关（两个独立旋钮）

`python -m anysole.train` 现在有两个独立 dropout 参数，都会写入 checkpoint 配置（eval/infer 自动读取）：

- `--dropout 0.1`：模型内部 nn.Dropout（fusion/pose/traj transformer）。`0.0` 关闭。
- `--dropoutVT DROP_V_PCT DROP_T_PCT`（或逗号写法 `--dropoutVT 20,30`）：**模态 dropout**，
  V 丢弃 DROP_V_PCT%、T 丢弃 DROP_T_PCT%（映射为 config_probs `[1-v-t, t, v]`，即 VT/V-only/T-only）。
  `--dropoutVT 0,0` 关闭模态 dropout。

重训实验示例（关全部 dropout + 更多轮次）：

```bash
conda activate touch_gait
python -m anysole.train --contact-method joint_and \
  --dropoutVT 0,0 --epochs 400 \
  --out-dir results/AnySole/anysolev1_joint_and/checkpoints
```

> 注意：重训后必须重跑 `python -m anysole.eval --model-name <MODEL_NAME> --contact-method ...` 刷新 BVH/npz/metrics（见 R_Test3 说明）。
>
> 上述 dropout 旋钮仅适用于 anysolev1 扩散线；主线 anysolev2 为回归式模型，忽略这些参数
> （train.py 会提示 knobs are ignored by modal anysolev2）。
