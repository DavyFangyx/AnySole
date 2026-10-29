# Results Display

集中存放展示产物与展示层脚本。展示行为按目的分三册，互不耦合：

- **Etest/**：E 系列实验（E1–E8）的展示产物——表 = CSV、图 = PNG，
  命名规范 `用途_图表编号_实验编号_内容名称`（Main_/Appendix_/Full_），
  详见 `README_e_test.md`（命名规范与逐实验参数约定）；生成/刷新 = `script/e_results_export.py`。

- **第一部分：数据检验（D_TestN）** —— 处理**原始数据**（对应 AnysoleWorkspace）：
  原始 GT 的可视化、自检、标签判定与协议探针。逐实验说明见 `README_d_test.md`。
- **第二部分：结果检验（R_TestN）** —— 处理**实验结果**（对应 results）：
  模型输出的可视化、参数对照、轨迹与专项消融。逐实验说明见 `README_r_test.md`。
- **各模型对应的评价指标**（能力矩阵、指标唯一实现位置、`—` 规则、V2T 层级）见
  `README_metrics.md`。

分层定位（2026-10-28 用户裁定）：**训练层**（train+val）与**测试层**（test）走
`configs/` 的 conf 管线（入口见 `configs/Z_README.md`）；**本层（display）**把 test
结果整理成表图，不进 queue、不经过 scheduler。

实验代码位于 `script/`（唯一被 git 跟踪的目录），生成物落在对应的实验目录。
本文已按 D/R 编号组织；P1-4 的脚本平铺改名已落盘（命令照抄可用），
新旧对应见下方对照表。

可用环境变量 `ANYSOLE_RESULTSDISPLAY` 覆盖本根目录。

## display 层 CLI（三层解耦，2026-10-28 用户裁定）

三层：**训练层**（train+val）与**测试层**（test）走 `configs/` 的 conf 管线
（入口见 `configs/Z_README.md`）；**本层（display）**把 test 结果整理成表图，
不进 queue、不经过 scheduler：

- E 系列（E1–E8）表图导出与驱动 CLI：见 `README_e_test.md` 的运行指令
- D/R 系列逐实验 CLI：见 `README_d_test.md` / `README_r_test.md`

## 编号对照

| 编号 | 说明 | 脚本 |
| --- | --- | --- |
| D_Test1 原始数据可视化 | GT/输入渲染（BVH/SMPL） | `d_test1_data_viz.py` |
| D_Test2 数据集自检 | 数据集自检 | `d_test2_dataset_check.py` |
| D_Test3 接触检测 | 接触标签判定（含 `smpl_contact` = SMPL 网格足底 <2cm） | `AnysoleWorkspace/tool/contact_labels.py`；`d_test3_contact.py` |
| D_Test4 基线触觉审计 | 四基线触觉格式审计（审计只读） | `d_test4_baseline_tactile.py`（数据源 = `build_shared.py` 产物） |
| D_Test5 鞋垫偏移补偿（**待适配**） | 鞋垫漂移补偿器测试 | `d_test5_insole_drift.py` |
| D_Test6 地面估计与坐标系审计 | 行走地面 ROI 审计（`AnysoleWorkspace/tool/.../floor_audit.py`） + 生产 floor 查看 | `floor_audit.py`；`d_test6_floor_view.py` |
| D_Test7 遮挡管线可视化 | cam3 bbox 底部遮挡（f∈{10,20,30,50}%）：原始帧 + HRNet 输入 crop，与训练提取逐位相同 | `d_test7_occlusion.py` |
| D_Test8 GT 足底热图/CoP × 触觉热图/CoP | 2×2 同帧：GT 足底热图 / 触觉热图 / GT CoP / 触觉 CoP（同一 `cop_from_grid`） | `d_test8_smpl_gt_cop.py` |
| D_Test9 BVH × SMPL 脚部对照 | 2×4 同帧：双源全身/脚侧视/接触网格/同框对照/高度时间线×接触标签（关节定义与趾间隙审计） | `d_test9_bvh_smpl_compare.py` |
| R_Test1 模型可视化 | 姿态动画（选中 + 生成什么统一入口） | `r_test1_visualize.py`（缺省单独生成；`--compare` 同 mode 1×N 横排：输入一次 + 各模型预测 + 每协议一栏 GT） |
| R_Test2 参数对照 | 跨模型/参数对照评估 | `r_test2_compare.py`（`--by-mode` 模式分块；原生协议 + 能力门控） |
| R_Test3 轨迹可视化 | 轨迹对比动画 | `r_test3_traj.py`（`--compare` 并排轨迹；配对协议 GT） |
| R_Test4 触觉生成 V2T | V2T 触觉生成 | `r_test4_v2t.py`（四层层级 brief 7 + 叶 5） |

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
│   │   ├── d_test6_floor_view.py       # D_Test6：生产 floor 查看（floor 系点云/RGB+ROI/高度直方图；审计工具在 AnysoleWorkspace）
│   │   ├── d_test7_occlusion.py        # D_Test7：遮挡管线可视化（原始帧 + HRNet crop；occ 缓存作闸门）
│   │   ├── d_test8_smpl_gt_cop.py      # D_Test8：GT 足底热图/CoP × 触觉热图/CoP（2×2 同帧）
│   │   └── d_test9_bvh_smpl_compare.py # D_Test9：BVH × SMPL 脚部对照（2×4 + 高度时间线）
│   ├── ── 结果检验（R_TestN）──
│   │   ├── r_test1_visualize.py         # R_Test1：姿态动画（选中 + 生成什么；--compare 同 mode 横排）
│   │   ├── r_test2_compare.py          # R_Test2：跨模型/参数对照评估（只出指标）
│   │   ├── r_test3_traj.py             # R_Test3：轨迹对比动画 + 静态图
│   │   └── r_test4_v2t.py              # R_Test4：V2T 触觉生成
│   │   └── mpl_fonts.py              # CJK 字体公共设置（中文标注不豆腐块）
├── DataTest/                           # D 组：数据检验产物（D_TestN）
│   ├── D1Test_data_viz/                # D_Test1 输出（<session>/<smp24|bvh23>/）
│   ├── D2Test_dataset_check/           # D_Test2 输出（自检 JSON + 均值姿态 npz）
│   ├── D3Test_contact/                 # D_Test3 输出（按方案分目录）
│   ├── D4Test_baseline_tactile/        # D_Test4 输出（每 session 一个 <session>_adapted_tactile.gif/.mp4）
│   ├── D5Test_insole_drift/            # D_Test5 输出（补偿前后对比动画 + theta_summary.csv）
│   ├── D6Test_floor/                   # D_Test6 输出（<date>_floor_audit.png + floor_candidates.csv + <date>_<subject>_floor_view.png）
│   ├── D7Test_occlusion/               # D_Test7 输出（<sid>_occ<FFF>.gif/.mp4）
│   ├── D8Test_smpl_gt_cop/             # D_Test8 输出（<sid>_smpl_gt_cop.gif/.mp4）
│   └── D9Test_bvh_smpl_compare/        # D_Test9 输出（<sid>_bvh_smpl.gif/.mp4）
├── ResultTest/                         # R 组：结果检验产物（R_TestN）
│   ├── R1Test_visualize/               # R_Test1 输出（AnySole/MMVP/MotionPRO/Step2Motion 分栏 + compare/<mode> 并排）
│   ├── R2Test_compare/                 # R_Test2 输出（只出指标，不做动画 + by_mode/ 模式分块）
│   ├── R3Test_traj/                    # R_Test3 输出（+ compare/<mode> 并排轨迹）
│   └── R4Test_v2t/                     # R_Test4 输出（tgen 汇总 + 热力图动画）
├── Test1_ComplementTest/               # 实验一：V 与 T 是否互补（C1–C3，各含 <model>/ 子目录）
│   ├── C1_specialist/                  # C1：专才分工（specialist_table.csv / specialist_forest.png / summary.json）
│   ├── C2_fusion/                      # C2：完整输入融合（fusion_* 产物）
│   └── C3_missing_branch/              # C3：缺失分支分工（main_branch_* 产物）
└── Test2_MissingTest/                  # 实验二：缺一路/连续缺失下主线能否工作（M1–M3，各含 <model>/ 子目录）
    ├── M1_v2t/                         # M1：从 F_V 读出 T 负责分量（branch_* 产物）
    ├── M2_t2v/                         # M2：从 F_T 读出 V 负责分量（branch_* 产物）
    └── M3_rho_grid/                    # M3：ρ 网格连续缺失鲁棒性（heatmap/slices/criteria）
└── Test6_VdegradTest/                  # 实验六：V 削弱难度轴（V削弱实验报告.md 两套件）
    ├── V3_Bv1/                          # 融合主线套件（delta_table / physical_delta / win_rates / delta_panels）
    └── V3_Dv1/                          # 干净通道套件（+ decile_table / decile_panels）
```

## 公共约定

### 时间对齐约定（动捕 ↔ 触觉/视频）

视频、触觉与动捕之间存在时间偏差与漂移。偏差来自各设备独立起录：每个 session 的
人工复核表 `AlignReviews_csv/<session>.csv` 记录常量偏移 `偏移量(s)`（= 视觉时间 − 动捕时间）；
漂移则由触觉/视频时间轴按逐帧时钟重建（触觉 `t_us` 时间戳、视频文件名时钟）吸收。
所有动捕与触觉同帧对比的脚本，GT motion 都按同一补偿方式
重采样到 40 Hz 会话网格：`t_mocap = t_grid − offset_s`，其中 `t_grid = visual_start_s + n/40`

### 模型协议约定（原生 SMPL-24 / BVH-23 双协议，无 common19）

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

## 数据生产实验与分析入口（历史）

数据生产实验统一由 `configs/` 调度（正式入口见 `configs/Z_README.md`，当前
实验线 = V7/U 家族，实验定义见 `z_note/实验计划.md`，结果文档在
`results_display/Etest/`）。任务由 `configs/tools/runner.py` 负责 checkpoint 准备、
val/test 正式评估和结果归档。

已归档的数据生产实验（`singlemodal_eval` / `rho_grid_eval` / `vdegrad` /
C1–M3 分析）不再新增：模型级结果已迁至 `results/cells/`（cell 命名），
`results/experiments/` 已整体删除（2026-10-26）。
