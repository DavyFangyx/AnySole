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
| D_Test4 基线触觉审计 | 四基线触觉格式审计 | `d_test4_baseline_tactile.py` → `AnysoleWorkspace/tool/generate_baseline_tactile.py` |
| D_Test5 鞋垫偏移补偿（**待适配**） | 鞋垫漂移补偿器测试 | `d_test5_insole_drift.py` |
| R_Test1 模型可视化 | 模型动画 | `r_test1_visualize_{anysole,mmvp,motionpro,step2motion}.py`；`r_test1_compare.py`（跨模型并排，模式对齐） |
| R_Test2 参数对照 | 跨模型/参数对照评估 | `r_test2_compare.py`（`--by-mode` 模式分块） |
| R_Test3 轨迹可视化 | 轨迹对比动画 | `r_test3_traj.py`（`--compare` 并排轨迹） |
| R_Test4 触觉生成 V2T | V2T 触觉生成 | `r_test4_v2t.py` |
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
│   │   ├── d_test4_baseline_tactile.py # D_Test4：四基线触觉格式审计（生成器前置，产物落盘）
│   │   └── d_test5_insole_drift.py     # D_Test5：鞋垫漂移补偿器测试（待适配）
│   ├── ── 结果检验（R_TestN）──
│   │   ├── r_test1_visualize_{anysole,mmvp,motionpro,step2motion}.py   # R_Test1：模型动画
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
│   └── D5Test_insole_drift/            # D_Test5 输出（补偿前后对比动画 + theta_summary.csv）
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

## 模型协议约定（SMPL / BVH 双协议）

展示与对照层（D_Test1 与 R_Test1/2/3）按文件格式自动检测协议，无协议 flag：

| 协议 | 模型 | 产物约定 | 检测规则 |
| --- | --- | --- | --- |
| SMPL-24 | AnySole / MotionPRO / MMVP pressure_toolkit / MMVP VP-MoCap | `predictions/eval_motion/<session>[_<config>].npz`，统一包含 `joint_xyz_world`、`joint_names`、`valid_mask`；AnySole 仍保留原生 SMPL 字段 | `joint_xyz_world` + 24 个 `joint_names` → SMPL-24 |
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
  模型比"啥也不干"还差，更像 bug（脚本会自动与 `anysolev1_joint_and/metrics/test.json` 对比）。

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

产出（`DataTest/D3Test_contact/`）：每个方案一个子目录（`<method>/`）：
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
# 只跑 test
python results_display/script/d_test4_baseline_tactile.py --split test
python results_display/script/d_test4_baseline_tactile.py --session S12072 --force
python results_display/script/d_test4_baseline_tactile.py --session S12072 --gen mp4
# 只落盘不渲染
python AnysoleWorkspace/tool/generate_baseline_tactile.py --split test
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
| `r_test1_visualize_motionpro.py` | MotionPRO 触觉输入/预测/GT 三栏对比动画 | `ResultTest/R1Test_visualize/MotionPRO/<checkpoint tag>/` |
| `r_test1_visualize_step2motion.py` | Step2Motion 足底压力/生成 BVH/GT 对比动画 | `ResultTest/R1Test_visualize/Step2Motion/gait_model/` |
| `r_test1_visualize_anysole.py` | AnySole 主模型与消融（足底压力/预测动作/SMPL GT；自动检测格式） | `ResultTest/R1Test_visualize/AnySole/<modal>/<config>/` |
| `r_test1_visualize_mmvp.py` | MMVP 两条方法线的轻量动画（消费 eval_motion npz，缺预测显式报告） | `ResultTest/R1Test_visualize/` |

```bash
conda activate touch_gait

# motionpro 可视化
python results_display/script/r_test1_visualize_motionpro.py
--session S14103
--checkpoint results://MotionPRO/checkpoints/imagepressure2smpl/init/5e-05/imagepressure2smpl_best.pth

# step2motion 可视化
python results_display/script/r_test1_visualize_step2motion.py
--self-test

# 默认 主模型 + 消融，全部 config
CUDA_VISIBLE_DEVICES=4 python results_display/script/r_test1_visualize_anysole.py --model-name V4B --contact-method joint_and --gen gif --render bone,mesh --session S12042
--model-name V4B
--contact-method bvh_h,bvh_soft,tactile_abs,pat_offset,joint_and
--config-id VT2M

# 历史 run 对照
--model-name F0b --contact-method joint_and
```

`--render bone,mesh`：面板渲染类型，逗号分隔；`bone` = 骨架面板（默认），`mesh` = 纯 SMPL
表面面板（不叠骨架）。两种类型像 gif/mp4 一样**分目录输出**，互不混叠：
`ResultTest/R1Test_visualize/AnySole/<model-name>/<config>/{bone,mesh}/{gif,mp4}/`。
无 SMPL 参数的文件（BVH，如 Step2Motion）只有 bone 可渲染，mesh 自动跳过并告警。

### R_Test1 并排对比（compare/，模式对齐）

`r_test1_compare.py` 按生成模式生成一行式并排动画：同模式模型共屏、共享同一渲染器
（common19 火柴人），主模型蓝 / GT 橙 / 基线按注册表配色，每栏 footer 显示该模型逐帧 MPJPE。
输入栏随模式变：VT2M/T2M 触觉热力图，V2M 真实 RGB 帧（自动对比度增强）。
缺预测的模型显示灰色占位并写明缺失路径。

| 模式 | 行布局 |
| --- | --- |
| VT2M | 触觉 \| AnySole VT2M \| MotionPRO \| MMVP_pressure_toolkit \| GT |
| V2M | 视频帧 \| AnySole V2M \| MMVP_FPP-Net \| GT |
| T2M | 触觉 \| AnySole T2M \| Step2Motion \| GT |

模式归属的唯一声明点：`script/models_modes.yaml`（AnySole 的模式 = config-id，无需声明；
基线必须声明，解析不出模式的行不进比较）。跨模式行永不共屏。

```bash
conda activate touch_gait
# 单 session 冒烟
python results_display/script/r_test1_compare.py --mode VT2M --session S13013 --gen gif
# 单模式 × val 全量
python results_display/script/r_test1_compare.py --mode V2M --split val
# 三模式全出（默认 val，--split test 换正式 36 集）
python results_display/script/r_test1_compare.py --model-name V4B --contact-method joint_and
```

产出（`ResultTest/R1Test_visualize/compare/`）：`<mode>/{gif,mp4}/<session>_<mode>_compare.{gif,mp4}`。
单个模型可视化目录不受影响。

## R_Test2 参数对照

`r_test2_compare.py` 是 R_Test 的统一数值评估入口（不是模型推理入口）。它对 AnySole、MotionPRO、Step2Motion 和 MMVP 按 VT2M/V2M/T2M/V2T 分块评估。Motion 模式使用 canonical common19 joint/pose/trajectory/temporal/surface 指标；V2T 单独评估压力重建和接触。
产出（`ResultTest/R2Test_compare/`）：
`comparison_per_session.csv`（逐会话明细）、`comparison_summary.csv`（精简表：仅模型 × 指标）、
`comparison_summary.png`（精简表的表格图）、`evaluation.log`（运行信息 + 各模型 checkpoint）。

```bash
python results_display/script/r_test2_compare.py \
  --manifest AnysoleWorkspace/manifests/session_manifest.csv \
  --split-csv AnysoleWorkspace/splits/default/splits.csv \
  --model-name V4A \
  --contact-method joint_and \
  --config-id VT2M,V2M,T2M,V2T \
  --split test --by-mode --write-model-metrics --surface-metrics --force

# 或者完全自动扫描 results/ 下的自包含模型目录
python results_display/script/r_test2_compare.py --auto-scan
```

模型清单默认由脚本扫描 `results/` 自动构建；也可用 `--models-config <yaml|json>` 指定固定清单。

### R_Test2 模式分块（--by-mode）

按 `script/models_modes.yaml` 把汇总表拆成 VT2M/V2M/T2M/V2T 四块，每块只含同模式行；
无法解析模式的行不进任何块并记入 `evaluation.log`（绝不猜测）。基线未导出统一
predictions 时行状态为 missing（显式占位行）。没有 checkpoints 的优化型基线也会正常注册；
`checkpoints/` 只对训练型模型有意义。没有有效帧的 session 会记录为
`excluded_no_valid_frames`，不计入模型聚合。平表 CSV 末尾追加 mode/family 两列。

`--write-model-metrics` 会把同一套指标写入各模型的
`results/<Model>/metrics/<split>_comparison.json`；原生模型日志保留不改。

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

**数据来源（双协议）**：

- SMPL 协议模型（AnySole）：`anysole.eval` 导出的原生 SMPL motion NPZ
  （`predictions/eval_motion/<session>_<config>.npz`）内嵌轨迹字段
  `pred_pelvis_trans` / `gt_pelvis_trans`（与 `root_ate_mm` 指标严格同源）。
- BVH 协议模型（Step2Motion 等）：`predictions/<run>/<session>_gen.bvh`，
  预测轨迹取 BVH 根关节路径，GT 用 `motion_io.load_session_gt`
  （SMPL 优先、BVH 回退）的根关节。

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

### R_Test3 并排轨迹对比（--compare，模式对齐）

每模式一行轨迹面板：所有面板共享同一 `TrajProjector`（union 全部预测 + GT 定相机），
可直接目测各模型的轨迹偏差；主模型蓝 / GT 橙 / 基线注册表配色，footer 显示各模型整段 ATE。
静态图为 top-down + 高度曲线双轴叠加（各模型配色 + ATE 图例）。缺预测的模型在行内跳过
并记 warning（行内其余模型照常渲染）。

```bash
# 冒烟
python results_display/script/r_test3_traj.py --compare --session S13013 --mode VT2M
# 三模式全出（默认 val，--split test 换正式 36 集）
python results_display/script/r_test3_traj.py --compare --model-name V4B --contact-method joint_and
```

产出（`ResultTest/R3Test_traj/compare/`）：`<mode>/{gif,png}/<session>_<mode>_traj_compare.{gif,png}`。

## R_Test4 触觉生成（V2T）

`r_test4_v2t.py` 默认只读取已经导出的标准 V2T archive，不重新推理模型。它展示
**视觉→触觉（V2T）生成器**的结果；旧的即时推理诊断保留为显式 `--source infer`。

V2T 指标统一到每脚 31×11 网格：AnySole 的 4×12 网格只在评估层双线性重采样，FPP-Net 的
31×11 网格直接使用。两者都计算压力、力、CoP 和接触指标；接触统一定义为每脚 canonical
grid 中至少一个归一化压力单元大于 0.5。archive 中的原生 contact 字段保留作溯源，但不作为
跨模型主指标来源。

三种条件（与 eval.py 的 `--config-id` 同名）：

| 条件 | 输入 | 意义 |
| --- | --- | --- |
| `VT2M` | 真 V + 真 T | 重建 sanity（上界参考） |
| `V2M` | 真 V + 零 T | **V2T 生成本身（主指标）** |
| `T2M` | 零 V + 真 T | 触觉自重建（输入端 sanity） |

产出（`ResultTest/R4Test_v2t/`）：`tgen_summary.csv`、`tgen_report.json`、31×11 统一网格的
逐格误差 `cells/*.npz`/`*.png`，以及热力图动画。

与 `anysole.eval` 的关系：`anysole.eval` 和 FPP-Net 导出器共同写标准 V2T archive，R_Test4
只消费这些 archive；R_Test2 与 R_Test4 使用同一套 V2T 指标定义。

```bash
conda activate touch_gait
python results_display/script/r_test4_v2t.py \
  --source archives --model-name V4A --contact-method joint_and --split test
python results_display/script/r_test4_v2t.py \
  --source archives --session S10103 --model-name V4A --contact-method joint_and
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

三份 fseries（主线、V 专才、T 专才）由 `singlemodal_eval` 队列任务落盘到
`results/experiments/singlemodal_eval/{V3_3B,V3_3B_vonly,V3_3B_tonly}/metrics/<split>_fseries.json`，
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
