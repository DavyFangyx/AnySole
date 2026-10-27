# Gait 触觉步态生成项目

本项目从**鞋垫压力 + 单目视觉**生成 3D 人体运动：主模型 AnySole（扩散式
V+T→M 生成）与四个基线（MotionPRO、Step2Motion、pressure_toolkit、
VP-MoCap (FPP-Net + PoseTransOpt)）在同一套数据、同一份 split、同一套
评价指标口径下对比。

> 2026-09-26 评估整改后的终态协议：主模型与 SMPL 系基线使用原生 **SMPL-24**
> 对 SMPL GT 评测；Step2Motion 使用原生 **BVH-23** 对 BVH GT 评测；V2T 压力
> 为四层层级（brief 8 + 叶 5，含接触级）；缺失能力显示 `—`。协议细节见
> `z_note/EVALUATION_PROTOCOL_AUDIT.md`。

## 统一前提（所有模型共用）

- 相机固定 **cam3**、采样 **40 Hz**、同一份 session split
  （当前默认 train/val/test = 92/12/36，三组互斥，`AnysoleWorkspace/protocol/splits/default/splits.csv`）。
  2026-10-23 前生成的历史结果仍绑定当时的 104/36/36、val=test split，不随当前文件改写。
- 中间数据、GT 与适配数据一律在 `AnysoleWorkspace/`；任何模型不得自行划分数据。
- 主模型 AnySole 只读写 SMPL；BVH 仅供 Step2Motion 基线。

## 目录分工

```text
gait/
├── README.md                        # 本文件：项目介绍与目录分工（无执行命令）
├── anysole/                         # 主模型源码（训练/评估/推理入口）
│   ├── model_fix_note/command_manual.md   # 主模型训练评估命令手册
│   └── configs/v1.yaml              # 主模型默认配置
├── Baselines/                       # 基线模型源码（只放源码，不写运行结果）
│   ├── MotionPRO/  Step2Motion/     # 学习型基线（SMPL / BVH）
│   ├── pressure_tookit/             # 优化型基线（目录保留历史拼写 tookit）
│   ├── VP-MoCap/                    # FPP-Net（V2T） + PoseTransOpt（V2M）
│   └── command_manual.md            # 基线训练/评估/方法运行手册
├── AnysoleWorkspace/                # 数据与过程产物工作区（唯一数据工作区）
│   ├── README.md                    # 中间数据准备手册（全部数据构建命令）
│   ├── raw/  protocol/  shared/  model_inputs/  work/  assets/  reports/
│   └── tool/                        # 数据构建代码（唯一入口）
├── results/                         # 最终结果区（checkpoint/预测/评测结果）
│   ├── README.md                    # 主模型与基线的文件树说明
│   ├── AnySole/                     # 主模型结果（<modal>_<contact>/[<variant>/]）
│   ├── baselines/                   # 基线结果（每模型 predictions/ + metrics/）
│   ├── experiments/  logs/  backup/ # 队列实验 / 运行日志 / BVH 时代归档
├── results_display/                 # 展示与跨模型比较（只消费 results/ 的正式结果）
│   ├── README.md                    # 各展示任务（D_TestN / R_TestN）使用说明
│   ├── README_metrics.md            # 各模型对应的评价指标说明（能力矩阵/口径/落点）
│   ├── models_modes.yaml            # 跨模型比较的模式注册表（唯一声明点）
│   └── script/                      # 各 R_Test/D_Test 消费端脚本
├── configs/                         # 后台训练/实验队列（单一共享队列，见 configs/Z_README.md）
├── metrics.py                       # 公共指标公式的唯一实现（各端只 import，不重写）
├── tests/                           # 公共指标与协议消费端的数值测试
└── z_note/                          # 过程文档：评估整改任务书/审计/历史记录
```

三个结果相关根目录的职责（评估整改任务 02 固定）：

- `Baselines/`：只放模型源码、原始配置、运行入口，**不写正式运行结果**。
- `AnysoleWorkspace/`：数据输入、适配数据、初始化、优化过程文件等可重建中间产物。
- `results/`：checkpoint、最终预测、最终评测结果；展示端只从这里消费。
- `results_display/`：展示与跨模型比较，只消费 `results/` 的正式结果。

## 方案 A 的正式 SMPL 评估

方案 A（`visual` / `view_consistency.v1`）的正式动作评分统一使用工作区
[`2_Code/smpl_evaluation/anysole_a_native.py`](../2_Code/smpl_evaluation/anysole_a_native.py)，
直接调用 `2_Code/smpl_evaluation` 的指标、GT 插值和 SMPL 几何实现。
它对15个已冻结正式 run 在三基线相同的完整原生 RGB 帧上重新推理：
T20/stride10 遍历整段、融合窗口预测后连续积分，计算 W100、Root ATE 和
Foot sliding 等规范指标，不重新训练或选择 checkpoint。Root ATE 明确计入
固定 native-world 初始根位置 p0=0 的误差，不借 GT 初始化。
命令、支持和完整新表见
[方案A组间性能与三基线对比](../3_Result/AnySole/方案A_组间性能与三基线对比.md)。
`anysole_a.py` 保留为旧40Hz保存输出的历史重评分入口。

`anysole/utils/eval_protocol.py` 的 `visual-v1` 动作函数及旧训练 selector
保留历史复现用途；其 root0/旧时钟/分段定义不能作为本次规范成绩。
压力没有对应的 SMPL 标准；原40Hz压力分数与相关性数值修复单独记录，
不能视作本次原生 RGB 时序的压力评分。
下面的旧评估入口说明不覆盖方案 A 的这一权威评估约定。

## 指标实现落点（旧协议入口）

同名指标只有一个实现，各端只 import：

- 公式唯一实现：根 `metrics.py`（`anysole/utils/metrics.py` 是它的转发 shim）。
- 主模型评估端 `anysole/utils/eval_protocol.py` 与展示端
  `results_display/script/utils/compare_core.py` 调用同一批函数计算。
- 基线模型不实现公式：只导出 `predictions/`，公共比较统一由
  `results_display/script/r_test2_compare.py`（compare_core → metrics.py）计算。
- V2T 压力指标的公共口径在 `compare_core.pressure_metrics`（31×11 归一化网格，
  四层层级（网格/力/CoP/接触）：表列 brief 8 键、叶 5 键进明细）；主模型训练明细在原生 4×12
  布局上计算（`anysole/eval.py:_tactile_corr`）。
- 各模型对应的启用/不适用指标详见 `results_display/README_metrics.md`。

## 快速入口

| 要做什么 | 看哪里 |
| --- | --- |
| 准备数据（清洗/序列/特征/深度/关键点/CLIFF 初始化） | `AnysoleWorkspace/README.md` |
| 训练/评估主模型 AnySole | `anysole/model_fix_note/command_manual.md` |
| 训练/评估/运行基线 | `Baselines/command_manual.md` |
| 跨模型比较与可视化（R_Test1/2/3/4） | `results_display/README.md` |
| 各模型评价指标与能力矩阵 | `results_display/README_metrics.md` |
| 后台队列跑实验 | `configs/Z_README.md` |
| 协议与口径审计 | `z_note/EVALUATION_PROTOCOL_AUDIT.md` |

## 环境

各步骤使用的 conda 环境按其用途隔离：`touch_gait`（主模型训练/评估/展示）、
`mmvp`（pressure_toolkit / FPP-Net / PoseTransOpt / CLIFF 拟合）、`depthpro`
（深度估计）、`bbox_scan`（MMPose 关键点）。环境体检：

```bash
python AnysoleWorkspace/tool/workspace.py doctor
```

> 历史说明：本文由早期 `README_usage.md`（已归档删除）整理而来；原执行命令已按职责拆分到
> `AnysoleWorkspace/README.md`（数据准备）、`Baselines/command_manual.md`
> （基线训练评估）与 `anysole/model_fix_note/command_manual.md`（主模型）。
