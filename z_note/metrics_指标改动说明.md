# metrics 指标体系改动说明（v3 · 2026-09-26 · 评估整改任务 01 终态）

> 本文与 `z_note/评估/01_统一指标求解器与模型能力任务书.md` 的锁定口径一致；
> 旧版（v2）中的 brief 含 `contact_f1`、foot sliding 顶点版等描述已按新裁定废弃。
> 键名与唯一实现以 `anysole/utils/metrics.py`（主模型自持 utils）为准；基线侧
> 数据协议/评估协议统一放在 `Baselines/utils/`（2026-09-26 用户裁定结构），
> 公式一律从 anysole 侧唯一实现引用，任何位置不得存在第二份公式。

## 1. 目标与锁定口径

1. 每 split 两个同源 JSON：`<split>.json`（明细，全部指标 + 诊断）与
   `<split>_brief.json`（精简，论文候选指标）。
2. 所有公共指标公式集中到 `anysole/utils/metrics.py`（唯一实现），模型侧
   evaluator 与基线侧工具不得本地重定义同名指标。
3. 缺失能力输出缺失值（JSON 缺键 / NaN），展示统一 `—`，绝不写 0。
4. common19 已删除：SMPL 模型以原生 SMPL-24 对 SMPL GT、Step2Motion 以原生
   BVH-23（23 语义关节，去 EndSite）对 BVH GT 评测，两种表示不再裁剪。
5. 正式 contact 对当前所有 motion 模型为不适用（`—`）：没有经过训练且显式
   导出的 contact prediction。训练正则与诊断保留在明细，标注为 diagnostic。
7. `root_orientation_deg` / `root_orientation_drift_deg` 是旧
   `yaw_abs_deg` / `yaw_drift_deg` 的正式名称，公式只存在于 `metrics.py`
   （`root_orientation_error_deg` / `root_orientation_drift_deg`），
   Step2Motion 用解析出的 BVH Hips/root rotation 调用同一求解器。
8. 公共 `foot_sliding_mm` 为关节版（SMPL 左右 ankle/foot；BVH 左右
   Foot/ToeBase）；顶点版保留为 SMPL 专属诊断 `foot_sliding_vertex_mm`。
9. **评估资格以 §2 各模型生成清单为唯一依据**：模型/方法原生生成什么就
   评什么，清单外一律不评估（`—`）。

## 2. 各模型生成清单（评估资格的唯一依据）

评估只允许评价**生成清单内**的量：模型/方法原生生成什么，就评什么；
清单外一律不评估（`—`），不补造、不回填、不套模板。基线分两类，平等按
清单评估：**模型**（可训练，输出来自生成头）与**方法**（拟合/优化算法，
输出来自 pipeline）。

| 模型/方法 | 类型 | 生成清单（原生输出） | 可评估指标 | `—`（不评估） |
|---|---|---|---|---|
| AnySole | 模型 | SMPL pose、root translation、root rotation；触觉重建（4×12/脚压力图） | 关节 / 旋转（MPJAE）/ 轨迹 / 朝向 / temporal / foot sliding；V2T 压力层级 | PVE、shape/vertex 类（**待处理**：模型当前不生成 beta，导出档 betas 为 GT 会话 beta）、contact（**待处理**：soft_contact 非显式输出） |
| MotionPRO | 模型 | SMPL pose（24 关节旋转，含全局朝向）、shape(beta)、translation、表面 | 全套 motion + PVE（预测表面） | contact |
| Step2Motion | 模型 | BVH-23 关节位置、Hips translation/rotation | MPJPE / PA / W / WA、root、temporal、foot sliding | MPJAE（无旋转输出）、PVE、shape、contact |
| FPP-Net（V2T 基线） | 模型 | 压力图（31×11/脚）；SMPL 接触图（接触头串行派生自压力输出，仍为网络输出头） | V2T 压力层级 + 接触层（contact_smpl_mse [brief]、contact_smpl_bce [叶]） | motion 指标；contact_f1（**待处理**：缺二值接触 GT） |
| PoseTransOpt / VP-MoCap（V2M 基线） | 方法 | SMPL pose（24 关节旋转，含全局朝向/root rotation）、beta、translation（PoseTransOpt 优化输出） | 全套 motion + PVE | contact |
| pressure_toolkit | 方法 | SMPL pose、translation、root rotation、joints、**shape（拟合 beta → 表面）** | 关节 / 旋转 / 轨迹 / 朝向 / temporal / foot sliding；**PVE（拟合表面）** | contact |

### 待处理（有对应生成、但接口/GT 尚未就绪，暂不评估）

| 项 | 模型 | 原因 | 就绪条件 |
|---|---|---|---|
| PVE / shape 评估 | AnySole | **当前不生成 beta**：模型输出只有 24 关节 6D 姿态 + translation（POSE_DIM=144，无 beta 输出头）；导出档 `betas` 为 GT 会话 beta（`betas_source="ground_truth_session"`），GT-beta PVE 已废除不再补评 | 若需 PVE：先给模型加 beta/shape 生成头（模型结构改动），接入导出/评估接口后按清单评估 |
| 正式 contact | AnySole | `soft_contact` 是关键点推导的训练正则（`soft_contact_from_keypoints()`，诊断），非显式接触输出接口 | 明确显式接触输出接口后评估 |
| contact_f1 | FPP-Net | 预测侧可二值化，但 **GT 侧没有二值接触**：f6_soft 是软标签体系（0.05/0.30/0.70/0.95）无天然二值分界；硬决策 motion_f6 是运动学状态机、不是模型输出 | 明确二值接触 GT 后评估 |

> **FPP-Net 与 PoseTransOpt（VP-MoCap）是两个独立基线**：各有一条注册表
> 条目（`MMVP_FPP-Net` = V2T、`MMVP_VP-MoCap` = V2M）、各自的结果根
> （`baselines/FPP-Net/` 与 `baselines/VP-MoCap/`）。VP-MoCap 的 V2M 输出
> 来自 PoseTransOpt（其上游消费 FPP-Net 的预测，但 V2M 不归因于 FPP-Net
> 单模型；展示名保留生产链 `VP-MoCap (FPP-Net + PoseTransOpt)`）。
> 命令手册 `Baselines/command_manual.md` §3/§4 同此分工。

裁定要点（2026-09-27 用户裁定，取代此前"来源等级"口径）：

- **FPP-Net 接触属于生成清单**：网络结构确认接触头由预测压力串行解出
  （`TemporalKpSMPLNet_Series_mlp.py:114-121`：press_output → cont_output，
  与 GT 标签生成链同构，把"压力→接触"映射内嵌进网络），但它是网络的
  输出头，因此 SMPL 接触的连续重建可评估（contact_smpl_mse/bce）；
  **contact_f1 待处理**（缺二值接触 GT，见待处理表）。
- **AnySole 的 shape 与 contact 均为待处理**（见待处理表），当前 `—`；
  GT-beta pose-only PVE 例外**废除**，不再用 GT 回填补评。
- **pressure_toolkit 生成 shape**：其原生拟合 pipeline 本来就产出 SMPL
  形状数据，因此 SMPL 形状相关评估（PVE，拟合表面对 GT 表面）**启用**。

## 3. 两文件口径与组结构

| 文件 | 组 | 内容 |
|---|---|---|
| `<split>.json`（明细） | T2M / V2M / VT2M / robust_vdrop / robust_tdrop | 全部协议指标：全身标量 + 部位级诊断 + 接触/顶点足滑等诊断键 + 协议元数据（checkpoint/modal/contact_method/split/protocol_seed/tw/sample_steps/robustness/forward_axis/diagnostic_fields） |
| `<split>_brief.json`（精简） | T2M / V2M / VT2M / **V2T** / robust_* | 同源：每 motion 组只保留第 5 节白名单键；V2T = 触觉功能组（第 4 组，取明细 V2M 触觉行，层级 brief 6 + 叶 4）；组序固定 T2M/V2M/VT2M/V2T/robust_vdrop/robust_tdrop，缺组不补 |

明细中的诊断键（不进 brief、不进正式跨模型表）由 payload 的
`diagnostic_fields` 显式声明：`contact_f1`、`contact_acc`、`contact_recall`、
`air_recall`、`contact_balanced_acc`、`foot_sliding_vertex_mm`、
`shape_vertex_std_mm`。

## 4. 键名与唯一实现

| 公共键 | 唯一实现 | 模型生成数据（评估对象） | GT / 参考数据 |
|---|---|---|---|
| `mpjpe_mm` / `pa_mpjpe_mm` | `mean_point_error`+`pelvis_align` / `pa_mpjpe` | 预测 joints | GT joints |
| `w_mpjpe100_mm` / `wa_mpjpe100_mm` | `windowed_world_mpjpe` | 预测 joints + 预测 root translation | GT joints + GT translation |
| `root_ate_mm` / `root_rte_percent` | `root_trajectory_metrics` | 预测 root translation | GT root translation |
| `root_orientation_deg` / `root_orientation_drift_deg` | `root_orientation_error_deg` / `root_orientation_drift_deg` | 模型/方法输出的 root rotation | GT root rotation |
| `mpjae_deg` | `matrix/rotation_error_degrees` | 模型直接输出的 joint rotations | GT rotations；BVH 反解旋转不得获得 |
| `accel_error_m_s2` / `jitter_*_m_s3` | `temporal_metrics` | 预测 joints + 真实时间轴 | GT joints + 时间轴 |
| `foot_sliding_mm`（关节版） | `foot_sliding_joints` | 预测足关节轨迹 | GT 足关节 + GT 接触判定（GT 足速度 < 0.3 m/s） |
| `pve_mm` | `mean_point_error`(vertices) | 模型/方法生成的 SMPL 表面 | GT 表面（vertices） |
| `contact_f1` | 无（当前所有模型不适用） | 二值接触预测（当前无） | 二值接触 GT（**当前没有**，待处理，见 §2） |
| V2T 层级（brief 7 + 叶 5，四层） | `pressure_metrics` + `contact_smpl_metrics`（31×11 网格 + SMPL 接触图） | pressure_pred（AnySole 4×12 重采样 / FPP-Net 31×11）；接触层另需 contact_smpl_pred | pressure_gt；contact_smpl_gt（f6_soft） |

能力门控实现在 `results_display/script/utils/compare_core.py`：
`CAPABILITY_FIELDS` + `METRIC_REQUIRES` + `applicable_metrics()`；注册表
（models_modes.yaml）声明 protocol / capabilities（sources 仅作溯源记录，
**不再作裁定依据**——评估资格以 §2 生成清单为准）。

## 5. 精简版字段名单（论文候选，一组一族一代表）

每 motion 组（T2M/V2M/VT2M）10 键：

| 键 | 定义（口径） | 单位 |
|---|---|---|
| `mpjpe_mm` | 骨盆对齐后逐关节 L2 误差均值（原生 24 / 23 关节） | mm |
| `pa_mpjpe_mm` | 逐帧 similarity Procrustes 对齐后同上 | mm |
| `w_mpjpe100_mm` | 100 帧滑窗世界系 MPJPE（前 2 帧刚性对齐） | mm |
| `pve_mm` | SMPL 顶点（骨盆对齐）逐顶点 L2 误差均值；仅生成表面的模型/方法（MotionPRO / VP-MoCap / pressure_toolkit） | mm |
| `mpjae_deg` | 逐关节相对旋转测地角均值；仅直接输出旋转的模型 | degree |
| `root_ate_mm` | 根逐帧位置误差均值 | mm |
| `root_orientation_drift_deg` | 根朝向相对首帧的累积漂移误差 | degree |
| `jitter_pred_m_s3` | 预测三阶差分（jerk）模长均值 | m/s³ |
| `jitter_gt_m_s3` | GT 三阶差分模长均值（成对参照） | m/s³ |
| `foot_sliding_mm` | 关节版：GT 接触帧上预测足关节位移均值 | mm |

V2T 组（第 4 组，层级结构：brief 7 键 + 叶 5 键，四层）：

| 键 | 定义 | 单位 | 层级 |
|---|---|---|---|
| `T_corr` | 逐帧展平 cell 向量 Pearson 相关均值 | — | 网格 [brief] |
| `T_rmse` | 网格逐 cell 差值均方根 | 压力单位 | 网格 [brief] |
| `T_mse` | 网格逐 cell 差值平方均值（= T_rmse²，与 FPP-Net 原生 pressure MSE 同名） | 压力单位² | 网格 [叶] |
| `T_mae` | 网格逐 cell 绝对误差均值 | 压力单位 | 网格 [叶] |
| `pressure_force_rmse` | 每脚合力误差 RMSE | 压力单位 | 力 [brief] |
| `pressure_force_r2` | 两脚合力之和整段时序的全局 R²（非逐帧均值） | — | 力 [brief] |
| `pressure_force_mae` | 每脚合力误差 MAE | 压力单位 | 力 [叶] |
| `pressure_cop_error_left` / `_right` | 左 / 右脚 CoP 位置误差（仅合力 > 0 帧） | grid units | CoP [brief] |
| `pressure_cop_error_mean` | 双脚 CoP 误差均值（派生） | grid units | CoP [叶] |
| `contact_smpl_mse` | SMPL 逐顶点连续接触图 L2（pred=FPP-Net 接触头，gt=f6_soft 软标签广播） | — | 接触 [brief] |
| `contact_smpl_bce` | 同上二值交叉熵（与 FPP-Net 原生训练损失同式） | — | 接触 [叶] |

> **V2T 四层从属图（2026-09-27 起为正式结构）**
>
> 前三层派生自同一份逐帧压力网格 `pressure_pred/gt`（AnySole 原生 96 格，
> 跨模型统一 31×11 重采样后）；接触层派生自连续 SMPL 接触图
> `contact_smpl_pred/gt`（FPP-Net 专属）。每层一个 L2 代表 + 一个
> 形态/全局量进 brief，其余为叶节点（算且存入明细，不进 brief）：
>
> ```text
> 压力网格 (T, 2足, 31×11)
> ├── 逐点比较 ───────────► T_rmse [brief] / T_corr [brief] / T_mse [叶] / T_mae [叶]
> ├── 每脚求和 → 每脚合力 ► pressure_force_rmse [brief] / pressure_force_mae [叶]
> │        └─ 两脚合力时序 ─► pressure_force_r2 [brief]（全局统计，非逐帧均值）
> └── 每脚质心 → CoP ─────► pressure_cop_error_left/right [brief] / _mean [叶]
> SMPL 接触图 (T, 2足, N顶点)
> └── 逐顶点比较 ─────────► contact_smpl_mse [brief] / contact_smpl_bce [叶]
> ```
>
> 派生链：force 是网格**求和**降维、CoP 是网格**加权质心**、r2 是合力时序
> 的全局统计量、T_mse 是 T_rmse 的平方（确定性派生）。brief（test_brief
> 第 4 组与 R_Test2/R_Test4 表列）只取 7 个 brief 键；叶 5 键保留在明细
> `test.json` 与逐 session 输出，payload 用 `v2t_leaf_keys` 标注。
> 接触层 2026-09-27 已纳入：FPP-Net 数据管线 contact GT 已从
> `press2Cont th=0.5` 切换为 f6_soft（foot 级软值广播到网格与顶点，
> `getVertsPress(soft=True)` 均值映射不二值化）；导出 archive 新增
> `contact_smpl_pred/gt`（float16 连续图）；AnySole 无接触输出，两键为
> `—`（not_applicable）。

**诊断出局（保留在明细）**：`wa_mpjpe100_mm`、`root_rte_percent`、
`root_orientation_deg`（drift 已代表）、contact 全部、`accel_*`、
`shape_vertex_std_mm`、`foot_sliding_vertex_mm`、`seam_jump_*`、
`joint_limit_viol_*`、部位级 MPJPE/PA-MPJPE。

## 6. Step2Motion（原生 BVH-23）指标明细

| 指标 | 能否计算 | 说明 |
|---|---|---|
| mpjpe_mm / pa_mpjpe_mm | 可以 | BVH-23 原生 23 关节对 BVH GT（Hips 为骨盆） |
| w_mpjpe100 / wa_mpjpe100 / root_ate / root_rte | 可以 | 前提：translation head 真实预测（非 GT fallback） |
| root_orientation_deg / _drift_deg | 可以 | BVH Hips/root rotation 调公共求解器；前向轴按实测约定传入 |
| accel / jitter | 可以 | 同一时间轴、同一 23 关节 |
| foot_sliding_mm（关节版） | 可以 | LeftFoot/RightFoot/LeftToeBase/RightToeBase |
| mpjae_deg | 不适用（`—`） | BVH 局部旋转语义与 SMPL 不可比，不得反解/混用 |
| pve_mm / shape/vertex 类 | 不适用（`—`） | 无网格输出；不得用给定 β 补造 |
| contact_f1 | 不适用（`—`） | 无显式训练导出的 contact prediction |

no-IMU 只是实验探针：不注册、不进入正式排名与 R_Test 自动扫描；
GT translation fallback 不参与轨迹/世界系指标。

## 7. 历史文件回填（已执行）

`anysole/backfill_brief.py` 已扩展并执行（2026-09-26）：

- `results/AnySole`：12 个 test.json / test_brief.json 对；
- `results/experiments`：30 个 detail/brief 对；
- 迁移内容：`yaw_*` → `root_orientation_*`（公式一致，纯改名）、旧顶点版
  `foot_sliding_mm` → `foot_sliding_vertex_mm`（公共关节版需重跑 eval 后
  才有，旧值不冒充）；payload 记录 `key_migration` 与 `diagnostic_fields`；
  brief 经新白名单重派生（V2T 层级 brief 7 + 叶 5、无 contact）。
- `results/backup/AnySole_BVH_backup/` 等历史归档未动（属任务 04 保留范围，
  见任务报告清理候选）。
- 旧 fseries / `*_seed*.json` 归档此前已清理，本轮扫描无残留。

## 8. 基线评估自产指标（2026-09-26 用户裁定）

"评估不仅要产出预测文件，还要有对应的评估结果"：

- 基线评估入口 `python -m Baselines.utils.evaluate --model <name|alias> --split test`
  读取已导出 predictions + 配对 GT（同一会话网格、同一 manifest/split），用
  `Baselines/utils/solver.py`（公式全部来自 anysole 侧唯一实现）计算并落盘
  `results/baselines/<Model>/metrics/<split>_comparison.json`（schema v2：逐
  session + 加权汇总 + protocol/capabilities/denied/applicable + split/manifest
  hash）+ `_comparison.csv`。
- 数值与 R_Test2 参考行实现逐位一致（`tests/test_baseline_utils.py` 有真实
  数据交叉锁）；R_Test2/R_Test4 定位为消费端。
- 各模型短别名：motionpro / pressure_toolkit / vp_mocap / step2motion / fpp_v2t。
- AnySole 主模型仍由 `anysole/eval.py` 自产明细+brief，不变。

## 9. 上下游适配清单（任务 01 边界外，交由对应任务/主 Agent）

| 位置 | 需要 | 归属 |
|---|---|---|
| `r_test1/2/3/4` 消费端 | 移除 common19 用法、按原生协议重写（metrics() 新签名、v2t_metrics 9 项→2026-09-27 层级 brief 7 + 叶 5、load_v2t_archive 无 contact） | 任务 03 |
| `models_modes.yaml` | 已由任务 02 更新（protocol/capabilities/sources 已就位） | 任务 02 |
| `anysole/rho_grid.py:301` 自检键 | `yaw_abs_deg` → `root_orientation_deg`（当前静默跳过） | 未分配，报告已提交 |
| `results_display/script/utils/missing_rate_analysis.py` COMPONENTS | `("yaw_abs_deg",)` → `root_orientation_deg` | 未分配，报告已提交 |
| `results_display` 其余读明细键处 | 核对 `root_orientation_*` 新键 | 任务 03 |
| 重跑 eval 回填公共关节版 `foot_sliding_mm` | 旧文件该键空缺，需重算 | 训练侧 |

## 10. 不足与待办（2026-09-27 生成清单裁定后）

### 10.1 待处理（有对应生成、但接口/GT 尚未就绪，暂不评估）

| 项 | 模型 | 原因 | 就绪条件 |
|---|---|---|---|
| PVE / shape 评估 | AnySole | **当前不生成 beta**：模型输出只有 24 关节 6D 姿态 + translation（POSE_DIM=144，无 beta 输出头）；导出档 betas 为 GT 会话 beta（betas_source=ground_truth_session），GT-beta PVE 已废除不再补评 | 若需 PVE：先给模型加 beta/shape 生成头（模型结构改动），接入导出/评估后按清单评估 |
| 正式 contact | AnySole | `soft_contact` 是关键点推导的训练正则（`soft_contact_from_keypoints()`，诊断），非显式接触输出接口 | 明确显式接触输出接口后评估 |
| contact_f1 | FPP-Net | 预测侧可二值化，但 GT 侧没有二值接触：f6_soft 是软标签体系（0.05/0.30/0.70/0.95）无天然二值分界；硬决策 motion_f6 是运动学状态机、不是模型输出 | 明确二值接触 GT 后评估 |

就绪后恢复点：contact_f1 加入 `V2T_LEVELS["contact"]` 与 `V2T_BRIEF_KEYS`
（求解器与测试已留注释标明）；AnySole 接入 beta 导出后，registry 行开启
`predicted_surface/predicted_shape` 即按清单自动启用 PVE。

### 10.2 已知不足（数据/文件缺口，本轮裁定遗留）

1. **pressure S13013 旧 archive 元数据**：`surface_source=template_reconstruction`
   / `public_surface_metrics=false` 仍是旧口径——正式收口重导出后自动更新为
   `method_optimization`/`true`；重导出前该文件的 PVE 会被 archive public-surface
   contract 拒绝。
2. **AnySole 历史 42 对 JSON**：pve_mm 值留在明细（已入 `DIAGNOSTIC_KEYS`，
   brief 不再显示），但历史文件的 `diagnostic_fields` 未补 `pve_mm`；下次
   重跑 eval 自然更新（backfill 的 `diagnostic_fields` 只在缺字段时生成）。
3. **公共关节版 `foot_sliding_mm` 历史 JSON 空缺**：无法从旧值确定性派生，
   待重跑各模型 eval 回填（训练侧重算流程）。
4. **FPP-Net sidecar 旧 GT 口径**：现有 `pred_contact_smpl/` sidecar 的
   contact GT 是旧 `press2Cont th=0.5` 口径；正式导出前必须重跑三 split
   推理（收口顺序 step 1）产出 f6_soft 口径 sidecar。
5. **正式 test split 全量重建依赖上游**：pressure 拟合 17/36、PoseTransOpt
   与 FPP-Net 统一导出 0/36——R_Test1/2/3/4 正式重建待这些上游完成。

### 10.3 本轮执行状态（2026-09-27）

- AnySole `12` 对、`results/experiments` `30` 对 detail/brief 已按当前
  白名单重新回填；`results/backup` 未触碰。
- 历史 `*_fseries.json`、`*_seedN.json` 在正式结果范围内未发现；不存在的旧
  `results/baselines/pressure_tookit/`、`results/VP-MoCap/` 和
  `results/offline/pressure_toolkit/` 不伪造、不处理。
- `results/baselines/pressure_toolkit/predictions/eval_motion/S13013.npz`
  仍带旧 `surface_source=template_reconstruction` / `public_surface_metrics=false`
  元数据，待正式重导出后再进入 PVE 评估，当前不删除原始 archive。
- `AnysoleWorkspace/derived/pressure_toolkit/` 过程树约 `113G`，因正式 test
  split 尚未全量完成，本轮只登记为 `BLOCKED`，不删除。
- `pred_contact_smpl` 旧 sidecar 仍保留在 VP-MoCap 过程树，待 FPP-Net
  `f6_soft` 三 split 重导出完成后核验并清理；不按目录名直接删除。
- R_Test 派生目录中当前未发现任务 04 所列的旧 compare 目录；现有单个 GIF
  和旧 contact CSV 不作为 canonical 指标输入，待新结果生成后再按 manifest
  逐项处理。
