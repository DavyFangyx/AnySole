# 模型评测协议审计（评估整改终态）

> 日期：2026-09-26（评估整改任务 01–03 完成后更新）
> 本文记录**现行**评测协议与消费端行为。2026-09-26 前的 common19 内容
> 只在「旧协议记录（已废弃）」一节保留，不参与任何现行命令。

## 1. 现行协议结论

1. 数值评测与并排可视化**不再使用 common19**。SMPL 模型用原生 SMPL-24 预测对
   SMPL GT；Step2Motion 用原生 BVH-23 预测对 BVH GT；两种表示永不按数值关节
   下标对齐或裁剪成公共关节集。
2. 指标公式的唯一实现是根 `metrics.py`（`anysole/utils/metrics.py` 为 re-export
   shim）；跨模型展示层经 `results_display/script/utils/compare_core.py` 调用同一
   套求解器，不在 R_Test 脚本里重定义公式。
3. 同名指标统一公式、单位、时间轴（session 网格 `t_mocap = visual_start_s +
   i/fps − offset_s`）、有效帧（manifest valid_frame_indices ∧ 预测 valid_mask）
   与聚合规则（按有效帧数加权平均）。
4. 模型没有某项真实输出时，该指标为 `not_applicable`，CSV 留空、表格显示 `—`
   （绝不写 0）。GT 回填、模板重建、拟合 shape、逆解旋转、运动学/压力阈值推导
   的 contact 都不能使指标获得资格。
5. 正式 contact：当前没有 motion 模型训练并显式导出 contact prediction，正式
   `contact_f1` 一律 `—`；诊断推导仅存在于训练监控与 D_Test3。
6. V2T 正式指标固定 9 项：`T_corr`、`T_mae`、`T_rmse`、
   `pressure_force_mae/rmse/r2`、`pressure_cop_error_left/right/mean`。
   R_Test2 与 R_Test4 消费同一个 `compare_core.v2t_metrics` 求解器。
7. `pve_t_mm` 已取消；`root_orientation_deg` / `root_orientation_drift_deg` 是
   原 yaw 公式的正式名（唯一实现 `metrics.root_orientation_error_deg/drift_deg`）；
   `foot_sliding_mm` 为关节版（SMPL ankle/foot、BVH Foot/ToeBase），顶点版改名
   `foot_sliding_vertex_mm` 仅作 SMPL 专属诊断。

## 2. 能力门控与状态

注册表 `results_display/models_modes.yaml` 是模式/协议/能力的唯一声明点，
每个正式条目带：

```yaml
protocol: smpl24 | bvh23 | pressure
capabilities:   # joint_positions / direct_joint_rotations / root_translation /
                # root_rotation / predicted_surface / predicted_shape /
                # pressure_prediction / contact_prediction
sources:        # model_prediction / method_optimization（可进入公共指标）
                # ground_truth_fallback / template_reconstruction /
                # method_fitting / derived_for_visualization（仅诊断/可视化）
```

消费端规则（`r_test2_compare.py` 的 `denied_capabilities` / `applicable_for`）：

- 能力声明为真但必要字段缺失、声明协议与文件不符、帧数与 session 不符 →
  行状态 `invalid` 并记录原因；
- 预测文件不存在 → `missing`；
- 能力缺失或来源被拒 → 指标 `not_applicable`（`—`），逐会话
  `metric_reasons` 列记录每项原因（写入 CSV 与 `*_comparison.json`）。

## 3. 各模型现行能力矩阵

| 模型/方法 | 协议 | 真实输出 → 启用指标 | `—` 指标 |
| --- | --- | --- | --- |
| AnySole motion | smpl24 | pose/translation/rotation → MPJPE、PA-MPJPE、W/WA-MPJPE、MPJAE、root ATE/RTE、root orientation、temporal、关节版 foot sliding；PVE 按已批准 pose-only 协议（GT beta，`--surface-metrics` 时） | 正式 contact；shape/vertex |
| MotionPRO | smpl24 | pose+shape+translation+rotation → 同 SMPL 全套 + PVE（模型表面） | 正式 contact |
| MMVP pressure_toolkit | smpl24 | 拟合 pose/translation/rotation → 关节、轨迹、朝向、temporal、foot sliding | PVE、shape/vertex（模板重建）、正式 contact |
| MMVP_VP-MoCap (FPP-Net + PoseTransOpt) | smpl24 | 优化 pose/beta/translation → 同 SMPL 全套 + PVE | 正式 contact |
| Step2Motion 主干 | bvh23 | BVH-23 joints + 预测 Hips translation/rotation → MPJPE、PA-MPJPE、W/WA-MPJPE、root ATE/RTE、root orientation、temporal、foot sliding | MPJAE（不逆解旋转）、PVE、shape/vertex、正式 contact |
| FPP-Net V2T | pressure | pressure prediction → V2T 9 项 | motion 指标、正式 contact |
| Step2Motion no-IMU 探针 | — | 不注册、不进入主排名与自动扫描（实验区保留） | — |

## 4. 各展示任务现行行为（消费端）

- **R_Test1**（`r_test1_compare.py`）：每模型画原生骨架 + 原生边，标题带
  `SMPL-24 native` / `BVH-23 native`；逐帧 MPJPE 只对模型与其配对同协议 GT 计算；
  混合行按出现协议分别给 `GT Motion — SMPL-24 native` / `GT Motion — BVH-23
  native` 栏；缺失预测显示占位与原因，不退用其他 GT。
- **R_Test2**（`r_test2_compare.py`）：按原生协议评测完整骨架；汇总表含
  `protocol` 列（`smpl24-native`/`bvh23-native`）；`—` 显示；`--write-model-metrics`
  写 schema `mmvp_native_metrics_v2`（含 evaluation_protocol、capability 摘要、
  GT source、split/manifest hash、metric 缺失原因）；旧 v1（common19）文件不被
  当作新结果读取，重跑时被 v2 取代。
- **R_Test3**（`r_test3_traj.py --compare`）：只纳入 `root_translation=true` 且
  轨迹来源合法的正式模型；SMPL 模型配 SMPL root GT，BVH 模型配 BVH/Hips root
  GT；`root_translation_source=ground_truth_fallback` 等标记 → 拒绝进入并记原因。
- **R_Test4**（`r_test4_v2t.py`）：正式 summary/report/CSV/aggregate 严格 9 项；
  archive 的 contact 字段容忍但不消费；与 R_Test2 同一 pressure 求解器，相同
  输入的 9 项数值逐项一致。

## 5. 路径规范

- 结果区 canonical：`results/baselines/{MotionPRO,Step2Motion,pressure_toolkit,
  FPP-Net,VP-MoCap}/`，每个正式 baseline 有 `predictions/` 与 `metrics/`。
- `Baselines/pressure_tookit/` 是历史源码仓库名，只保留在源码路径；workspace、
  结果、注册表一律 `pressure_toolkit`。
- FPP-Net V2T 与 VP-MoCap V2M 分根：`baselines/FPP-Net/predictions/v2t/<session>.npz`、
  `baselines/VP-MoCap/predictions/eval_motion/<session>.npz`。
- AnySole 行按 `<model>_<contact>/<run>/predictions/eval_motion/` 展开
  （`--variant` 指定 run；省略则扫描全部 run）。

## 6. 重新生成流程（重训/新导出后）

1. `python -m anysole.eval --model-name <M> --contact-method <C>` 刷新
   `predictions/eval_motion/`（含 V2T archive）与 `metrics/`。
2. baseline 侧：`AnysoleWorkspace/tool/export_baseline_motion.py` 按模型导出，
   `validate_baseline_exports.py --split test --model all` 校验。
3. 消费端重建：
   `r_test2_compare.py --split test --by-mode --write-model-metrics --surface-metrics --force`
   → `r_test1_compare.py --split test`、`r_test3_traj.py --compare --split test`、
   `r_test4_v2t.py --source archives --split test`。

## 7. 旧协议记录（已废弃，仅审计留存）

以下为 2026-09-26 评估整改前的历史描述，**已废弃**，不得与现行命令混写：

- common19 语义关节映射（`COMMON_JOINTS`/`SMPL_COMMON_NAMES`/`BVH_COMMON_NAMES`/
  `COMMON_EDGES`/`select_common_*`）与 `joint_set=common19` 元数据：已从
  `compare_core.py` 与 R_Test 脚本删除；其派生结果（R_Test1/2/3 compare 产物、
  含 contact 的旧 R_Test4 CSV/JSON、schema v1 `*_comparison.json`）列于任务 03
  待清理清单，按 `z_note/评估/04_联合验收与历史清理任务书.md` 处置（该任务书已随 2026-10-01 U10 废止删除；现行说明见 `z_note/评估/评估配置说明书.md`）。
- `yaw_abs_deg`/`yaw_drift_deg` → 已改名 `root_orientation_deg`/
  `root_orientation_drift_deg`（历史 JSON 已键名回填并记录 `key_migration`）。
- 顶点版 `foot_sliding_mm` → 已改名 `foot_sliding_vertex_mm`（诊断）。
- `pve_t_mm` → 已取消。
- V2T 的 contact 四项（contact_f1/acc/recall/air_recall）→ 不再作为正式 V2T
  指标输出。
- 旧 mixed-GT 轨迹比较（所有模型共用 `load_session_gt` 根轨迹）→ R_Test3 已改
  配对协议 GT。
