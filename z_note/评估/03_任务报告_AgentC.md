# 任务 03 交付报告（Agent C：results_display 原生协议适配）

- 任务：03_results_display原生协议适配任务书.md（Agent C）
- 状态：**complete**
- 日期：2026-09-26
- 前置依赖：任务 01（Agent A，complete）、任务 02（Agent B，partial——代码链路完成，61GB 旧树物理迁移待拟合批次结束）

## 1. 修改文件

| 文件 | 改动 |
|---|---|
| `results_display/script/r_test2_compare.py` | 全量重写：删除 common19 裁剪/元数据；按原生协议评测（`smpl24-native`/`bvh23-native`）；能力门控（`capability_map`/`denied_capabilities`/`applicable_for`/`mask_metrics`）；`missing`/`invalid`/`not_applicable` 三态区分；逐会话 `metric_reasons` 列；汇总表加 `protocol` 列；CSV 缺失值留空、PNG 显示 `—`；`--write-model-metrics` 写 schema `mmvp_native_metrics_v2`（旧 v1 报 legacy 并取代，绝不读回）；V2T 严格 9 项；AnySole 按 `<model>_<contact>/<run>` 展开（`--variant`）；`--auto-scan` 弃用（registry 是唯一声明点） |
| `results_display/script/r_test1_compare.py` | 全量重写：每模型原生 joints/边（`native_edges`）+ 标题 `SMPL-24 native`/`BVH-23 native` 标记；逐帧 MPJPE 只对配对同协议 GT（`native_frame_mpjpe_mm`）；混合行按出现协议分别给 GT 栏；BVH 预测重采样到会话网格；缺失面板保留且写明原因 |
| `results_display/script/r_test3_traj.py` | `--compare` 重写：`translation_allowed()` 门控（root_translation=true + 来源非 GT 回填/模板重建，检出 `root_translation_source=ground_truth_fallback` 拒绝并记日志）；每模型配自己协议 GT（SMPL→SMPL pelvis，BVH→BVH Hips，均重采样会话网格）；面板/静态图标协议；静态图按协议分组坐标轴；单模型路径 BVH 也改配 BVH GT |
| `results_display/script/r_test4_v2t.py` | 正式 summary/report/CSV/aggregate 严格 9 项（删 contact 四列）；`v2t_metrics` 新签名（3 参数）；`load_v2t_archive` 无 contact；AnySole 条目 variant-aware；`--source infer` 诊断入口保留 |
| `results_display/script/r_test1_visualize_mmvp.py` | 删除旧标签 `MMVP_FPP-Net`（V2T-only，无 motion 输出）；EDGES 硬编码改为 `native_edges(names)`；GT 改 `protocol_gt(row,"smpl24")` 配对；`array_from_file` 改从 compare_core 导入 |
| `results_display/README.md` | 消费端部分重写：协议约定节（无 common19、能力门控、`—`、V2T 9 项、路径规范、重新生成流程）；R_Test1/2/3/4 各节；旧协议明确标「旧协议，已废弃」 |
| `z_note/EVALUATION_PROTOCOL_AUDIT.md` | 重写为终态协议审计：现行协议/能力矩阵/各展示任务行为/路径规范/重建流程；历史 common19 内容压缩进「旧协议记录（已废弃）」一节 |
| `tests/test_protocol_consumer.py`（新增） | 53 断言：能力门控、来源拒绝、mask 机制、archive contact 容忍、V2T 严格 9 项、R_Test2 与 R_Test4 同求解器数值一致、missing/invalid 状态、common19 协议拒绝 |

### 跨边界最小修复（Agent A 报告已标记为未分配所有权，本任务按“历史遗留清理”用户指示实施）

| 文件 | 改动 |
|---|---|
| `results_display/script/utils/missing_rate_analysis.py` | COMPONENTS 别名 `("yaw_abs_deg",)` → `("root_orientation_deg",)`（否则 global_yaw 组件取不到值） |
| `anysole/rho_grid.py` | corner_check 自检键 `yaw_abs_deg` → `root_orientation_deg`（原静默跳过） |

## 2. 新增/变化的接口（消费端）

- `r_test2_compare.ANYSOLE_MOTION_CAPS/ANYSOLE_V2T_CAPS/ANYSOLE_APPROVED`（pose-only PVE 批准例外）、`denied_capabilities(entry, archive_meta)`、`applicable_for(caps, denied, approved)`、`mask_metrics(...)`、`archive_meta(path)`、`anysole_run_dirs(model_dir, variant)`、`evaluate_motion_row/evaluate_pressure_row`（可单元测试）。
- `r_test3_traj.translation_allowed(entry, path) -> (bool, reason)`。
- 指标公式全部来自任务 01 冻结接口（`compare_core.metrics(protocol=..., forward_axis=...)`、`v2t_metrics` 9 键、`load_v2t_archive`、`native_edges`、`native_frame_mpjpe_mm`、`applicable_metrics`/`read_capabilities`），R_Test 脚本内无第二套公式。

## 3. 对任务 01/02 接口的验收记录

### 任务 01 公共比较核心（任务书 §5.1）

1. common19 常量/选择器/专用逐帧 MPJPE 已删（compare_core 仅存“已删”注释）✅
2. SMPL/BVH 对应 GT loader（`protocol_gt`）、原生关节名（`LEGACY_BVH_NAMES` 过滤 23 语义关节）、原生 skeleton edges（`native_edges`，SMPL-24 树 + 实测 Skeleton3 层级）✅
3. `protocol + capabilities + 实际字段` 分派：`applicable_metrics`/`read_capabilities`（两 schema 键名兼容）+ 消费端 `denied_capabilities` 补来源拒绝；`not_applicable`/`invalid`/`missing` 已区分 ✅
4. 正式 motion contact 不从 joints 推导（`contact_from_joints` 仅诊断，r_test2 不再调用）；BVH 反解旋转不启用 MPJAE（能力 false → `—`）✅
5. V2T 求解器只返回 9 项；聚合不把 NaN 计入分母（`aggregate` 用 isfinite 过滤）✅

### 任务 02 注册表（任务书 §5.6）

- 5 个正式条目均含 `protocol`/`capabilities`/`sources`/`display_name` ✅
- pressure_toolkit 指向 `results://baselines/pressure_toolkit/...`（正确拼写）✅
- V2M 行名 `MMVP_VP-MoCap`、display_name `VP-MoCap (FPP-Net + PoseTransOpt)` ✅
- 无 Step2Motion no-IMU 探针 ✅
- 消费端显示一律用 `display_name`（面板/表行），V2T 根 `results://baselines/FPP-Net/predictions/v2t`（无 `_V2T` 后缀）✅

## 4. 执行的检查/测试（证据）

1. `tests/test_protocol_consumer.py`（新增）：**53 passed, 0 failed**。
2. `tests/test_public_metrics.py`（任务 01，回归）：**86 passed, 0 failed**。
3. BVH 前向轴实测（S13013，任务书 §5.2 第 7 条 + Agent A 交接第 3 条）：display 帧 BVH Hips 旋转 −Y 列与 SMPL GT 朝向中位圆差 4.4°（±180° 公翻转在圆差公式下精确抵消）→ `PROTOCOL_FORWARD_AXIS = {"smpl24": 2, "bvh23": 1}`。
4. 端到端冒烟（单 session S13013、`/tmp/r03smoke/` 临时 split/models 配置）：
   - **SMPL**：MotionPRO `smpl24-native` status=ok n_valid=339，全套 motion 指标 + PVE（模型表面）有值；
   - **SMPL（能力受限）**：MMVP pressure_toolkit PVE 为 `—`（`metric_reasons: pve_mm=not_applicable`），其余指标照常；
   - **BVH**：Step2Motion `bvh23-native` status=ok，mpjae/pve 为 `—`，root orientation/drift、root ATE、foot sliding 已补算；
   - R_Test3 --compare 的 ATE 与 R_Test2 root_ate_mm 逐模型一致（MotionPRO 223 vs 222.75mm；pressure_toolkit 2960 vs 2959.68mm；Step2Motion 942 vs 942.09mm）——轨迹展示与数值评测同源；
   - `--write-model-metrics` 落盘 schema v2（evaluation_protocol/capabilities/applicable/gt_source/split+manifest hash），冒烟产物已删（非 canonical split 的临时文件）；
   - **V2T**：无真实 archive（FPP-Net/AnySole V2T 待上游导出），用合成 archive 走 R_Test2 pressure 行与 R_Test4 同求解器路径，9 项数值逐项一致（测试断言）。
5. 静态检查：4 个 R_Test 脚本 + README + 审计文档无 common19 正式用法（仅“已删除/已废弃”标注）；全 display 脚本 `py_compile` 通过；无 `select_common*`/`COMMON_JOINTS`/`frame_mpjpe` 等旧符号残留引用。

## 5. 旧派生产物清单与清理（用户已授权）

机器可核对清单：`z_note/评估/migration/cleanup_manifest_task03.json`（逐文件路径+大小）。

| 路径 | 状态 | 说明 |
|---|---|---|
| `results_display/ResultTest/R1Test_visualize/compare/` | 已删（17 文件，1.57MB） | common19 派生，可重建 |
| `results_display/ResultTest/R2Test_compare/` | 已删（12 文件，0.46MB） | 含遗留 `r_test2_compare_metrics.csv` |
| `results_display/ResultTest/R3Test_traj/compare/` | 已删（6 文件，0.58MB） | 混合 GT 派生 |
| `results_display/ResultTest/R4Test_v2t/` | 不存在 | — |
| `results/**/metrics/*_comparison.json`（v1/common19） | 盘上已无此类文件 | 扫描结果为 0 |

清理日志：`z_note/评估/migration/cleanup_log_task03.md`（注明用户授权与可重建方式）。
**未删除**：单模型 R_Test1/R_Test3 输出、checkpoints、原始 predictions、原生 metrics、`results/experiments/**/_V2T.npz` 实验预测、`results/backup/` 全部保留。

## 6. 最终能力矩阵（真实输出 → 启用指标 → `—` 指标）

| 模型 | 协议 | 启用指标 | `—` 指标 |
|---|---|---|---|
| AnySole motion | smpl24 | MPJPE/PA-MPJPE/W/WA-MPJPE/MPJAE/root ATE+RTE/root orientation/temporal/foot sliding；PVE（pose-only + GT beta，`--surface-metrics`） | 正式 contact |
| MotionPRO | smpl24 | 全套 SMPL motion + PVE（模型表面） | 正式 contact |
| MMVP pressure_toolkit | smpl24 | 关节/轨迹/朝向/temporal/foot sliding（method_optimization） | PVE、shape/vertex、正式 contact |
| MMVP_VP-MoCap (FPP-Net + PoseTransOpt) | smpl24 | 全套 SMPL motion + PVE（method_optimization） | 正式 contact |
| Step2Motion 主干 | bvh23 | MPJPE/PA-MPJPE/W/WA-MPJPE/root ATE+RTE/root orientation/temporal/foot sliding（BVH-23 原生 + Hips 真实 translation/rotation） | MPJAE、PVE、shape/vertex、正式 contact |
| FPP-Net V2T | pressure | V2T 9 项 | motion 指标、正式 contact |

## 7. 未完成项 / 交给任务 04 与后续

1. 正式 test split 全量重建（`r_test2_compare.py --split test --by-mode --write-model-metrics --surface-metrics --force` 等）依赖上游导出：AnySole 各模型重跑 `anysole.eval`、FPP-Net V2T 与 VP-MoCap V2M 预测导出（Agent B 未完成项 3）。
2. pressure_toolkit 其余 35 个 test session 预测拟合/导出完成后，R_Test2/3 行才完整（当前仅 S13013）。
3. Agent B 的 61GB 旧树物理迁移与删除候选，仍按任务 02/04 流程执行（与本任务无关，未触碰）。
4. `workspace.py doctor` 14 条历史 ERROR（Step2Motion/MotionPRO 簿记遗留）报主 Agent 定夺（Agent B 报告第 7 节）。
5. 跨边界修复已实施并记录（见 §1）；如主 Agent 认为 `missing_rate_analysis.py`/`rho_grid.py` 应另有归属，仅需复核这两处一行级改动。
6. A/B 分析脚本（`component_analysis.py` 等）的 `--fs-*` CLI 旧名（fseries 旧称）不在本任务范围，未动。
