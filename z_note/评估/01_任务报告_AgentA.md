# 任务 01 交付报告：统一指标求解器与模型真实输出能力

- 任务：01_统一指标求解器与模型能力任务书.md（Agent A）
- 状态：**complete**（实现 + 数值测试 + 历史回填；删除类动作已按任务书集中到任务 04）
- 日期：2026-09-26

> **结构更正（2026-09-26，用户裁定）**：指标公式的唯一实现移回
> `anysole/utils/metrics.py`（主模型自持 utils），根目录 `metrics.py` 已删除；
> 基线侧数据协议/评估协议统一放入 `Baselines/utils/`，公式引用 anysole 侧唯一
> 实现。原报告表中"`metrics.py`（根目录）"一行按下述更正理解。

## 追加（2026-09-26 用户后续裁定：Baselines/utils + 基线评估自产指标）

### 结构

- `anysole/utils/metrics.py`：公式唯一实现（主模型自持 utils）；根目录
  `metrics.py` 已删除，根目录不再有任何 .py 散件。
- 新增 `Baselines/utils/`（基线侧统一数据/评估协议）：
  - `motion_io.py` / `bvh_aligner.py`：原始运动/BVH 读取器（canonical 归位；
    `results_display/script/utils/{motion_io,bvh_aligner_pose}.py` 改为 re-export shim）
  - `protocols.py`：SMPL-24（引用 anysole.types 单一来源）/ BVH-23 关节表、
    骨架树、协议标识、骨盆/足关节约定、前向轴（smpl24=2, bvh23=1，Agent C 实测）
  - `gt_loading.py`：会话网格 GT 加载、预测 archive 读取、split/manifest 助手
  - `capabilities.py`：能力 schema + 必要输入 + provenance 门控（DENIED_SOURCES）
  - `solver.py`：协议感知指标组装（零公式，全部来自 anysole.utils.metrics）+ V2T 九项
  - `evaluate.py`：**基线评估入口** — predictions + 求解器 → 逐 session + 汇总 →
    `results/baselines/<Model>/metrics/<split>_comparison.json`（schema v2）+ csv
- `results_display/script/utils/compare_core.py`：改为展示层 + re-export 面
  （消费端 import 全部不断链，求解器为同一函数对象）。

### 基线评估自产指标（用户裁定：评估不仅要产出预测文件，还要有对应的评估结果）

- 各基线通过 `python -m Baselines.utils.evaluate --model <name|alias> --split test`
  自产 metrics；R_Test2 定位为消费端。
- 实测（真实数据）：Step2Motion S13013（bvh23-native，mpjpe 32.40mm，
  mpjae/pve/contact 为 —）；MMVP pressure_toolkit S13013（smpl24-native，
  模板 surface → PVE 为 —）均落盘成功。
- **交叉验证**：与 r_test2 参考行实现逐位一致（8 项指标 diff=0.00e+00，双协议）。
- 测试：`tests/test_baseline_utils.py` 新增 33 断言（re-export 同一性、
  协议表、provenance 门控、evaluate 入口、真实数据交叉锁），三套测试
  85 + 53 + 33 全过。

### 交接（供后续任务/主 Agent 分配）

1. `AnysoleWorkspace/tool/export_baseline_motion.py`（任务 02 领地）：导出后串联
   `Baselines.utils.evaluate`，或命令手册将两者写成同一条评估命令链。
2. `r_test2/r_test4`（任务 03 领地）：`--write-model-metrics` 对基线退役，改默认
   读取基线自产 `metrics/<split>_comparison.json`（schema v2 字段已兼容）。
3. `workspace.py doctor` / `validate_baseline_exports.py`（任务 02 领地）：metrics
   校验从"目录存在"升级为"comparison 文件存在且 schema v2、V2T 恰 9 项"。
4. `Baselines/command_manual.md`（任务 02 领地）：补"评估产出 predictions + metrics"。
5. 注册表已移至 `results_display/models_modes.yaml`（evaluate.py 默认路径已适配）；
   如希望生产侧不读展示目录，可再考虑迁至 `Baselines/utils/`（需任务 02 确认）。
6. `00_评估整改总控.md` §2 的"公式集中到根目录 metrics.py"一行需同步改为
   `anysole/utils/metrics.py`（主 Agent 文件，待确认后改）。

## 修改文件

| 文件 | 改动 |
|---|---|
| `metrics.py`（根目录） | 唯一实现所在地：新增 `root_orientation_error_deg` / `root_orientation_drift_deg`（旧 yaw 公式迁移，唯一实现）、`foot_sliding_joints`（公共关节版）、`foot_sliding_vertices`（旧顶点版改名，诊断）；`METRIC_UNITS` 删除 `pve_t_mm`；新增 `MOTION_METRIC_NAMES`（含 root_orientation 两键，去掉 shape_vertex_std_mm） |
| `anysole/utils/metrics.py` | 改为根 `metrics.py` 的 re-export shim（消灭双实现；全部 importer 仅 eval_protocol 与 compare_core，已验证） |
| `anysole/utils/eval_protocol.py` | yaw 公式删除、改调公共求解器；输出键 `yaw_abs_deg/yaw_drift_deg` → `root_orientation_deg/root_orientation_drift_deg`；`foot_sliding_mm` 改关节版 + 顶点版以 `foot_sliding_vertex_mm` 诊断输出；`V2T_NAMES` 严格 9 项；`summary_metrics` 改白名单（`BRIEF_MOTION_KEYS` 10 键，contact 一律不进 brief）；payload 新增 `diagnostic_fields` 显式标注诊断键 |
| `results_display/script/utils/compare_core.py` | 删除 common19 全套（COMMON_JOINTS/SMPL_COMMON_NAMES/BVH_COMMON_NAMES/COMMON_EDGES/select_common_joints/select_common_rotations/common_edges_for/frame_mpjpe_mm）；新增 `PROTOCOL_LABELS`（smpl24-native/bvh23-native）、`normalize_protocol`、`native_edges`（SMPL-24 树 + 实测 Skeleton3 层级）、`native_frame_mpjpe_mm`；`bvh_joints` 返回原生 BVH-23（去 EndSite、父节点重映射，实测 S10103_gen.bvh 验证）；`metrics()` 按 protocol 分派骨盆对齐/足关节/root orientation，去掉 contact 计算与参数；`v2t_metrics` 严格 9 键、`load_v2t_archive` 不再要求 contact 字段；`METRICS`/`MODE_METRICS` 无 contact；新增能力门控 `CAPABILITY_FIELDS`/`CAPABILITY_ALIASES`/`METRIC_REQUIRES`/`applicable_metrics` |
| `anysole/backfill_brief.py` | 新增阶段 2 键名回填（`yaw_*`→`root_orientation_*` 纯改名；旧顶点版 `foot_sliding_mm`→`foot_sliding_vertex_mm`，公共关节版不冒充），payload 记录 `key_migration` 与 `diagnostic_fields`，brief 重派生 |
| `z_note/metrics_指标改动说明.md` | 重写为 v3 终态口径（锁定决策、10 键白名单、V2T 9 项、Step2Motion 明细、能力门控、回填记录） |
| `tests/test_public_metrics.py` | 新增：任务书 §5.2 数值测试，86 项断言全过（含 SMPL-24/BVH-23 pred=GT 零误差不经 common19、yaw 恒等/偏置/线性漂移/±180° 环绕/与旧 torch 公式数值等价、缺失能力 NaN、能力守卫、V2T 恰 9 项、关节版 foot sliding、common19 符号删除、真实 BVH-23 过滤） |

## 新增或变化的接口（供任务 03 消费）

- `compare_core.metrics(pred, gt, keep, fps, *, protocol, names=None, pred_rotations=None, gt_rotations=None, pred_vertices=None, gt_vertices=None, times=None, forward_axis=None)` — protocol 必填（接受 `smpl24-native`/`bvh23-native`/`smpl24`/`bvh23`）；contact 参数已删除。
- `compare_core.v2t_metrics(pred_pressure, gt_pressure, valid=None)` — 只返回 V2T_METRICS 9 键 + `n_valid_frames`。
- `compare_core.load_v2t_archive(path)` — 只要求 pressure_pred/pressure_gt/valid_mask；旧 contact 字段容忍但不返回。
- `compare_core.array_from_file` — 协议标识返回 `smpl24-native` / `bvh23-native`。
- `compare_core.bvh_joints(path, row=None)` — 返回 `(joints[23], names, parents)`。
- `compare_core.native_edges(names_or_protocol, parents=None)`、`native_frame_mpjpe_mm(pred, gt, *, pelvis_indices=None, protocol=None)`。
- `compare_core.applicable_metrics(capabilities, *, approved=())` + `METRIC_REQUIRES` + `read_capabilities`（接受任务 02 注册表的两种键名拼写）。
- `metrics.root_orientation_error_deg / root_orientation_drift_deg / foot_sliding_joints / foot_sliding_vertices / MOTION_METRIC_NAMES`。
- `eval_protocol.summary_metrics`（白名单）、`BRIEF_MOTION_KEYS`、`V2T_NAMES`(9)、`DIAGNOSTIC_KEYS`；明细 payload 新增 `diagnostic_fields`。

## 执行的检查/测试

1. `tests/test_public_metrics.py`（touch_gait 环境）：**86 passed, 0 failed**。
2. 真实数据验证：`results/baselines/Step2Motion/predictions/gait_model/S10103_gen.bvh` 实测为 28 通道（23 语义关节 + 5 EndSite），`bvh_joints` 过滤后恰 23、父节点重映射正确、无 EndSite。
3. 历史回填执行并核验：
   - `results/AnySole` 12 对、`results/experiments` 30 对 detail/brief 迁移完成（含 `all_models`、`singlemodal_eval`）；
   - 抽验 F0b test.json：旧 yaw 键消失、`root_orientation_deg=6.2385` 等值保留、`foot_sliding_vertex_mm=13.2335`（旧值改标）、公共 `foot_sliding_mm` 空缺（待重跑 eval）、`key_migration`/`diagnostic_fields` 已记录；brief V2T 恰 9 键、V2M 白名单 10 键。
   - 回填幂等（重复 dry-run 显示 keys already current）。
4. 静态检查：4 个核心脚本无 `yaw_abs_deg`/`yaw_drift_deg`/`fseries`/`pve_t_mm`/common19 残留（仅文档性提及）；`pve_t_mm` 已从 `METRIC_UNITS` 消失；`MODE_METRICS["V2T"]` 恰 9 项。

## 验收结果（对照任务书 §5）

- §5.1 静态检查：全部通过（见上）。
- §5.2 数值测试：1–7 全通过。其中 yaw 新实现与旧 torch 公式逐帧等价（max 差 ≤ 1e-5）；pred=GT 时 SMPL-24 与 BVH-23 全部误差指标为 0 且不经 common19（common19 符号已从 compare_core 删除）。
- 附加：能力守卫（去掉 root_translation 后 root ATE/W-MPJPE 不可用；contact 永不适用）。

## 未完成项 / 需后续任务执行

- 公共关节版 `foot_sliding_mm` 的数值回填：需重跑各模型 eval（旧文件该键空缺、brief 中暂缺该列）；属训练侧/重算流程。
- `results/backup/AnySole_BVH_backup/` 等历史归档中的旧键（`yaw_abs_deg` 等）：未迁移（任务 04 永久保留范围），列为清理/迁移候选。

## 建议清理项（不得自行删除，交由任务 04）

- `results_display` 下 common19 派生旧结果（R_Test1/2/3 compare 目录、R_Test4 含 contact 的 CSV/JSON、schema v1 comparison 文件）：由任务 03 生成精确清单。
- `results/backup`、`results/experiments` 中旧口径 JSON 的键名迁移（可选，按追溯需求定）。
- 本次未删除任何文件；回填仅改名/改标/重派生，旧值全部保留并可溯源（`key_migration`）。

## 交给下一 Agent（任务 03）的注意事项

1. `r_test1_compare.py` / `r_test2_compare.py` 对已删除符号的 import 会立即报错（COMMON_JOINTS、select_common_joints、common_edges_for、frame_mpjpe_mm 等）——这是 01 冻结后的预期状态，需按任务书 §5.2/§5.3 用原生协议接口重写；r_test2 的 `metrics()` 调用需传 `protocol=` 且不再传 contact 参数。
2. `r_test4_v2t.py` 读 contact 键处用 `.get()`，不会崩，但 fieldnames 需按 9 项清理。
3. BVH 侧 root orientation 的前向轴约定需用真实数据做一次 smoke（`forward_axis` 参数默认 +Z=SMPL 约定；Skeleton3 BVH 的轴请实测后固定传入），任务书 §5.2 第 6 条。
4. `models_modes.yaml`（任务 02）已含 protocol/capabilities/sources 且键名两拼写并存，`read_capabilities`/`applicable_metrics` 已兼容。
5. 跨边界（未分配所有权）文件需主 Agent 重新分配：`anysole/rho_grid.py` 自检键（`yaw_abs_deg`→`root_orientation_deg`，当前静默跳过不报错）；`results_display/script/utils/missing_rate_analysis.py` 的 COMPONENTS 别名（`("yaw_abs_deg",)` → `root_orientation_deg`，否则 global_yaw 组件取不到值）；`anysole/backfill_brief.py` 已由本任务扩展（原文件未在所有权表中，如另有安排请知会）。
