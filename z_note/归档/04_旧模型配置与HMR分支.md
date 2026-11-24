# 旧主模型、配置、历史文档与 HMR2.0/E3

partial

先读 [00_总览与执行顺序.md](00_总览与执行顺序.md)。本任务先只读梳理；01–03 确认恢复点及正式结果依赖后，才能提出具体移出清单。

## 候选范围

| 范围 | 初步处理方向 | 必须同时检查 |
|---|---|---|
| `z_note/_历史_已废弃/` | 用户明确指出可移出 | 所有历史说明引用、Git 可恢复版本、其中是否有最后一份方法来源 |
| `z_note/HMR2.0/`、`hmr2_tta/`、旧 E3 | 以整条实验线退役为目标 | 两个 submodule、`.gitmodules`、E3 导出器、资产与 HMR2.0 结果；先剥离正式 E6 公共依赖 |
| F0/F2/F4、V3/V4/V5/V6 旧模型 | 按 ID 调查 | 模型入口、共享模块、两级 registry、训练/评估分支、loss、配置、cell、checkpoint、测试与文档 |
| V7 AR/CTX/SD/早期 gate 等 | 查是否只是历史探索 | V8 导入、初始化、BiFoot/E9 与附录引用，不按目录前缀整删 |
| 旧 V8 模型与 s10、branches | 逐用途核对 | 最终 E4 消融、无枚举基座、Router/TGate/DualGate、补充图、历史比较来源 |
| `configs/` 与 `anysole/configs/` | 按正式管线收敛 | queue/running、paused、done/failed、默认注册、YAML 继承及所有写方 |

## 已知不能误删的依赖

1. `anysole/models/v8_cv7_physimu.py` → `v7_a.STRUCTURE`、`common/model_v7.py` → `common/model_v4_1.py`。旧模型入口可退役，不代表相同版本名的公共实现可删。
2. `results_display/script/e6_ft_contribution.py`、`e6_attention_teaser.py`、`e6_part_error_teaser.py` 等导入 `anysole_tta.phase_common.load_contact`；`phase_common` 又导入 `hmr2_tta.paths`、`data.assemble_npz`。HMR 退役前需提供最小公共依赖保留或迁移方案，逐个适配全部消费者。
3. E6 还从 `results/diagnostics/20261112_v8_gate_frozen/train_tgate_win_hb.py` 导入 `_fk_pos`。diagnostics 中的源码不能按“诊断产物”整删。
4. 用户退役的是 HMR2.0/E3 的多视角实验；E9 的 BiFoot-TTA 仍是论文组成部分，名称包含 TTA 不是退役依据。
5. `configs/paused_v3b_imu/` 有 15 份 conf，`paused_v8_cv8u_pressure/` 有 1 份；初查 queue/running 无任务 conf。清理时仍须复查实际运行，不启动、取消或恢复任何任务。
6. 72 个未跟踪纹理遮挡 YAML 与现有 Q14 修改先纳入 01 快照和 02 来源核对，不能作为垃圾配置移除。

## 调研要求

- 用全仓 `rg` 分别搜模型 ID、模块名、结果路径、YAML、conf、注册别名和 import；包含 ignored 源码、`.tmp_scripts`、Baselines、tests。列出全部读写方，不只模型目录内互引。
- 对旧模型建立 ID 到入口/配置/初始化权重/结果/消费者的对应表。跨版本继承、冻结载体、预训练初始化和动态 `import_module` 必须展开。
- 正式主线至少覆盖 Residual 与其基座、no-enum 基座、训练消融、Router、TGate、DualGate、BiFoot 及 Q 项实际使用的模型；以 02/03 最终清单为准。
- 区分可直接移出的叶子资产、需解除依赖的旧代码、必须保留的公共实现。迁移公共函数时先列出现有合适落点和所有适配方，不为清理新增抽象框架或另写指标公式。
- 对 `anysole/registry.py`、`configs/z_gen/model_registry.py`、`common.py`、`run.sh`、`anysole/train.py`、`eval.py`、`utils/losses.py`、展示注册及文档中的旧入口提出逐个适配清单。注册表移除与源码/结果退役要一致。
- `results_display/script/legacy/` 和其它旧脚本是否仍被正式生成器 import、调用或提供常量需核对。旧 V8 表仍在用时，不先移走其导出所需数据。
- HMR submodule 与独立仓库内容由 01 保证可恢复；不要直接 `submodule deinit`、修改 gitlink 或删除 `.git/modules`。

## 验收与交付

在本文件追加精确候选清单、全部读写方、依赖迁移方案、恢复位置、受影响测试/命令和容量；把数据/大结果路径交给 06，把基线路径交给 05。

调研阶段只做静态检查。执行阶段需验证保留模型的 registry/builder、CLI/配置读取、既有核心数值测试以及与 02/03 对应的固定输入评测；完整模型/环境不具备时明确记录，不能用仅 `py_compile` 代替复现验收。

不得因发现旧测试依赖就直接删除测试或放宽验证。修改共享 registry/CLI/文档由 07 统一安排，避免多个 Agent 并发改同一文件。

---

## 调研结论（2026-11-23~24，只读调研；未移动/删除/提交任何文件）

状态：`partial`。六个分片（HMR/E3、F/V3–V6、V7/V8/s10/branches、configs、legacy/tests、历史文档）完成并经主线抽样核验；2026-11-24 用户对 6 项待裁定逐项拍板，文档类移出已执行（见 §11），源码/注册表/权重退役移交 07/06。数据/大结果容量移交 06，基线路径移交 05。

### 1. 保留边界（正式线依赖，退役时不可触碰）

| 对象 | 依据 |
|---|---|
| `anysole/models/v8_cv7_physimu.py` | V8_IMU_Residual 主模型 + FT 消融四变体（CleanOnly/woEnum/woAdapt/DecoderOnly）+ NoEnum 基座 |
| `v8_cv7_physimu_router.py` / `v8_cv5_physimu.py` / `v8_cv7_physimu_gate.py`（+`v8_cv7_physimu_decgate`） | E4 结构消融行 Router/TGate/DualGate |
| `anysole/models/v7_a.py`、`v7_c2sd2.py` | 正式入口 import：`v8_cv7_physimu.py` 顶部 `from v7_a import STRUCTURE`；`v8_cv5.py` 整文件 re-export、`v8_cv5_physimu.py:12` import v7_c2sd2 |
| `common/model_v7.py`、`common/model_v4_1.py`、`common/model_v2.py` | model_v7 是正式入口 import 源；model_v4_1 被 `model_v7.py:35-38` import（V41ContactHead/_temporal_encoder）；model_v2 被 `models/__init__` 再导出、train/eval/infer 与 tests 使用 |
| `common/model_v3h/v3f/v5_1.py`、`utils/constraints{,_g,_h}.py`、`utils/losses.compute_losses_v5_1` | 被 `eval_protocol.py:70-75`、`train.py:70-72`、`eval.py` 无条件 import（仅旧 ckpt 分支触发）。文件小；保留，或由 07 剥离旧模式分支后移除 import |
| `anysole/models/v3_3b_physimu.py` | `tests/test_v3b_physimu.py:10` 直接 import `build`；V3_Bv1 线注册活跃（2026-10-31 用户重注册） |
| `anysole_tta/phase_common.py`、`e6_phase_analysis.py` | E6 正式链公共件，20+ 个 results_display 脚本消费（e6_ft_contribution/e6_attention_teaser/e6_part_error_teaser/e6_reconstruction_teaser/e6_response_ratio/e6_v8_best_sessions/e6_vgate_router_k_teaser/e6_paired_stats/e6_group_compare/e6_v8_phase_teaser/s_gate_diag/e2_t_bifoot_residual/bifoot_yaw_tb_metrics/e5_t_bifoot_scenarios 等 + .tmp_scripts 3 处） |
| `hmr2_tta/paths.py`、`hmr2_tta/data/assemble_npz.py` | E6 链经 phase_common 导入面（详见 §2 迁移方案） |
| `results/diagnostics/20261112_v8_gate_frozen/` 内 `train_tgate_win_hb.py::_fk_pos`、`train_router_gate.py::build_router`、`e5_v8_smpl_gain_heatmap.py` | E6 脚本 sys.path 导入（`e6_ft_contribution.py:47-50`、`e6_vgate_router_k_map.py:52`）——diagnostics 不能按"诊断产物"整删 |
| s10 线 | `vgate_s10_export.py` + `20261114_imu_eval_v2(_all)_s10` + 两个 s10 展示树；Q7 长时窗表依赖 s10 树 |
| `results_display/script/visual_warmup_compare.py` | `tests/test_visual_warmup_compare.py` 直接 import |
| `.tmp_scripts/` 79 个跟踪文件 | Q2/Q10 复现入口、s10 复现入口（run_cv7_s10_motion.sh、run_formal_eval_v2_s10.sh） |

### 2. HMR2.0/E3 线（整线退役候选）

**退役候选（全部 git 可恢复；结果资产移交 06）：**

- 源码 `hmr2_tta/{eval/,models/,data/surface_sel35.py}`（全仓零消费者；`models/` 是唯一导入 4D-Humans `hmr2` 包的位置）
- `anysole_tta/{easymocap_baseline,calib_baseline,prep_kp2d24,prep_rtmpose26,e_teaser,eval_cam3_init,run_tta,e7_driver}.py`——唯一消费方 `e_results_export.py:1216-1217`（E3 分支 lazy import，try/except 兜底返回 "—"）
- `results_display/script/e_results_export.py`（E1–E7 旧导出器；输出树 `results_display/Etest/` 已不存在）
- 结果资产 `results/baselines/HMR2.0/` 8.8G：checkpoints 7.9G（b2a/b2b/b2c 各 2.6G 等）、data/npz 342M、data/ttok 308M、data/pressure 21M、predictions/zero+tta_theta 45M、旧 metrics+logs
- `z_note/HMR2.0/` 文档 35M（不含两个 gitlink 工作区），git 历史可恢复
- 分支 `origin/feat/e3` 独有提交 `52911ce`：E3 实验代码唯一位置（anysole/tta.py、data/e3_grouped.py、e3_observations.py、tests/test_e3_*.py 等 23 文件，HEAD 均无）→ **保留分支引用**，勿删分支
- gitlink：`.gitmodules` 两条（4D-Humans@efe18de、EasyMocap@e6006fd）。不 deinit、不改 gitlink、不删 `.git/modules`（00 约束）；移除方式需用户确认

**必须保留的公共实现与正式消费面：**

- `hmr2_tta/paths.py` 的 `PROJECT_ROOT/facts_session_dir/resolve_uri/calibration_json/RESULTS_ROOT`（注：`paths.py:7` 硬编码 PROJECT_ROOT，迁移时改 `__file__` 相对）
- `hmr2_tta/data/assemble_npz.py` 的 `load_manifest_rows/load_calibration_chain`。4D-Humans 依赖只在 `build_smpl_layer()`（`assemble_npz.py:104`）惰性 import，唯一调用方是退役线 `hmr2_tta/eval/eval_zero.py`——E6 路径不经过它
- E6 链实际导入面：`phase_common.py:18-19`（`paths.{PROJECT_ROOT,facts_session_dir,resolve_uri}` + `assemble_npz.load_manifest_rows`）、`e6_phase_analysis.py:30`（`paths.PROJECT_ROOT`）；`load_calibration_chain` 的消费方（easymocap/calib_baseline、prep_kp2d24）全部在 E3/E7 退役线内

**迁移方案（二选一，07 统一执行）：**

1. 最小保留：留 `hmr2_tta/{__init__,paths,data/assemble_npz}.py` 3 文件；paths.py 改动态 PROJECT_ROOT。缺点：3 文件内仍残留 build_smpl_layer 的惰性 4D-Humans 引用
2. 迁入 anysole/（更干净）：`load_manifest_rows` 迁 `anysole/data/manifest.py`（其仅依赖 `AnysoleWorkspace.tool.workspace`，无 4D-Humans）；路径符号迁 `anysole/paths.py`；改 `phase_common.py` 2 行 + `e6_phase_analysis.py` 1 行 import；间接消费者（20+ e6 脚本）零改动。验收 = 重跑 `e6_ft_contribution.py` 单折固定输入（02/03 口径），不能仅 py_compile

**E7 已裁定退役（2026-11-24 用户：E7 不在论文）**：`run_tta/e7_driver/eval_cam3_init` 与 `predictions/anysole_tta_stage1_V7_Cv5_test`（84M）、`predictions/easymocap`（99M）、`data/rtmpose26`（21M）、`data/kp2d24`（19M）随 E3 整线退役（结果/数据移交 06）。注意：BiFoot-TTA（E9，`20261111_variant_sweep`）与本线无关，不误伤（04 已知项 #4）。

**受影响面**：tests/ 对 hmr2_tta 零导入（退役不触碰测试）；受影响 CLI = run_tta、eval_tta、eval_zero、assemble_npz、surface_sel35、prep_kp2d24/rtmpose26、easymocap_baseline、calib_baseline、e_teaser、e7_driver、e_results_export（全为 E3/E7 旧线）。

### 3. F0/F2/F4 与 V3/V4/V5/V6（按 ID）

**可移出叶子入口（24 文件，KB 级，git 可恢复；无运行消费者）**：`f0b/f2/f2p4/f4a/v3_2/v3_3a/v3_4a/v3_4b/v3_4c/v4a/v3_3e/v3_3f/v3_3g/v3_3h/v4b/v5_1/v4_1`（入口）+ `v3_3b_{imu,imu_nogate,physimu_nogate,query}/v3_3c/v3_3d`。注意区分：`v4_1.py` 入口可退役，但 `common/model_v4_1.py` 必须保留（model_v7 依赖）；`v3_3b_physimu.py` 必须保留（tests 导入）。`common/model_v3e.py/model_v3g.py/model_v4b.py` 仅各自入口 import，与入口同退役。

**需解除依赖（07 适配清单）**：`configs/tools/runner.py:93,115`（V3_Gv1/V3_Cv1 分支）；`anysole/registry.py:39-114` 旧别名约 60 条 + `configs/z_gen/model_registry.py` 旧网格生成代码（`common.py:26-68,594`，V3_Bv1_cam*/tw_scan 35 格/V4_Av1/V5_Av1 格/λ 扫描 24 格）。注意：V3_Bv1（IMU/PhysIMU/NG 四折）与 V8_Cv8U 系列虽仍是注册条目（用户 2026-10-31 重注册、11-12 注册），**2026-11-24 用户已裁定该线为死线不再继续**——registry 条目、paused conf（`configs/paused_v3b_imu/` 15 份、`paused_v8_cv8u_pressure/` 1 份）与权重随退役；`get_builder` 动态 import_module，旧 ckpt 重评依赖注册——注册表移除与源码/结果退役一致。

**数据/权重（移交 06）**：`results/AnySole/V3_Bv1*` 18 树 ~5.0G；`results/cells/` 旧模型目录（V3/V4/V5 各 4.38/2.89/3.89GiB 含变体）。

### 4. V7 探索与旧 V8 变体、s10、branches

**V7 探索入口 ~33 个**（`v7_c/v7_c2*` 族，registry 有注册、不在 02 正式生成链）。非零消费者：`anysole_tta/run_tta.py:92-98` TTA_MODELS（V3_Bv1+V7_Cv1..Cv5）、`e6_phase_analysis.py`（V7_Cv4/Cv5/Cv7_s1 cells）、`e_teaser.py`（默认 `--cell V7_Cv5_s1`）、tests（test_v7_ar2/test_v7_ctx/test_v7_ar2_rollout/test_v7_foot_loss/test_v1_v3_models）。退役需 07 同步改 TTA_MODELS 注册与测试导入（04 要求：不得直接删测试）。

**V8 正式五模型**（保留）：v8_cv7_physimu（主+FT 消融+NoEnum）、v8_cv7_physimu_router、v8_cv5_physimu、v8_cv7_physimu_gate（+decgate）。eval 树实测：`20261114_imu_eval_v2_all` = {Residual,Router,TGate,DualGate}；`_v2` 另加消融四变体+_smoke；s10 树只含 Residual ✓。

**V8 历史变体（成组退役候选）**：v8_cv5/cv4/cv7/cv5_imu/cv7_imu/cv5t1/cv5po/cv5vg/cv5vig/cv5vug/cv8/cv8u 族/cv7_physimu_tgate + 诊断脚本组（`z_gate_diag.py`、`stage2_train_gate.py`、`build_vug_ckpts.py`、`v8_cv8u_*_probe.py`、`v8_cv8u_gate_diagnose.py`、`v8_cv8u_oracle_probe.py`）。

**s10：保留**（正式补充线，Q7 依赖；容量约 246M：展示树 16M+31M、eval 树 42M+93M+64M）。`Etest_crossfit_v8_s10`（31M）随旧树退役判定处理。

**branches**：`feat/b-distill`、`feat/view-consistency` 独有提交为 HMR2.0/E3 同期多视角探索（52/49 文件），随该线退役，保留 ref 即可；`fix/pipeline-contracts` 独有提交 ab2a679 的修复（runner.py `--variant`、test_pipeline_contracts.py）**主线未落地**，保留 ref（01 已登记）。

**权重（移交 06）**：`results/AnySole/V7*` 78.5G（含 V7_Cv5_gb 11G）、`results/cells/V7*` 369 目录 13G；`results/AnySole/V8*` 31.4G——保留面 = `V8_Cv7_PhysIMU_f*` 四折基座 + `V8_Cv7_PhysIMU_NoEnum_*` 共 4.1G，其余 ~27G（Cv2/Cv4/Cv5/Cv6/Cv8 系四折基座）为历史；`results/cells/V8*` 533 目录 26G，正式来源 = `V8_Cv7_PhysIMU_f*_s1` 4 目录。

### 5. configs/ 与 anysole/configs/

- 复核 04 已知项 #5：`configs/queue`=0、`running`=0 ✓；`configs/paused_v3b_imu/` 15 份、`configs/paused_v8_cv8u_pressure/` 1 份（00830__V8_Cv8U_SG_fS5S6S11_s1）——两线已裁定死线（2026-11-24），paused conf 随退役；执行阶段仍须复查实际运行进程，不启动/取消/恢复任何任务
- `done`=750、`failed`=17、`logs`=315。failed 中 4 个为 `V8_Cv7_PhysIMU_NoEnum` 四折（00833–00836，从未调度成功；正式 NoEnum 基座来自 diagnostics FT 消融）。**`configs/done/` 已按用户裁定移出**（→ `_trash/20261124/configs_done/`，`.gitkeep` 保留，目录可在下次调度时重建）；`failed/`、`logs/` 未裁定暂留
- 调度机制：`scheduler.sh` 单队列原子 mv → `run.sh` 按 MODEL 分发（anysole/motionpro/step2motion/pressure_toolkit/fpp_train/fpp_infer/posetransopt）；任务 conf source `defaults.conf`（默认 `CONFIG=anysole/configs/v1.yaml`）
- `anysole/configs/` 253 yaml：**v1 是配置系列名（E3 基础口径），不是退役模型 V1，不能按前缀退役**。v1_* 家族（fold 四折/cam/occ/gap/wo05/bv2/bv3/ar2）+ tmp_v8_scene* 6 个。退役候选：`v1_wo05.yaml`、`v1_bv2/bv3.yaml` 三族（仅被 registry 已删条目引用）；`tmp_v8_scene*` 与 Q14/E5 场景线核对（移交 02/07）
- 04 已知项 #6 更新：**72 个 Q14 tex conf 已由 commit 3fb33ca 入库**（现 112 个 tex 全部 git 跟踪），不再属"未跟踪"，更不能作为垃圾配置移除 ✓
- conf 写方：`configs/z_gen/gen.py`（per_action_folds/wo05 等子命令）、gen_e1/e2/e4/e5、gen_full_package、gen_u_only、`tools/create_queue.py`、tools/tune.py、手工 cp
- 两级 registry：`configs/z_gen/model_registry.py`（2668 行 MODELS 规格表）import 时执行 `_drop_legacy_entries()`+`_drop_all_but_current()` 两轮清理，运行时只保留当前线；`anysole/registry.py`（819 行）旧条目全部仍在 → 07 统一移除，与入口退役一致

### 6. legacy 脚本、历史链脚本与 tests

- `results_display/script/legacy/`：16 文件、288K、全部 git 跟踪。**15/16 可移出**；唯一写方适配 = `AnysoleWorkspace/tool/smoke_baseline_integration.py:150`（compile 语法检查清单，非运行 import）→ 移出时同步从 smoke 工具删该行（07）。核心实现已被现行 R 系统一入口吸收（`r_test1_visualize.py:75` 注明"唯一实现点"）
- **例外（不可直接移出）**：`legacy/r_test1_visualize_step2motion.py` 被独立基线仓库 `Baselines/Step2Motion/src/visualize_compare.py:16-18` 经 importlib 加载（文件缺失即 `raise ImportError`），且 `Baselines/Step2Motion/src/test.py:20` 运行时 `from visualize_compare import ...`。迁移方案：渲染器提升到 `results_display/script/` 根层，同步改 `visualize_compare.py` 的 `_RENDERER` 路径——涉及独立基线仓库修改，归属 05/07 协调（本任务已移交）
- 其余历史引用（无需运行时适配，退役时 07 同步文档）：`Baselines/决策/04_E_评估收尾.md:522`、`z_note/评估/评估配置说明书.md:105,116`（旧 display 链描述）；`anysole/model_fix_note/command_manual.md:97,168,533`（历史命令示例，用户维护文档注意"用户改过的 md 不要动"）；`results_display/script/utils/ridge_probe.py:2`、`utils/render_common.py:3-5`（docstring 历史溯源，本体是现行公共件）；`.claude/settings.local.json:82-83`（历史命令 allowlist 陈旧条目）
- 历史链脚本 9 个（~100K）：`e2_crossfit_export`（`e2_crossfit_export.py:75` 硬编码 RootAbs，重跑即回退，02/03 已裁定）、`e2_t_bifoot_table_replace`、`e5_t_bifoot_reference_replace`、`e5_wham_align_to_e2`（已废弃 WHAM_OFFSETS 平移）、`e5_motionpro_export`、`e5_sceneocc_v7v8_export`、`v3b_imu_crossfit_export`、`d_test4_baseline_tactile`、`fill_step2motion_gt_anchored`。正式链已换 e2_v2_table/e5_v2_export（02 结论）；`results_display/README.md:29-32` 记载旧链需同步更新（07）；`fill_step2motion_gt_anchored` 为基线 GT 锚列一次性填充脚本，退役前确认当前树基线行无需重填
- e6 脚本 5 处死 sys.path（`z_note/HMR2.0/hmr2_gait` 目录不存在，实际 import 全来自 anysole_tta/anysole）：`e6_paired_stats.py:35`、`e6_group_compare.py:31`、`e6_best_sessions.py:27`、`s_gate_diag.py:41`、`e6_gate_values_occlusion.py:30`——随 HMR2.0 退役清理
- E6 正式链对 diagnostics 的导入（不可整删，04 已知项 #3 具体化）：`e6_ft_contribution.py:47-48` → `20261112_v8_gate_frozen/train_tgate_win_hb.py::_fk_pos`、`:50` → `e5_v8_smpl_gain_heatmap.py`；`e6_vgate_router_k_map.py:52` → `train_router_gate.py::build_router`
- tests（26 个）对退役候选的依赖：`test_bifoot_standalone.py`（sys.path 导入 diagnostics `20261111_variant_sweep/bifoot_standalone.py`——BiFoot 提升为 git 包时改 import，01 已记录）；`test_vtbridge.py`（`20261110_vtbridge/se2_smoother.py`，doc16 负结果）；旧模型测试 10 个（test_v1_v3_models/test_v3b_imu/test_v3b_physimu/test_v5_conditioning/test_v7_ar2(_rollout)/test_v7_ctx/test_v7_foot_loss/test_e8_null_t，依赖 v1/v3/v7 入口、utils.losses、train 旧函数、registry.get_builder、common.model_v2/v7）。04 要求不得因旧测试依赖就删测试——**需用户裁定：旧模型退役后这些测试随模型退役，还是保留为历史验证**（公共件 utils/train/registry/common 保留）
- `.tmp_scripts` 命中旧模型关键词的 18 文件：全部 V8 时代运行脚本（router/FT 消融、E6 phase、s10、cv5 训练）+ 2 个旧辅助脚本（gate_parts_reexam.py、make_router_eval_ckpts.py）；Q2/Q10 与 s10 复现入口必须保留（02 已裁定）

### 7. z_note/_历史_已废弃/

- 42 文件、780K、全部 git 跟踪、无未跟踪文件；**零运行依赖**（全仓 rg 仅文档引用）。用户已明确可移出
- **结论：可整体移出**（按用户规则移入 `_trash/<日期>/`；git 可恢复）。**已于 2026-11-24 执行**（→ `_trash/20261124/z_note_历史_已废弃/`），3 处链接处理：
  1. ✅ `z_note/README.md:9`：按用户"怎么方便怎么来"改为指向 `归档/00_总览与执行顺序.md`；引用纪律改"数字只引 z_note/归档/ 与 results/cells/*/metrics/"（现状.md 随目录移出，git 历史可恢复）
  2. ✅ Q14 说明.md:20,66：`E5_视觉退化扩展说明.md` 恢复至 `z_note/` 原位置（76c2dbd 移动前路径），两处链接改为 `z_note/E5_视觉退化扩展说明.md`（最小路径替换，未动其余内容）
  3. `z_note/multimodal_fusion/V7_方案/V7_C门控修改方案.md:13` 相对链接本就已断（唯一副本在废弃 V7旧方案/）——移出不改变现状，未处理（07 可选）
- 唯一方法来源：V4_A/V5_A/V6_A 方案、T 信息量判定、codex 对话、E5 视觉退化扩展说明的**最后磁盘副本**在本目录；git 历史原路径可恢复（移动前基线 `2873254`、`76c2dbd`、`88a95fd`，`git checkout <SHA> -- <path>` 可回原位置）
- 另：`configs/z_gen/common.py:766`、`model_registry.py` 多行、`anysole/data/make_gap_cache.py:3`、`occ_mix.py:3`、`anysole/models/common/model_v4_1.py:4`、`v3_3d.py:5`、`position_head.py:3`、`e_results_export.py:589` 等代码注释引用已消失的旧路径（历史说明注释，非运行依赖）——退役时不必逐一修，属 07 可选项

### 8. 执行阶段验收（07 统一协调；本任务不动手）

1. 保留模型 registry/builder 验证：`get_builder` 构建 E4 正式五模型 + NoEnum 基座成功
2. CLI/配置读取：run.sh 正式链 conf 解析不变
3. 固定输入评测：02/03 口径下重跑 `e4_v2_tables.py`、`e6_ft_contribution.py` 单折，与现表逐位一致
4. hmr2_tta 公共面迁移后重跑 `e6_ft_contribution.py` 单折验收（不能仅 py_compile）
5. 完整模型/环境不具备时明确记录（00 约束）
6. 退役顺序建议：先 HMR/E3 线（依赖面已最小），再旧模型入口+registry（07 同步），最后旧权重/结果（06 按数据规则）；每步先列精确路径+迁移目的地等用户确认

### 9. 核验命令与原始输出（主线抽样核验）

```text
phase_common.py:18-19  hmr2_tta.paths(PROJECT_ROOT/facts_session_dir/resolve_uri)+assemble_npz.load_manifest_rows ✓
e6_phase_analysis.py:30 hmr2_tta.paths.PROJECT_ROOT ✓
rg hmr2_tta（排除 anysole_tta）→ anysole/results_display/.tmp_scripts/tests/configs/AnysoleWorkspace/Baselines 零命中 ✓
assemble_npz.py:104 hmr2.models.smpl_wrapper 惰性 import 仅 build_smpl_layer()；调用方仅 hmr2_tta/eval/eval_zero.py ✓
v8_cv7_physimu.py: from v7_a import STRUCTURE; common/model_v7.py:35-38 import model_v4_1 ✓
eval_protocol.py:73-75 六条旧模型 import 无条件 ✓
eval 树：_v2_all={DualGate,Residual,Router,TGate}；_v2 另加消融四变体+_smoke ✓
configs/paused_v3b_imu=15、paused_v8_cv8u_pressure=1（00830）、queue=0、running=0 ✓
legacy 16 文件 git 跟踪；smoke_baseline_integration.py:150 唯一外部引用 ✓
e6_ft_contribution.py:47-50 → _fk_pos + e5_v8_smpl_gain_heatmap ✓
hmr2_gait 死 sys.path ×5（e6_paired_stats:35/e6_group_compare:31/e6_best_sessions:27/s_gate_diag:41/e6_gate_values_occlusion:30）✓
README.md:9 [现状.md](现状.md) 断链 ✓；Q14 说明.md:20,66 活链接到 _历史_已废弃 ✓
e_results_export.py:1216-1217 lazy import calib/easymocap_baseline；results_display/Etest 不存在 ✓
run_tta.py:92-98 TTA_MODELS={V3_Bv1,V7_Cv1..Cv5} ✓
容量：results/baselines/HMR2.0 8.8G；AnySole V3_Bv1* 5.0G / V7* 78.5G / V8* 31.4G（正式基座 4.1G）；cells V7 13G/V8 26G；z_note/HMR2.0 35M；legacy 288K；_历史_已废弃 780K；done 3.1M+failed 76K+logs 1.7M
```

### 10. 用户裁定记录（2026-11-24）与移交

| 项 | 裁定 | 落实 |
|---|---|---|
| ① E7 是否在论文 | **不在论文** | E7 随 E3 整线退役（§2 已更新；结果/数据移交 06） |
| ② V3_Bv1 与 V8_Cv8U 线 | **死线，不再继续** | registry 条目、paused conf、权重随退役（§3/§5 已更新；07 移除注册、06 移权重） |
| ③ 旧模型测试 | 未答复 | 暂按 04 要求保留测试不动；07 执行源码退役时再裁定 |
| ④ 现状.md 链接 | **怎么方便怎么来** | 已执行：README 改指 `归档/00`，现状.md 随目录移出（§7） |
| ⑤ Q14 说明.md 链接 | **怎么方便怎么来** | 已执行：E5_视觉退化扩展说明.md 恢复 `z_note/`，链接最小替换（§7） |
| ⑥ configs/done | **不用，可删除** | 已执行：移入 `_trash/20261124/configs_done/`（§5；.gitkeep 保留） |

- 移交 06：`results/baselines/HMR2.0/` 8.8G、`results/AnySole/{V3_Bv1*,V7*,V8 历史}`、`results/cells/` 旧模型与 V7/V8 非正式目录、`predictions/{anysole_tta_stage1_V7_Cv5_test,easymocap}`、`data/{rtmpose26,kp2d24}`
- 移交 07：HMR/E3 源码删除组 + 公共依赖迁移（§2 方案 2 推荐）；24 个旧模型入口 + ~33 个 V7 探索入口 + V8 历史变体与诊断脚本组；registry 条目移除（anysole/registry.py 旧别名 + z_gen MODELS 死线/旧网格）；legacy 15 文件 + 历史链脚本 9 个 + e6 死 sys.path 5 处 + smoke 工具 compile 清单行 + results_display/README 生成链描述；§8 执行验收（registry/builder、固定输入重跑）
- 移交 05：`Baselines/Step2Motion/src/visualize_compare.py` 渲染器路径迁移（`r_test1_visualize_step2motion.py` 提升到 results_display/script/ 根层后同步改 `_RENDERER`）；Baselines 仓库内与 E3 相关的部分（如有）
- 移交 01：branches 退役 ref（feat/b-distill、feat/view-consistency、feat/e3、fix/pipeline-contracts）——保留 ref 不删分支

### 11. 已执行记录（2026-11-24）

只读调研之后执行的全部为"用户已裁定"项，未触碰源码/注册表/权重：

```text
mkdir _trash/20261124
mv z_note/_历史_已废弃/E5_视觉退化扩展说明.md → z_note/          （恢复 76c2dbd 移动前原位置）
mv z_note/_历史_已废弃 → _trash/20261124/z_note_历史_已废弃/       （原 42 文件，恢复 1 个后余 41，git 可恢复）
mv configs/done → _trash/20261124/configs_done/                    （750 conf，3.1M；.gitkeep 留在原位）
Q14 说明.md：2 处链接 z_note/_历史_已废弃/E5_... → z_note/E5_...   （最小路径替换）
z_note/README.md：现状.md 行改指 归档/00；目录职责删 _历史_已废弃 行；引用纪律同步
git 影响：_历史_已废弃 41 个跟踪文件在工作区显示为删除（待用户提交）；configs/done 本就被忽略（0 跟踪）
未执行：源码删除、registry 修改、权重/结果移动、分支处理（分别移交 07/06/01）
```
