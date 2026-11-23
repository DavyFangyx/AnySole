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
