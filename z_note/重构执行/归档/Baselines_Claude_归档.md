# Baselines 执行归档（Agent：Claude，2026-09-29）

本次会话工作范围：应要求编写跨 5 基线（MotionPRO / Step2Motion / FPP-Net / PoseTransOpt / pressure_tookit）的输入与监督信号一览文档。纯文档任务：零 commit、零代码修改、零数据产物。

## 1. 提交清单

| 短 hash | 仓库(主库/Baselines) | 主题 | 类型(D/T1/T2/M/审计/文档) |
|---|---|---|---|
| —（本次会话 0 条 commit） | — | — | — |

## 2. 修改条目

本次会话未修改任何原生代码/数据（铁律#2：每条修改需三件套，本次无 commit 故无条目可写）。唯一新增文档见 §4-1 声明。

| 修改编号 | 类型 | 位置(文件:行) | 原生行为 → 当前行为 | 对应裁定/条目号 | 证据(测试名+结果) | 状态 |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

## 3. 数据产物

| 产物 | 路径 | 计数/大小(磁盘实测) | artifact.json(有/无) | 源 hash |
|---|---|---|---|---|
| —（无数据产物） | — | — | — | — |

## 4. 挂账与偏差

1. **未提交工作区改动（自查#2）**：`Baselines/模型输入与监督信号.md`（untracked，2026-09-28 17:32 新建，磁盘实测 23,252 bytes / 255 行）——性质：新增跨 5 基线文档（每模型输入原始采样/模型张量、监督信号 pred-监督两侧、输出、关节协议、本地适配口径；5 个模型依据 4 个并行调研 agent 的代码级核验 + 语义改动对照.md）；归属：用户 2026-09-28 口头任务「在 Baselines 写一个 md 列出 5 个模型的训练消耗与监督信号」；待主会话审核后统一提交。另注：`Baselines/utils/`（untracked）与主库 `M anysole/*`、`M results_display/*`、`D Baselines_Backup/*` 均**非本次会话改动**（其他会话/历史遗留），不在本归档范围。
2. 本次审计核实既有挂账（复述，均已登记于语义改动对照.md / 09 报告，未新增登记）：
   - MotionPRO：`smpl.npy` 生成链空缺（C2 阻塞）——语义改动对照.md MotionPRO 节；
   - Step2Motion：adapter `.pt` 为 MotionDatasetState 格式，原生 `dataset.py` 无法加载——语义改动对照.md Step2Motion 节；
   - FPP-Net：原生生效 loss 484 vs 192 维不匹配（必崩）、press2Cont 阈值 0.5/0.7 待拍板、pixel_weight 待核——语义改动对照.md FPP-Net 节；
   - pressure_tookit：当前数据树不足（depth 15 / mask 2 张），正式结果 blocked——语义改动对照.md pressure_tookit 待核节；
   - 口径事实：当前 Baselines 树以原生代码为准，「当前行为」多数落于 `Baselines_old/` 与生产 adapter——已在文档「口径说明」中声明。
3. 铁律#1 遵守：未回填 `Baselines/语义改动对照.md`；未 commit、未 push、未动数据。
