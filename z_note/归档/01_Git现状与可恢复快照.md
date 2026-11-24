# Git 现状与可恢复快照

partial

先读 [00_总览与执行顺序.md](00_总览与执行顺序.md)。本任务优先于口径修订与清理。目标是在 rebuttal 时既能恢复清理前现场，也能单独恢复已投稿结果对应版本。

## 已核对的现场

- 主仓：`SMPL_Motion`，HEAD `3fb33ca`，领先本地 `origin/SMPL_Motion` 1 个提交。远端引用是本地缓存，本轮没有联网 fetch，不能当作远端实时状态。
- `Baselines/` 是独立仓库，主仓将其忽略；HEAD `d8225cf`，分支 `main`，初查工作区干净。主仓 commit 不保存它的内容。
- `z_note/HMR2.0/4D-Humans`、`EasyMocap` 是主仓 gitlink，对应 `efe18deff163b29dff87ddbd575fa29b716a356c`、`e6006fd3814d5f8ad45a7ce965beb9bcc90767f2`。`git submodule status` 对 4D-Humans 显示 `-`，但 `git -C` 能读到独立仓库及 HEAD，需核对登记/恢复状态；不能把 gitlink 当作上游源码备份。
- 主仓 `.git` 约 6.5G，基线 `.git` 约 126M；是否含历史大文件需只读查明，暂不做 history rewrite、GC 或 prune。
- 初查有 3 个 tracked 修改、3 个 tracked 删除、72 个未跟踪配置。修改位于 `generate.py`、`q14_tex_occlusion_tables.py`、V8_IMU 论文版说明；删除是 `_trash/20261115/E6_s10_old2/` 内三个已跟踪文件。不能重置、丢弃或误认为本轮删除。
- `.tmp_scripts` 有 79 个跟踪文件；其余临时源码、`results/diagnostics` 内源码、最终展示树、权重/预测大多被忽略，Git 快照不能自动覆盖。

## Git 分支初查

`git merge-base --is-ancestor <ref> HEAD` 与 `git rev-list --left-right --count HEAD...<ref>` 的结果：

| 引用 | SHA | HEAD 独有提交 | 引用独有提交 | 初步处理 |
|---|---|---:|---:|---|
| `BVH_Motion` / `origin/BVH_Motion` | fe46b3b | 254 | 1 | 保留独有提交的可恢复引用，查 BVH 时代来源 |
| `origin/feat/b-distill` | d48305d | 87 | 5 | 查独有提交与实验用途，再提退役 |
| `origin/feat/e3` | 52911ce | 40 | 1 | 用户明确 E3 为旧线，仍先保存独有提交 |
| `origin/feat/view-consistency` | 2b223d6 | 87 | 5 | 查独有提交；不因与另一分支计数相同就视为重复 |
| `origin/fix/pipeline-contracts` | ab2a679 | 136 | 1 | 查合同修复是否已以别的提交落地主线 |
| `origin/main` | bccb60f | 257 | 0 | 已是 HEAD 祖先；保存引用后再评估是否移除分支名 |
| `origin/merge/metrics-into-smpl-motion` | 9008784 | 87 | 0 | 已是 HEAD 祖先，同上 |
| `origin/metrics` | 4aa5103 | 135 | 0 | 已是 HEAD 祖先，同上 |

现有标签 `review-baseline-20260928` 不是本轮论文恢复点。尚未创建新的快照、标签、bundle 或提交。

## 调研与封存要求

1. 记录各仓库完整 SHA、分支、标签、远端配置、worktree、submodule、工作区/暂存区差异。查看 reflog 与独有提交，必要时检查 patch 等价性；祖先关系只证明提交包含关系，不证明实验不再需要。
2. 对主仓和独立仓库分别规划可恢复备份：保留全部必要 refs/提交，并保存工作区与 index 差异、未跟踪和被忽略的必要源码。不能用一个标签声称保存了 dirty working tree；不要使用 `git add .` 把海量数据或用户不相关变更混入提交。
3. 用户确认的“当前现场”与“投稿版本”分别锁定。先从最终目录、Q1/Q10、日志和论文使用表反查版本；如果论文/LaTeX 不在仓库，列为缺少输入，不推断已投稿数字。不得自动改稿或改写投稿证据。
4. 明确哪些资产需 Git、哪些需独立包：最终 FT 与 BiFoot 源码/权重、基线独立源码、准确配置/splits/manifests、指标实现、逐折/逐会话原始结果、最终采用的图表、环境版本、外部数据位置与校验值。源码位于忽略目录仍必须入恢复方案。
5. 用校验清单关联代码 SHA、dirty patch、数据/权重/配置/manifest、输出及现有运行命令。日志里旧绝对路径保留原文，恢复时另给路径映射，不修改历史证据。
6. 区分训练复现、固定权重重新推理、固定预测重新评测、原始评测重新生成表图四层。逐层说明必须保留什么、当前实际验证到哪层；GPU/依赖/数据不具备时如实记录，不能把“能 clone”当作实验已复现。
7. 实际创建快照时先扫描受影响路径的所有读写方，限定写入范围。保留投稿快照，再在独立分支/副本做口径修订；不得 amend、force push 或改写已有投稿历史。

## 交付与验收

将实际调查结论追加于本文件：每个仓库/资产的保留位置与恢复命令、完整版本清单、遗漏项、是否已有异机或独立目录可读备份。只提出需要封存的具体对象，不在调研阶段删除任何 refs。

至少附以下只读命令的完整原始输出；真正生成备份后，再附 verify 与独立恢复输出：

```bash
git status --short --untracked-files=all
git diff --stat
git diff --cached --stat
git branch -avv
git worktree list
git tag -l
git submodule status
git -C Baselines status --short
git -C Baselines branch -avv
git ls-files .tmp_scripts
```

若使用 Git bundle，必须 `git bundle verify <实际文件>`，在隔离目录恢复并核对各 refs、必要历史对象、工作区差异与资产校验值。数据和权重不能依靠网络“将来再下”作为唯一备份。未完成此项，不进入删除执行。

---

## 第 4 点盘点：Git / 独立包 / 忽略 三分类（2026-11-23）

partial。只清点、只记录，未执行任何 add/commit/移动；解跟踪见 §D。

### A. 需入 Git：源码 / 配置 / 文档（已跟踪 ✓，除注明"待 add"）

| 资产 | 位置 | 状态 |
|---|---|---|
| 主模型全部源码（V8_IMU_Residual=`v8_cv7_physimu.py`、v7/v8 全系、common/） | `anysole/models/`（106 文件）、`anysole/`（共 342 文件含 eval/infer/train/registry） | 已跟踪 |
| 指标实现（PA-MPJPE/MPJPE/轨迹/物理量等） | `anysole/utils/metrics.py`、`eval_protocol.py`、`evaluation_contract.py`、`vision_smpl_metrics.py`；`anysole_tta/`（11 文件）；表图生成器 `results_display/script/`（173 文件） | 已跟踪 |
| 协议：四折 splits/manifests/注册表/标定 | `AnysoleWorkspace/protocol/`（77 文件：`splits/crossfit_woS13/` 四折 csv+manifest、motionpro_v4_registry、step2motion_registry、calibration、`visual_baselines_unit_manifest.sha256`） | 已跟踪 |
| 训练/评测配置 | `anysole/configs/` 181 已跟踪；**72 未跟踪**（Q14 纹理变体，见下） | 待 add（记录） |
| 论文口径文档 | `z_note/multimodal_fusion/V8_IMU_论文版_模型与训练.md`（工作区有用户未提交修改，保持不动） | 已跟踪 |
| 调度器工具 | `configs/`（43 文件：bg.sh/defaults.conf/run.sh 等；queue/running/done/failed/logs/generated 忽略） | 已跟踪 |
| 历史临时脚本 | `.tmp_scripts/` 79 已跟踪文件（Q2/Q10 引用）；目录已忽略，新文件不入 git | 保持现状 |
| 本归档任务文档 | `z_note/归档/` 8 个 md | 待 add（记录） |

**72 个未跟踪 Q14 纹理配置**（`anysole/configs/v1_occ{010,030,040,060,070,080}_tex_{crate,fabric,wood}_fold_{S5S11S12,S5S6S11,S5S6S8,S6S7}.yaml`）：与已跟踪的 40 个 `occ{020,050,090}` 纹理 conf 同族（由 `v1_occ050_fold_*.yaml` 单键派生），应按惯例入 git；本轮只记录不执行 add。

### B. 源码位于忽略目录、必须入恢复方案（记录，暂不移动）

1. **BiFoot 源码（最终采用）**：`results/diagnostics/20261111_variant_sweep/` 内 `bifoot_standalone.py`（`tests/test_bifoot_standalone.py` 直接 sys.path 导入）、`bifoot_model.py`、`bifoot_train.py`、`bifoot_eval*.py`、`bifoot_theta_s_b.py`、`bifoot_token_tta.py`、`bifoot_se2_*.py` 等约 15 个 py。建议恢复方案：提升为 git 跟踪包（如 `anysole/bifoot/`），并同步改 tests 导入路径。
2. **FT 训练源码**：`results/diagnostics/20261112_v8_gate_frozen/` 下 `finetune_router_enum.py`、`finetune_router_random.py`（FT 主训练脚本；同目录另有 `finetune_router_enum_std.py` 与多个门控实验 `train_*.py`），依赖 `train_router_gate.py`、`train_tgate_win_hb.py`。注意：`20261114_imu_ft_ablation{,_v2}/` 仅存消融权重与日志，无 Python 源码。
3. **E2/E5/E6/E9 最终表图生成与评测记录**：`results/diagnostics/20261111_final_protocol/`、`20261111_halfplane_occ/`、`20261114_imu_eval_v2_all/`（主模型评测输入记录，引用 FT ckpt 路径）、`20261119_q10_unified_eval/`、`20261120_q14_tex_*` 等（被 `results_display/script/` 消费）。
4. **BiFoot 航向变体源码**：`results/diagnostics/20261115_imu_yaw_branch/bifoot_yaw_train.py`——被已跟踪的 `results_display/script/bifoot_yaw_tb_metrics.py`（L20 `from bifoot_yaw_train import clip_arrays, infer_starts, _pad`）直接导入，必须入恢复方案。
5. **Q14 纹理 demo 预览生成脚本**：`.tmp_scripts/q14_tex_demo.py`——磁盘存在但未跟踪（目录被忽略），复用 `occlude_lower_strip` 同款 recipe，输出到 `results/diagnostics/20261120_q14_tex_occlusion/demo`。E5/E9 demo 主生成链为已跟踪的 `results_display/script/{smpl_9part_viz,e5_demo_teaser,e9_vision_touch_strip}.py` 与 `.tmp_scripts/e5_demo_*.py`，另有 `results/diagnostics/20261111_final_protocol/e9_tta_final.py`、`e9_teaser_scene_model.py`（E9 teaser 组合面板，见 §B.3 目录条目）。

### C. 需独立包（权重/结果/图表/环境/数据校验，不进 git，已由现有 ignore 覆盖）

1. **主模型权重**：
   - 基座预训练四折：`results/AnySole/V8_Cv7_PhysIMU_f*/…/checkpoints/ckpt_best.pt|ckpt_last.pt`（每折 ~500M）；
   - **最终 FT 四折**：`results/diagnostics/20261112_v8_gate_frozen/router_ft_only_f{S5S11S12,S5S6S11,S5S6S8,S6S7}_v2rerun.pt`（82M×4；`20261114_imu_eval_v2_all/V8_IMU_Residual/` 评测记录引用；同目录另有 `_last`、`v3`、`router_ft_enum_*` 变体）；
   - FT 消融变体四折：`20261114_imu_ft_ablation{,_v2}/V8_IMU_Residual_{CleanOnly,woEnum,DecoderOnly}_f*.pt`。
2. **BiFoot 权重**：`results/diagnostics/20261111_variant_sweep/ckpt/`（`bifoot_standalone_f*.pt` 四折 + `abl_*` 消融 + `theta_sb/theta_v`，合计约 200MB）。
3. **逐折/逐会话原始结果**：`results/diagnostics/`（28G）、`results/AnySole/`（116G）、`results/baselines/`（445G 正式预测）、`results/cells/`（49G）。
4. **最终采用的表图树**：`results_display/Etest_crossfit_physimu_vgate/`（60M：E2/E4/E5/E6/E9 主表图 + Supplement + demo）；`results_display/Etest_crossfit_v8/`（29M，E2/E5 生成器仍读取旧树基线表）与 `models_modes.yaml` 暂保留其被消费部分。
5. **基线独立源码**：`Baselines/` 独立仓库（HEAD d8225cf，工作区干净，远程 Anysole_Baselines）——保持独立 repo，不入主仓。
6. **基线环境版本**：`Vision_Baseline/visual_baselines/environments/`（GVHMR/WHAM/Tactile 的 conda.explicit + snapshot）——忽略目录内，必须入恢复方案。
7. **主环境版本（缺失）**：`touch_gait` 环境在仓库内无任何 spec 文件 → 本轮需生成 `pip freeze` / `conda env export` 并归入独立包。
8. **外部数据位置与校验值**：`AnysoleWorkspace/hf-raw/`（HF 私有库 THU-SIGS-EILAB/AnySole_raw，README 已记录推送协议与血缘）；本地全树 sha256 manifest **缺失**（现有 `protocol/visual_baselines_unit_manifest.sha256` 只覆盖视觉基线单元）→ 需生成 hf-raw 校验清单。
9. 视觉基线交付包：`baselines_deliverables/`（11G，MANIFEST.md/checkpoints/predictions/metrics）；`Vision_Baseline/visual_baselines/`（240G：history 122G + reference_results 118G）。与 `results/baselines/` 的重复关系留 3B/3C 核对。

### D. 保持忽略（机器本地，覆盖已完整，无需新条目）

results/ 整体、results_display 非 script 部分、AnysoleWorkspace/{work,shared,assets,model_inputs,hf-raw,reports}、wandb/、Wandb_Analyzer/、Vision_Baseline/、baselines_deliverables/、.tmp_scripts/（新文件）、_trash/、configs/{queue,running,done,failed,logs,generated,failed_trash}、z_note/exec、z_note/probes、anysole/ReferenceWorks/*/（含 PhysPT）、Download/、.hf_staging/、全局 *.csv/*.png/*.npz/*.pt/*.log 等——`git check-ignore` 抽查均命中。历史 splits（`z_note/data_spilts_fixpipe/splits_test_*.csv`）由全局 *.csv 覆盖，当前四折口径以 `protocol/splits/crossfit_woS13/` 为准。

**已执行的 ignore 操作**（工作区/索引层面，未提交）：`git rm --cached` 解跟踪 `_trash/20261115/E6_s10_old2/` 内 3 个工作区已删除的跟踪文件（目录本就在 `/_trash/` 忽略内）：
`detail/contribution/Main_Table1_E6_FT_Contribution_byPart.md`、`detail/router_k/Main_Table1_E6_RouterK_byPart.md`、`detail/teaser_selection/E6_TeaserSelection_S11051.json`。撤销方式：`git restore --staged` 该 3 个路径。

### 核查输出摘要（只读命令原始值）

```text
git status --short --untracked-files=all 行数 = 86（3 M + 3 D + 72 ?? tex conf + 8 ?? 归档 md）
git ls-files 按顶层目录：anysole 342 / AnysoleWorkspace 233 / results_display 174（script 173）/ z_note 163 / .tmp_scripts 79 / configs 43 / tests 26 / anysole_tta 11 / hmr2_tta 9 / _trash 3（解跟踪后 0）
anysole/configs：253 yaml = 181 跟踪 + 72 未跟踪；其中 fold_ 族 108 跟踪、tex 族 40 跟踪
AnysoleWorkspace/protocol：77 跟踪（splits/crossfit_woS13 四折 csv+manifest 在列）
results 容量：baselines 445G / AnySole 116G / cells 49G / diagnostics 28G / logs 340M / MotionPRO 2.4M
Baselines 独立库：HEAD d8225cf，工作区干净
主模型 FT 权重引用链：20261114_imu_eval_v2_all/V8_IMU_Residual/* → router_ft_only_f*_v2rerun.pt（20261112_v8_gate_frozen）
FT 训练源码：20261112_v8_gate_frozen/finetune_router_{enum,random}.py（依赖 train_router_gate.py、train_tgate_win_hb.py）；20261114_imu_ft_ablation{,_v2} 无 .py
BiFoot 航向源码：20261115_imu_yaw_branch/bifoot_yaw_train.py ← results_display/script/bifoot_yaw_tb_metrics.py 导入
```
