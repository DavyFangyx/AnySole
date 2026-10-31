# configs/ 离线实验调度说明

定位：**训练层 + 测试层的 conf 队列执行手册**（三层解耦，2026-10-28 用户裁定）。
实验是模型通用的——模型通过 `--models` 参数传入，不是按模型编号建实验。

- **训练层**：模型训练（train 训练 + val 验证）→ ckpt —— `gen_full_package.py` / `gen_u_only.py`
- **测试层**：用训练层 ckpt 在 test 上验证评估 —— `gen_e1/e2/e4/e5.py`
- **display 层**：把 test 结果整理成表图 —— 脚本与 CLI 见 `results_display/README.md`

实验定义在 `z_note/实验计划.md`（E1–E8）；结果在 `results_display/Etest/`（按 E1–E8 分目录）。

一个 `.conf` 的粒度：一个 cell = 模型 × 条件（遮挡档 / ckpt 口径 / 训练配比）。
conf 命名 `NNNNN__<model_id>.conf`；结果目录 `results/cells/<model_id>/`。
实验 = cell 的组合（E2 的 cell 是 E1/E5 的子集，跨实验共享同一结果目录、先到先评）。

## 1. 队列机制（执行手册）

状态机：`queue/ → running/ → done/ | failed/`。scheduler 原子 `mv` 抢任务，
多 worker（每卡一个或多个，`bg.sh`）共享同一队列、互不重复。

skip 两级（2026-10-26 用户裁定「conf 与结果一一对应」）：

- 生成期：cell 的 conf 已在 queue/running/done/failed 任一状态、或结果目录
  已有 complete 产物 → 不生成——先做 E5 再做 E2，E2 的共用格自动跳过
- 执行期：runner 对每个 conf 复核——结果 complete 且 ckpt sha256 一致且各
  split metrics 齐备 → 整 conf 跳过（结果目录零写入）
- 两级 skip 都由 `--force` / `FORCE=true` 绕过（强制重跑，含重训）

| 动作 | 命令 |
|---|---|
| 生成任务 | `python configs/z_gen/gen_<入口>.py`（见 §3）；底层条目 `python configs/z_gen/gen.py <条目>` |
| 启动 worker | `CUDA_VISIBLE_DEVICES=N bash configs/bg.sh`（同卡重复执行 = 追加 worker） |
| 队列状态 | `python configs/tools/status.py` |
| 手动单跑 | 改 `configs/scratch.conf` 三行 → `bash configs/run.sh configs/scratch.conf`（`DRY_RUN=1` 只打印命令） |

公共参数（所有 gen 脚本可用；按层生效）：

- `--models a,b`：**模型是参数**（实验通用）；底层条目必须是其 `MODEL_IDS` 子集，
  否则报错——旧模型未接入实验管线时运行报错即止，不迁就
- `--concat-method f6_soft / f6_soft_time_accel_v1 / f6_soft_time_accel_state_v2`：
  接触标签版本（默认 `f6_soft`；后两者映射到 `*_tafix` / `*_statefix` 条目；
  id 已带版本后缀时不叠加）
- `--seed 1,2,3`：多种子（训练层；1=基名、≥2=追加 `_sN` 派生条目——已注册
  当前默认训练线的 `_s2/_s3`）
- `--ckpt last/best`：ckpt 口径（测试层；best 映射到 `*_evalbest*` 条目——
  已注册 Cv1–7 的 `_evalbest` / `_evalbest_occ{010-050}`）
- `--wandb` / `--checkpoint-all`：**只对训练层生效、默认 false**（训练期
  专用开关，不写即关；测试层 conf 全部 eval-only，写了也不生效）
- `--force`：绕过生成期 skip 并写 `FORCE=true`；`--dry-run`：只打印将写入的 conf

conf 通用键（生成器写入；实验级默认在 `common.py` 的 `EXPERIMENTS` 表）：

- `FORCE=false`：true 绕过两级 skip + ckpt 复用检查，强制重跑（含重训）
- `WANDB` / `CHECKPOINT_ALL`：训练期专用（同公共参数说明）

注册位置：模型条目 `configs/z_gen/model_registry.py`（`MODELS` 表）；
实验条目 `configs/z_gen/common.py`（`EXPERIMENTS` 表）；
公共默认 `configs/defaults.conf`。条目全量清单：`python configs/z_gen/gen.py`（无参数）。

## 2. 公共基础配置（默认）

| 配置项 | 实际值 | 说明 |
|---|---|---|
| 模型架构 | `anysolev2` | 注册表默认显式传入；覆盖 YAML 中的旧默认值 |
| YAML | `anysole/configs/v1.yaml`（当前默认线） | 数据、窗口、损失默认（历史 wo05 轴保持 `v1_wo05.yaml`） |
| Splits | `AnysoleWorkspace/protocol/splits/default/splits.csv`（2026-10-29 用户裁定：当前所有模型训练评估统一 canonical） | train 92 / val S12 / test S13+S14 = 36 会话 |
| 接触标签 | 默认 `f6_soft` | 训练评估统一；版本切换 = `--concat-method`（见 §4） |
| 相机 | `cam3` | E5 遮挡档 = cam3 + occ010/020/030/050（`v1_cam3_occ*.yaml`） |
| 窗口长度 / stride | `tw=20` / `20` | 每窗口 20 帧、不重叠 |
| batch size | `256` | 公共默认 |
| 学习率 | `1e-4` | constant schedule |
| 训练轮数 | `740` | 覆盖公共默认 `800` |
| train seed | `1` | 训练随机种子 |
| dropout | `0.1` | 网络 dropout |
| diffusion steps | `1000 / 50` | train / sample |
| pose 表示 | `6d` | SMPL pose 6D 表示 |

| 训练条件采样概率（模态 dropout） | `config_probs = [0.50, 0.25, 0.25]` | （VT/V-only/T-only）；
`lambda_pose 3.0 / lambda_traj 1.0 / lambda_contact 1.0 / lambda_air 2.0`（trec/vrec 0）。

选点：ckpt 四阈值门（V≤40 / T≤90 / VT≤40 / ΔPA≤0.5）+ 候选池全留
（`checkpoint_policy=multi_metric`，`CHECKPOINT_ALL` 开关）；正式口径 = ckpt_last。

## 3. 运行命令（训练层 + 测试层）

```bash
cd /data/fangyuxuan/projects/gait
# ---- 训练层：模型训练（train + val）→ ckpt ----

# 全量套餐：当前默认七个 U（--wandb/--checkpoint-all 默认 false，不写即关）
python configs/z_gen/gen_full_package.py --concat-method f6_soft --seed 1
# 仅 U
python configs/z_gen/gen_u_only.py --models V7_Cv5 --concat-method f6_soft --seed 1,2,3
# --wandb true --checkpoint-all true
# 多种子示例：--seed 1,2,3
# --tw N

# ---- 测试层：用 ckpt 在 test 上验证评估（eval-only）----

# E1 专家对比（默认 Cv5 last 的 V/T/VT 面板）
python configs/z_gen/gen_e1.py --concat-method f6_soft --seed 1
# E2 主性能评估（口径可切：--ckpt last/best）
python configs/z_gen/gen_e2.py --models V7_Cv5 --concat-method f6_soft --ckpt last --seed 1
# E4 消融组
python configs/z_gen/gen_e4.py --concat-method f6_soft --seed 1
# E5 遮挡交叉评估（默认 occ 四档；口径可切：--ckpt last/best）
python configs/z_gen/gen_e5.py --models V7_Cv5 --concat-method f6_soft --ckpt last --seed 1

# ---- 调参（需要时用）----

# lr × λ_contact × schedule 18 格
python configs/z_gen/gen.py v7c_tune18

# ---- 底层条目直发（新模型补评估等场景）----

# 列全部注册条目
python configs/z_gen/gen.py

# 只发指定 cell
python configs/z_gen/gen.py <条目> --models a,b --concat-method f6_soft

# ---- 队列运维 ----

# 启动 worker（每卡可多个，重复执行 = 追加）
CUDA_VISIBLE_DEVICES=6 bash configs/bg.sh

# 看队列状态
python configs/tools/status.py
```

## 4. 当前默认口径（2026-10-29）

当前新训练采用新 SMPL24 动作来源，沿用已有压力与帧对应规则。默认接触包
`f6_soft_time_accel_state_smpl24_v1`，条目带 ``；加速度修正版
参照 `f6_soft_time_accel_smpl24_v1`，条目带 `_tafix_smpl24`。两套包各 140 会话。

当前结构比较 = Cv1–7 七个统一模型，不默认训练 S_V/S_T：`gen_full_package.py`
默认七个 U、`gen_u_only.py` 默认 Cv5、`gen_e1.py` 默认同一 Cv5 last 的
V/T/VT 面板。历史专家只在显式指定历史 ID 时运行。正式比较用 last；
`CHECKPOINT_ALL` 是候选保存开关，不能单独决定评估读 best 还是 last。

接触标签版本一览（`--concat-method` 三取值 ↔ 条目后缀；smpl24 两包为当前
默认来源，经模型 id 直选 `*` / `*_tafix_smpl24`，不走
`--concat-method`）：

| 版本 | 条目后缀 | 说明 |
|---|---|---|
| `f6_soft`（默认） | 无 | 主管线默认，训练评估统一 |
| `f6_soft_time_accel_v1` | `_tafix` | 时间轴重建 + 加速度二阶差分修正（10-27） |
| `f6_soft_time_accel_state_v2` | `_statefix` | 上者 + 缓慢抬脚状态修复（10-27 晚间） |
| `f6_soft_time_accel_state_smpl24_v1` | `` | 当前默认（新 SMPL24 来源） |
| `f6_soft_time_accel_smpl24_v1` | `_tafix_smpl24` | 当前参照（新来源加速度修正） |

旧 `f6_soft`、`*_tafix`、`*_statefix` 条目保留原义；旧修正版包绑定旧 BVH，
不能配新动作来源使用，旧 ckpt 也不得当作新来源训练结果。新结果按接触版本
隔离目录。新包注册后的入口先只评验证集，测试集只做 Dataset 读取验收；
E2 清晰档与 E5 四遮挡档合起来覆盖五档条件。只预览不入队：各 gen 入口
`--dry-run`；去掉 `--dry-run` 才入队。

## 5. 注册约定

- 命名：conf 文件、模型本体文件、result 目录、wandb tag 全链同名（当前主线
  `V7_CvN` 家族；旧模型（V3_Bv1 等）仍在注册表，走 `gen.py` 直发）
- 选点：统一 ckpt_last（epoch 740，记 sha256）；候选池 + 四阈值门 = 训练健康度诊断
- 评估口径：`eval_splits=["val","test"]` 显式声明——「val 先评、test 后跑」只用于
  选点前筛查，正式统计两 split 齐出
- 专家：S_V/S_T 固定各自 ckpt_last，不参与 U 的选点
- conf 参数：`WANDB`、`CHECKPOINT_ALL` 默认 false（训练层需要时显式 true）

> 历史条目（wo05 / tw_scan 等波次实验、v7_first_round 候选线、第二轮评估线）
> 保留在注册表中供复现直发（`gen.py <条目>`），不入上文入口清单。
