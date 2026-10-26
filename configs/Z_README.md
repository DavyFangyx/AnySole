# configs/ 离线实验调度说明

定位：**当前有效实验的清单 + conf 队列的执行手册**。

- 实验的定义（目的/设置/判据/产物）在 `z_note/实验计划.md`（E1–E5 五个实验）；
- 实验结果的数字与判读在 `z_note/E_Test/`（ABC对照 / C版本对照）；
- 这里只登记「有哪些实验条目、如何生成与运行」。

一个 `.conf` 的粒度：一个 cell = 模型 × 条件（遮挡档 / ckpt 口径 / 训练配比），
conf 命名 `NNNNN__<model_id>.conf`，结果目录 `results/cells/<model_id>/`——
实验 = 这些 cell 的组合（E2 的 cell 是 E1/E5 的子集，跨实验共享同一结果
目录，先到先评）。历史结果已全量迁至 `results/cells/`（2026-10-26，448 个
cell）；`results/experiments/` 连同归档与重复运行隔离区已整体删除（迁移记录
见 `results/.migration_20261026.log`）。

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
| 生成任务 | 实验级：`python configs/z_gen/gen_<实验>.py`（§3 清单）；底层条目：`python configs/z_gen/gen.py <条目>` |
| 启动 worker | `CUDA_VISIBLE_DEVICES=N bash configs/bg.sh`（同卡重复执行 = 追加 worker） |
| 队列状态 | `python configs/tools/status.py` |
| 手动单跑 | 改 `configs/scratch.conf` 三行 → `bash configs/run.sh configs/scratch.conf`（`DRY_RUN=1` 只打印命令） |

生成器两层（按实验分 gen，公共参数一致）：

- 实验级 `configs/z_gen/gen_full_package.py` / `gen_u_only.py` / `gen_e1.py` /
  `gen_e2.py` / `gen_e5.py`——一个实验一个脚本，模型列表在脚本内
  `DEFAULT_MODELS`（换家族时改脚本或用 `--models` 覆盖）
- 底层条目 `configs/z_gen/gen.py <条目>`——通用入口，无参数 = 列出全部条目
  （含模型数、wandb/checkpoint_all 默认、波次）；`--wave N` 发波次条目
  （wo05 / tw_scan / per_action_folds，闸门不满足即退出）

公共参数（两层通用）：

- `--models a,b`：只发指定 cell（底层条目必须是其 `MODEL_IDS` 子集，否则报错）
- `--wandb` / `--checkpoint-all` / `--force`：覆盖实验级默认，直写 conf
- `--force`：绕过生成期 skip 并写 `FORCE=true`
- `--dry-run`：只打印将写入的 conf，不落盘（生成前先看一遍）

conf 通用键（生成器写入；实验级默认在 `common.py` 的 `EXPERIMENTS` 表）：

- `FORCE=false`：true 绕过两级 skip + ckpt 复用检查，强制重跑（含重训；复用 done/ 历史 conf 前必查）
- `WANDB=true/false`：true 时 runner 注入 `wandb_mode=online`（实验级默认取 `EXPERIMENTS[].WANDB`）
- `CHECKPOINT_ALL=true/false`：true=候选全留协议；false=只滚存最近过线者

模型条目：`configs/z_gen/model_registry.py`（`MODELS` 表）；实验条目：
`configs/z_gen/common.py`（`EXPERIMENTS` 表）；公共默认：`configs/defaults.conf`。

## 2. 公共基础配置（V7 线）

| 配置项 | 实际值 | 说明 |
|---|---|---|
| 模型架构 | `anysolev2` | 注册表默认显式传入；覆盖 YAML 中的旧默认值 |
| YAML | `anysole/configs/v1_wo05.yaml` | 数据、窗口、损失默认（wo05 划分） |
| 划分 | `AnysoleWorkspace/protocol/splits/default/splits_wo05.csv`（yaml 内写 `protocol://splits/default/splits_wo05.csv`） | train 88 / val S12 / test S13+S14 = 36 会话 |
| 接触标签 | `f6_soft` | 训练和评估统一使用 |
| 相机 | `cam3` | E5 遮挡档 = cam3 + occ010/020/030/050（`v1_cam3_occ*_wo05.yaml`） |
| 窗口长度 | `tw=20` | 每窗口 20 帧 |
| 训练 stride | `20` | 不重叠窗口 |
| batch size | `256` | 公共默认值 |
| 学习率 | `1e-4` | constant schedule（tune18 扫描除外） |
| 训练轮数 | `740` | V7 线统一（覆盖公共默认 `800`） |
| train seed | `1` | 训练随机种子 |
| dropout | `0.1` | 网络 dropout |
| diffusion steps | `1000 / 50` | train / sample |
| pose 表示 | `6d` | SMPL pose 6D 表示 |

U 训练采样概率与损失（`v1_wo05.yaml` + V7 条目 `train_args`）：

```text
config_probs = [0.50, 0.25, 0.25]        # [VT, V-only, T-only]
lambda_pose = 3.0    lambda_traj = 1.0
lambda_contact = 1.0 lambda_air = 2.0    # trec/vrec = 0
```

选点：ckpt 四阈值门（V≤40 / T≤90 / VT≤40 / ΔPA≤0.5）+ 候选池全留
（`checkpoint_policy=multi_metric`，`CHECKPOINT_ALL` 开关）；正式口径 = ckpt_last
（E2/E5 对照表并列 best/last 双口径）。

## 3. 实验清单（2026-10-26 用户裁定）

训练套餐与实验解耦：训练按需选一档；实验（E1/E2/E5）单独选择、不捆绑。

### 训练套餐

| 套餐 | 生成脚本 | 训练内容 | 用途 |
|---|---|---|---|
| 全量 | `gen_full_package.py` | U + S_V + S_T（同架构、同数据、同预算 740 epoch 单 seed） | 一次训齐，E1 随时可补 |
| 仅 U | `gen_u_only.py` | U（统一模型，[0.5,0.25,0.25]） | 结构筛选阶段 |

### 实验（单独选择）

| 实验 | 生成脚本 | 前置 | 内容 |
|---|---|---|---|
| E1 | `gen_e1.py` | U + 专家 | U@T vs S_T@T、U@V vs S_V@V、U@VT（附录表 A 五行口径） |
| E2 | `gen_e2.py` | U | 三块十指标 vs 外部基线（表 2） |
| E5 | `gen_e5.py` | U | occ000/010/020/030/050 五档交叉评估，固定 ckpt_last（表 3） |

### 当前实验条目（底层清单）

实验条目 = `common.py` 里 `EXPERIMENTS` 表的键（`MODEL_IDS` 即生成范围），
由 `gen.py <条目>` 生成；上面的实验/套餐脚本是这些条目的组合入口
（模型、WANDB、checkpoint 保存方式都是参数）。

| 实验条目 | 内容 |
|---|---|
| `v7_first_round` | 候选 A/B/C ×（cam3 / cam3_occ010）训练 |
| `v7c_tune18` | V7_Cv1 调参 18 格：lr × λ_contact × schedule |
| `v7_variants_precheck` | V1–V3 四格 6-epoch 端到端预检 |
| `v7_variants_clean` | V1/V2/V3/V1_enum/Cv5 740-epoch 正式训练 |
| `v7_variants_stats` | V1–V3 统计：cam3 test + occ 四档（eval-only） |
| `v7_experts` | E1 专家 S_V/S_T（v7_a 骨架，与 C 非同架构，已被 Cv5 版取代） |
| `v7_cv5_experts` | E1 专家 S_V/S_T（v7_c2 双门骨架 = U 同架构） |
| `v7_r2_eval` | 第二轮 E2/E5：固定四门选点 + occ 四档 |
| `v7_ckpt_caliber` | E2/E5 best/last 双口径对照 27 格（eval-only） |

U = `V7_Cv5`（双门 + 枚举三条件）；S_V/S_T = `V7_Cv5_SV1` / `V7_Cv5_ST1`
（同骨架，`config_probs` 分别为 `[0,1,0]` / `[0,0,1]`）。

### 注册约定

- 模型名：统一编号格式（`V7_CvN` 家族）——conf 文件、模型本体文件、result 目录、wandb tag 全链同名
- 选点：统一 ckpt_last（epoch 740，记 sha256）；候选池 + 四阈值门 = 训练健康度诊断，不参与选点（ckpt 规则第三版）
- 评估口径：`eval_splits=["val","test"]` 显式声明——「val 先评、test 后跑」只用于选点前筛查，正式统计必须两 split 齐出
- conf 参数：`WANDB=true`、`CHECKPOINT_ALL=true`
- 专家：S_V/S_T 固定各自 ckpt_last，不参与 U 的选点

## 4. 运行命令

```bash
cd /data/fangyuxuan/projects/gait
# 生成（写入 configs/queue/；按需逐条执行，先 --dry-run 看一遍）
python configs/z_gen/gen_full_package.py   # 全量套餐：U + S_V + S_T
python configs/z_gen/gen_u_only.py         # 仅 U（结构筛选阶段）
python configs/z_gen/gen_e1.py             # E1 专家对比（前置 U 已训）
python configs/z_gen/gen_e2.py             # E2 主性能评估（三块十指标）
python configs/z_gen/gen_e5.py             # E5 遮挡交叉评估（occ 五档）
python configs/z_gen/gen.py                # 底层条目通用入口（无参数 = 列清单）
# 启动 worker（每卡一个或多个，同卡重复执行 = 追加）
CUDA_VISIBLE_DEVICES=4 bash configs/bg.sh
# 看队列状态
python configs/tools/status.py
```
