# AnySole 离线实验配置说明

本文只说明三组进入离线队列的数据生产实验，以及 A0–B3 如何消费这些结果。

一个 `.conf` 的粒度是：一个模型 + 一份明确配置

## 1. 公共基础配置

| 配置项 | 实际值 | 说明 |
|---|---|---|
| 模型架构 | `anysolev2` | scheduler 显式传入；覆盖 YAML 中的旧默认值 |
| YAML | `anysole/configs/v1.yaml` | 数据、窗口、损失和采样默认值 |
| 接触标签 | `joint_and` | 训练和评估统一使用 |
| 相机 | `cam3` | AnySole cam3 数据 |
| 窗口长度 | `tw=20` | 每个输入窗口 20 帧 |
| 训练 stride | `20` | 不重叠窗口 |
| batch size | `256` | 公共默认值 |
| 学习率 | `1e-4` | constant schedule |
| 训练轮数 | `400` / `740` | F 系 400；V3 系 740（registry 逐模型覆盖公共默认 `800`） |
| train seed | `1` | 训练随机种子 |
| dropout | `0.1` | 网络 dropout |
| diffusion steps | `1000 / 50` | train / sample |
| pose 表示 | `6d` | SMPL pose 6D 表示 |

`v1.yaml` 中的输入和损失为：

```text
tactile_input = raw108       v_input = hrnet       no_imu = false
lambda_pose = 3.0            lambda_traj = 1.0
lambda_kp = 1.0              lambda_trec = 0.1
lambda_vrec = 0.1            lambda_con = 0.0
```

## 2. `singlemodal_eval`：主线和单模态专才

```bash
python configs/z_gen/singlemodal_eval.py --models V3_3B
```

### 2.1 主线 `V3_3B`

| 配置项 | 有效值 | 含义 |
|---|---|---|
| model name | `V3_3B` | checkpoint 和结果目录名称 |
| tactile encoder | `foot_conv` | 触觉分支编码器 |
| pose parts | `9` | 9 个身体分区 |
| soft parts | `true` | 使用软分区 |
| gate | `sigma` | 使用 sigma gate |
| lambda assign | `0.05` | 分区分配损失权重 |
| lr warmup | `0.05` | 前 5% 训练过程 warmup |
| grad clip | `5.0` | 梯度裁剪上限 |
| f2 repr | `false` | 不启用 F2 表征 |
| train | `true` | registry checkpoint 存在则跳过训练；缺失则从零训练 |
| checkpoint | `results/AnySole/V3_3B_joint_and/checkpoints/ckpt_last.pt` | registry 自动推导 |

主线训练采样概率的顺序固定为 `[VT, V-only, T-only]`：

```text
config_probs = [0.50, 0.25, 0.25]
```

这表示主线 checkpoint 同时见过完整输入和两种缺失模态条件；它不是只在
完整 VT 条件上训练的模型。

### 2.2 `V3_3B_vonly`

这不是删除 T encoder 的新网络，而是与 `V3_3B` 结构完全相同、只改变
训练条件采样概率的派生模型：

```text
config_probs = [0.00, 1.00, 0.00]
                 VT     V-only T-only
```

因此每个训练 batch 只采样 V-only 条件。它独立写入自己的 checkpoint 目录，
并由 scheduler 按 registry 血缘自动确定 warm-start checkpoint。

### 2.3 `V3_3B_tonly`

同样保持全部网络结构不变，只改训练条件采样：

```text
config_probs = [0.00, 0.00, 1.00]
                 VT     V-only T-only
```

每个 batch 只采样 T-only 条件，输出到独立的 `V3_3B_tonly` checkpoint 目录。

三份配置的本质区别是：

```text
V3_3B        : [VT, V-only, T-only] = [0.50, 0.25, 0.25]
V3_3B_vonly  : [VT, V-only, T-only] = [0.00, 1.00, 0.00]
V3_3B_tonly  : [VT, V-only, T-only] = [0.00, 0.00, 1.00]
```

每个模型 conf 会执行 checkpoint 准备、正式 val、正式 test，并将结果写入
`results/experiments/singlemodal_eval/<model>/`。

## 3. `rho_grid_eval`：帧级连续缺失配置

```bash
python configs/z_gen/rho_grid_eval.py --models V3_3B
```

`rhoV` 和 `rhoT` 是模态 token 的帧级保留率，不是训练 dropout 比例：

```text
保留真实 token 的概率 = rho / 100
替换为 null token 的概率 = 1 - rho / 100
```

四个角的含义为：

| rhoV | rhoT | 输入条件 |
|---:|---:|---|
| 100 | 100 | 完整 VT |
| 100 | 0 | V-only |
| 0 | 100 | T-only |
| 0 | 0 | 空输入 |

### 3.1 网格配置

| 项目 | 设置 | 含义 |
|---|---|---|
| 模型 | `V3_3B` | 使用 V3_3B checkpoint |
| rhoV | `0,20,40,60,80,100` | V 分支帧级保留率 |
| rhoT | `0,20,40,60,80,100` | T 分支帧级保留率 |
| train | rhoV/rhoT=`0,100`，seed=`0` | 生成训练表征/运动产物 |
| val | 全网格，seed=`0,1,2` | 估计验证均值和方差 |
| test | 全网格，seed=`0` | 最终报告口径 |
| reuse | `true` | 已有 cell 产物则复用 |

每个组合都生成独立 conf：

```text
TASK=rho_grid
MODEL_ID=V3_3B
GRID_SPLIT=val
GRID_SEED=1
RHO_V=20
RHO_T=40
```

它只执行这一格：

```bash
python -m anysole.rho_grid \
  --split val --seeds 1 --rho-v 20 --rho-t 40 --reuse
```

当前共生成：

```text
train: 2 × 2 × 1 =   4 个 conf
val:   6 × 6 × 3 = 108 个 conf
test:  6 × 6 × 1 =  36 个 conf
总计                 148 个 conf
```

结果合并写入：

```text
results/experiments/rho_grid_eval/V3_3B/grid_metrics_train.json
results/experiments/rho_grid_eval/V3_3B/grid_metrics_val.json
results/experiments/rho_grid_eval/V3_3B/grid_metrics_test.json
```

## 4. `all_models`：全部 12 基座训练 + 正式评估

```bash
# 全部 12 个注册模型
python configs/z_gen/all_models.py
python configs/z_gen/all_models.py --models F4a,V4B
```

每个模型一个 conf（`TASK=model_run`）：runner 先从头训练（独立性重构，
无 warm-start 血缘），训练完成后跑正式 val/test，结果写入：

```text
results/experiments/all_models/<model>/{metrics,predictions,run.log,...}
```

训练 checkpoint 仍落在 registry 地址 `results/AnySole/<model>_joint_and/<variant>/`。
同一个模型若已被其它实验训出 checkpoint，runner 检测到后自动跳过训练、只做
正式评估（这也是 `singlemodal_eval` 复用自己的 V3_3B 训练的机制）。

| 项目 | 设置 | 含义 |
|---|---|---|
| 模型集合 | `MODELS` 表全部 12 基座 | 生成器直接取 registry 表，新增模型自动纳入 |
| train | `true`（全部） | 每个模型从零独立训练 |
| epochs | F 系 `400` / V3 系 `740` | registry 逐模型覆盖公共默认 `800` |
| 重发防护 | queue/running/done 已有同名 conf 则跳过 | 防止重复排队 740 轮训练；`--force` 强制重发 |
| failed 恢复 | 不拦截 | 修复原因后 `--models <id>` 重发即可 |

### 4.1 V4B 的结构输入（数据版分区，无模型依赖）

V4B 的 `--part-json` 指向**原始训练数据运动学聚类**导出的分区文件（2026-09-24
裁定：模型无关——不依赖任何已训练模型/ckpt）：

```text
results/AnySole/partitions/partitions_kinematic_b_K9.json
```

生成器 `z_note/probes/probe_part_cluster_b_data.py` 对训练窗口逐关节运动学特征
（速度/加速度/相关性，Ward 聚类 + silhouette 扫描，同方案 B 方法）直接落盘
K9/K10/K11/K14 多个分区文件，供 V4B 及对照使用。若文件缺失，重跑该 probe 即可
（无需任何模型训练），随后：

```text
python configs/z_gen/all_models.py --models V4B
```

## 5. A0–B3 分析与数据对应关系

A0–B3 不生成 queue conf，只读取 `results/`：

| 编号 | 分析名称 | 对比设置 | 要回答的问题 | 输出 |
|---|---|---|---|---|
| A0 | 单模态能力分离 | V 专才 vs T 专才 | 两路是否各自擅长不同运动分量 | 分量差值森林图 + 数值表 |
| A1 | 融合完整性 | 主线 VT vs 最佳单模态专才 | 是否保留各自优势、产生协同或受到干扰 | 融合增益森林图 |
| A2 | 跨模态表征对齐 | 同时刻 F_V–F_T vs 错位时刻 | 两种单模态输入是否映射到相近运动状态 | 配对胜率点图 + 相似度分布 |
| B1 | V-only 全能性 | 主线 V vs V 专才 vs 空输入 | 主线模型只给 V 时是否仍是合格 V 模型 | 三列对比表 |
| B2 | T-only 全能性 | 主线 T vs T 专才 vs 空输入 | 主线模型只给 T 时是否仍是合格 T 模型 | 三列对比表 |
| B3 | 连续缺失鲁棒性 | rhoV × rhoT 全网格 | 从四个角到任意帧级缺失是否稳定 | rho 热力图 + 两条边界曲线 |

| 分析 | 数据来源 | 脚本 |
|---|---|---|
| A0 | `singlemodal_eval` | `results_display/script/singlemodal_analysis.py` |
| A1 | `singlemodal_eval` | `results_display/script/complement.py` |
| A2 | `singlemodal_eval` | `results_display/script/dropout_ablation.py` |
| B1 | `singlemodal_eval` + `rho_grid_eval` | `results_display/script/v2t_upper.py` |
| B2 | `singlemodal_eval` + `rho_grid_eval` | `results_display/script/trust.py` |
| B3 | `rho_grid_eval` | `results_display/script/rho_grid_analysis.py` |

## 6. 运行命令

```bash
python configs/z_gen/all_models.py
python configs/z_gen/singlemodal_eval.py --models V3_3B
python configs/z_gen/rho_grid_eval.py --models V3_3B
CUDA_VISIBLE_DEVICES=0 bash configs/bg.sh
python configs/tools/status.py
```

每个 worker 和该 worker 领取的任务共用一份独立日志，日志名按「卡号 + worker
编号」自动递增：

```text
configs/GPU4_worker1.log
configs/GPU4_worker2.log   # 同一张卡重复执行 bg.sh 追加的第二个 worker
configs/GPU5_worker1.log
```

成功任务进入 `done/`，失败任务进入 `failed/`。
