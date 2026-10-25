# checkpoint 保存、候选复评与正式选点规则

日期：2026-10-25（第二版：四阈值门机制——V/T/VT 三门 + ΔPA 税门，取代第一版的双条件防守方案）。适用任务：一个 U checkpoint 支持 V / T / VT；先检查 E2、E5，再进行 E1。本文件不修改实验计划的达标判据（E2/E5/E1 的判据在 `z_note/实验计划.md`，判定发生在 test 上）。

**注意：当前保存机制默认 val 性能是与 test 性能相关性很强（≈），但是实际上肯定是有偏差的，因此目前的方案注意以阈值门为主。**

## 1 机制（单一通道，2026-10-25 用户裁定）

**第一阶段（训练时，候选池）**：`--checkpoint-policy multi_metric` 下，每轮 **val 检查**凡**三门通过**的 epoch 权重全部保留
（val PA-MPJPE：V ≤ 40、T ≤ 90、VT ≤ 40），不设数量上限、不做采样；

三门不过的 epoch 仅当它是某模式 PA/GlobalMPJPE 全程最优（冠军）时保留一份（这一份仅仅是用于研究）。旧的双条件 survey 系列、Pareto cap 采样已从代码中删除。

**第二阶段（训练后，正式选点）**：候选池 + ckpt_last/ckpt_best 做原生 val 十指标复评后，`checkpoint_review` 按注册规则选点——**四个阈值全部通过**（V ≤ 40、T ≤ 90、VT ≤ 40、ΔPA = val PA(VT) − val PA(V) ≤ 0.5）的候选中，取 **VT PA 最小**者；无人通过 → **报告失败，不选**。选中的一份 U 固定权重，跑 test E2/E5 各一次。

| 产物 | 含义 | 状态 |
| --- | --- | --- |
| `ckpt_last.pt` | 最近完成 epoch 的训练状态 | 保留（评估锚、结论点）；文件存在不等于训练完成，必须核对 epoch 与 config.epochs |
| `ckpt_best.pt` | 按 val/tau0_mpjpe/VT 的快照 | 保留但**降级**：仅作防有限梯度爆炸的备份（F0b_warm ep370 / V3_2_part9 ep271 教训），不再承担任何选点语义 |
| `checkpoints/multi_metric/` | 候选池：三门通过者全留 + 门不过者的 PA/Global 冠军（manifest.json 记账，v2） | 保留，选点的唯一来源 |
| `checkpoints/survey/`（历史） | 双条件 survey 系列 | **已弃用**：保存代码已删除；历史 37 份权重只读保留，仅经 survey_eval 复评，不再产生新文件 |
| 原生 val 十指标明细及 receipt | 候选正式比较依据 | 需同会话、同帧轴、同窗口、同评估协议，并核对 checkpoint 哈希 |

训练监控的旧 `val/mpjpe` 是世界坐标误差，不能抄到正式骨盆对齐 MPJPE 一列。旧配置、权重、指标不回写。

## 2 选点规则与阈值（2026-10-25 用户裁定）

- 三门（val PA-MPJPE，健康门）：**V ≤ 40、T ≤ 90、VT ≤ 40** mm。只挡明显欠训，不与 E2 的 test 侧达标判据混用。
- 税门（无明显退化）：**ΔPA = val PA(VT) − val PA(V) ≤ 0.5** mm——与三门同属阈值指标（四个阈值）。四个阈值全部通过的候选中取 VT PA 最小；无人通过 → 报告失败，不从三个不同 checkpoint 拼接 V/T/VT 行。
- 选点只在 val 上做。**val≈test 为已登记假设**：历史量级 V/VT +4~6 mm、T +4.5~20 mm 漂移大，不做事先常数修正。选中后 test 只跑一次固定权重；test 未过即实验不达标，**不因 test 结果回头换 ckpt**（测试集不用于选择）。
- 专家（S_V/S_T/S_VT）只列十指标候选清单，不参与选点判定；E1 的专家对比用各自固定完整权重。

正式比较必须列出完整十指标：PA-MPJPE、MPJPE、GlobalMPJPE、W-MPJPE100、MPJRE、traj_ATE、RootOrientationDrift、FootSliding、GroundPen、Jitter_pred。局部、世界系、物理指标分别解释，不用加权总分抵消某一侧的明显退化。

候选优先检查：

- E2：同一 U 的 V/T/VT 三条件都可用；不同骨架、训练集或有效帧的外部基线不能直接作严格排名。
- E5：同一 U 在 cam3 与 occ010/020/030/050 下的 VT−V；重点看根轨迹、全局误差、足部位置、滑步和穿透，同时报告局部姿态代价及触觉干预。遮挡严重度不能成为换 checkpoint 的理由。
- E1：先固定 U，再与同数据、同架构、同监督、可比训练预算的三个专家比较。专家也固定完整权重，不能逐指标挑选专家。

## 3 新训练如何启用

在新运行的原训练命令后追加：

```bash
--checkpoint-policy multi_metric --checkpoint-role auto
```

三门按默认 40/90/40；需要覆盖时用 `--checkpoint-gate-v/--checkpoint-gate-t/--checkpoint-gate-vt`。使用新模型名或新输出目录，不把新规则混入已经完成的旧运行。

角色按 `config_probs=[VT,V,T]` 推断：混合输入为 U，纯 V / T / VT 分别为对应专家。新 U 建议使用明确的三条件采样接口，并让它统一管理缺失掩码；不要同时用多套 null 增强却仍将角色登记为“始终双模态专家”。现有角色推断不能识别所有额外增强，需人工核对 `occ_mix`、`v_cond_dropout`。

## 4 验证与测试的顺序

1. 核对源运行完整结束。候选可以来自早期 epoch，但其所属训练必须已完成；epoch=3/740 的 last 不合格。
2. 只在 val 上重评可用候选，保留逐会话、逐窗口明细以及权重、配置、划分与评估代码版本。
3. `checkpoint_review` 用注册规则（四个阈值：V/T/VT 三门 + ΔPA 税门，全过者取 VT 最优）出判定：选中的一份 U 与其哈希由用户固定；无人通过即报告失败。
4. 用固定权重执行 test E2 与 test E5（各一次），之后再执行 E1。历史 test 已被反复查看，旧数据上的再次评估属于历史重评，不能宣称为新的盲测。

候选池验证计划入口：

```bash
python -m anysole.checkpoint_candidates --archive /path/to/checkpoints/multi_metric/manifest.json --config /path/to/run.yaml --out /path/to/new_validation_directory --device cuda:0
```

默认只生成计划，`--execute` 才运行验证。执行前必须确认源运行完整结束；该独立工具不能代替 runner 的全部完成度检查。历史 best/last/survey 的重评使用显式 `--ckpt`，命令见 [重执行清单与候选](实验重执行清单与候选checkpoint.md)。不伪造 manifest 来绕过真实存档情况。

```bash
python -m anysole.checkpoint_review --manifest /path/to/review_manifest.json --out /path/to/new_comparison_directory
```

review manifest 的 `rules` 字段：`null` = 只出候选与十指标差值（E1 专家对比用）；给定 `{"thresholds": {"V": 40, "T": 90, "VT": 40, "dpa": 0.5}}` = 出选点判定（E2/E5 选点用，四个阈值全过者取 VT PA 最优）。一个 manifest 一份 U 的候选集（多 epoch），外部参考用固定权重文件。

## 5 历史结果与数据边界

保存规则只影响后续能够保留哪些权重，不能恢复已经删除或从未保存的 epoch（V7_Cv1 的 ep730 即历史丢失实例，队列已以 670/680 替代）。仅缺选点口径时，优先复用现存权重重评；仅当必须取得缺失的关键权重、修正训练监督或更换训练划分时，才重新训练。

同一 U 的正式表必须从同一个 checkpoint 的完整评估 JSON 取所有输入行。旧报告把 T-only 的 last 修正与 V/VT 的 best 并列，可作为诊断记录，不能原样当作新规则的统一模型主表。

GlobalMPJPE / ATE 仍受当前逐窗 GT `trans_anchor` 评估协议影响，不等于长序列自由积分。GT floor 只用于评分与用于后处理是两种不同的实验，后者不能算触觉贡献。S12102 无有效窗口、不同外部基线有效帧不一致等情况必须原样报告，不补齐、截断、填零或静默删除。

## 6 文件保留与清理

本轮不删除、不搬迁任何已有 checkpoint。候选池内的自动 unlink 现在只作用于「三门不过且不再是 PA/Global 冠军」的轮次——这是选点规则本身划定的域外，不是数量淘汰；旧机制中「cap 采样淘汰、survey 滚动 10 份」的自动删除路径已随代码删除。对现有历史权重仍按「先列清单、确认后移动到 /_trash/<日期>/」执行，不启动任何会覆盖历史文件的自动清理。

## 7 验收命令

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -B -m unittest discover -s tests -p 'test_checkpoint_*.py' -v
```

覆盖三门候选池保留（通过者全留、门不过冠军兜底、淘汰）、正式选点判定（四阈值门+VT 最优、无人通过报失败、专家不判定）、训练完成度、配置一致性、训练失败传播和 val 默认复评。
