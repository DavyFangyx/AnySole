# checkpoint 保存、候选复评与正式选点规则

日期：2026-10-25（第三版 2026-10-26 修订：**选点统一 ckpt_last（epoch 740）**——四阈值门机制保留为候选池防守与诊断，不再作选点依据）。适用任务：一个 U checkpoint 支持 V / T / VT；先检查 E2、E5，再进行 E1。本文件不修改实验计划的达标判据（E2/E5/E1 的判据在 `z_note/实验计划.md`，判定发生在 test 上）。

**注意：当前保存机制默认 val 性能是与 test 性能相关性很强（≈），但是实际上肯定是有偏差的，因此目前的方案注意以阈值门为主。**

## 1 机制（单一通道，2026-10-25 用户裁定）

**第一阶段（训练时，候选池）「存不存」**：`--checkpoint-policy multi_metric` 下，每轮 **val 检查**按 `checkpoint_all` 开关决定保留：

- `checkpoint_all=true`：凡**三门通过**（val PA-MPJPE：V ≤ 40、T ≤ 90、VT ≤ 40）的 epoch 权重全部保留，不设数量上限、不做采样；三门不过的 epoch 仅当它是某模式 PA/GlobalMPJPE 全程最优（冠军）时保留一份（这一份仅仅是用于研究）。
- `checkpoint_all=false`（**默认，2026-10-26 裁定**）：候选池只滚存**最近一个过线者**——本轮过线则覆盖存本轮，没过线则保留前一个最近的过线者（late 尖峰不会顶掉最近的健康点），不存冠军、不存全量。E2/E5 选点需要完整选域的实验在 conf 里设 `checkpoint_all=true`。

**第二阶段（训练后，正式选点）「选谁」**（2026-10-26 修订）：**统一选 ckpt_last（epoch 740）**——两个实测反例（C 的 ep665、V1 的 ep665 规则选点在 test 侧均劣于 last：T2M MPJPE 161.9→153.3、occ010 Δ踝足 −1.9→−3.4）证明 val VT PA 选点不优于 last；候选池 + 四阈值门保留为训练健康度防守与诊断，不再参与正式选点。选中的 ckpt_last 固定权重（记 sha256），跑 test E2/E5 各一次。

| 产物 | 含义 | 状态 |
| --- | --- | --- |
| `ckpt_last.pt` | 最近完成 epoch 的训练状态 | 保留（评估锚、结论点）；文件存在不等于训练完成，必须核对 epoch 与 config.epochs |
| `ckpt_best.pt` | 按 val/tau0_mpjpe/VT 的快照 | 保留但**降级**：仅作防有限梯度爆炸的备份（F0b_warm ep370 / V3_2_part9 ep271 教训），不再承担任何选点语义 |
| `checkpoints/multi_metric/` | 候选池：三门通过者全留 + 门不过者的 PA/Global 冠军（manifest.json 记账，v2） | 保留，选点的唯一来源 |
| 原生 val 十指标明细及 receipt | 候选正式比较依据 | 需同会话、同帧轴、同窗口、同评估协议，并核对 checkpoint 哈希 |

训练监控的旧 `val/mpjpe` 是世界坐标误差，不能抄到正式骨盆对齐 MPJPE 一列。旧配置、权重、指标不回写。

## 2 选点规则与阈值（2026-10-25 用户裁定）

- 三门（val PA-MPJPE，健康门）：**V ≤ 40、T ≤ 90、VT ≤ 40** mm。只挡明显欠训，不与 E2 的 test 侧达标判据混用。
- 税门（无明显退化）：**ΔPA = val PA(VT) − val PA(V) ≤ 0.5** mm——与三门同属阈值指标（四个阈值）。**四阈值门 = 候选池防守与训练健康度诊断，不参与正式选点**（2026-10-26 修订：选点统一 ckpt_last）。
- **正式选点 = ckpt_last（epoch 740）**：训练完整结束后固定该权重（记 sha256），test 只跑一次；test 未过即实验不达标，**不因 test 结果回头换 ckpt**（测试集不用于选择）。val≈test 为已登记假设（历史量级 V/VT +4~6 mm、T +4.5~20 mm 漂移大）——两例实测中 last 反超规则选点，故不做 val 侧微调选点。
- 专家（S_V/S_T/S_VT）只列十指标候选清单，不参与选点判定；E1 的专家对比用各自固定完整权重。

正式比较必须列出完整十指标：PA-MPJPE、MPJPE、GlobalMPJPE、W-MPJPE100、MPJRE、traj_ATE、RootOrientationDrift、FootSliding、GroundPen、Jitter_pred。局部、世界系、物理指标分别解释，不用加权总分抵消某一侧的明显退化。

候选优先检查：

- E2：同一 U 的 V/T/VT 三条件都可用；不同骨架、训练集或有效帧的外部基线不能直接作严格排名。
- E5：同一 U 在 cam3 与 occ010/020/030/050 下的 VT−V；重点看根轨迹、全局误差、足部位置、滑步和穿透，同时报告局部姿态代价及触觉干预。遮挡严重度不能成为换 checkpoint 的理由。
- E1：先固定 U，再与同数据、同架构、同监督、可比训练预算的三个专家比较。专家也固定完整权重，不能逐指标挑选专家。

## 这样组织 ckpt 规则的原因：
1. 震荡情况（val PA，每 5 轮一次检查）

模型: A
末段(560–740) V 振幅: 3.7 mm
T 振幅: 11.7
VT 振幅: 4.4
末值 dpa: +3.04（税）
────────────────────────────────────────
模型: B
末段(560–740) V 振幅: 23.3（ep600 单点尖峰 V
55.4）
T 振幅: 20.1
VT 振幅: 16.5
末值 dpa: +2.89（税）
────────────────────────────────────────
模型: C
末段(560–740) V 振幅: 4.4
T 振幅: 12.5
VT 振幅: 4.5
末值 dpa: −0.34（零税）

2. 机制对震荡的表现——可以，这正是它要防的

- 尖峰被门挡掉：B 的 ep600（V 55.4 > 40）门不过，已确认不在候选池里；早期欠训点（ep5–140）同理只按冠军留一份供研究。选点永远碰不到这些点。
- 震荡中的好点一个不丢：A 121 / B 109 / C 131 份全留（7.9–9.7G），末段最后 10 次检查全部过门。
- 选点不依赖 last 的运气：四阈值预演——C → ep665（VT 31.49、dpa −0.24，和上一轮一致），不是 last 的 33.33；B 的 last T 67.3 比末段最优 61.2 差 6mm，但机制从池里挑最好的。
- 一个后果需要你确认：A/B 的末段 dpa 恒为 +3 左右（税），零税带里只剩中段点，所以按四阈值规则 A 会选 ep210、B 选 ep205（V 35–37 的中段），而不是末段。如果 A/B 只当对照行、继续用 last 出表（E5 第一轮那样），就没影响；如果要按新机制正式选 A/B，就是这个结果。

**机制结论：可以。震荡期「门挡尖峰 + 全留好点 + 带内选 VT 最优」正好把「last 落在震荡哪一侧」的偶然性从选点里剔掉了。**


## 3 新训练如何启用

在新运行的原训练命令后追加：

```bash
--checkpoint-policy multi_metric --checkpoint-role auto
```

三门按默认 40/90/40；需要覆盖时用 `--checkpoint-gate-v/--checkpoint-gate-t/--checkpoint-gate-vt`。`checkpoint_all` 默认 false（只滚存最近过线者）；需要全留协议时加 `--checkpoint-all`。使用新模型名或新输出目录，不把新规则混入已经完成的旧运行。

角色按 `config_probs=[VT,V,T]` 推断：混合输入为 U，纯 V / T / VT 分别为对应专家。新 U 建议使用明确的三条件采样接口，并让它统一管理缺失掩码；不要同时用多套 null 增强却仍将角色登记为“始终双模态专家”。现有角色推断不能识别所有额外增强，需人工核对 `occ_mix`、`v_cond_dropout`。

## 4 验证与测试的顺序

1. 核对源运行完整结束。候选可以来自早期 epoch，但其所属训练必须已完成；epoch=3/740 的 last 不合格。
2. 只在 val 上重评可用候选，保留逐会话、逐窗口明细以及权重、配置、划分与评估代码版本。
3. `checkpoint_review` 对候选池做四阈值诊断（V/T/VT 三门 + ΔPA 税门），出具健康度清单；**正式选点 = ckpt_last**，其哈希由用户固定。
4. 用固定权重执行 test E2 与 test E5（各一次），之后再执行 E1。历史 test 已被反复查看，旧数据上的再次评估属于历史重评，不能宣称为新的盲测。

候选池验证计划入口：

```bash
python -m anysole.checkpoint_candidates --archive /path/to/checkpoints/multi_metric/manifest.json --config /path/to/run.yaml --out /path/to/new_validation_directory --device cuda:0
```

默认只生成计划，`--execute` 才运行验证。执行前必须确认源运行完整结束；该独立工具不能代替 runner 的全部完成度检查。历史 best/last 的重评使用显式 `--ckpt`，命令见 [重执行清单与候选](实验重执行清单与候选checkpoint.md)。不伪造 manifest 来绕过真实存档情况。

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
