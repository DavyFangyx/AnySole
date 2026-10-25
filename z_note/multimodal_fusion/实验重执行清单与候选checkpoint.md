# 实验重执行清单与候选 checkpoint

日期：2026-10-25。核对注册实验、`results/experiments/*/metrics/test.json`、对应 val JSON，以及 checkpoint 内的 epoch、训练预算、输入配比和划分路径。本轮只修改执行入口、准备清单与候选，不启动训练或批量测试，不覆盖历史结果。

## 1 执行结论

当前仅继续现有调度中 tw_occ_scan 的原 8 个历史缺格收尾，不重复入队；其他历史训练和评估不整批重跑。先读取 C1 的 val E2/E5，按结果决定其余候选及遮挡验证；固定 U 后执行 5 次 test E2/E5，最后才条件性验证 E1。保存机制不能追溯恢复未留存 epoch，历史诊断不伪称按新机制重训。

| 实验 | 本轮实查 | 不需要重训的部分 | 需要补的工作 |
| --- | --- | --- | --- |
| `tw_scan` | 注册 39 格，39 格有 test；包括 32 评估格与 7 E 载体格 | 10-24 已完成 32 评估格统一协议重评，含 session_detail；不重训、不重复补评 | 仅当另做全 last 面板时，才考虑 B/Av4 的两个 cam3 tw20 last 补评；本轮不发 |
| `tw_occ_scan` | 原 8 个历史缺格正在现有串行调度收尾；本次复查暂有 7 份 test 文件 | 保留已经完成的格，不重新发射整组 | 仅核查现有调度与文件状态；不补零，不把运行中的临时文件变化当成新重训任务 |
| `per_action_folds` | 65 份原始 test 与 session_detail 全部 best 同源 | 原始 65 格不重训、不重评；已有 V4/V5 cam3 十个 fold 的 last 诊断 | 展示文档中 cam3 T2M 已有 last 修正，需单独明确其诊断口径；不能据此要求重跑原始 65 格。全族 50 格 last 扩展本轮不发 |
| `prior_lambda_scan` | 注册 24 格，18 格有 test | 18 格均已有 last 重评；不用重复训练、重复推理 | 直接汇总两口径。V5_Av4 的 lp2/3/4 × lt0/1 六格失败，先保留缺格，不为补齐表格立即重训 |
| `ckpt_last_rebaseline` | 54/54 格有结果 | 已完成，不重训、不重复全量复评 | 作为选点敏感性和整行取数的来源 |
| `ckpt_last_reread` | 15/15 格有结果 | 已完成，不重训、不重复全量复评 | 含 V4 wo05-drop 的 cam3+四遮挡档，以及 V4/V5 的 cam3 folds |
| `wo05` | 12/12 格有结果 | 可复用原权重 | 正式候选先补严格 val 明细；V4 drop/tonly 的 ep5 best 只能作欠训诊断，不能据此宣称坍塌 |
| `wo05_survey` | 4 个源运行完成，现存 37 份 survey 权重 | 不为候选搜索重新训练 | 下方先筛 4 份；旧 survey 测试只覆盖少量格，不能拿多候选 test 选赢家 |
| 旧 `survey_eval` test 波 | 178 个配置已搁置；done 共 11 个，包含 7 个旧 test 与后续 4 个候选 val | 旧 185 个 test 计划作废，已完成 test 只作诊断，不继续执行搁置项 | 候选只用 val 筛选；固定一份 U 哈希后才执行正式 test |
| `cond_coop` | 旧 test 与持续更新的 last 不能直接对应；本次两项 last 已到 740，另有一项仍为 3/740 | 保留现存权重与历史诊断，不自动重训 | 本轮仅核查状态，不另发批量补评；是否并入后续候选另行决定 |
| `v4b_carrier` / `v4b_carrier_last` | 各 2 格有结果 | best/last 两口径均有，不重复跑 | 属于折中结构诊断，不能当新设计已通过 E1 |
| `v4b_finetune` | 审查结束时 4/4 格有结果，源训练完成 | 保留权重，不重复训练 | canonical 与 wo05 不能混比；载体训练履历与公平对照未核实，暂不放入本轮 wo05 主候选 |
| `e1_av4` / `e1_experts` | 分别 4/4、15/15 格有结果 | 本轮不重训专家 | E2/E5 有兑现才用现存专家权重对照，最多 6 次候选验证；若权重或公平口径不足，报告限制，不自动发重训 |
| 外部基线 | 复用既有训练产物，含动作05的训练集差异已登记 | 不重训、不重跑 | 训练集、骨架与有效帧差异继续报告，不能借旧数字宣称严格同协议排名 |
| 旧 offset 前探针、残差/几何消融 | 历史诊断产物 | 保留历史解释，不整批重跑 | 不能把删除分支、旧数据或 GT floor 后处理的数字迁入新主表 |
| E3、E4 | 正式流程尚未完整启动 | 不存在应“重跑”的完整正式实验 | 继续按实验计划暂缓，不因本轮 checkpoint 规则扩展任务 |

这里“有 test”只表示结果文件存在，不能替代完成度和来源检查。不同实验的补评会引用同一源权重，表中格数不能相加解释成训练次数。

本次只读复查有两项文件变化：`V5_Av4_fS5S6S8` 的 `ckpt_last.pt` 当前缺失，64 份其他 fold last 可核验到完整预算；这不推翻原始 65 份 best 的同源性，也不自动要求重训。`tw_occ_scan/V5_Av4_cam3_occ030_v3off_tw10/metrics/test.json` 当前也缺失，相比原 8 缺格快照多一项；先核对现有调度和产物变化，不补值、不追加队列。

## 2 真正需要重训或补训练的条件

只有以下情况建议重新训练：源运行失败且无法继续；所需的关键 epoch 权重没有保存；要改变训练监督；要更换训练划分；或要取得按 multi_metric 机制保留的新完整学习轨迹。

时间实验有独立于 checkpoint 的划分问题：历史说明记录 cam3 tw20 四个评估锚点用过 train=104、val≡test 的旧划分，其余时间格主要是 train=92。重评能统一测试指标，不能消除训练集差异，也不能把旧 val 当干净的新选点集。若要严格受控的 canonical 时间曲线，应补训这四个 tw20 评估锚点；V4/V5 若仍沿用旧 E 载体，还需一个同划分 E 载体。不要直接与 wo05 的 train=88 主实验混合。这项补训排在主候选之后。

动作 fold 的训练集本来随 fold 变化，不能统一替换成 wo05。先保留原 fold 定义，不为了某个动作好看移除其失败会话。04/05 动作只有 k=4 的覆盖上限，不填成五折。

### 此前三项未完成运行的复查

| 模型 | 本轮读取的 last epoch / 预算 | 处理 |
| --- | --- | --- |
| `V5_Av5_cam3_occ010` | 740 / 740 | 本次已到完整预算，先核对评估来源，不自动重跑 |
| `V5_Av5_nocond` | 740 / 740 | 同上 |
| `V5_Av5_nocond_cam3_occ010` | 3 / 740 | 日志有 OOM，且未完成；不能用旧高误差判结构坍塌。若确认没有运行，再另开运行补训 |

epoch 是审查时点快照，执行前重新读取，不能按本表强行覆盖正在运行的训练。

### 暂缺 test 的 8 个时间×遮挡格

```text
V3_Bv1_cam3_occ030_v3off_tw15
V5_Av4_cam3_occ020_v3off
V5_Av4_cam3_occ020_v3off_tw5
V5_Av4_cam3_occ020_v3off_tw10
V5_Av4_cam3_occ020_v3off_tw15
V5_Av4_cam3_occ030_v3off
V5_Av4_cam3_occ030_v3off_tw5
V5_Av4_cam3_occ030_v3off_tw15
```

六个 λ 缺格为 `V5_Av4_wo05_lp{2,3,4}_lt{0,1}`。不立即发射重训队列；先区分仍在跑、完整权重待评、失败欠训三种情况。

## 3 四份候选：先供人工检查

均为现存 survey 快照，所属源训练已到 ep740。V5 三份来自同一个 wo05-drop 运行，V 流从零训练；V4 的 E 载体配置记录为 wo05。以下数值来自快照内的训练验证监控，尚不是新流程的完整十指标复评。

| ID | 运行 / epoch | val PA：V / T / VT，mm | val Global：V / T / VT，mm | 保留理由 |
| --- | --- | --- | --- | --- |
| C1 | `V5_Av4_wo05_survey_drop` / 735 | 32.769 / 65.273 / 32.658 | 65.175 / 166.425 / 64.049 | T Global 与 VT PA 较好的代表，先看这一份 |
| C2 | 同运行 / 690 | 33.173 / 64.546 / 33.244 | 63.932 / 172.385 / 64.014 | 现存该运行 survey 的 T PA 最低点，用于观察姿态与轨迹的取舍 |
| C3 | 同运行 / 720 | 32.625 / 68.490 / 32.729 | 64.098 / 180.130 / 63.533 | V PA、VT Global 较好的代表，不把单指标极值拼成一行 |
| C4 | `V4_Av1_wo05_survey_drop` / 105 | 29.828 / 76.057 / 30.026 | 56.418 / 161.650 / 56.674 | 视觉较强的旧结构参照；冻结载体使其不适合承载 E1 的 T→V 证明 |

这些候选不保证通过 E2 或 E5；也不代表所有十指标的 Pareto 最优。没有根据它们的 test 数字排序。V5 的 epoch715 是另一项 V Global 极值，但先不扩张候选数量，必要时再复核。

路径均保留历史双层 checkpoints 目录：

```text
results/AnySole/V5_Av4_wo05_survey_drop_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/checkpoints/survey/survey_ep735.pt
results/AnySole/V5_Av4_wo05_survey_drop_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/checkpoints/survey/survey_ep690.pt
results/AnySole/V5_Av4_wo05_survey_drop_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/checkpoints/survey/survey_ep720.pt
results/AnySole/V4_Av1_wo05_survey_drop_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_lpos3_wu0.05_gc5_ep740_bs256_sd1/checkpoints/checkpoints/survey/survey_ep105.pt
```

## 4 最小执行顺序与命令

第一阶段：先读 C1 的 cam3 val 三条件面板，再决定是否保留及是否需要 C2–C4。当前四份 cam3 val 文件都已存在，直接复用，不重复执行下方示例。随后仅对保留候选补 occ010/020/030/050 val。四份全部保留时连同 cam3 共最多 20 次评估，不需要训练。每次开启十指标及逐会话/逐窗明细，输出到新目录。

下面只执行 C1 的 cam3 验证；从仓库根目录运行。

```bash
GAIT_PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
GAIT_CKPT=results/AnySole/V5_Av4_wo05_survey_drop_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/checkpoints/survey/survey_ep735.pt
GAIT_OUT=results/experiments/checkpoint_review_20261025/C1/cam3
test ! -e "$GAIT_OUT" || exit 1
"$GAIT_PY" -B -m anysole.eval --config anysole/configs/v1_wo05.yaml --ckpt "$GAIT_CKPT" --split val --config-id V2M,T2M,VT2M --device cuda:0 --contact-method f6_soft --no-robustness --no-write-motion --dump-detail --protocol-out "$GAIT_OUT/val.json" --metrics-out "$GAIT_OUT/val_brief.json"
```

执行前按规则核对源训练完成度并记录实际 checkpoint 哈希；输出目录已存在时停止，不继续评估。其余候选替换路径和输出 ID。遮挡配置分别为 `anysole/configs/v1_cam3_occ010_wo05.yaml`、020、030、050，checkpoint 保持同一个，输出到对应条件子目录。

V5 可用现有 `--z-src V` 在相同权重和输入上关闭触觉专家，作为 E5 触觉贡献对照。时间错位、跨会话错配若没有现成正式接口，先登记为待实现，不能用文字宣称已经验收。

第二阶段：用户固定一份 U 与其哈希，使用这份权重跑 test cam3+四遮挡档，共 5 次评估，形成 E2/E5 的完整面板。外部基线只有在骨架、评分帧和监督边界明确后才能作严格比较；本轮未重新跑外部基线。

第三阶段：若 E2/E5 有兑现，再检查 E1。优先核对同架构 `V5_Av4_wo05_vonly`、`_tonly`、基名 VT 专家的现存 best/last，按各自输入做验证，最多 6 个候选评估。先固定三个完整专家权重，再与同一 U 比较；若关键权重已经丢失或输入监督不公平，先报告限制，本轮不启动专家重训。V4 冻结载体参照不承担正迁移承诺。

## 5 此轮不立即执行的内容

不重新发射全部时间窗、动作 fold 或 λ 网格；不另开 occ050 两格训练，兑现验证放在候选的遮挡 val 阶段；不发 λ 六缺格、fold 全族 50 格 last 扩展或 E1 专家重训；E3/E4 保持暂缓。四个 tw20 锚点补训只作为主候选之后、严格受控时间曲线所需的可选项。不把 canonical、wo05、旧 train104 的权重混成同一受控实验，不批量用 test 选赢家，不修改已有主表数字。新模型由用户按 [模型设计思路](模型设计思路.md) 设计，再进行接口与可验收性审查。
