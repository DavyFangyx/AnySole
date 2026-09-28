# 17-E3 公共 contact GT 口径裁定材料

生成日期：2026-09-28
定位：14 号方案 E3 的**裁定材料**（选项表 + 消费端影响 + 与已定稿项关系 + R35 子问题）。只读核对，未改任何代码/数据。

---

## 0. 一句话问题

「公共 contact GT」目前**只有 FPP-Net 一个消费者**（训练损失 + V2T 接触级指标 + PoseTransOpt 上游），
但它的语义在四处描述互相冲突、磁盘上的实际数据又是第三个版本——需要拍板：**用哪套规则生成接触 GT、是否排除 R35 故障格、以及接触层指标是公共指标还是诊断。**

## 1. 现状盘点

### 1.1 五模型的接触监督/输出口径（来自《模型输入与监督信号》§接触监督汇总 + 注册表）

| 模型 | 接触监督 | 是否对外输出 contact | 现在的定义 |
| --- | --- | --- | --- |
| FPP-Net | **有**：press2Cont 顶点级二值（主监督，生产链末端） | 是（接触头，压力输出 → 192 维 SMPL 足部顶点） | 训练 GT 口径 2026-09-27 被切到 f6_soft，用户已拍板**方案 A：回退 press2Cont**（`z_note/评估/FPP-Net接触GT语义错配_问题报告_20260927.md`） |
| PoseTransOpt | 消费 FPP 的 4-flag（toe/heel 四区域），自身不产 contact | 否（注册表 V2M 条目 contact=false） | `dataset_mmvp.py:46-95`：每顶点 >0.5 二值化 → 脚内 toe/heel 四区域 flag |
| MotionPRO | 有：contact (T,10) **仅作损失权重** | 否 | 不构成 prediction，生成清单未列 |
| Step2Motion | 无 | 否 | — |
| pressure_toolkit | 9-region 承重区域标签（拟合先验） | 否 | 只用二值接触；注册表 contact=false |
| AnySole（主线） | 有：`contact_method` ∈ {joint_and, motion_f6, f6_soft, pressure_f6} 生成鞋垫标签 | 否（无接触头） | 主线的 contact 指标（`contact_f1/acc/balanced_acc/recall`）是**诊断键**（`eval_protocol.py:DIAGNOSTIC_KEYS`） |

**结论：公共 contact GT 的真实消费面 = FPP-Net 自己**。其余五模型要么不评接触（Step2Motion/FPP 之外的运动模型），要么接触只进诊断（AnySole），要么接触是输入不是输出（PoseTransOpt/pressure_toolkit/MotionPRO）。

### 1.2 三个消费端（全部指向 FPP-Net 侧）

| # | 消费端 | 读取路径 | 当前 GT 实参 |
| --- | --- | --- | --- |
| 1 | FPP-Net 训练损失 | `lib/Dataset/PressDataset/PED_tempKPCont.py` → `trainer_tempkpSMPLCont.py:61-62`：`loss_cont = 0.8 × BCE(pred_cont, contact_smpl)` | 现磁盘为 `contact_f6_soft.npy` 的**二值 motion_f6**（09-27 16:13 被外部会话二值化，33/144 session 两脚全 1）；方案 A 拍板后应回 press2Cont |
| 2 | V2T 接触级指标 `contact_smpl_mse`(brief)/`contact_smpl_bce`(叶) | `solver.contact_smpl_metrics(pred=archive.contact_smpl_pred, target=archive.contact_smpl_gt)`；R_Test2 `evaluate_pressure_row` 与 R_Test4 都走这一支 | 随 archive 携带，即 sidecar 内容——**当前是 th=0.7 的 press2Cont**（R3 实测，非文档所称 th=0.5） |
| 3 | PoseTransOpt 上游契约 | `pred_contact_smpl/<frame>.npy`（推理输出，逐帧） | 与 GT 无关；但 GT 语义决定模型能否学到脚内结构（f6_soft 广播 → 四区域设计被架空） |

### 1.3 磁盘实测：四处口径互斥（同一件事的四种说法）

| 位置 | 表述 |
| --- | --- |
| `export_v2t.py:17-18/:133` | “The retired f6_soft broadcast GT is gone: `contact_gt_source` is now the **native press2Cont binary vertex contact**”；实测 archive 里 `contact_gt_source = press2Cont_binary_vertex` |
| `models_modes.yaml` V2T notes | 「接触级 pred 来自 FPP-Net 接触头（model_prediction），**gt 来自 f6_soft 软标签**」 |
| `Baselines/utils/solver.py:186`（`contact_smpl_metrics` docstring） | 「target from the **f6_soft** per-foot soft labels broadcast to vertices」 |
| `z_note/metrics_指标改动说明.md:134` / `:251` | :134「`contact_smpl_mse` … gt=**f6_soft 软标签广播**」；:251「现有 sidecar 的 contact GT 是**旧 `press2Cont th=0.5` 口径**，正式导出前必须重跑三 split 推理产出 f6_soft 口径 sidecar」 |
| **磁盘实测（R3，2026-09-28）** | 随附 sidecar 顶点 GT 实为**已退役 th=0.7**（GT=0 而 sig>0.7 = 0 帧；GT=0 而 sig>0.5 = 4785 帧）；且 38/140 session 用**旧权重**（S8×9/S12×11/S13×18；侧车写于 09-27 22:58–22:59，早于双足承重修复） |

→ 代码里没有任何一层能"知道"GT 的 th 或权重版本：`contact_smpl_metrics` 对目标**无条件接受**（对 {0,1} 二值或 [0,1] 连续都算 MSE/BCE，公式不变）。**这意味着 E3 的口径改动是纯数据侧改动（sidecar 重生成），零公式改动、零代码改动**——但也意味着口径漂移不会报错，只能靠元数据与人工核验。

### 1.4 一个已经落地的裁决（供 E3 引用）

`contact_f1` **不是**正式指标（`solver.py:61-65`）：需要二值接触 GT，而「f6_soft 是软标签体系、motion_f6 是运动学状态机不是模型输出」——挂在生成清单「待处理」。
注意：**若采用选项 A（press2Cont 顶点级二值 GT），`contact_f1` 的阻塞条件即消失**（二值 GT 存在了），此时是否把 `contact_f1` 恢复为正式键，是 E3 的连带决策。

## 2. 已定稿的相邻项（E3 不得推翻，只能衔接）

| # | 已定稿项 | 出处 | 与 E3 的关系 |
| --- | --- | --- | --- |
| S1 | **f6_soft → press2Cont（方案 A）**：FPP-Net 训练接触 GT 回退原生 press2Cont 顶点级 | FPP 报告 §0/§6（用户拍板） | 训练侧已定；**E3 只需裁定评估侧（§6 step 4 明确待再拍板）** |
| S2 | press2Cont **阈值 0.5 已定方向** | doc 14 §E3；指标文档 :251 记「旧 press2Cont th=0.5」 | 但磁盘 sidecar 实测是 th=0.7 → **"0.5 已定方向" 与现数据不一致，需在 E3 一并明确"重生成即用 0.5"** |
| S3 | **pixel_weight = 现场计算 + 双足承重站立帧修正**（上游静态已证伪） | T1 #2 已裁定（09-28），已执行（S8 59647→68681、S12 69615→81537、S13 25093→26935） | 权重直接决定 `mean_press` 与接触阈值 → 38 个旧权重 session 的 sidecar **必须重算**才与 S3 一致 |
| S4 | **D7 故障格结论**：右脚 cell 35 单通道间歇故障；跨 session 尺度对齐勿用 max/p99，用非零均值/p90；格级过滤判据 = ON 帧 CV<0.4 + 连续 ON>10 帧 + 该格 max>3×同脚其余格 p99 | doc 10 D7 子节 + R3 | 决定 Q3（是否排除）与其标定口径 |

## 3. 待裁定的三个问题（互相独立，可分别拍板）

- **Q1（评估侧 GT 源）**：V2T 接触级指标 `contact_smpl_mse/bce` 的 GT 取 **press2Cont 顶点级**（与训练损失 S1 恢复同式）还是保留 **f6_soft/脚级**？——FPP 报告 §6 step 4 原样待裁。
- **Q2（是否设"公共" contact GT）**：接触层是否作为**公共指标**（进 brief 第 7 键、R_Test2/R_Test4 表列）还是**降级为诊断**（只留明细）？
- **Q3（R35 故障格）**：接触判定（以及压力 GT 生成）是否**必须排除**右脚 cell 35 故障格？排除方式与范围？

## 4. 选项表

| 选项 | 内容 | 训练侧 | 评估侧 GT | 成本 | 风险 |
| --- | --- | --- | --- | --- | --- |
| **A** | 公共 contact GT = **FPP press2Cont 顶点级二值（th=0.5）**；训练与评估同源同式 | press2Cont（与 S1 一致） | press2Cont 顶点级 | 重生成 sidecar（140 session × 3 split 级联：重训 FPP → 三 split 重推理 → 重导出） | 与 S2 一致；`contact_f1` 阻塞解除；**但必须同时处理 Q3**（否则 1.48% 假阴性 + 0.313% 假接触直接进 GT） |
| **A′** | 同 A，但**沿用现侧车 th=0.7**，不重生成 | press2Cont th=0.7 | th=0.7 | 仅重推理（不重标） | 与 S2「0.5 已定方向」冲突；th=0.7 会让接触判定更保守（漏检轻载接触），且无法解释给读者 |
| **B** | **不设公共 contact GT**：接触层整体降级为诊断（`contact_smpl_mse/bce` 移出 brief，只留明细；R_Test2/R_Test4 表列去掉该键） | 训练仍用 press2Cont（S1 不变） | 仅诊断输出，不进正式表 | 最低（改 brief 白名单 1 处 + 文档） | 丢失唯一一个"触觉→运动"链路的接触可解释性；FPP-Net 论文侧少一个指标；PoseTransOpt 的脚内结构优势不再被量化 |
| **C1** | 双轨：**训练 press2Cont、评估 f6_soft 脚级**（即 09-27 之后的现状） | press2Cont | f6_soft 脚级广播 | 0 | 已被 FPP 报告证伪：评估口径与训练不同源、且脚级广播的指标"整脚同值"，BCE 可被常量预测刷分；**不推荐** |
| **C2** | 双轨：训练 press2Cont、评估 **motion_f6 脚级二值** | press2Cont | motion_f6 二值 | 低 | 同上但更硬（0/1 常量风险更大，33/144 session 全 1）；诊断价值仅"接触时段是否对上" |

> 备注：C1/C2 都是"训练/评估不同源"，与 S1 的**恢复顶点级语义**意图相反——S1 的初衷（FPP 报告 §4.3）正是"评估必须同源，才能保住 PoseTransOpt 四区域设计前提"。

## 5. 消费端影响矩阵（逐模型 × 逐消费端）

| 消费端 | 是否消费公共 contact GT | 选项 A/A′ 的影响 | 选项 B 的影响 | C1/C2 的影响 |
| --- | --- | --- | --- | --- |
| FPP-Net 训练 | 是（S1） | 回 press2Cont 顶点级，与 S1 一致 | 不变 | 不变 |
| FPP-Net V2T 接触级指标 | 是 | GT 与训练同源；数值语义 = 顶点级接触准确率 | 指标移出 brief（表列少 1 键 → 11 键）；明细仍可看 | 指标保留但语义降为脚级（BCE 可被常量刷分） |
| PoseTransOpt 区域 flag | **否**（消费 pred 不消费 GT） | 间接受益（模型能学到脚内差异） | 无影响 | 设计前提被架空（FPP 报告 §4.3） |
| R_Test4 展示 | 是（`contact_smpl_pred/gt` 连续图） | 图例/口径标注需更新为 press2Cont | 图仍在（诊断），表列去掉 | 脚级广播图（无脚内信息） |
| AnySole（V2T 行接触两键） | 否（`contact_prediction=false` → `—`） | 不变 | 不变 | 不变 |
| AnySole 自身 contact 诊断 | 否（用自有 `contact_<method>.npy` 标签） | 不变 | 不变 | 不变 |
| MotionPRO / Step2Motion / pressure_toolkit / VP-MoCap（V2M 行） | 否（四者都不评 contact） | 不变 | 不变 | 不变 |

**要点：无论选哪个选项，其余五模型的指标表都不会变化**——影响面被限制在 FPP-Net 行 + 展示层。这是本裁定"可以晚拍、但必须与 FPP 重导出一并执行"的原因。

## 6. 与已定稿项的衔接动作（拍板后一次做完）

1. **sidecar 重生成**（数据侧，零代码）：`PED_tempKPCont.py` 的 GT 路径回退 press2Cont(th=0.5) + `getVertsPress(soft=False)`；用**修正后的 pixel_weight（S3）**重算，覆盖 38 个旧权重 session；清理 VP-MoCap 树里为解燃眉建立的 280 个 `contact_f6_soft.*` 软链。
2. **FPP 重训 → 三 split 重推理 → 重导出**（收口顺序 step 1；依赖 FPP ckpt，见 E5 草稿）。
3. **元数据自证**：`export_v2t.py` 已写 `contact_gt_source=press2Cont_binary_vertex` 与 `formal_contact_metrics=contact_smpl_mse/contact_smpl_bce`；建议**把 th 与权重版本也写进元数据**（如 `contact_gt_threshold=0.5`、`pixel_weight_revision=double_support_20260928`），否则 1.3 节的"静默漂移"会再次发生。
4. **文档同步**：`solver.contact_smpl_metrics` docstring、`models_modes.yaml` V2T notes、指标文档 :134/:251 四处表述统一。

## 7. Q3 专章：R35 故障格是否必须从接触判定中排除

### 7.1 事实（R3 + D7，全部实测）

| 项 | 数字 |
| --- | --- |
| 故障位置 | 右脚源通道 34（1-based 列 35 = 4×12 脚内第 3 行第 11 列）→ 映射到 31×11 的 8 格 (4..7, 5..6)，逐值直通（4204 帧抽样 0 处不一致） |
| 顶点影响面 | 独占 **3/96** 右脚顶点（SMPL 6653/6661/6701）= 192 维接触向量的 **1.56%**；0 个部分污染 |
| 通道 ①（最重）：权重膨胀 | 判据命中 **53/139** session（31×11 判据 56/139）；S11 `mean_press` 91.4→209.0（**2.29×**）、S7 130.8→234.9（**1.80×**）；全库 **256,041 / 17,258,724 = 1.48%** 的（帧,格）接触判定被压成**假阴性**（S11 6.08%、S7 3.30%）；两个标定 session（S7011/S11011）均被命中 |
| 通道 ②：输入饱和 | 8/484 维在 10,390 帧（**20.05%**，91 session）sigmoid>0.99；尖峰/环邻 17.3×（最坏 61.5×） |
| 通道 ③：二值假接触 | th=0.5 轻载假接触 **160 帧（0.313%**，47 session）；完全腾空 0 帧；多为"抬升的地板"（中位 212，>1023 仅 3.4%） |
| 值域登记 | 31×11 无 clip 直通（max 10,392）；AnySole/MotionPRO 为 `/1023` 有界截断；FPP 无 1023 假设（`Σraw` 无界 + per-subject 自适应均值）→ **同一故障只把 FPP 打死** |
| D7 侧 | 89/140 session 该格 max>1023、77/140 为全场最大格、承载 >1023 质量 **76.5%**；clip 压平质量占比中位 9.7%/最大 42.9% |

### 7.2 关键判断：故障同时落在「输入」和「GT」两侧

- **输入侧**：FPP 的 `mean_press` / 接触阈值（通道 ①）、sigmoid 饱和（通道 ②）。
- **GT 侧**：接触 GT 与压力 GT 由**同一份 raw 触觉**生成 → 假阴性 1.48% 与假接触 0.313% **会进入 GT 本身**；V2T 的压力级指标（T_corr/T_rmse/force/CoP）同样对着被污染的 GT 比较（故障格 8/341 ≈ 2.3% 的右脚格）。

### 7.3 三个处理方案

| 方案 | 内容 | 影响 | 代价 |
| --- | --- | --- | --- |
| **R-a（推荐）源头置零 + 重算权重** | 在 raw→48 列入口把 cell 35 置零（或用 D7 判据逐 session 屏蔽该格），随后重算 pixel_weight（S3 已裁定流程）与全部 GT | R3 明言「源头置零后权重重算即自愈」：通道 ① 消失、GT 假阴性归零；通道 ②③ 随输入一起消失 | 需在**共享 facts 层**改一次（所有模型共同受益，但动的是数据协议 → 需用户确认范围）；被屏蔽格在指标里记 `null`（现有帧级 null-token mask 机制可复用） |
| **R-b 保留 + 登记为已知系统偏差** | 不改数据，在正式结果附「已知系统偏差清单」：右脚接触 GT 约 1.48% 假阴性 + 0.313% 假接触；右脚压力 GT 含 1 个故障格（2.3% 格位） | 零改动、零返工 | 右脚指标系统性偏保守；跨模型比较时 FPP 被特别放大（Σraw 无界） |
| **R-c 只修标定 session** | 仅对两个标定 session（S7011/S11011）+ 38 个旧权重 session 重算权重，其余保留 | 成本最低的"半修" | **不彻底**：通道 ① 的 1.48% 假阴性在其余 session 仍在；且 53/139 命中面意味着"其余"仍是绝大多数 |

### 7.4 对 Q3 的直接回答（供拍板）

- **要求 1（建议无条件执行）**：**权重标定链必须排除故障格**——两个标定 session 都被命中，若不排除，9/9 受试者的 `pixel_weight` 全部被抬高（这正是 38 个旧权重 session 的成因）。此项属 S3 的必然延伸，建议直接定。
- **要求 2（建议执行，但需用户确认范围）**：**接触判定（GT 生成）应排除故障格**。理由：接触 GT 是"顶点级二值"，故障格的 3/96 顶点在通道 ③ 下会**凭空产生接触**（160 帧假接触全封闭在这 3 个顶点里），这是**系统性错误标注**而不是噪声；且 Q3 与 Q1/A 强耦合——选 A 而不排除，等于把 1.48% 假阴性写进公共 GT。
- **要求 3（登记项）**：若用户选择保留（R-b），必须在 E5 的「已知系统偏差清单」里写清数字与方向（右脚、假阴性为主、3/96 顶点），且**跨模型对比结论不得基于右脚单格差异**。
- **判据基准待明确**：53/139（48-cell 基准）vs 56/139（31×11 基准）——两个基准的命中集合不同，需指定**以哪个基准判定"故障 session"**（建议 31×11，与消费端一致）。

## 8. 建议（供拍板的最小问题集）

| # | 问题 | 建议 |
| --- | --- | --- |
| Q1 | V2T 接触级指标 GT 源 | **取 press2Cont 顶点级（与 S1 同源）**——理由：S1 的初衷就是训练/评估同源；零公式改动；且能顺带解除 `contact_f1` 阻塞 |
| Q2 | 接触层是否公共指标 | **保留在 brief 第 7 键**（有唯一消费者的公共指标仍是公共指标），但 E5 交付时在 README 标注"该键只对 FPP-Net 行有值，其余行 `—`" |
| Q3-1 | 权重标定是否排除故障格 | **必须排除**（S3 的必然延伸；否则 9/9 受试者权重继续被抬高） |
| Q3-2 | 接触判定/GT 是否排除故障格 | **建议排除**（源头置零），范围需用户确认（共享 facts 层改动 vs 仅 FPP 侧）；若不排除则登记为已知偏差 |
| Q3-3 | 故障判定基准 | 指定 **31×11**（与消费端一致），并登记与 48-cell 基准的差异（53 vs 56 session） |
| Q4 | sidecar 阈值 | **重生成即用 th=0.5**（S2 已定方向，现磁盘 th=0.7 与文档不符），并把 th 与权重版本写进 archive 元数据 |
| Q5 | `contact_f1` 是否随 A 恢复为正式键 | 建议**先不恢复**：A 落地后二值 GT 存在，但 AnySole 无接触预测 → 该键永久只有 FPP 一行有值；先留在诊断，等 E5 再评估 |
