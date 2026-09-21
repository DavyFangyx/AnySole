> **已归档（2026-09-20）**：细节留档，不再维护。当前状态与结论见
> ../model_fix_note.md，当前基座命令见 ../command_manual.md。

# AnySole V3 设计方案（触觉兑现 · 功能导向修订版）

> **本文档取代 fix_plan_v2.md 的 §F5**，主线目标从"部位解码器"重构为
> **"兑现 F4 触觉潜能"**（用户裁定 2026-09-18/19）。v2 其余章节继续有效：
> §F6a/F6b（接触标签与滑步损失）、§F7（σ）、§F8（AMASS 精修器）按本计划新排期
> 继承；v2 §F5（part9 解码器）**作废**，其实现留档（part_decoder.py 保留不删）。
>
> **V2 的教训**（本计划的设计约束）：
> 1. 基线锚定错位——v2 的 F5 假设 F1/F4b 已完成，实际基线是 F4a/F2p4；
> 2. 单步混多组件——F5 一步动了编码器+解码器+输出+数据增强+超参，失败无法归因；
> 3. F5 四 run 失败的已定位根因：`encode_stream()` 缺 LayerNorm（T token 量级
>    1400×V token → 交叉注意力 one-hot 退化 → LN 常数擦除），次要：aux 头删除、
>    稳定性处方未启用、V 流未部位化（因 F1 取消）。
>
> **本文档是给后续 AI 会话的实施手册**。每一步完成并验收后回填结果；未回填 =
> 未完成。所有路径在 `/data/fangyuxuan/projects/gait`，env = touch_gait。

---

## 0. 起点与基线（所有数字 = eval 协议实测，2026-09-18 回填）

主开发基座 = **F4a_footconv**（v1 表示 + foot_conv + V1 解码器）；f2 移植基座 =
**F2p4_combo**。参照 bar：F0b_seed2 / F2_f2rep（线性 T 编码器的 T2M 水平）。

| 口径 | F0b_seed2 | F4a（主基座） | F2 | F2p4（移植基座） |
|---|---|---|---|---|
| VT2M MPJPE / PA | 63.7 / 25.1 | **55.5 / 23.0** | 65.4 / 24.7 | 63.4 / 23.1 |
| VT2M 下肢 / 踝脚 | 69.4 / 80.1 | **60.3 / 70.2** | — | 66.5 / 74.2 |
| V2M MPJPE / 踝脚 | 66.3 / 83.8 | 63.5 / 80.0 | — | 68.7 / 80.5 |
| V2M contact_f1 | 0.661 | 0.681 | — | 0.660 |
| T2M MPJPE / PA | **116.2 / 37.4** | 123.8 / 40.3 | **89.7 / 35.8** | 99.7 / 40.7 |
| T2M 上半身 / 手臂 | 114.2 / 136-141 | 122.3 / 148-151 | — | 101.6 / 118-122 |
| T2M 踝脚 | 139.9 | 147.0 | — | 110.2 |

关键事实（F4a vs F0b 分部位）：foot_conv 在 **VT 配置兑现均匀**（下肢 −9.1、
双脚 −10.0、双腿 −8.3），但 **T-only 全身退化 +7.6，且退化最重在上半身
（手臂 +10~15、手 +14.6）**。T2M PA 40.3 也是四配置最差。

通路分解探针（`z_note/probes/probe_f4_tactile_path.py`，F4a ckpt，ridge 线性天花板，mm）：
F=58.2 / v_tok=74.3 / t_tok1(forward)=110.6 / t_tok3(stream)=114.4 / 融合前拼接=73.0。
结论：**融合没有毁掉触觉信息（F 比任何单流好 15mm），瓶颈不在融合容量**；触觉
单流全身姿态天花板 ~110-130mm（增强器定位确认）；t_tok3 比 t_tok1 无读出优势；
**触觉最擅长的接触/相位在训练中完全无监督**（λ_con=0 且标签 83.4% 假接触）——
这是"潜能未兑现"的实锤。

---

## 1. 目标与机制框架（功能导向）

用户裁定目标：**模态互补 + 模型全能**（功能，非纯指标）。

- **模态互补**：V 有空间全局信息、T 有局部精确信息；融合表征应是"完整运动状态"。
- **模型全能**：混合训练后，V-only 输出带"触觉风格"的精度（遮挡/精确场景），
  T-only 输出带"全局风格"的完整姿态（靠先验补全）。

结构语言翻译为三个机制，**"全能"是三者合起来的涌现**：

| 机制 | 结构改动 | 解决的问题 |
|---|---|---|
| A 流级胜任 | 双向流内深监督 + 接触监督 | 流级搭便车（T 流从不学全局、V 流从不学接触） |
| B 信任调度 | per-part σ 软门控（V/T/先验三路） | 缺失/不可信部位自动走先验路径 |
| C 先验存储 | 条件先验路径 g_∅；不足时接 AMASS 精修器 | T-only 补全局、V-only 补精度的知识源 |

性能（MPJPE）是副产品；每步验收以功能信号为主（§2.3 功能验收协议）。

---

## 2. 实验规范

### 2.1 单变量纪律（V2 教训 2 的直接修正）

- 每步只动一个组件；新开关默认关闭（= 上一步行为）；失败即弃、ckpt 留档、代码不回改。
- **流结构改动步骤**（V3-2/3）只允许"目标结构 + 保持其它一切与基线相同"；
  不得捆绑局部窗口/跳数偏置/部位自注意力等可选件（原 F5 的失败捆绑件，逐项
  通过后再单列对照）。

### 2.2 预算与稳定性处方

- 短预算 = **3700 步**（batch 256 ≈ 740 epoch；batch 64 ≈ 195 epoch，步数对齐）。
- 优化器：**只加头不动结构的步骤**（V3-1）保持 F4a 原配置（Adam lr 1e-3 +
  cosine、grad clip 5.0）——单变量；**动了结构的步骤**（V3-2/3）在同样 lr/clip
  外加 **前 5% 步数线性 warmup**（计划 v2 §F5 的稳定性处方，上一轮未启用）。
- **NaN-grad 守卫**（train.py 小改，全步骤生效）：clip 前检查任一参数梯度非有限
  → 跳步并降 lr。防 F5B_gated 式单 NaN 梯度经 clip_grad_norm_ 传染全模型。
- batch：结构不动步骤维持 256；结构步骤若 OOM 降 64（必要时梯度累积 4×64 等效）。
- 损失量级：新损失项首 epoch 加权后 ≈ 主损失 L_pose 的 10–30%（v2 §2 规则）。

### 2.3 功能验收协议（每步必跑，主判据）

- **F1 互补性**：VT 下肢误差 < min(V, T) 下肢误差 − 5mm（互补 = 优于任一单流；
  F4a 现状 VT 60.3 < V 68.9 已成立，增益大小看趋势）。
- **F2 V-only 触觉风格**：V-only 踝脚误差与 VT 差距 ≤ 12mm（现 80.0 vs 70.2 = 9.8，
  保持并随 VT 进步）；V-only contact_f1 ≥ 0.75（现 0.681）；**鲁棒性验证集**
  （V 连续丢 20–40% 帧，v2 §2 已定义未实现，V3-0 起实现）接触 F1 较无退化掉幅
  ≤ 0.10。
- **F3 T-only 全局风格**：V3-1 门槛 = T2M 追平线性 T 水平（v1 基座 ≤116.2、f2 基座
  ≤92）；上半身加速度分布误差 ≤ V2M 的 1.5×（F4a T2M 32.4 vs V2M 18.4 = 1.76×）；
  关节限位违规率 0（现 0，保持）。
- **F4 门控**（V3-3 起）：T-only 时 arm/spine 部位 g_∅ 激活率 > 0.1；VT 时
  foot g_T − arm g_T > 0.2（部位分化，非全局单值）。
- **ridge 通路探针每步必跑**：F / v_tok / t_tok1 三行天花板表（脚本已有，收敛进
  results_display/script/ 正式化）；T2M 改善必须伴随 t_tok1 行下肢天花板同步下降
  （否则是标签泄漏而非流级能力）。

### 2.4 已否决的设计（记录在案）

- **p_drop_both（双流全屏蔽）**：回归模型 MSE 下先验被逼逼近具体 GT → 平均
  姿态塌缩。否决；"先验路径"的训练信号来自 g_∅ 的条件性（§3.4）。
- **encode_stream 3-token 流进 V3 主线**：无读出优势（114 vs 110）+ 缺 LN bug。
  V3 全部用 foot_conv 的 `forward()` 单 token。encode_stream 的 LN 作为代码卫生
  修复照做（防他人误用），V3 不调用。
- **V3-3 门控接原始流（v_tok/t_tok）**：融合正是触觉兑现处（F 58.2 vs 拼接 73.0），
  门控必须作用在**融合后 F 的模态掩码视图**上，不可绕过融合。

---

## 3. 步骤

### V3-0 接触标签体系（零训练，先于一切；继承 v2 §F6a 全文）

- 实现 `results_display/script/f6_contact_labels.py`：motion_f6 状态机 + pressure_f6
  承重 + f6_soft 软标签，注册进 contact_methods.py；落盘 `contact_f6_soft.npy`。
- 验收门 = v2 §F6a 的标签验收门（4 反例翻转、接触率先验、压力互证、事件级时差），
  一字不改。
- 依赖：V3-1 起的接触监督全部用 f6_soft。

### V3-1 流级深监督（机制 A；主基座 F4a；tag `v31_ds`）

**唯一改动**：在 F4a 结构上加两个只读头 + 两个损失项。模型结构、主损失、优化器、
config 采样（25/25/50 维持现状）全不动。

1. T 深监督头（只读 `t_tok = t_enc.forward()` 输出，B,tw,d）：
   2 层 transformer encoder → 腿/脚 8 关节（15–18,19–22）6D + 接触 logit（2）+ 3 维
   轨迹。损失：标准化空间 MSE + BCE(f6_soft 软目标) + 轨迹 MSE。
2. V 深监督头（只读 v_tok）：接触 logit（2）。损失：BCE(f6_soft)。（下肢 6D 作为
   对照开关默认关，V 流主要缺的是接触语义。）
3. 深监督在流存在时计算（T 头：VT/T 行；V 头：VT/V 行）；λ_ds 按 §2.2 量级规则
   设，两项分开设权（T 项 ≈ L_pose 的 30%，V 接触项 ≈ 10%）。

- **验收**：F3 门槛（T2M ≤116.2 且踝脚 ≤140）；V2M/VT2M 不劣化（V ≤66.5、VT ≤58）；
  F2 门槛（V-only contact_f1 ≥0.75）；T 编码器梯度占比 vs V3-0 基线上升 ≥1.5×
  （测法同 v2 §F4b 梯度比探针）。
- **失败**：T2M 不达标 → 查 f6_soft 标签质量与 λ_ds 量级，不达标弃深监督方向
  （保留标签），直接上 V3-3 测门控本身。
- **通过后**：V3-1 的 T 头接触 logit 与 V3-3 的脚部 g_T 共享语义（接触 = 触觉的
  词汇表，也是门控可解释性的锚）。

### V3-2 9 部位最小移植（结构基座；tag `v32_part9min`；可与 V3-1 并行实现）

**唯一改动**：pose_head 的 3 组查询（body/left/right，现状）→ 9 部位查询，部位
划分用冻结的 `PART_JOINTS`（root(1)/躯干(4)/头颈(2)/双臂各 4/双腿各 2/双脚各 2）。
其余一字不动：交叉注意力读**同一个 fused F**（40 token）、traj head、aux head、
损失、优化器（+warmup，§2.2）。

- 实现落点：`pose_head.py` 加 `n_parts=9` 模式（query (1,tw,9,dim)、9 个部位
  unembed 头、拼回 138 维）；`model_v2.py` 加 `--pose-parts 9` 开关（默认 3 = 现状）。
- **验收**：与 F4a 打平——VT2M ≤58、V2M ≤66.5、T2M ≤126（F4a±2 内）；三配置任何
  一项显著劣化 → 弃 9 组粒度，V3-3 降级在 3 组上做。
- **目的**：为 V3-3 提供门控基座；本步**不实现任何门控**。

### V3-3 σ 软门控（机制 B；依赖 V3-1、V3-2 都过门；tag `v33_sigma_gate`）

**唯一改动**：V3-2 的交叉注意力改为**对 fused F 的双模态掩码视图**（F 的前 20
token = V 源、后 20 = T 源，`fusion.forward` 的拼接顺序即此）+ 每部位 σ 路由 +
条件先验路径。逐层：

```
z ← self-attn(9×tw 查询)            # 部位+时间上下文 = 条件先验（机制 C 的宿主）
e_V = cross-attn(z, F, mask=V 源 token)
e_T = cross-attn(z, F, mask=T 源 token)
σ_V, σ_T = MLP_σ(z)                  # per-part per-stream，2 标量
g_V, g_T, g_∅ = softmax([σ_V, σ_T, τ_p])     # τ_p 可学习 per-part 先验偏置
z += g_V·e_V + g_T·e_T              # g_∅ = 不注入模态信息，保留 z（条件先验！）
FFN
```

- **τ_p 初始化**：foot/leg/root 偏 T（τ 低）、torso/head/arm 偏 V——沿用原
  part_decoder 的 gate_bias 思路（v2 §F5 的少数可继承件）。
- **σ 监督（显式，不依赖掩码）**：β-NLL（β=0.5），r_p = 该部位标准化空间误差
  （stopgrad 主输出），log σ 截断 [−6,3]；前 10% 步 σ 固定=1。**不用"掩码自然
  产生 σ 梯度"的说法**——现硬掩码（内容置零+ℓ=−∞）下 σ 梯度恒零；软掩码
  （null 流照常通过）可作对照开关。
- **验收（先 F4 后 F3/F2/F1）**：T-only 时 arm/spine g_∅ > 0.1 且 foot g_T 明显
  高于 arm；VT 时 foot g_T − arm g_T > 0.2；VT−V 下肢增益 ≥ V3-1 同指标且
  T2M/V2M 不劣化于 V3-1；鲁棒性验证集（V 丢帧）优于无门控对照。
- **失败处置一**：g_∅ 不激活 → 先验路径容量不足 → 提前 V3-5 轻量 AMASS。
- **失败处置二**：σ 塌缩成常数 → 查 r_p 是否在变（段丢弃/遮挡强度），调 β。

### V3-4 f2 表示移植（V3-1~3 的机制在 F2p4 基座上重跑；tag `v34_f2`）

- 改动 = `--f2-repr` + warm-start 自 F2p4_combo；机制代码零改动（深监督头/门控
  天然支持 4 维轨迹与 tilt，实现时按 f2 口径适配输出维度）。
- 验收：T2M ≤92 且 PA ≤36（向 F2 的 89.7/35.8 看齐）；VT2M ≤65；F1–F4 全协议。
- 数值警戒：f2 的 f2_to_world + FK 归一化脆弱（F5B_gated NaN 实录），NaN-grad
  守卫必开，监控 train/nonfinite_skips。

### V3-5 先验存储（机制 C；触发式）

- V3-3 的 g_∅ 通路健康 → 接 v2 §F8a AMASS→BVH23 精修器预训练（可与 V3-1~3 并行
  启动，见 §4），F8b 接入时机照 v2。
- V3-3 的 g_∅ 不激活 → 提前做轻量版：AMASS 子集（走/跑/蹲）微调 9 部位解码器
  的 prior 路径（冻结流编码器，只训 τ_p + 无模态注入时的输出）。

### V3-6 接触/滑步损失（继承 v2 §F6b；V3-3 通过后）

- L_contact（f6_soft BCE，作用于脚部接触 logit）+ L_skate + L_height（v2 §F6b 全文
  不变，前 10% 步线性升权）。验收同 v2。

---

## 4. 并行与依赖

- **V3-0 标签**：一切接触监督的前提，最先做。
- **V3-1 与 V3-2 可并行**（一个动损失一个动结构，互不重叠；都过门才开 V3-3）。
- **V3-5 的 AMASS 预训练**（v2 §F8a）可与 V3-1~3 全程并行（只依赖数据转换脚本，
  不依赖主模型）。
- **严格串行**：V3-0 → {V3-1, V3-2} → V3-3 → V3-4/V3-6。
- 修复项（随时可做，不影响主线）：encode_stream 补 LN（代码卫生）、token 量级
  smoke 断言（|v_tok|/|t_tok| ∈ [1,100]，防"形状对量级错"再过检）、NaN-grad 守卫。

---

## 5. 命令手册（env = touch_gait）

通用模板（`--device` 按 nvidia-smi 空闲卡填）：

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --epochs <N> --out-dir results/AnySole/<STEP>/checkpoints \
  --wandb_mode online --wandb_experiment_tag <tag> --wandb_eval_interval 10 \
  --loss-cap 1.0 --device <gpu>
```

| 步 | 关键 flag（叠加到模板） | 预算 | tag |
|---|---|---|---|
| V3-0 | 无（脚本，非训练） | — | f6a_labels |
| V3-1 | `--t-encoder foot_conv --init-from F4a_footconv --lambda-ds-t 0.3 --lambda-ds-v 0.1`（待实现） | 3700 | v31_ds |
| V3-2 | `--t-encoder foot_conv --pose-parts 9 --init-from F4a_footconv`（待实现） | 3700 | v32_part9min |
| V3-3 | 叠加 V3-1+V3-2 开关 + `--gate sigma`（待实现） | 3700 | v33_sigma_gate |
| V3-4 | V3-3 全开关 + `--f2-repr --init-from F2p4_combo` | 3700 | v34_f2 |
| V3-6 | + `--lambda-contact 1 --lambda-skate <w> --lambda-height <w>` | 3700 | v36_contact |

**验收口径（每步同表）**：功能验收协议 F1–F4（§2.3）+ eval 全套 fseries 指标 +
ridge 通路探针表 + 训练面板（loss 分量、grad_norm、nonfinite_skips、T/V 编码器
梯度比）。

---

## 6. 实施顺序清单（AI 逐项打勾）

- [ ] 修复项：encode_stream 补 LN + token 量级 smoke 断言 + NaN-grad 守卫（train.py）
- [ ] V3-0：f6_contact_labels.py（motion_f6/pressure_f6/f6_soft + 验收门）→ 过门
- [ ] V3-1：流级深监督（T 头/V 头 + λ_ds 量级规则 + 梯度比探针）→ 验收 F3/F2
- [ ] V3-2：pose_head n_parts=9 最小移植 → 验收打平 F4a
- [ ] V3-3：双掩码交叉注意力 + σ 路由（β-NLL 监督 + τ_p 条件先验）→ 验收 F4
- [ ] V3-4：f2 移植（F2p4 基座）→ 验收
- [ ] V3-5：先验（按 V3-3 门控结果触发：正式 F8a 或轻量 AMASS finetune）
- [ ] V3-6：L_contact/L_skate/L_height（v2 §F6b）
- [ ] （并行）F8a AMASS→BVH23 转换 + refiner 预训练（v2 §F8a）
- [ ] 鲁棒性验证集实现（V 丢帧 20–40% + T 丢段，v2 §2 定义，V3-0 起可用）

## 7. 决策树（用户设计，已按 §2.4 修正）

```
V3-1（深监督 + f6_soft 接触标签）
    ↓
T2M 追平线性 T 水平（v1 ≤116.2 / f2 ≤92）？
  ├─ YES → V3-2 → V3-3
  └─ NO  → 查 f6_soft 标签质量 / λ_ds 量级 / T 头是否泄漏位置信息
           （深监督失败不影响 V3-2 结构步骤，但 V3-3 的门控价值需重新评估）

V3-3 上线后：T-only 时 arm/spine g_∅ > 0.1？
  ├─ YES → 先验路径通了 → V3-6 接触滑步损失 → 接 F8
  └─ NO  → prior 容量不足 → 提前 V3-5 轻量 AMASS finetune
```

---

## 附：与 fix_plan_v2.md 的关系（供后续会话）

- 本计划 §V3-0/V3-6 继承 v2 §F6a/F6b；§V3-3 的 σ 监督继承 v2 §F7 的 β-NLL；
  §V3-5 继承 v2 §F8；**v2 §F5 作废**。
- 已定位根因档案：`memory/f5-root-cause-tstream-scale-bug.md`；
  探针脚本：`z_note/probes/probe_f4_tactile_path.py`（通路分解，每步必跑）。
- F0b/F2 两个线性 T 编码器的 run 保留为 T2M 参照 bar，不再重训。
