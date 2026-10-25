# V7_Av1 方案（候选 A：独立证据 + 共享生成；2026-10-25 设计，已实现、未发射训练）

载体 = `z_note/multimodal_fusion/模型设计思路.md`（§2 共同骨架 + §3 候选 A）。三候选 = V7_Av1（A 本体）/ V7_Bv1（A + 接触参与根运动）/ V7_Cv1（A + 窗口级可靠性路由）——每版只新增一种机制。

## 1. 设计主张与机制规则

- **独立证据流**：V/T 各自做充分时序建模为 M_V / M_T，不先放进一个无隔离 Transformer；融合只发生在共享运动解码状态中。
- **共享生成**：一个运动 query 集合（根 + 9 部位）分别 cross-attend M_V 与 M_T；三条件（V / T / VT）= 同一前向的三种输入组合，同一解码器与输出头。
- **缺失模态硬关闭**：分支输出乘 per-sample 存在掩码 0（不是软掩码）；输入置零、注意力掩码、损失掩码由同一个决定（config_id）产生。
- **运动学一致**：输出 = 旋转 + 根速度；关节位置由 FK 统一计算（评估侧）。模型不独立修关节点。
- **T-only 上身 = 共享运动先验回退**：上身部位 query 在 T-only 时 attend 不到有效证据 → 学到的条件先验；不强制蒸馏视觉教师的上肢具体动作（观测歧义）。
- **无载体、无修正头、无推理侧约束**：从零训练，全部参数可训。

## 2. 模型结构（`anysole/models/common/model_v7.py`）

### 2.1 数据流（tensor 形状，d=256、tw=20 默认）

```
V_feat (B,tw,2051) ──v_enc──▶ v_tok (B,tw,256) ──v_temporal 4L──▶ M_V (B,tw,256)
T_raw (B,tw,96) + T_phys (B,tw,12) ──foot_conv──▶ t_tok (B,tw,256)
                                          │──▶ contact_logits (B,tw,2)
                                          └──t_temporal 4L──▶ M_T (B,tw,256)

运动 query：root_query (1,tw,1,256) + part_queries (1,tw,9,256)（learned + 时间 PE）
  ──▶ 共享解码器 6 层：每层 = query 自注意力 + xattn(M_V) + xattn(M_T) + FFN
       缺失模态 = 该分支输出 × v_present/t_present (B,1,1) ∈ {0,1}
  ──▶ z_root (B,tw,1,256) ──root_head──▶ v_hat (B,tw,3) 根速度 ──cumsum/FPS──▶ trans_hat (B,tw,3)
  ──▶ z_parts (B,tw,9,256) ──9 部位线性头──▶ pose (B,tw,144) 6D 旋转（hard PART_JOINTS 拼装，
       姿态统计归一化与既有头同款）

关节位置 = FK(pose, trans_hat + trans_anchor, offsets, parents)——评估协议侧统一计算。
```

### 2.2 组件

- **V/T 时序编码**：`v_temporal` / `t_temporal` = 4 层 TransformerEncoder（复用 model_v4_1 的 `_temporal_encoder`）。
- **共享解码器**（`V7MotionHead`，挂名 `pose_head`——train.py 按名拟合姿态统计）：`_V7DecoderLayer` 自实现（自注意力 + 双交叉注意力 + FFN，norm_first）；query = 根 1 + 部位 9（learned 参数 + `embeddings.time_pe`）。
- **读出**：9 部位线性头（6D，`part_place_idx` 拼装 POSE_DIM）；根头 = Linear→GELU→Linear(3)（根速度）。
- **接触头**：`V41ContactHead`（复用 model_v4_1），f6_soft 口径。
- **存在掩码**：`v_present = config_id != CONFIG_T`、`t_present = config_id != CONFIG_V`（前向可被 `v_present/t_present` 参数覆盖——诊断/干预接口）。

### 2.3 三条件开关（同一前向）

| 条件 | 输入 | 掩码 | 读出 |
|---|---|---|---|
| V2M | T 置零（condition_inputs） | t_present=0 | pose + 根速度（仅 V 证据） |
| VT2M | V+T | 1/1 | 同上（双证据） |
| T2M | V 置零 | v_present=0 | 同上（T 证据 + 共享先验） |

## 3. 训练

- **采样**：config_probs [0.5, 0.25, 0.25]（明确三条件配比，模型设计思路 §4；不叠加 occ_mix）。
- **损失**（`compute_losses_v7`，losses.py 内 v7 路由，train.py 零改动）：
  - L_pose 3.0：单一生成器输出的完整运动监督（当前条件全样本；T-only 也接受正式运动标签）。
  - L_traj 1.0：根速度 + 多尺度位移（`_trajectory_losses` 口径）。
  - L_contact 1.0：f6_soft pos-weighted BCE + air 加权 2.0——仅 T 在场窗口（缺失 T 不要求凭零输入重建）。
  - 掩码：全部由 config_id 同一决定产生（输入置零 / 注意力 / 损失）。
- **优化**：Adam + cosine LR、lr_warmup_frac 0.05、grad_clip 5.0；从零训练 740 epoch（从零口径，设计文档 §9 训练时长注记）。
- **checkpoint 规则**（ckpt 规则 §3）：`checkpoint_policy=multi_metric --checkpoint-role auto --checkpoint-pareto-keep 8`；registry which=last。

## 4. 评估（预注册口径，无结果数字——未发射训练）

- **验收顺序**（模型设计思路 §7）：三条件与缺失掩码核对 → 敏感性检查（同一样本换真实 T 的输出变化；敏感性只证明使用、不证明正确）→ val E2/E5 → 冻结 U 后 test → 有互补再 E1。
- **E2**：同一 U 的 V/T/VT 三条件十指标；外部基线同数据口径（跨骨架不严格排名）。
- **E5**：cam3 + occ010/020/030/050 交叉评估的 VT−V；重点根轨迹、全局误差、足部位置、滑步、穿透，并报局部姿态代价 + 触觉干预行。
- **判据**：第一轮不承诺十项全赢；零税 = 关键指标不明显退化的待验收目标（非逐位恒等——两种语义不得混用）。
- 干预：`--z-src` 为 V5 专用；V7 用输入置零 + 存在掩码覆盖接口做同口径干预（前向 `v_present/t_present` 参数）。

## 5. 落地文件与实现状态

- `anysole/models/common/model_v7.py`（骨架 + `V7MotionHead` + `_V7DecoderLayer`）→ `anysole/models/v7_a.py`（入口）→ `anysole/utils/losses.py`（`compute_losses_v7` + generic 入口 v7 路由）→ 双 registry（V7_Av1 × cam3/cam3_occ010）+ `configs/z_gen/common.py`（`v7_first_round` 实验，6 格，注册未发射）。
- CPU 冒烟过：三条件前向有限、硬关闭正确、损失有限、梯度贯通；dry-run 训练命令验证过。
- 未发射训练（用户裁定发射时机）。

## 6. 版本

- **V7_Av1**：候选 A——独立证据 + 共享生成（本方案本体）。
