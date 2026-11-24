# V8_IMU_Residual：模型结构与训练配方（论文整理版）

日期：2026-11-15。用途：面向论文 Method / Implementation Details / Ablation 三节的整理稿；只保留可写入文章的结构、算法与训练配方，不写代码位置、权重路径、实验管线状态与结果数字。

---

## 1. 问题与符号

输入为一段长度为 \(T=20\)（40 Hz，0.5 s）的多模态窗口：

- 视觉特征 \(V\in\mathbb{R}^{T\times 2051}\)：HRNet 逐帧特征；视觉可能被遮挡或缺失。
2048+3 = 2051 维（V_FEAT_DIM，定义在 anysole/types.py）
  - 2048 维：CLIFF-HRNet backbone 对裁剪后人体图像的外观特征（extract_hrnet.py 里的 xf）
  - 3 维不是原始 bbox 坐标，而是 CLIFF 论文里那套经焦距归一化的 bbox 位置/尺度编码：
    a. (center_x − 图像宽/2) / 608 × 2.8
    b. (center_y − 图像高/2) / 608 × 2.8
    c. (bbox_size − 0.24×608) / (0.06×608)，其中 bbox_size = scale × 200

- 触觉压力 \(T_{\mathrm{raw}}\in\mathbb{R}^{T\times 96}\)：左右足各 48 个压力格。
96 = 48（左足）+ 48（右足），每足为 4×12 网格布局；原始为 16-bit 克值鞋垫读数，经 `clip(·, 0, PRESSURE_CLIP)/PRESSURE_CLIP` 归一化到 [0,1]（`anysole/data/pressure.py` `normalize_raw`）。

- 物理量 \(T_{\mathrm{phys}}\in\mathbb{R}^{T\times 12}\)：足底接触/力/包络等低维物理特征。
12 = 6（左足）+ 6（右足），每足 6 维顺序为 \([\mathrm{cop}_x,\mathrm{cop}_y,\mathrm{force},\mathrm{envelope},\mathrm{spatial},\mathrm{temporal}]\)（`anysole/data/pressure.py` `_foot_phys`）：
  a. \(\mathrm{cop}_x,\mathrm{cop}_y\)：由 48 格压力质心估计的足底压力中心（沿足宽/足长两方向）
  b. \(\mathrm{force}\)：该帧 48 格压力求和，即总载荷
  c. \(\mathrm{envelope}\)：力值在当前帧及前后各 1 帧内的滑动最大值，描述载荷包络
  d. \(\mathrm{spatial}\)：相邻格压力差（横向+纵向）的空间梯度均值
  e. \(\mathrm{temporal}\)：力值的帧间差分

- IMU \(T_{\mathrm{imu}}\in\mathbb{R}^{T\times 12}\)：双足 IMU 通道（传感器局部系含重力比力，g 单位 + 局部系角速度，rad/s）。
12 = 6（左足）+ 6（右足），每足为 acc3 + gyro3。

输出为 SMPL-24 运动：

- 姿态 \(\hat{x}\in\mathbb{R}^{T\times 144}\)（24 关节 × 6D 旋转）；
- 世界系根速度 \(\hat{v}\in\mathbb{R}^{T\times 3}\)，积分得到根轨迹 \(\hat{\tau}\)；
- 足接触 logits \(\hat{c}\in\mathbb{R}^{T\times 2}\)。

核心假设：**视觉在遮挡下会退化，而触觉/IMU 不受遮挡影响**；模型需要在视觉可靠时充分利用视觉，在视觉退化时由触觉/IMU 补位，同时不能让人为引入的触觉噪声在 clean 上伤害视觉。

---

## 2. 模型结构

### 2.1 总览

模型为**双分支编码 + 共享运动解码**：

```text
V  ── 视觉编码器 ── M_V ─┐
                          ├─ 共享运动 query（root + 9 parts）
T_raw/T_phys/T_imu ── PhysIMU 编码器 ── M_T ─┘
                          │
                     6 层运动解码器
                     （V/T 交叉注意力 + 模态在场掩码）
                          │
        root head / pose head / contact head / anchor head（旁路）
```

模型不使用显式的 V/T 可靠性门或竞争路由；V/T 交叉注意力输出按标准残差方式加回 query，且不做门控缩放（\(g_V=g_T=1\)），模态缺失由在场掩码处理。这是 `V8_IMU_Residual` 名称中“Residual”的含义。

### 2.2 视觉分支

- HRNet 2051 维/帧 → 线性投影到 \(d=256\)；
- 4 层时序 Transformer（8 头，dropout 0.1）→ 视觉记忆 \(M_V\in\mathbb{R}^{T\times d}\)；
- 视觉在场变量 \(m_V\in\{0,1\}^{T}\)：缺失帧的 V 交叉注意力输出置零（硬关闭）。

### 2.3 触觉/IMU 分支（PhysIMU 两阶段融合）

压力、物理量与 IMU 的物理语义和尺度不同，因此不直接拼接，而是两阶段融合：

1. **物理状态**：\(h_p = \mathrm{LN}\left(W_p [\,h_{\mathrm{phys}}; h_{\mathrm{imu}}\,]\right)\)，其中 \(h_{\mathrm{phys}}\) 为 \(T_{\mathrm{phys}}\) 的 MLP 编码，\(h_{\mathrm{imu}}\) 为 IMU 的 MLP 编码；
2. **触觉 token**：\(h_t = \mathrm{LN}\left(W_o [\,h_{\mathrm{raw}}; h_p\,]\right)\)，其中 \(h_{\mathrm{raw}}\) 为 \(T_{\mathrm{raw}}\) 经 FootConv 的逐帧 token。

融合后的 \(h_t\) 保持 \(T\times d\)，再经 4 层时序 Transformer 得到触觉记忆 \(M_T\)。两阶段融合的动机是：先在同一物理状态空间内对齐压力与 IMU，再让原始压力修正接触细节；LayerNorm 保证融合 token 与视觉 token 同尺度，避免某一模态在交叉注意力中主导。

### 2.4 共享运动解码器

- 运动 query：1 个 root query + 9 个部位 query，共 10 个 slot；
- 6 层解码器，每层：
  - query 自注意力（窗口内非因果）；
  - 对 \(M_V\) 的交叉注意力，乘视觉在场掩码 \(m_V\)；
  - 对 \(M_T\) 的交叉注意力，乘触觉在场掩码 \(m_T\)；
  - FFN。
- 同一个解码器处理 V-only、T-only、VT 三种配置；枚举预训练（§3.1）让共享 query 学会从任一模态组合读取运动证据。

### 2.5 输出头

- **根头（SD2 四通道）**：输出 \([s, d_x, d_z, v_y]\)（水平速率、水平方向两通道、垂直分量），组装为世界系根速度 \(\hat{v}\)；\(\hat{\tau}=\mathrm{cumsum}(\hat{v})/\mathrm{fps}\)。四通道读出把“走多快”“往哪走”“上下起伏”解耦，避免三通道回归中方向与速率相互干扰。
- **姿态头**：9 个部位 token → 6D 旋转，拼成 144 维 SMPL-24 姿态。
- **接触头**：由触觉 token 预测双足接触 logits。
- **平移锚头（旁路）**：从视觉记忆的窗口均值回归一个绝对平移锚 \(\hat{a}\)；锚头梯度不回传主干（stop-gradient），输出不进入 \(\hat{\tau}\)。它的作用是提供绝对尺度/平移线索，同时不改变主轨迹路径。

---

## 3. 训练配方

### 3.1 基座预训练：V/T/VT 枚举

**目标**：让共享解码器在三种模态配置下都能工作，为后续遮挡微调提供初始化。

- 数据：SMPL-24、40 Hz、20 帧窗口、步长 20；F6 soft 接触标签。
- 训练：740 epoch，batch 256，Adam，lr \(10^{-4}\)，warmup 5%，grad clip 5。
- 模态枚举：每个窗口枚举 V-only、T-only、VT 三种配置；损失按
  \[
  p=(p_{VT},p_V,p_T)=(0.5,0.25,0.25)
  \]
  加权。VT 占一半，保证主配置的拟合；V/T 各四分之一，保证单模态可用。
- 损失：
  \[
  \mathcal{L}_{\mathrm{base}}
  = 3\mathcal{L}_{\mathrm{pose}}
  + \mathcal{L}_{\mathrm{traj}}
  + \mathcal{L}_{\mathrm{kp}}
  + \mathcal{L}_{\mathrm{contact}}
  + 2\mathcal{L}_{\mathrm{air}}
  + \mathcal{L}_{\mathrm{SD2}}
  + \mathcal{L}_{\mathrm{anchor}},
  \]
  其中 \(\mathcal{L}_{\mathrm{pose}}\) 为姿态参数损失，\(\mathcal{L}_{\mathrm{traj}}\) 为根速度/轨迹损失，\(\mathcal{L}_{\mathrm{kp}}\) 为世界系关键点损失，\(\mathcal{L}_{\mathrm{contact}}\) 为足接触 BCE，\(\mathcal{L}_{\mathrm{air}}\) 为腾空相损失，\(\mathcal{L}_{\mathrm{SD2}}\) 为根头四通道的恒等式配平项，\(\mathcal{L}_{\mathrm{anchor}}\) 为平移锚头损失（权重 1.0，stop-gradient）。

**设计动机**：枚举训练等价于系统性的模态 dropout；它让模型见过“只有 V”“只有 T”“V+T”三种输入，而不是只在 VT 上训练后再期望它在遮挡下自行退化。VT 权重 0.5 是主配置与单模态可用性之间的折中。

### 3.2 遮挡感知微调

**目标**：在保持视觉编码器的前提下，让触觉编码器与解码器适配“视觉被遮挡”的分布，并显式约束“VT 不应在 clean 上差于 V”。

- 初始化：§3.1 的基座；结构不变。
- 数据：同一批窗口的四档视觉条件——clean、20%、50%、100% 视觉遮挡；触觉/IMU 不变。遮挡只作用于视觉侧，因此四档的差异纯粹来自视觉可靠性。
- 可训练参数：触觉原始压力编码器（含其内部时序层）、物理量 MLP、触觉时序编码器、姿态头（含根头）；视觉编码器/时序、IMU 编码器/融合、平移锚头冻结。这样做的动机是：视觉表征已在基座中学好，微调只需让触觉侧与解码器适配遮挡分布，避免小数据微调破坏视觉特征。
- 训练：20 epoch，batch 64，Adam，lr \(10^{-4}\)；四档条件按顺序优化（每个 mini-batch 一次更新）。
- 条件权重：clean \(0.5\)，20%/50%/100% 各 \(1/6\)。clean 占一半，保证不因遮挡训练而牺牲 clean；三档遮挡合计一半，覆盖退化谱。
- 每档损失：同一 batch 内枚举 V/T/VT 三路，**等权 \(1/3\)**：
  \[
  \mathcal{L}_c=\frac{1}{3}\left(\mathcal{L}_V^{(c)}+\mathcal{L}_T^{(c)}+\mathcal{L}_{VT}^{(c)}\right),
  \qquad
  \mathcal{L}_m^{(c)}=\mathcal{L}_{\mathrm{PA}}(m,c)+\mathcal{L}_{\mathrm{MPJPE}}(m,c).
  \]
- clean 相对约束：
  \[
  \mathcal{L}_{\mathrm{rel}}
  =\max\left(0,\ \mathcal{L}_{VT}^{(\mathrm{clean})}
  -\mathrm{sg}\left[\mathcal{L}_V^{(\mathrm{clean})}\right]\right),
  \qquad \lambda_{\mathrm{rel}}=1.0,
  \]
  其中 \(\mathrm{sg}[\cdot]\) 为 stop-gradient。该项只惩罚“VT 比 V 差”的窗口，不奖励“VT 比 V 好”，避免把融合推向过度依赖触觉。
- 总损失：
  \[
  \mathcal{L}_{\mathrm{FT}}
  =\sum_{c\in\{\mathrm{clean},20,50,100\}} w_c\,\mathcal{L}_c
  +\lambda_{\mathrm{rel}}\,\mathcal{L}_{\mathrm{rel}}.
  \]
- 模型选择：在验证集上按验证损失选 best checkpoint；测试集不参与选点。

**设计动机**：遮挡微调解决的是“视觉退化时触觉如何补位”，clean 相对约束解决的是“触觉不能在视觉可靠时添乱”。两者分开：前者作用于三档遮挡，后者只作用于 clean。

### 3.3 推理配置

- 主模型推理使用 **VT**（视觉 + 触觉/IMU）；V-only 作为对照，T-only 也可用。
- 模态缺失由在场掩码处理；推理不需要显式的可靠性门或路由。
- 根轨迹由根速度积分得到；锚头输出不进入主轨迹。

### 3.4 四折协议与模型选择

采用**受试者不相交**的四折：

| 折 | 验证受试者 | 测试受试者 |
|---|---|---|
| F1 | S12 | S5, S6, S11 |
| F2 | S13 | S5, S11, S12 |
| F3 | S14 | S6, S7 |
| F4 | S7 | S5, S6, S8 |

每折的训练集为其余受试者；验证集只用于 best-checkpoint 选择；测试集只在最终评估时使用。四折报告 mean±std。

---

## 4. 设计选择与消融

主模型 `V8_IMU_Residual` 的消融围绕三条设计轴：**预训练**、**微调范围**、**融合/门控**。训练策略消融（§4.1）沿用 §3 的基座与微调配方；融合/门控变体（§4.2）各自在表中注明训练方式。

### 4.1 训练策略消融

| 变体 | 设计改动 | 回答的问题 |
|---|---|---|
| `V8_IMU_Residual` | 完整模型：enum 基座 + 四档遮挡微调 | 主模型 |
| `V8_IMU_Residual_woEnum` | 基座不做 V/T/VT 枚举，只训 VT；再做同样的遮挡微调 | 枚举预训练是否必要 |
| `V8_IMU_Residual_DecoderOnly` | 微调只训姿态头（含根头），触觉编码/时序冻结 | 增益来自触觉表征适配还是仅解码器适配 |
| `V8_IMU_Residual_CleanOnly` | 微调只用 clean，不看遮挡 | 遮挡训练是否必要 |

设计动机：

- **woEnum**：无 enum 基座只见过 VT，V-only 输入是分布外；如果 woEnum 微调后仍不如完整模型，说明枚举预训练提供的“单模态可用性”是必要的。
- **DecoderOnly**：如果只训姿态头就能达到完整微调的效果，说明增益主要来自解码器适配，而非触觉表征；反之说明触觉编码器需要适配遮挡分布。
- **CleanOnly**：clean-only 在 clean 上应持平或略好，但在遮挡下应明显退化；这是“遮挡训练必要”的直接证据。

### 4.2 融合/门控变体

| 变体 | 设计改动 | 训练方式 | 回答的问题 |
|---|---|---|---|
| `V8_IMU_Router` | 逐帧逐组竞争路由 \(k=\sigma(s_v-s_t)\)；V 支路乘 \(1-k\)，T 支路乘 \(k\) | 冻结基座，只训 router | 逐帧软路由是否优于无门全量融合 |
| `V8_IMU_Router_Full` | 路由 + 联合微调（路由、姿态头、触觉编码/时序） | 基座 + router + 联合微调 | 路由与表征联合适配是否有额外增益 |
| `V8_IMU_TGate` | 双独立 T 门（root/pose 两组），传感分支门控 | 独立 740 ep 基座（双 T 门） | 传感分支的 T 信任门是否必要 |
| `V8_IMU_DualGate` | 四组（root/upper/left/right）独立 \(g_V,g_T\) 后置门 | 冻结基座 + 门控训练 | 独立 V/T 后置门控是否优于无门融合 |

设计动机：

- **Router**：用逐帧逐组的 \(s_v,s_t\) 竞争分数学习“这一帧更信 V 还是更信 T”；\(k\) 是软混合权重，不是硬二选一。router-only 检验冻结表征上的路由上限；full 检验路由与表征联合适配。
- **TGate**：该变体的双 T 门按 root/pose 两组独立学习 T 信任；检验“传感分支门控”这一设计。
- **DualGate**：在冻结基座上外挂四组独立 \(g_V,g_T\) 后置门；检验后置门控能否在不改表征的情况下提升遮挡鲁棒性。
- 主模型刻意不使用显式门/路由：让 enum 预训练与遮挡微调把“何时信 T”学进共享解码器，减少门控塌缩与额外监督的需求；上述变体用于验证这一选择。

---

## 5. 实现细节汇总

| 项目 | 基座预训练 | 遮挡微调 |
|---|---|---|
| 初始化 | 从零 | 基座 |
| epoch | 740 | 20 |
| batch | 256 | 64 |
| optimizer | Adam | Adam |
| lr | \(10^{-4}\) | \(10^{-4}\) |
| warmup / grad clip | 5% / 5 | — |
| 模态配置 | V/T/VT 枚举，\(p=(0.5,0.25,0.25)\) | V/T/VT 枚举，等权 \(1/3\) |
| 条件 | — | clean/20%/50%/100%，权重 \(0.5,1/6,1/6,1/6\) |
| 可训模块 | 全部 | 触觉原始编码（含内部时序层）、物理编码、触觉时序、姿态头（含根头） |
| 冻结模块 | — | 视觉编码/时序、IMU 编码/融合、锚头 |
| 模型选择 | — | 验证集 best |

数据与表示：SMPL-24、40 Hz、20 帧窗口、步长 20；HRNet 2051 维视觉特征；96 维压力 + 12 维物理量 + 12 维 IMU；F6 soft 接触标签。

评估指标（实验节使用）：PA-MPJPE、MPJPE、GlobalMPJPE、RootOrientationDrift、FootSliding、GroundPen、Jitter、MPJRE；四折 mean±std。

---

## 6. 论文写作对应建议

- **Method**：§2（结构）+ §3.1–3.3（预训练、遮挡微调、推理）。重点写清三条设计轴：PhysIMU 两阶段融合、无门全量融合 + 模态枚举、遮挡微调 + clean 相对约束。
- **Implementation Details**：§5 表格 + §3.4 四折协议；补充优化器、学习率、batch、窗口/步长、特征维度、接触标签。
- **Ablation**：§4.1（训练策略）+ §4.2（融合/门控）；按“预训练是否必要 / 微调范围 / 门控是否必要”三组组织，只保留与这三条设计轴相关的变体。
- **Figure 建议**：一张总览图（§2.1 数据流 + §2.3 两阶段融合放大图）；一张训练流程图（§3.1 枚举 → §3.2 四档遮挡微调）；消融表按 §4 的两张表组织。
- **写作口径**：主模型推理用 VT；V-only 作为对照；Method 只写最终配方，不写训练管线历史或实验状态。
