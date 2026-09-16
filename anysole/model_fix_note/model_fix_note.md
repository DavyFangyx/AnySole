# AnySole V1 模型修复笔记（E 系列实验）

> 时间：2026-09-15 ~ 2026-09-16。基线：wandb run yg01hyju（anysolev1_joint_and, 2000 epoch）。
> 每个实验的 wandb tag：e1_taumax100 / e2_tnorm / e3_nocontact / e4_normdiff / e5_sched200。

## 起点问题（yg01hyju 基线，2000 epoch）

- pose head 塌缩成"均值姿态输出器"：tau0（喂 GT 姿态）MPJPE 145mm ≈ 均值姿态基线 147mm；
  对输入 x_τ、扩散步 τ、条件 F 几乎全部无响应；val/mpjpe 162mm 自 epoch~300 平台不动。
- 根因链：TimestepEmbedding 路径无归一化 → |t_tok| 训练中无界增长（0.11→37-40，50× token 量级）
  → head 学会把 τ 信号整体消掉（τ 盲）→ 无法门控 → 对大多数 τ 输入是噪声 → 唯一可行的
  单函数策略 = 忽略输入 → 输出 ≈ f(F) ≈ 常量。

## E 系列改动与结果

### E1（--tau-max 100，500 ep，已停）
- 改动：训练期 τ ~ U(0,100) 低噪声带。
- 结果：tau0 仅 145→113mm，身份映射未学会；L_con 爆炸到 2.17。
- 结论：**τ 分布假说证伪**——问题不是训练噪声太大，是结构性的 τ 盲。

### E2（t-token 修复 + 损失再均衡）
- 改动：
  1. `embeddings.py`：TimestepEmbedding MLP 尾部加 `LayerNorm(dim)`（锁死量级，杜绝无界增长）；
  2. `pose_head.py`：t_tok 加 `×0.05`（|t_tok|≈0.8，与 pose token 0.7 同级）；
  3. `v1.yaml`：λ_pose 1→3、λ_con 0.1→0.01。
- 结果（250 ep 被杀）：**tau0 145→66mm，门控恢复成功**；val/mpjpe 162→146mm 且仍在降。
- 教训：`total loss` 的"0.3"大头是 L_con（BCE 爆炸到 61-87），不是 loss_pose。

### E3（λ_con=0，800 ep）
- 改动：λ_con 0.01→0.0（接触项在 0.01 权重下仍是 dominant 噪声源）。
- 结果：tau0 23-31mm；VT2M MPJPE **141.8mm**；root_ate 0.043 恢复；train loss_pose 0.011
  （首次低于均值解下界 0.014）。PA-MPJPE 81.2mm 与基线 80.8 持平（**F 读出天花板，
  ridge-on-F 基线 85mm**）。contact_acc 0.048（接触项移除的代价：脚浮空）。
- 新瓶颈（probe_jitter.py 实测）：窗口内逐帧抖动 **116mm/帧**（GT 8.7）；窗口拼接跳变
  **213mm**（数据集 stride=20 无重叠，导出直接 concat）。

### E4（扩散噪声按 pose_std 缩放 = Step2Motion 归一化机制）
- 改动：
  1. `train.py`：q_sample 噪声 ×`pose_std`（N(0,1) 噪声加在 std=0.118 的原始 6D 空间
     是 8.5× 信噪失配——Step2Motion 是 normalize 后扩散，缩放噪声等价）；
  2. `diffusion.py`：采样 x_T 初始化同步 ×`pose_std`，并以 ckpt 的 `noise_scaled`
     标志区分新旧模型（旧 ckpt 自动按未缩放采样，不破坏既有指标）；
  3. `train.py`：cosine LR（1e-3→0，按 epoch 比例）；
  4. `eval.py`：导出窗口边界 4 帧 crossfade + 6D 重正交化（修 213mm 拼接跳变）。
- 结果（~480 ep）：**head 单步去噪 τ=0/999: 23/57mm**（E3 是 98mm 全平——head 真正修好了，
  loss 低是真实的）；但 **50 步 DDIM 链 92-166mm，比 E3 还差**。
- 关键发现：**链步数越少越好**——10 步 80.4mm > 20 步 81.5 > 50 步 92.4 > 100 步 107。
  机制：x0 预测链每步注入一次 head 的 F 读出误差并累积；E4 的 head 会 echo 输入
  （0.12-0.63），链把逐帧噪声逐步回收，末步放行 → 逐帧掷骰子 →"又乱又跳"。
  已改 `diffusion_sample_steps` 50→20。
- 解释"为什么只改信噪比、输出反而乱"：噪声缩小让 echo 变便宜，模型学会回显；
  而 DDIM 链假设模型输出完整干净姿态。E3 的"稳定"是常量输出的假稳定。

### E5（Step2Motion 同款 T=200 linear schedule）
- 改动：
  1. `diffusion.py`：新增 linear schedule（β 1e-4→0.02），`GaussianDiffusion` 默认改 linear；
  2. `v1.yaml`：`diffusion_train_steps` 1000→200、`diffusion_sample_steps` 50（后实测改 10）；
  3. 低通滤波（导出端 EMA）实现后按用户要求**删除**（治标不治本）；
  4. crossfade 保留（导出正确性修复）。
- 结果（850 ep）：**tau0 12mm（历次最好，全关节 ≤15mm）**；但链 50 步 141mm / 10 步 114mm；
  抖动降到 27mm/帧。E4 同期（cosine, 10 步）曾 80mm——linear-200 的链静态误差反而更高
  （x_T 纯噪声起步，而训练 t=199 顶格 abar≈0.13 有 36% 信号，起步即 OOD）。
- **"飞天脚"定位（per-joint，50 步链）**：误差随离骨盆距离单调放大——
  脊柱 15 / 头 87 / 髋 95-128 / **踝 168-202mm**。旋转误差沿运动链累积，脚是杠杆臂末端。

## 当前代码状态（已合入工作区）

| 文件 | 改动 |
|---|---|
| `models/embeddings.py` | TimestepEmbedding + LayerNorm |
| `models/pose_head.py` | t_tok ×0.05 |
| `models/traj_head.py` / `losses.py` | **零改动**（用户两次怀疑 loss 出错，均排除） |
| `train.py` | 噪声×pose_std、cosine LR、ckpt 存 `noise_scaled` 标志 |
| `diffusion.py` | x_T 按 noise_scaled 缩放、linear schedule、默认 T=200 |
| `eval.py` | 窗口边界 crossfade 4 帧 + 重正交化 |
| `configs/v1.yaml` | λ_pose=3.0、λ_con=0.0、diffusion_train_steps=200、diffusion_sample_steps=10 |

探针脚本（touch_gait 环境）：`z_note/probe_anysole_v1_readout.py`（读头行为）、
`probe_anysole_v1_followup.py`（全 val 基线）、`probe_ttoken_erasure.py`（t-token 量级）、
`probe_group_std.py`（分组方差）、`probe_jitter.py`（链抖动/拼接跳变，CKPT 路径可改）。

## 待解决问题（截至 2026-09-16）

1. **飞天脚（核心未决）**：链生成姿态在远端肢体误差大（踝 168-202mm@50 步 / ~100mm@10 步），
   视觉表现为摆动腿脚"飞天"。head 精修能力（tau0 12mm）与凭空生成能力（链 114mm）差距悬殊。
2. **链质量对采样步数反直觉敏感**：步数越多越差（10 步最优）。50 步默认值在此 head 上
   不是好选择；步数与 schedule 的最优组合未系统搜索。
3. **PA-MPJPE 卡 ~81mm**：F 单视角读出天花板（ridge 基线 85mm），与训练时长无关。
4. **脚浮空/滑步**：λ_con=0 后 contact_acc 0.048；接触项需以软形式（MSE on soft-contact
   概率、或 hinge）放回，避免 BCE + 饱和 sigmoid 的梯度爆炸问题（E1/E2 的 L_con 61-87）。

## 初步诊断（E4/E5 飞天脚的成因链）

1. 链从纯噪声起步，唯一姿态信息来源是 F（V+T 融合特征）；单视角 cam3 对远端肢体
   （尤其遮挡侧）信息不足 → F 的生成极限 ≈ ridge 85mm。
2. 旋转误差沿运动链逐段放大：髋/膝各差几度 → 踝/腕放大到 168-202mm。
3. DDIM 链每步把 head 的 F 读出误差注入 x_t 并累积，步数越多越差（10 步 114mm 优于 50 步 141mm）。
4. E3 的"稳定"是均值塌缩假象；E4/E5 的"乱/飞天"是 head 修好后暴露的**链生成能力极限**，
   不是回归。

## 下一步候选（E6）

1. **Step2Motion inpainting（首选）**：窗口重叠（stride < tw），链初始化时前一半帧用
   上一窗口的预测做种子，模型只精修/续写后半——把 12mm 的精修能力用于续写而非凭空
   生成，同时自然解决窗口拼接连续性。参考其 `metrics.py` 的 prev_x_t 机制。
2. **schedule 回退**：linear-200 的 x_T 纯噪声起步与训练分布（36% 信号）不一致；
   cosine-1000 + 10-20 步在 E4 上曾到 80mm，可作对照。
3. 接触项软形式放回；训练拉满 2000 epoch 看 F 读出是否继续收敛。
