# E6: 对照 Step2Motion 实测核验（2026-09-16）

> 对 E5 checkpoint（epoch 887, E5_sched200）与 Step2Motion gait_model 的 head/采样器/续写
> 做了代码级对照实验。探针：`z_note/probe_sampler_continuation.py`、
> `probe_continuation_fullval.py`、`probe_head_generation_floor.py`、`probe_warmstart.py`。
> 结论：**采样器与 schedule 不是主因（修正 E5 笔记的归因）；head 的高 τ 行为 + 6D 表示 +
> 窗口结构才是。** Step2Motion 的优势 = 位置空间扩散 + 100 帧 stride-1 训练 + 续写推理 +
> 高噪声级零回声的 head。

## 实测数据（同 64 val 窗 / 全 34 session）

### 1. head 单步行为（决定链命运的量）

| | anysole E5 (tau=199) | Step2Motion (t=200) |
|---|---|---|
| on-marginal 输入（36% 信号） | kp 44.1mm | L1 0.122 |
| **纯噪声输入（采样条件）** | **kp 131.8mm** | **L1 0.651** |
| 纯噪声输入 echo slope | **0.293** | **0.007** |
| tau=0 干净输入 | 11.1mm | — |

- anysole 的 head 在训练边缘分布上很好（44mm），但把 29% 的纯噪声直接回显进输出；
  Step2Motion 的 head 对纯噪声几乎零回声（只读鞋垫条件）。
- 回声本身不是病：Bayes 最优去噪器对 x_tau 的斜率应为 √ᾱ（tau=199 时 ≈0.366）。
  病在 (a) 链从纯噪声起步 → 输入 OOD，回声把 OOD 噪声注入链；(b) F 读出本身弱，
  纯噪声下 one-shot 只有 131.8mm。
- tau=0 时 echo 0.96 是**正确**的身份映射行为，不是捷径缺陷（修正 Test9.1 的解读）。

### 2. 采样器换 Step2Motion 同款不解决问题

- DDIM-10: 161.6mm / DDIM-50: 178.6 / DDPM-200 posterior: 180.9（同一 head）。
- 即：把采样器换成 Step2Motion 的 200 步 DDPM posterior，没有任何改善。
  笔记中"换成 Step2Motion 采样机制即可"的暗示被证伪——采样器不是杠杆。

### 3. 续写（prev_x_t inpainting）部分有效

- S7013 连续 8 窗：DDIM 161.6 → **83.6mm**（续写），seam 213→52mm。
- 全 34 session 平均：续写 171.3 vs DDIM-10 162 —— 平均无净收益，逐会话方差极大。
- 结论：续写对"好会话"巨大，对"坏会话"无助——它放大了 head 单窗生成质量的方差，
  不是通用解药。

### 4. 暖启动（链起点对齐训练边缘分布）小有帮助

- x_T = √ᾱ·prior + √(1-ᾱ)·noise（prior=均值姿态）：197.0 → 160.7mm（同 probe 内对比）。
- f(F) prior 反而更差（177.5）——f(F) 里含回声噪声。
- 注意：同变体换 x_T 噪声种子可差 35mm（161.6 vs 197.0）——链对 x_T 极其敏感，
  好模型不该如此；这是"head 在 OOD 输入上输出 OOD"的又一佐证。

### 5. 训练量与窗口

- anysole: 1167 窗/epoch ≈ 5 step/epoch × 887 ep ≈ **4.4k 步**；20 帧非重叠窗。
- Step2Motion: 23.7k 帧 stride-1 100 帧窗 ≈ 93 step/epoch × 100 ep ≈ **9.3k 步**，
  每 epoch 见到全部对齐方式（20× 窗口数）。
- 两套数据同源（AnysoleWorkspace gait pt 的帧数 23786/11937/11937 与 anysole
  splits 的帧数一致）→ Step2Motion 的 normalizer/导出管线可直接复用。

### 6. 既有指标佐证

- E5 metrics（10 步）：VT2M 162.4 / V2M 191.5 / **T2M 211.4**——触觉通路基本是坏的
  （Step2Motion 仅凭同款鞋垫做到 pose MPJPE 133mm）。
- per-joint 放大谱：Hips 60 → 前臂 259 → **手 398mm**（DDIM-10）——6D 旋转误差沿
  FK 链放大的签名；Step2Motion 位置空间无此放大（mpjpe_legs 71mm）。
- 抖动：anysole 46-56mm/帧（GT 8.7）vs Step2Motion mpjve_legs 123mm/s≈3mm/帧。

## 修正 E5 笔记的归因

1. "linear-200 的 x_T 纯噪声起步与训练分布不一致" —— 现象对，但 Step2Motion 同样
   纯噪声起步、同样 36% 顶格信号，差异在 head 对 OOD 的稳健性（echo 0.007 vs 0.293），
   不在 schedule。
2. "DDIM 链每步注入 F 读出误差，步数越多越差" —— 半对：根因是 head 在链状态
   （OOD）上输出劣化 + 回声，步数只是表象。
3. E6 候选 #1（inpainting 首选）——实测部分有效，单用不够。

## 解决方案（按优先级）

**A. 零重训（1-2 天）：**
1. eval/infer 改续写式推理（prev_x_t，探针已实现）+ 暖启动（均值姿态 prior）。
2. 训练期监控三件套：echo slope vs √ᾱ 曲线、tau=199 纯噪声 one-shot MPJPE
   （131.8 ↔ 链 162，最好的链质量预报器）、续写 seam jump。

**B. 重训对齐 Step2Motion（治本）：**
1. 表示：6D 旋转 → 根相对 3D 位置（69 维），复用其 normalizer 与
   skeleton_pos_to_rot/BVH 导出。消除 FK 放大，输出空间=指标空间。
2. 窗口：tw 20→100，训练 stride 1（数据量 ×20，1167→23.7k 窗/epoch）。
3. 触觉编码：对齐 Step2Motion 的 per-group（脚跟/脚掌压力、IMU、力、COP）
   per-frame 对齐 embedding + cross-attn；修 T2M 211mm。
4. 损失：L1 位置空间；接触软损失（hinge/MSE）放回（修 contact_acc 0.056）。
5. 训练期 val 用最终采样器（Step2Motion 用 10 步 DDPM posterior 选 ckpt）。

**C. 若保留 6D 表示：** A 全部 + λ_kp 提权 + 窗口 stride 训练；上限仍受 FK 放大限制。
