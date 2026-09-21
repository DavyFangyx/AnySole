# E6.6 / E6.7 计划：触觉路径重建 + 训练目标权重清单（E3 基线，CLI 单步叠加）

> **目标**：把 echo slope（E5 实测 0.293 vs S2M 0.007）打下来，让条件（尤其触觉）驱动输出。
> **机理**（E6 笔记）：echo 是条件弱时 Bayes 最优去噪器的自然行为——x0 后验均值对 x_tau 的
> 斜率 = √ᾱ（S2M τ=200 时 0.366；cosine-1000 的 τ=999 时 ≈0）。S2M 做到 0.007 不是因为
> 网络形状好，而是每帧都有强耦合的鞋垫条件可依赖，不需要看 x_tau 就能给出合理 x0。
> 所以治理 echo 的唯一正路 = **把触觉条件做强**，而不是给损失打补丁。echo slope 只作
> 为验收指标全程监控。
>
> **v1.yaml 全程不动**（保持纯 E3 默认值）。所有实验差异由 CLI 传入，CLI 覆盖值随
> checkpoint 保存（现有机制），eval/infer 从 ckpt 读配置、无需重复传参。

---

## 工作流：叠加 + 单步回退

1. **叠加链**（每条命令都是"到当前步为止的全量 flags"，直接照抄即可）：

   ```
   E3_base → E6.6a → E6.6b → E6.1_pos →（可选 E6.2）→ E6.3_longwin → E6.7_loss
   ```

2. **每步验收**：与该步的"上一步 checkpoint"同表对比（验收口径见文末）。200 epoch
   筛查；筛查有明显收益再补足（6D 系列补到 800 ep，pos 系列 200-400 ep），然后才
   进入下一步。
3. **回退规则**：某步验收劣于上一步 → **弃用该开关**，用上一步的命令集（去掉该 flag）
   继续叠下一个开关。失败的 checkpoint 留档不删；代码不回改（新开关默认关闭，天然
   无副作用）。
4. **特例**：若 6.6a 无收益，**不回退到 E3**——6.6b 本身包含 `--tactile-input s2m50`，
   直接进 6.6b 验证"路径稀释"假设；仅当 6.6b 也劣于 E3 才整体回退触觉两步，从 E3
   直接叠 E6.1。

## 开关一览（CLI）

| 开关 | CLI | 默认 | 状态 |
|---|---|---|---|
| 触觉通道 | `--tactile-input raw108 \| s2m50` | raw108 | **已实现**（2026-09-16） |
| 触觉直达 | `--tactile-direct` | 关 | **已实现**（2026-09-16，需 s2m50，有守卫） |
| 表示切换 | `--modal anysolev1_pos` | anysolev1 | 已有（E6.1） |
| 窗口/步长 | `--tw` / `--stride` / `--batch-size` | yaml | `--stride` 已加入 train.py，`--tw`/`--batch-size` 已有 |
| 损失权重 | `--lambda-pose/-kp/-traj/-trec/-vrec/-con/-pose-vel/-bone` | yaml | 已加入 train.py（E6.7） |

---

## 0. 触觉现状 vs Step2Motion（代码级对照）

| | Step2Motion（强） | AnySole E3（弱） |
|---|---|---|
| 通道 | 25 维/脚：**16 压力（4×12 池化、分 heel[8]/toes[8]）+ 3 加速度 + 3 角速度（由 BVH 脚部运动学合成，世界系、去重力）+ 总力 + CoP** | 54 维/脚：48 格原始压力（展平，无 heel/toes 结构）+ 6 维物理量（cop/force/envelope/spatial/temporal）。**无 IMU** |
| 编码 | 8 组各自一个 MLP（`model.py:93-148`）→ 8 条 per-frame token 流 | 单层 `Linear(108→256)` + LayerNorm（`encoders.py:41`） |
| 注入 | 每层 cross-attn，8 个头各对一条模态流（`insole_multiheadattention.py`） | 与 V 一起过 4 层融合，pose head 只 cross-attn 到融合后的 F |
| 对齐 | 条件与输出逐帧对齐（T=100） | 同为逐帧（tw=20），但经融合后被稀释 |
| 结果 | 同款数据 pose MPJPE 133mm（T2M 口径） | T2M one-shot 154mm、T2M 链 211mm |

**根因三条**：
1. **通道缺失**：无 IMU。S2M 的合成 IMU 是"GT 脚部运动学"的逐帧读数，对脚部运动是
   最强的条件信号；你的 48 格压力只有"踩没踩地"级别的信息。
2. **编码粗糙**：108 维一锅端进一个 Linear；4×12 网格的 heel/toes 空间结构丢失
   （S2M 按 Moticon 16 通道池化后分组嵌入）。
3. **路径稀释**：V 信号强（V2M 191 < T2M 211），4 层融合里 T 被 V 主导，T 通路的梯度
   长期欠更新；pose head 只能读融合结果，触觉从未直达 pose head。

---

## E6.6a 触觉通道对齐（只换通道，编码/路径不动）

**改动**：把 t_enc 的输入从 `cat([T_raw, T_phys])`（108 维）换成 **T_s2m（50 维）**，
与 S2M `input_dim=50` 完全同口径：

```
每脚 25 维 = pressure16（pool_48_to_16：heel[8] toes[8]）+ acc3 + gyro3 + force1 + cop2
```

- `pool_48_to_16`、`cop_from_grid`、`rest_foot_imu_axes`、`angular_velocity_from_rotmats`、
  `synthesize_imu`、`foot_features` **从 `Baselines/Step2Motion/src/process_gait.py` 照抄
  （约 150 行 numpy）**，不要自行"改进"：合成局部系 acc/gyro（左脚 Y 翻转、+G、/G），
  再按 S2M `dataset.py:136-148` 转回世界系并减重力——两段照抄保证与 S2M 训练输入同分布
  （数学上往返严格抵消，最终通道 = 脚部世界系加速度/角速度，但仍照抄以留审查痕迹）。
- 脚部世界位置二阶差分作 acc_world、脚部全局旋转差分作角速度；anysole 数据集构建时
  已有 `fk_pose6d_np` 的 kp 与 `rot6d_to_rotmat_np`，只需补一个返回全局旋转的小 FK
  （或照抄 `fk_local`）。dt = 1/FPS = 0.025s。
- 实现：新文件 `anysole/data/tactile_s2m.py`；`dataset.py` 每 session 构建时算好
  `T_s2m (T, 50)` 存入 session dict，`__getitem__` 随窗口裁剪。**不新建缓存文件**（构建
  开销与现有 FK 同量级）。`T_raw` 保留（仍作 L_Trec 的 target，不受影响）。
- 模型侧：`encoders.py` 的 t_enc `in_dim 108 → 50`；`model.forward` 签名
  `T_raw, T_phys → T_s2m`（`diffusion.py` / `train.py` / `eval.py` / `infer.py` 机械跟随）。
  `types.py` 加 `T_S2M_DIM = 50`。CLI 开关 `--tactile-input` 随本步一并接入 train.py。

**泄漏口径**（必须写进实验记录）：合成 IMU 由 **GT BVH** 推得，eval 的输入含 GT 派生
信息——**与 Step2Motion 的评估口径完全相同**（它的 133mm 就是这么测的）。真实部署时
用鞋垫硬件 IMU 替代；补一个消融：IMU 6 通道置零，量化对合成 IMU 的依赖。

**预期**：T2M one-shot 154 → 明显下降（IMU 给出摆动/支撑的逐帧脚部运动）；echo slope
下降。若 200 epoch 内 T2M 无变化，说明瓶颈在路径稀释（跳到 E6.6b）。

---

## E6.6b 触觉使用方式（per-group 编码 + 直达 pose head）

在 E6.6a 之上，改"怎么用"：

1. **新 `TactileEncoder`**（`anysole/models/tactile_encoder.py`）：8 组
   （左/右 × 脚跟压力8/脚掌压力8/IMU6/其他3）各自一个 `Linear(g→256)→GELU→Linear→LayerNorm`，
   + 时间 PE，8 条 per-frame 流**求和合并**为 `t_tok (B, TW, d)`（保持融合层输入形状不变，
   fusion 无感切换）。流内信息保留方式 = 分组编码（S2M 同款）；流间区分靠求和前各流
   自己的编码权重。
2. **pose head 直达触觉记忆**：`model.forward` 里 cross-attn 记忆从 `F` 改为
   `cat([F, t_tok], dim=1)`（40 → 60 token，`pose_head.forward` 一行不动）。
   config dropout 复用现有 null-token 机制（drop_t 时 t_tok 已被替换为 null_t，
   直达路径自动一并屏蔽，V/T/VT 三口径语义不变）。
3. CLI 开关：`--tactile-direct`（默认关）。

**预期**：T2M 是最大受益口径（T-only 时 pose head 有直达触觉记忆）；V-only 口径因
null_t 直达路径与 E3 等价（回归保护）。VT2M 也应下降（F 与 T_mem 双路）。

**后续升级（E6.6c，仅在 6.6b 不够时）**：照抄 `InsoleMultiheadAttention`（8 头每头
对一条流）替换求和合并，pose head 每层对 8 条流做 per-head cross-attn——S2M 原版结构。

---

## E6.1 / E6.2 / E6.3（纯 CLI，无新代码）

- **E6.1 表示**：`--modal anysolev1_pos`（位置空间 66 维，FK 放大消失；2ep 已实测
  90mm ≈ 6D 满训 97.8）。
- **E6.2 平滑+刚性**（可选，pos 模式）：`--lambda-pose-vel 10 --lambda-bone 1`
  （摆动腿被抹平则降为 3 / 0.3）。
- **E6.3 窗口**：`--tw 100 --batch-size 64 --stride 1`（监督密度 ×100，270 步/epoch，
  40-100 ep 饱和）。

---

## E6.7 训练目标清单与权重

当前全部损失（`losses.py`，v1.yaml 的 E3 权重，CLI 覆盖键见括号）：

| # | 损失 | 权重（E3） | 梯度流向 | 作用 | E6.7 建议（CLI） | 理由 |
|---|---|---|---|---|---|---|
| 1 | L_pose | **3.0** | pose_head + F + encoders | 扩散 x0（6d=MSE / pos=L1） | **保持 3.0** | 唯一直接训练 head 的损失，最高优先级 |
| 2 | L_kp | **1.0** | 同上 | FK 关键点几何监督 | **保持 1.0** | 旧 E7 教训：几何监督不能砍 |
| 3 | L_traj | 1.0 | traj_head + F + encoders | 速度 + 多尺度位移 MSE | **`--lambda-traj 0.3`** | 与 pose 竞争 F 容量；S2M 平移是独立网络，这里不应等权 |
| 4 | L_Trec | 0.1 | aux_head + F + encoders | 重建 96 维压力 | **`--lambda-trec 0.05`**（可后续移除） | 训练 F"复述输入"而非"服务 pose"；重建与 pose 目标冲突 |
| 5 | L_Vrec | 0.1 | aux_head + F + encoders（仅 T-only 行） | 重建 V 特征 | **`--lambda-vrec 0.05`**（可后续移除） | 同上 |
| 6 | L_con | **0.0** | pose_head（6d 经 FK） | 接触 BCE | **保持 0.0** | E3 口径；pos 模式不适用 |
| 7 | L_pose_vel | 0.0 | pose_head | 帧间平滑（E6.2） | pos 模式 **`--lambda-pose-vel 3~10`** | 压抖动；摆动腿被抹平则降 |
| 8 | L_bone | 0.0 | pose_head | 骨长刚性（E6.2） | pos 模式 **`--lambda-bone 0.3~1`** | 保骨架；同上 |
| — | traj_velocity_w / traj_delta_w | 1.0 / 1.0 | L_traj 内部 | 速度项与位移项配比 | 随 λ_traj 整体缩放即可 | 内部配比不必先动 |

**原则**：pose head 的梯度只有 L_pose + L_kp；但 F 的梯度来自全部 6 个损失。
E6.7 的第一步只降 3/4/5（不动 1/2/6），第二步按需移除 4/5 并观察 T2M。
**E6.6 系列期间权重保持 E3 不变**（单步纪律）。

**可选 E6.7b（τ 采样）**：cosine-1000 下均匀 τ 采样使大部分样本落在 ᾱ≈0 的深噪声区
（S2M 的 linear-200 均匀采样则全程有信号）。若 6.6 后高 τ 行为仍差，把 τ 采样改为
**ᾱ 均匀**（或加大低 τ 概率，现有 `--tau-max` 可先试低噪声带训练），直接减少深噪声
区的训练占比。

---

## Echo 监控（贯穿所有实验）

训练期探针接入 train.py 的 eval hook（每 `wandb_eval_interval` 记 2 个数，逻辑照抄
`z_note/probe_head_generation_floor.py`）：

1. **echo slope @ τ_max**：固定一个噪声种子，x_tau=纯噪声、τ=999 前向，
   x0_hat 对 x_tau 逐维线性回归斜率均值。参照线 = √ᾱ(999) ≈ 0（cosine-1000）。
   E5 实测 0.293（其 τ=199 参照线 0.366）。目标：随 6.6 逐步下降。
2. **top-τ 纯噪声 one-shot MPJPE**：同一前向的 kp 误差（E6 笔记：131.8 ↔ 链 162，
   是最好的链质量预报器）。目标：逼近 g(F) 地板（E3 的 T 口径 154）。

配套对照：训练面板 tau0（head 健康度）、T2M/V2M/VT2M one-shot、50 步链指标、
脚踝相对高度（踢腿判据：E3 -0.606/0.286）。

---

# 完整命令手册（每步：训练 / 评估 / 可视化，直接照抄）

> env = touch_gait，全部在 `/data/fangyuxuan/projects/gait` 下执行。
> `--device` 按当时 `nvidia-smi` 空闲 GPU 填（命令里已给出示例，注意勿与在跑实验撞卡）。
> 可视化目录约定：模型目录名 = `--modal` 与 `--contact-method` 的字符串拼接
> （`<modal>_<contact-method>`），eval 的 `--write-bvh` 必须与此一致。
> 每步 200 ep 筛查通过后，把 `--epochs` 补足（6D 系列 800，pos 系列 200-400）再进下一步。

## 步骤 0：E3 基线复跑（对照基准）

```bash
# 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --epochs 800 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E3_base/checkpoints \
  --wandb_mode online --wandb_experiment_tag e3_base \
  --wandb_eval_interval 10 \
  --device cuda:4

# 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E3_base/checkpoints/ckpt_last.pt \
  --modal anysolev1 --contact-method joint_and \
  --write-bvh results/AnySole/E3_base/predictions/eval_bvh \
  --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal E3 --contact-method base
```

## 步骤 1：E6.6a 触觉通道对齐（+ `--tactile-input s2m50`）

```bash
# 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --tactile-input s2m50 \
  --epochs 200 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E6.6a_tacch/checkpoints \
  --wandb_mode online --wandb_experiment_tag e6_6a \
  --wandb_eval_interval 10 \
  --no-imu \
  --device cuda:4

# 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.6a_tacch/checkpoints/ckpt_last.pt \
  --write-bvh results/AnySole/E6.6a_tacch/predictions/eval_bvh \
  --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal E6.6a --contact-method tacch --force
```

**验收**：T2M one-shot / echo slope 对比 E3_base。无收益 → 不回退，直接进步骤 2。

## 步骤 2：E6.6b 触觉直达（再 + `--tactile-direct`）

```bash
# 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --tactile-input s2m50 \
  --tactile-direct \
  --epochs 200 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E6.6b_tacdir/checkpoints \
  --wandb_mode online --wandb_experiment_tag e6_6b \
  --wandb_eval_interval 10 \
  --no-imu \
  --device cuda:5

# 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.6b_tacdir/checkpoints/ckpt_last.pt \
  --write-bvh results/AnySole/E6.6b_tacdir/predictions/eval_bvh \
  --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=6 python results_display/script/visualize_anysole.py --modal E6.6b --contact-method tacdir --force
```

**验收**：T2M/VT2M one-shot、echo slope 对比 E6.6a 与 E3_base。
劣于 E3_base → 整体回退触觉两步，从步骤 0 直接叠步骤 3（命令去掉两个 `--tactile-*`）。

## 步骤 3：E6.1 位置空间表示（再 + `--modal anysolev1_pos`，已有开关）

```bash
# 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --tactile-input s2m50 \
  --tactile-direct \
  --modal anysolev1_pos \
  --epochs 200 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E6.1_pos/checkpoints \
  --wandb_mode online --wandb_experiment_tag e6_1_pos \
  --wandb_eval_interval 10 \
  --device cuda:6

# 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.1_pos/checkpoints/ckpt_last.pt \
  --write-bvh results/AnySole/E6.1_pos/predictions/eval_bvh \
  --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=6 python results_display/script/visualize_anysole.py --modal E6.1 --contact-method pos
```

**验收**：MPJPE / PA-MPJPE / 抖动 / 脚踝高度对比 E6.6b。若回退了触觉两步，则对比 E3_base。

## 步骤 3.5（可选）：E6.2 平滑 + 骨长刚性（再 + 两个 lambda，仅 pos 模式）

```bash
# 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --tactile-input s2m50 \
  --tactile-direct \
  --modal anysolev1_pos \
  --lambda-pose-vel 10 --lambda-bone 1 \
  --epochs 200 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E6.2_smooth/checkpoints \
  --wandb_mode online --wandb_experiment_tag e6_2_smooth \
  --wandb_eval_interval 10 \
  --device cuda:4

# 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.2_smooth/checkpoints/ckpt_last.pt \
  --write-bvh results/AnySole/E6.2_smooth/predictions/eval_bvh \
  --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal E6.2 --contact-method smooth
```

**验收**：帧间抖动下降、无"摆动腿被抹平"（BVH 肉眼 + 逐关节速度）。
抖动不降或动作变僵 → 回退本步（去两个 lambda）继续。

## 步骤 4：E6.3 长窗口 + stride-1（再 + `--tw 100 --batch-size 64 --stride 1`）

```bash
# 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --tactile-input s2m50 \
  --tactile-direct \
  --modal anysolev1_pos \
  --tw 100 \
  --stride 1 \
  --batch-size 64 \
  --epochs 200 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E6.3_longwin/checkpoints \
  --wandb_mode online --wandb_experiment_tag e6_3_longwin \
  --wandb_eval_interval 10 \
  --device cuda:7

# 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.3_longwin/checkpoints/ckpt_last.pt \
  --write-bvh results/AnySole/E6.3_longwin/predictions/eval_bvh \
  --device cuda:6

# 可视化
CUDA_VISIBLE_DEVICES=7 python results_display/script/visualize_anysole.py --modal E6.3 --contact-method longwin
```

**验收**：全部指标对比 E6.2（或 E6.1）。tw=100 显存不足 → batch-size 降到 32/16。

## 步骤 5：E6.7 损失权重重构（再 + 三个 lambda）

```bash
# 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --tactile-input s2m50 \
  --tactile-direct \
  --modal anysolev1_pos \
  --tw 100 \
  --stride 1 \
  --batch-size 64 \
  --lambda-traj 0.3 --lambda-trec 0.05 --lambda-vrec 0.05 \
  --epochs 200 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E6.7_loss/checkpoints \
  --wandb_mode online --wandb_experiment_tag e6_7_loss \
  --wandb_eval_interval 10 \
  --device cuda:4

# 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.7_loss/checkpoints/ckpt_last.pt \
  --write-bvh results/AnySole/E6.7_loss/predictions/eval_bvh \
  --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal E6.7 --contact-method loss
```

**验收**：对比 E6.3。之后可试第二步：`--lambda-trec 0 --lambda-vrec 0` 完全移除重建损失。

---

## 单会话推理（任一步 checkpoint，导出可交互 BVH）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.infer \
  --ckpt results/AnySole/E6.7_loss/checkpoints/ckpt_last.pt \
  --modal anysolev1_pos --contact-method joint_and \
  --session S7013 --config-id VT2M \
  --device cuda:4
```

## 验收口径（每步必须同表）

1. **数字**：VT2M/V2M/T2M one-shot MPJPE、echo slope vs √ᾱ、τ=0 身份 MPJPE、50 步链
   MPJPE / PA-MPJPE / traj_ATE、训练面板 tau0；
2. **运动学（关键）**：导出 BVH 的逐关节误差、**脚踝相对高度均值/最大值**
   （踢腿判据：E3 为 -0.606/0.286，E4 为 +0.013/0.627）、帧间抖动
   （GT 8.7mm/帧，E5 46-56mm/帧）；
   测量脚本复用 `z_note/smoke_e6x_decoupling.py` 的 fk 读取方式。

## 实现文件清单

| 文件 | 改动 |
|---|---|
| `anysole/data/tactile_s2m.py`（新） | pool/synthesize_imu 等端口（照抄 process_gait.py）；**已实现+单测** |
| `anysole/data/dataset.py` | session 构建时算 T_s2m；item 增加该字段；**已实现** |
| `anysole/types.py` | `T_S2M_DIM = 50`；**已实现** |
| `anysole/models/tactile_encoder.py`（新，6.6b） | 8 组 per-group 编码 → 求和合并 t_tok；**已实现** |
| `anysole/models/encoders.py` | 6.6a：t_enc in_dim 108→50；6.6b：换 TactileEncoder；**已实现** |
| `anysole/models/model.py` | forward 签名换 T_s2m；6.6b：记忆 cat([F, t_tok])；**已实现** |
| `anysole/train.py` | `--tactile-input`/`--tactile-direct` 已加入；`--stride`/`--lambda-*` 已加入；echo 探针（待接入） |
| `anysole/diffusion.py` / `eval.py` / `infer.py` | T_s2m 贯通；**已实现+集成测试通过** |

**兼容性已验证**：E2–E6.3 全部现代 checkpoint 在新模型上 strict 加载通过（raw108 架构
逐参数一致，E3 行为不变）；`--tactile-direct` 无 s2m50 时有守卫报错。

## 风险与诚实声明

1. **合成 IMU 含 GT 信息**：eval 口径与 Step2Motion 相同（基线同泄漏），但论文里必须
   声明并给"IMU 置零"消融；真实部署依赖鞋垫硬件 IMU（AnySole 鞋垫是否真有 IMU 需确认，
   若没有，此方案的价值限于与基线对齐的学术口径）。
2. **E6.6 在 6D 上验证**（按单步纪律从 E3 出发）；最大单步收益预期仍是 E6.1 pos
   （实测 2ep 90mm）。若 6.6a 在 200 ep 内无收益，直接跳 6.6b，不要把预算花在通道上。
3. 命令里 `--device` 为示例值，执行前用 `nvidia-smi` 确认空闲卡，避免与在跑实验撞卡。
