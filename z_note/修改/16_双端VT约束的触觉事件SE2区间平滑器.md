# 双端 VT 约束的触觉事件 SE(2) 区间平滑器（2026-11-10）

## 0. 裁定

本文档取代「逐帧独立、各向同性加性误差 + 位置闭式投影」作为正式设计。

正式方法定义为：

> **T+IMU 因果轨迹主干 + 双端 VT 约束的 SE(2) fixed-lag smoother。**
> 视觉失效期间由 T+IMU 实时输出相对轨迹；视觉重新出现后，在有限遮挡区间内联合估计
> 根位置、航向、区间陀螺零偏和乘性步长尺度，重新发布该区间轨迹。

执行顺序冻结为：

| 阶段 | 内容 | 是否训练 |
|---|---|---|
| v1 | 逐帧 SE(2) 图；区间共享 `b_g`、`a`；固定物理协方差；双端 VT 硬约束 | 否（只作 train-only 统计标定） |
| v2 | 冻结轨迹主干，训练 covariance head；IMU-only / IMU+T 消融 | 是，只训练 covariance head |
| v3 | heel-strike 检测通过独立验收后，用触觉事件替换逐帧节点 | 视 v2 结果决定是否微调 covariance head |
| 后续 | 静止软因子、VT 软边界、`b_g/a` 随机游走 | 仅在前序诊断明确需要时 |

`S2M-VTBridge` **不实现**。S2M 只保留原始口径和当前已有的 `S2M-VT` 线性粘合基线；
本文通过自身消融拆开「通用 SE(2) 平滑收益」与「触觉机制收益」。

---

## 1. 问题与现状

### 1.1 目标场景

主场景是：

```text
VT 可见入口 → T+IMU 视觉失效段 → VT 重新可见出口
```

系统必须同时支持：

| 可用锚点 | 输出 |
|---|---|
| 前后 VT 都有 | T+IMU 原始轨迹 + 双端后验平滑；本文主场景 |
| 只有前 VT | 从入口绝对位姿开始的因果 dead reckoning；误差随时间增长 |
| 前后都无 VT | 任意局部坐标系下的相对轨迹；绝对原点和绝对 yaw 不可观测 |
| 只有后 VT | 离线时可向出口对齐；不作为实时主协议 |

因此，SE(2) smoother 是可选后端，不能成为 T+IMU 主干产生位移的必要条件。

### 1.2 当前实现

当前 `HybridTranslation`：

- 输入 44 维：双足原始 IMU 12、yaw 3、F2 前端 12、压力块 10、压力时钟 6、
  `v_phys` 1；
- 输出当前 yaw 帧二维逐帧位移；
- 损失为逐帧 L1、多尺度 cumsum、周期位移和周期 cumsum；
- 模型输入、结构和训练目标均没有 VT 入口/出口状态、遮挡边界或出口约束。

当前 `assemble()`：

1. 入口位置取 `vt[n0]`；
2. 用入口前 40 帧可见块标定全局 yaw；
3. 遮挡内积分 Hybrid 位移；
4. 出口计算 `delta = vt[e0] - pred[e0]`；
5. 将 `delta` 按时间线性 ramp 加回整段位置。

该装配把航向漂移、步长尺度误差和局部位移误差全部压成出口二维位置残差，不能区分误差来源。

### 1.3 当前公平结果

四折场景遮挡、相同 VT 前后锚下的动作均值如下（遮挡内 ATE，mm）：

| 基座 | 动作 | S2M-VT | Hybrid |
|---|---:|---:|---:|
| V7_Cv5 | 03 | 129.2 | **114.1** |
| V7_Cv5 | 04 | 144.7 | **105.6** |
| V7_Cv5 | 05 | **163.9** | 164.1 |
| V7_Cv5 | 08 | **159.2** | 170.7 |
| V7_Cv5 | 09 | 196.5 | **191.3** |
| V8_Cv5 | 03 | 155.2 | **123.1** |
| V8_Cv5 | 04 | 130.5 | **116.2** |
| V8_Cv5 | 05 | **128.3** | 135.3 |
| V8_Cv5 | 08 | **160.3** | 171.9 |
| V8_Cv5 | 09 | 218.2 | **210.6** |

Hybrid 稳定赢 03/04/09，未赢 05/08。下一步必须改善遮挡内部轨迹形状，而不是继续依靠出口位置粘合。

---

## 2. 为什么不用逐帧闭式位置投影

原候选式：

$$
d_t^*=\hat d_t+
\frac{\sigma_t^2}{\sum_j\sigma_j^2}
\left(\Delta p-\sum_j\hat d_j\right)
$$

只在以下条件下成立：

1. 误差是逐帧独立加性噪声；
2. 协方差各向同性；
3. 航向已知且不需要由出口位置共同估计；
4. 不存在跨时间相关的 gyro bias 和乘性步长尺度误差。

惯性轨迹的主要误差不满足这些条件：

- yaw gyro bias 积分后形成强时间相关的航向误差；
- 步长偏差主要表现为乘性尺度；
- 出口位置残差本身包含航向漂移信息；
- 等权横向修正会表现为整段侧滑；
- 静止段只要权重非零就可能被出口残差推动。

因此，闭式位置投影只保留为 v1a 的退化验收对象，不作为最终方法。

---

## 3. 坐标、索引与符号

### 3.1 区间索引

一个完整遮挡区间包含状态：

$$
t=0,1,\ldots,K
$$

- `t=0`：最后一个可靠入口 VT 状态；
- `t=K`：第一个可靠出口 VT 状态；
- 因子 `k=1,...,K` 连接 `k-1 → k`；
- 采样间隔 `dt=1/40 s`，实际实现必须读取数据口径，不得默认补帧。

发现 VT、IMU、压力或预测长度不一致时立即报告；不得截断、补齐或伪造。

### 3.2 坐标约定

- $p_k\in\mathbb R^2$：世界 x-z 平面根位置；
- $\psi_k$：世界 yaw，单位 rad；
- $\hat d_k=[d_{f,k},d_{l,k}]^\top$：主干输出的 yaw 帧前向/左向位移；
- $\Delta\hat\psi_k$：IMU/yaw 通道给出的 `k-1 → k` 航向增量；
- `wrap(·)`：映射到 $(-\pi,\pi]$。

为与现行 `yaw_frame_disp/_assemble_vec` 逐位一致，定义：

$$
A(\psi)=
\begin{bmatrix}
\sin\psi & -\cos\psi\\
\cos\psi & \sin\psi
\end{bmatrix}
$$

则 yaw 帧位移到世界位移为：

$$
\Delta p_k=A(\psi_{k-1})d_k
$$

实现时不得换成另一套标准二维旋转矩阵后再靠符号试错。

---

## 4. v1：不训练的 SE(2) 区间平滑器

### 4.1 优化变量

v1 使用逐帧节点：

$$
x_k=(p_k,\psi_k),\qquad k=0,\ldots,K
$$

每个遮挡区间另有两个共享变量：

$$
b_g\in\mathbb R,\qquad a\in\mathbb R,\qquad s=\exp(a)>0
$$

- $b_g$：区间常值 yaw gyro bias，单位 rad/s；
- $a$：log 步长尺度；
- $s$：乘性步长尺度。

首版不得增加逐帧 `b_{g,k}` 或逐步 `a_k`。

### 4.2 里程计因子

平移残差：

$$
r^p_k=
A(\psi_{k-1})^\top(p_k-p_{k-1})
-\exp(a)\hat d_k
$$

航向残差：

$$
r^\psi_k=
\operatorname{wrap}
\left[
\psi_k-\psi_{k-1}
-\left(\Delta\hat\psi_k-b_g\Delta t_k\right)
\right]
$$

位置与航向在同一个因子图中优化。禁止先独立做 yaw smoothstep，再做位置投影。

当 gyro bias 为区间常值、没有其他信息时，最优 yaw 修正自然随时间线性积累；不额外指定三次平滑函数。

### 4.3 固定物理协方差

v1 不学习协方差。身体/yaw 帧下的平移协方差为：

$$
\Sigma^p_k=
\begin{bmatrix}
q_f(\ell_k+\epsilon_\ell) & 0\\
0 & q_l(\ell_k+\epsilon_\ell)
\end{bmatrix},
\qquad
\ell_k=\|\hat d_k\|_2
$$

航向方差为：

$$
\sigma^2_{\psi,k}=q_\psi\Delta t_k+\epsilon_\psi
$$

要求：

- $q_f,q_l,q_\psi$ 只用当前 fold 的 train split 估计一次，随后冻结；
- $epsilon_\ell,\epsilon_\psi>0$，防止静止帧协方差奇异；
- 所有特征值设固定上下界；
- v1 只使用一个全局各向异性比 $q_f/q_l$；
- 不按动作、测试受试者或场景单独调参；
- `Sigma` 表示方差，文档和实现不得与标准差混用。

`q_f/q_l` 不是“永远禁止侧移”。它只表达正常直行下的全局软先验；05/08、转身和侧步仍由位姿状态与观测共同解释。

### 4.4 区间共享变量先验

必须加入：

$$
r_b=\frac{b_g-b_{g,0}}{\sigma_b},
\qquad
r_a=\frac{a-a_0}{\sigma_a}
$$

默认：

$$
b_{g,0}=0,\qquad a_0=0
$$

$\sigma_b,\sigma_a$ 只使用 train split 中 raw rollout 相对 GT 的误差统计估计并冻结。
没有这两个先验时，短距离、直线或近静止区间的 `b_g/a` 可能弱可辨识。

### 4.5 双端 VT 硬约束

v1 采用：

$$
p_0=p^{VT}_{in},\quad \psi_0=\psi^{VT}_{in},\quad
p_K=p^{VT}_{out},\quad \psi_K=\psi^{VT}_{out}
$$

硬约束通过从自由变量中消去 `x_0/x_K` 实现，不使用任意超大的 penalty 近似。

这是 v1 的刻意简化：先判断 SE(2) 联合解释是否优于线性位置粘合。VT 软边界和
`Sigma_in/Sigma_out` 放到后续阶段。

### 4.6 v1 目标函数

$$
\min_{\{p_k,\psi_k\},b_g,a}
\sum_{k=1}^{K}
\left[
(r^p_k)^\top(\Sigma^p_k)^{-1}r^p_k
+\frac{(r^\psi_k)^2}{\sigma^2_{\psi,k}}
\right]
+r_b^2+r_a^2
$$

v1 不加入：

- 静止根位移因子；
- 触觉事件节点；
- VT 软边界；
- robust kernel；
- `b_g/a` 随机游走；
- 任何训练参数。

若普通最小二乘被少数异常帧支配，先输出残差诊断；不得直接加 Huber 掩盖问题。

### 4.7 初始化与求解

初始化：

1. `psi_0 = psi_in^VT`；
2. `psi_k = psi_{k-1} + Delta_hat_psi_k`；
3. `p_k = p_{k-1} + A(psi_{k-1}) hat_d_k`；
4. `b_g=0, a=0`；
5. `p_K/psi_K` 仍由硬边界固定，不用 GT 初始化内部节点。

求解：

- double precision；
- Gauss–Newton；
- 首版最多 10 次迭代；
- 目标相对下降 `<1e-8` 或最大状态增量 `<1e-7` 时停止；
- 每次记录目标值、最大增量、正规方程条件数和终点约束残差；
- 若目标上升、矩阵非正定或出现 NaN，判定失败并报告，不回退到伪结果。

区间最多几百帧，首版不做稀疏求解器优化。机制成立后再优化速度。

---

## 5. v1a：逐位退化到当前 linear ramp

### 5.1 退化条件

设置：

1. 固定所有 $\psi_k=\hat\psi_k$；
2. 固定 $b_g=0$；
3. 固定 $a=0$；
4. $\Sigma^p_k=\sigma^2I$，所有帧完全相同；
5. 入口、出口位置为硬约束；
6. 不加入其他因子。

由于 $A(\psi)$ 正交：

$$
\|A(\psi)^	op\Delta p-\hat d\|_2^2
=
\|\Delta p-A(\psi)\hat d\|_2^2
$$

因此世界系每帧增量获得相同修正：

$$
\Delta p_k^*=\Delta\hat p_k+rac{1}{K}
\left(p_K^{VT}-\hat p_K\right)
$$

积分后：

$$
p_k^*=\hat p_k+rac{k}{K}
\left(p_K^{VT}-\hat p_K\right)
$$

即当前 `assemble(end_glue=True)` 的线性位置 ramp。

### 5.2 强制验收

在同一输入、同一 `n0/e0` 上比较旧实现与 v1a：

- `max_abs_position_diff <= 1e-6 m`；
- 入口、出口逐位一致；
- 事件外逐位等于 VT；
- 所有事件和所有四折 test 会话都通过。

任一项失败，停止 v1b；不得放宽阈值或改比较口径。

---

## 6. v1b：联合航向、零偏和尺度

v1a 通过后，依次开放：

1. 优化 $\psi_k$；
2. 优化区间共享 $b_g$；
3. 优化区间共享 $a$；
4. 等权各向同性改为 §4.3 的固定物理各向异性协方差。

每一步单独导出结果，禁止四项一次性打开后无法归因。

### 6.1 v1 消融臂

| 臂 | yaw | `b_g` | `a` | 协方差 |
|---|---|---|---|---|
| `LinearRamp` | 固定 | 0 | 0 | 等权各向同性 |
| `SE2-Yaw` | 优化 | 0 | 0 | 等权各向同性 |
| `SE2-YawBias` | 优化 | 区间共享 | 0 | 等权各向同性 |
| `SE2-YawBiasScale` | 优化 | 区间共享 | 区间共享 | 等权各向同性 |
| `SE2-Physical` | 优化 | 区间共享 | 区间共享 | 固定物理各向异性 |

### 6.2 v1 诊断

除遮挡内 ATE 外必须报告：

- 优化前出口位置/yaw 残差；
- 优化后硬边界残差；
- 每事件 `b_g` 与 `exp(a)`；
- `b_g/a` 是否触及先验的极端尾部；
- 世界系与身体系的纵向/横向残差；
- 入口、出口速度跳变；
- 轨迹曲率和帧间速度突变；
- Gauss–Newton 收敛率、迭代次数、条件数；
- 从 raw rollout 与 linear-ramp 两种内部初始化是否收敛到相同解。

横向位移只作按动作/regime 分层诊断，不能把所有真实侧步计为错误。

### 6.3 v1 阶段门

进入 v2 前必须同时满足：

1. v1a 对旧 linear ramp 的逐位退化验收通过；
2. v1b 所有测试事件数值收敛，无 NaN、奇异矩阵或漏事件；
3. `SE2-YawBiasScale` 相对 `LinearRamp` 的 pooled per-session 遮挡 ATE 改善；
4. 改善不是只来自单一动作或单一折；
5. 入口/出口速度跳变没有系统性恶化；
6. `b_g/a` 分布在 train 标定先验下可解释。

若第 3–6 项不过，保留 v1 为负结果，不训练 covariance head。

---

## 7. v2：冻结主干训练 covariance head

### 7.1 学到的量

v2 的 covariance head 输出每帧身体坐标下的过程噪声：

$$
Q_t=\operatorname{diag}
\left(
\sigma^2_{f,t},
\sigma^2_{l,t},
\sigma^2_{\psi,t}
\right)
$$

首版只预测对角项，不预测相关系数。输出通过：

$$
\sigma^2=\operatorname{softplus}(u)+\sigma^2_{min}
$$

并设置固定上界，保证正定和数值稳定。

只有在 NLL 和 held-out coverage 通过后，才能称其为“协方差/不确定度”；此前只能称为权重。

### 7.2 两个输入消融

两个 head 使用完全相同的网络形状：

```text
Linear(44, 64) → GELU → Linear(64, 3)
```

- `IMU-Sigma`：仅保留 `X[:,0:15]`，`X[:,15:44]` 替换为训练均值；
- `IMUT-Sigma`：使用完整 44 维。

两臂初始化、优化器、epoch、batch 和数据顺序一致。禁止一个 head 使用主干 hidden、另一个使用原始特征；
否则 T 信息边界不干净。

### 7.3 条件残差

covariance head 不学习已由区间共享零偏和尺度解释的系统误差。

对每个 train 区间，先用 GT 相对位姿 profile：

$$
(b_g^*,a^*)=
\arg\min_{b_g,a}
\sum_k\|e_k(b_g,a)\|^2
+\frac{b_g^2}{\sigma_b^2}
+\frac{a^2}{\sigma_a^2}
$$

再定义条件残差：

$$
e_k^{cond}=e_k(b_g^*,a^*)
$$

注意：

- `b_g/a` 只从该 train 区间 GT 相对运动拟合；
- 不使用平滑后的内部状态作为 NLL 标签；
- 不使用 test GT；
- 不把 `b_g/a` 从推理图中删除；
- 该步骤只是在 covariance 目标中剥离已显式建模的区间级系统误差。

### 7.4 窗口级 SE(2) NLL

禁止直接把所有逐帧残差当作独立样本做 NLL。

对非重叠固定窗口 $B=[u,v]$，用 SE(2) 预积分 Jacobian 传播逐帧过程协方差：

$$
\Sigma_B=
\sum_{t=u}^{v}J_{B,t}Q_tJ_{B,t}^\top
$$

$J_{B,t}$ 必须包含 yaw 误差对后续平移的影响。窗口条件残差为：

$$
e_B^{cond}=
\begin{bmatrix}
D_B^{GT}-\hat D_B(b_g^*,a^*)\\
\operatorname{wrap}
(\Delta\psi_B^{GT}-\Delta\hat\psi_B(b_g^*))
\end{bmatrix}
$$

训练目标：

$$
\mathcal L_{NLL}=
\frac12\sum_B
\left[
(e_B^{cond})^\top\Sigma_B^{-1}e_B^{cond}
+\log\det\Sigma_B
\right]
$$

窗口长度使用 `20/40/80` 帧。每个 epoch/区间随机选择一个长度并生成不重叠窗口，避免把同一误差通过大量重叠窗口重复计权。

### 7.5 训练纪律

1. 冻结 `HybridTranslation` 全部参数；
2. 冻结 v1 的 `q/epsilon/sigma_b/sigma_a` 统计口径；
3. 只训练 covariance head；
4. 选点只看 val NLL，不看 test ATE；
5. 固定 epoch 版本和 best-val 版本都保留；
6. test 只运行一次预注册评估；
7. 若 NLL 降低但 coverage/ATE 变差，不能宣称不确定度有效。

### 7.6 校准指标

held-out 必须报告：

- NLL；
- Mahalanobis 距离分布；
- 68% / 95% coverage；
- 预测方差与实际平方误差的 Spearman；
- 按动作、遮挡时长和运动 regime 分层的 coverage；
- covariance 特征值最小/中位/p95/最大；
- 触及上下界的比例。

二维平移的参考阈值：

$$
\chi^2_{2,0.68}\approx2.30,
\qquad
\chi^2_{2,0.95}\approx5.99
$$

yaw 首轮单独报告标准化残差；不要把未校准的三维联合 coverage 写成有效结论。

### 7.7 v2 消融与阶段门

| 臂 | 分配方式 | 回答什么问题 |
|---|---|---|
| `SE2-Iso` | 固定等权各向同性 | 联合位置/yaw 本身的收益 |
| `SE2-Physical` | 距离条件化，近静止权重高 | 不学习的物理分配能做到多少 |
| `SE2-IMU-Sigma` | 学习协方差，只看 IMU/yaw | IMU 能否判断局部可靠度 |
| `SE2-IMUT-Sigma` | 学习协方差，看完整 IMU+T | T 是否提供额外可靠度信息 |

允许写“T 通过协方差控制修正位置”的必要条件：

1. `IMUT-Sigma` held-out NLL 优于 `Physical` 和 `IMU-Sigma`；
2. coverage 更接近目标且没有靠大量触及方差上界取得；
3. 四折 pooled 遮挡 ATE 优于两者；
4. 至少不是由单一动作定符号。

任一条件不成立，只保留通过的较简单方案。

---

## 8. v3：触觉事件节点

### 8.1 前置阻塞项

当前压力 onset 与 BVH 足高接触的已知诊断：

- 帧级一致率约 0.84–0.85；
- 匹配上的 onset 中位误差为 0 帧；
- 压力 onset 数约为 BVH onset 的 2.4 倍；
- 主要问题是 chatter/重复触发，不是匹配事件的时间偏移。

因此，v3 在以下项目完成前不得开始：

- 接触进入/退出双阈值滞回；
- 最短支撑时间；
- 最短摆动时间；
- 左右脚交替约束；
- 极短伪周期合并；
- 跑动 flight 后 touchdown 只触发一次；
- 四折 train/val 独立统计事件数、匹配率和时序误差。

发现压力与 BVH 帧数、时间轴或缺失文件不一致时只报告，不重采样补齐。

### 8.2 图结构

节点改为：

```text
VT_in → touchdown_1 → touchdown_2 → ... → touchdown_M → VT_out
```

每个事件因子由其内部所有帧的 T+IMU 预测做 SE(2) 预积分；协方差通过相同 Jacobian 传播。

T 的机制作用变为：

1. 定义轨迹离散结构；
2. 定义 IMU/位移预积分区间；
3. 提供接触状态与 regime；
4. 条件化过程协方差；
5. 后续可触发足端约束，但不在 v3 首版加入。

### 8.3 帧级轨迹恢复

事件节点优化只给出 touchdown 时刻的状态。事件内部帧必须：

1. 保留原始主干逐帧相对形状；
2. 使用优化后的两个事件端点重新积分/插值该步的 SE(2) 修正；
3. 修正按预积分协方差分配，不按时间线性 ramp；
4. 保证事件端点与优化节点逐位一致；
5. 不产生跨事件速度跳变。

不得只画事件折线替代 40 Hz 轨迹。

### 8.4 v3 归因

| 臂 | 节点 | 协方差 |
|---|---|---|
| `Frame-Physical` | 逐帧 | 固定物理 |
| `TNode-Physical` | 触觉事件 | 固定物理 |
| `Frame-IMUT-Sigma` | 逐帧 | 学习 IMU+T |
| `TNode-IMUT-Sigma` | 触觉事件 | 学习 IMU+T |

只有 `Frame → TNode` 在相同协方差下有正增量，才能声称“触觉决定图结构”有效。

---

## 9. 后续项，不提前实现

### 9.1 静止软因子

ZUPT 严格约束的是接触足速度，不是骨盆速度。只有经独立标签确认的 no-step 静止块，才可对根位置加入：

$$
r_{static}=p_k-p_{k-1}
$$

它必须是软因子。禁止对所有 stance 或 double support 强制根不动。

### 9.2 VT 软边界

硬边界机制成立后，再改为：

$$
\|x_0\ominus x_{in}^{VT}\|^2_{\Sigma_{in}^{-1}}
+
\|x_K\ominus x_{out}^{VT}\|^2_{\Sigma_{out}^{-1}}
$$

`Sigma_in/out` 必须来自 crossfit VT 可见块实际误差、可见率和边界拟合残差的 train-only 标定，不能手填到结果变好。

### 9.3 `b_g/a` 随机游走

只有共享变量残差随时间呈系统变化时，才升级为：

$$
b_{g,k+1}=b_{g,k}+w^b_k,
\qquad
a_{k+1}=a_k+w^a_k
$$

增加自由度前必须先证明共享变量模型失配；不能因为优化可实现就提前加入。

---

## 10. 评估协议与主表

### 10.1 数据协议

- 四折 LSO woS13；
- 每折只用 train 估计固定统计量，只用 val 选 covariance head；
- test 不参与超参、先验、阈值或 epoch 选择；
- V7_Cv5 为主基座，V8_Cv5 作基座稳健性复核；
- 场景遮挡使用现有 `full_events`；
- blackout ladder 用 0.5/1/2/4/8 s 作长度压力测试；
- 每个结果同时报 pooled per-session 和 fold-level mean±std；
- 四折 std 大时不作显著性主张。

### 10.2 主臂

| 类别 | 臂 |
|---|---|
| 原始参照 | `Step2Motion`、`S2M-VT`、`VT` |
| 当前方法 | `Hybrid-Raw`、`Hybrid-LinearRamp` |
| v1 | `SE2-Yaw`、`SE2-YawBias`、`SE2-YawBiasScale`、`SE2-Physical` |
| v2 | `SE2-IMU-Sigma`、`SE2-IMUT-Sigma` |
| v3 | `TNode-Physical`、`TNode-IMUT-Sigma` |
| 端点下界参照 | `VT-Hermite`：只使用双端位置/速度，不使用 T+IMU |

`VT-Hermite` 用来证明模型不是只靠两个端点插值。它不是方法臂，不与主方法共享学习参数。

### 10.3 主指标

- 遮挡内 root ATE（主指标）；
- full-session ATE；
- 出口前 raw endpoint error；
- 入口/出口速度接缝；
- yaw RMSE；
- 曲率/速度突变；
- solver failure rate；
- 按动作、折、遮挡长度和 regime 分层结果。

### 10.4 最终主张门槛

完整方法相对 `S2M-VT`：

1. 四折 pooled per-session 遮挡 ATE 更低；
2. 五个动作至少四个更低；
3. 任一动作不得差于 `S2M-VT` 的 1.1 倍；
4. 优于 `VT-Hermite`，证明 T+IMU 提供内部轨迹形状；
5. `Hybrid-Raw` 不因后端训练而退化，保证无出口 VT 时仍可独立工作。

若只过部分门槛，论文只写对应范围，不追加 test 后调参。

---

## 11. S2M 对照边界

不实现 `S2M-VTBridge`。原因不是“S2M 数学上不能进入位姿图”，而是：

- 通用 SE(2) 平滑收益已由 `LinearRamp → SE2-Iso` 测量；
- 触觉图结构收益由 `Frame-Physical → TNode-Physical` 测量；
- 触觉可靠度收益由 `IMU-Sigma → IMUT-Sigma` 测量；
- 把 T 节点和 T 协方差接到 S2M 会改变其输入传感器与方法定义。

论文中使用以下准确表述：

> S2M 维持原始预测器及现有 VT 线性锚定基线。本方法的通用平滑收益、触觉事件拓扑收益和
> 触觉条件化可靠度收益通过内部逐级消融独立测量，因此不需要构造一个改变 S2M 输入定义的扩展方法。

---

## 12. 数据与实现纪律

1. 任何 VT、IMU、压力、GT、事件长度不一致均报告；禁止截断、插值补齐或复制末帧；
2. 任何新路径修改前先用 `rg` 列出读写方；
3. 训练、val、test 的统计量和缓存分开保存并记录来源；
4. 角度统一使用 rad，展示时才转 degree；
5. 世界 x-z、yaw 前向/左向矩阵必须写单元测试；
6. 所有 covariance 保存的是 variance，不是 standard deviation；
7. 所有逆矩阵使用 Cholesky/线性求解，禁止显式 `inverse()`；
8. 验证失败修代码，不放宽阈值；
9. 不修改旧结果文件；新结果写独立目录；
10. 不修改 README，除非用户另行要求。

---

## 13. 实现验收清单

### 13.1 数学单测

- `A(psi)` 与现有 `_assemble_vec` 在随机输入上逐位一致；
- `A(psi)^T A(psi)=I`；
- yaw wrap 跨 `±pi` 正确；
- 解析 Jacobian 与有限差分一致；
- v1a 与 linear ramp `max_abs <= 1e-6 m`；
- hard VT 端点残差在 double precision 下接近机器精度；
- covariance 所有特征值为正且在固定上下界内；
- NLL 的 logdet、Mahalanobis 项与手算小例一致。

### 13.2 合成场景

至少覆盖：

1. 零误差直线：优化前后不变；
2. 常值 gyro bias：恢复正确 bias，yaw 修正随时间线性积累；
3. 常值步长尺度：恢复正确 `exp(a)`；
4. bias + scale 同时存在：位置信息能联合区分；
5. 原地静止：没有出口残差时不得产生位移；
6. 真实侧移：各向异性先验不能把侧移压成直行；
7. `±pi` 转向；
8. 极短区间和近零位移区间；
9. 错误初始化仍收敛或明确报告不可辨识；
10. 故意奇异协方差必须验证失败，不能静默继续。

### 13.3 实数验收原始输出

实现者最终汇报必须包含以下命令的原始输出，不只写结论：

```bash
python -m pytest tests/test_vtbridge.py -q
python results/diagnostics/20261110_vtbridge/vtbridge_eval.py --stage v1a --folds all
python results/diagnostics/20261110_vtbridge/vtbridge_eval.py --stage v1b --folds all
```

v2/v3 命令在对应实现落地后补充，但不得改动 v1a 的逐位退化阈值。

---

## 14. 不做清单

- 不先做大模型或双向 Transformer；
- 不把 VT 端点直接拼进主干后让网络学插值；
- 不用 smoothstep 单独修 yaw；
- 不把 root 在所有 stance/double-support 中设为零速度；
- 不在 v1/v2 使用尚未去抖的 heel-strike 节点；
- 不把只通过 bridge loss 学到的相对权重称为 uncertainty；
- 不为 S2M 实现本文 SE(2) bridge；
- 不根据 test 动作逐项调协方差或先验；
- 不因某个指标变好而跳过轨迹形状和接缝检查；
- 不在机制未过阶段门时继续叠加后续组件。

---

## 15. 论文定位

若 v1–v3 全部通过，方法主张为：

> 视觉提供失效区间两端的稀疏绝对位姿；IMU 提供区间内的方向变化和短时动力学；
> 触觉首先定义可靠的步态事件与预积分结构，其次估计各运动区间的可靠度；
> SE(2) fixed-lag smoother 联合解释航向零偏、步长尺度与局部里程计误差，
> 在视觉重新出现后恢复连续、闭合且与传感器方向一致的遮挡轨迹。

若只有 v1 通过，则只写：

> 双端 VT 约束的 SE(2) 联合平滑优于位置级线性粘合。

若 v2 通过而 v3 未通过，则只增加：

> T 提升了局部里程计可靠度估计。

只有 v3 的固定协方差节点消融也有正增量，才能写：

> T 决定轨迹图的离散结构。
