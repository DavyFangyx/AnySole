# HMR2.0 多视角方法与模型要求

代码位置：`z_note/HMR2.0/4D-Humans`（HMR2.0 官方实现）。

## 1. HMR2.0 模型结构与数据流

单视角 HMR。一句话：**ViT 骨干提视觉特征 → 交叉注意力 Transformer decoder 把 1 个 token 聚合为 SMPL 参数 → SMPL 层解出网格/关节**。

```
输入图像 → 检测+裁剪 256×256 → 切左右各 32px → 256×192（ViTPose 预训练分辨率）
  → ViT backbone（patch16，embed 1280，depth 32，16 头）
  → 空间特征图 [B,1280,12,16] → flatten 192 token（context）
  → decoder：1 个 query token（零初始化）+ 可学习位置嵌入
      6 层 self-attn → cross-attn(Q←token, K/V←视觉特征) → FFN（dim 1024，8 头）
  → token_out [B,1,1024]
  → decpose(144=24×6D) / decshape(10) / deccam(3) 线性读出，残差叠加在数据集均值参数上
  → 6D 旋转转旋转矩阵，拆 global_orient(1)+body_pose(23)
  → SMPL 层（smplx + joint_map 25 关节 + joint_regressor_extra → 46 关节，6890 顶点）
  → 弱透视相机 (s,tx,ty) → cam_t → 2D 重投影
```

关键文件：`hmr2/models/hmr2.py`（前向 `forward_step`）、`hmr2/models/backbones/vit.py`（ViTPose-H ViT，输出空间特征图）、`hmr2/models/heads/smpl_head.py`（decoder 头）、`hmr2/models/components/pose_transformer.py`（decoder 本体）、`hmr2/models/smpl_wrapper.py`（SMPL 层）。

术语口径（对应论文 Fig.1）：

| 论文术语 | 位置 | 性质 |
|---|---|---|
| SMPL query tokens（粉色）| decoder 输入侧：零 token + 嵌入 + 可学习位置嵌入 | 信息≈零，仅位置嵌入可学习 |
| learned SMPL tokens（橙色）| 交叉注意力输出 token_out，MLP 之前 | 每图动态形成的图像条件化表示；「learned」指由可学习层生成，非固定可学习参数 |

HMR2.0 没有 DETR 式的固定可学习 SMPL token；token 内容每张图零起手、经 6 层交叉注意力从视觉特征现读现聚合。

## 2. 论文方法概要

《Monocular Models Are Strong Learners for Multi-View Human Mesh Recovery》（arXiv:2603.20391）：免多视角训练、免标定（可选校准）的多视角 HMR 框架，基于**测试时优化（TTA）**。

核心组件：

- **per-view learned SMPL token 作可优化变量**：前向一次得 token_out 为初值；TTA 时梯度只经最后的 MLP（single linear projection），不回传 ViT/decoder 权重——轻量。
- **virtual view**：额外一组可优化 SMPL 参数 Θ₀=(θ₀,β₀)，由各视角预测经 1-δ 滤波加权平均初始化（剔除离群关节旋转），梯度来自全部视角，优化后为最终统一重建。免校准模式下全局朝向保持视角相关，其余关节全部加权平均。
- **三个测试时损失**（每步梯度下降）：
  - L_2D：44 关节 + 35 解剖表面点重投影（Synthpose 检测器伪 GT）；
  - L_con = Σ_{i≠j} ‖θi−θj‖² + ‖βi−βj‖² + ‖Mi−Mj‖²（跨视角成对一致，可化简为星形 O(N)）；
  - L_reg：锚定单视角先验初值，防漂移。

结果（H3.6M，MPJPE/PA-MPJPE）：Ours(HMR2.0) 校准 31.6/21.5、免校准 45.2/23.0，超过需多视角训练与标定的 HeatFormer（29.5/22.4）。

关键性质：**训练与推理均为多视角**——TTA 要求推理时所有视角在场。这是与本文第 4 节要求的根本差异。

## 3. 论文方法如何应用到 HMR2.0

论文自己给出的接法：HMR2.0 前向一次，每视角取交叉注意力输出 token_out 作该视角优化变量初值；TTA 沿 `token_out → MLP → SMPL 参数 → SMPL 层 → 网格` 路径更新 token_out，虚拟视角聚合各视角估计。

在本项目设定下的转化——**推理只有单视角，多视角 TTA 不可用**，一致性必须从「推理时优化」移到「训练时正则」：

| 论文机制 | 转化 |
|---|---|
| 推理时优化 learned SMPL token | 训练时把一致性梯度反传进共享权重（v_enc/fusion），视角不变性在训练中烤进单分支 |
| virtual view（推理时聚合）| 训练时的 T 一致目标：各视角生成相同的 T / T 编码 |
| L_con 跨视角一致损失 | 训练损失 Σ‖T̂i−T̂j‖（+ 可选小权重 ‖θi−θj‖）|
| TTA 需要全视角在场 | 训练侧多视角、推理侧单视角（view dropout 保证分布一致）|

保留不动的概念：token_out 层（紧凑、图像条件化、MLP 前）是施加一致约束的正确层级；一致锚选在 T 的重建上（T̂），不硬对齐视觉 token——T̂ 一致是最小约束，只消灭 V 表示中「对 T 有分歧」的成分，保留遮挡下视角特有的有用信息。

## 4. 模型要求

- **推理**：任意单视角 V 输入；**无 T、无多视角**（推理条件 = V2M，T 走 null token）。
- **训练**：四视角 V（同帧轴，D1 已统一）+ T（鞋垫压力，作一致性监督）。
- **结构**：共享权重单分支——v_enc / foot_conv / fusion / pose·traj head 全部跨视角共享；每视角一个 T̂ aux 读出（复用现有 L_trec 压力重建头）。
- **训练窗口**：{V_cam1..V_cam4} + T + GT；view dropout 随机采 1–4 视角，一致性损失只在采样到的视角间计算（保证单视角推理落在训练分布内）。
- **损失**：Σ_i (L_pose(θi, GT) + L_trec(T̂i, T_GT)) + λ_cons·Σ_{i≠j}‖T̂i−T̂j‖ + 可选小权重 Σ‖θi−θj‖。
- **两个硬约束**：
  1. T̂ 必须保留对 T_GT 的重建监督——只给一致项会塌缩到常数解；
  2. L_cons 梯度必须反传到共享权重——只喂 aux 头则姿态估计不变。
- **目标场景**：遮挡相机（cam3_occ010–050 系列，现成产物）；干净相机作对照。
