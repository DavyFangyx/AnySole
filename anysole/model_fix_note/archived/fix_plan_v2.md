# AnySole V2 设计方案（F 系列 · v2 代码口径修订版）

> **本文档是给后续 AI 会话的实施手册**。每一步实现完成并验收后，在该步小节末尾回填
> 结果（wandb tag、指标、判据通过/失败）；未回填 = 未完成。
>
> **与 fix_plan.md（已作废）的差异**：原计划按 V1 文档中"BVH 已转 SMPL"的表述编写，
> 使用 SMPL 关节编号 / betas / AMASS 原生对接。**代码实际口径是 BVH-23（Skeleton3），
> 全链路无 SMPL**（`types.py:N_JOINTS=23`、`dataset.py:316` 有 SMPL 字段守卫）。
> 本版全部关节编号、FK、数据管线按代码真实口径重写，并补齐文件级实施细节。
> 原 fix_plan.md 仅作方法论参考（单变量纪律、σ_seed 判据、门控可解释性关卡保留）。

---

## 0. 代码现状速查（实施者必读，所有改动以此为真）

### 0.1 数据与表示（真实口径）

| 事实 | 位置 |
|---|---|
| 动捕 = 23 关节 Skeleton3 BVH，120Hz，YXZ 欧拉（72 通道），重采样到 40Hz | `anysole/data/bvh_io.py:114-131` |
| 目标表示 = 23 关节 6D 旋转 `pose_gt (B,TW,138)`，根轨迹 `trans_m/vel_gt (B,TW,3)` | `anysole/types.py:53-55`、`dataset.py:251-255` |
| FK = 静态 per-session offsets（23×3，米）+ parents + 6D→旋转矩阵，**无 betas 无 SMPL** | `anysole/geometry.py:121-147`、`bvh_io.py:31-33` |
| `kp_gt (B,TW,23,3)` = FK 从 BVH 导出（非测量），世界系 | `dataset.py:215` |
| 关键点损失 L_kp 世界系 FK；接触软标签经 FK→高度/速度→sigmoid | `losses.py:163-166`、`losses.py:47-54` |
| 触觉 = 原生率 CSV（`pressure_left/right.csv`，t_us+48 格）线性插值到 40Hz，`T_raw` 96 维归一化 + `T_phys` 12 维 | `data/pressure.py:87-107` |
| 接触标签 = `contact.npy`（`tactile_abs`）或 Test5 生成的 `contact_<method>.npy`，取列 6:8 | `dataset.py:80-90,257` |
| 窗口 = tw（默认 20）帧 @40Hz，stride 可配（默认=tw 不重叠）；`trans_anchor` = 窗口前 1 帧世界位置；`vel_gt[0]=0`（无前帧差分） | `dataset.py:139-151,309-315,238-242` |
| E6.1 pos 表示 = 22 非根关节根局部位置（66 维，session 第 0 帧朝向系） | `types.py:59`、`dataset.py:229-236` |
| E6.6 s2m50 / no-imu(38) 触觉通道（含合成 IMU，仅作上界参照，不进主线） | `types.py:65-68`、`data/tactile_s2m.py` |

### 0.2 BVH-23 关节与 9 部位划分（本计划全部使用）

`types.py:73-124`（JOINT_NAMES / JOINT_PARENTS 已冻结，bvh_io 有严格校验）：

| 部位 | 关节（BVH 索引：名称） | 每帧输出 |
|---|---|---|
| 根 | 0 Hips | tilt 6D（去 yaw）+ 轨迹 4D（ψ̇、朝向系水平速度 2、根高 h） |
| 躯干 | 1 Spine, 2 Spine1, 3 Spine2, 4 Spine3 | 24 |
| 头颈 | 5 Neck, 6 Head | 12 |
| 左臂 | 7 LShoulder, 8 LArm, 9 LForeArm, 10 LHand | 24 |
| 右臂 | 11 RShoulder, 12 RArm, 13 RForeArm, 14 RHand | 24 |
| 左腿 | 15 LUpLeg, 16 LLeg | 12 |
| 右腿 | 19 RUpLeg, 20 RLeg | 12 |
| 左脚 | 17 LFoot, 18 LToeBase | 12 |
| 右脚 | 21 RFoot, 22 RToeBase | 12 |

合计 23 关节 × 6D = **138 维（= 现有 POSE_DIM，恰好吻合）**，加轨迹 4、接触 2、压力 48（V2M 临时输出，F9 替换）。
关键性质：**BVH 的 1–22 号关节旋转是父系局部量，天然 yaw 不变**——F2a 的朝向规范化
只动 Hips 根旋转与轨迹表示，其余关节表示完全不动。

### 0.3 模块清单（改动落点）

| 文件 | 职责 |
|---|---|
| `anysole/types.py` | 冻结常量；BATCH_SHAPES 守卫 |
| `anysole/data/bvh_io.py` | BVH 读写/重采样/6D 转换（已带层级严格校验） |
| `anysole/data/dataset.py` | session 构建 + 窗口；stride 已支持 |
| `anysole/data/pressure.py` | 原生 CSV → 40Hz T_raw/T_phys |
| `anysole/geometry.py` | 6D↔旋转矩阵、FK（numpy/torch）、positions_to_6d_np（往返 0.00mm 已验证） |
| `anysole/models/embeddings.py` | time PE / modality / null token / timestep MLP |
| `anysole/models/encoders.py` | V/T 编码（LinearTemporalEncoder）+ null 替换 |
| `anysole/models/fusion.py` | 4 层融合 Transformer（40 token） |
| `anysole/models/pose_head.py` | 6 层 MDM decoder（t-token prepend、3 组投影、set_stats 标准化） |
| `anysole/models/traj_head.py` | 2 层 encoder，F 池化 → 速度 → cumsum |
| `anysole/models/aux_heads.py` | T_rec(96) / V_rec(2051) |
| `anysole/losses.py` | compute_losses + soft_contact_from_keypoints + traj 多尺度 |
| `anysole/diffusion.py` | GaussianDiffusion（q_sample/DDIM/续写）；**v2 不再调用，保留兼容旧 ckpt** |
| `anysole/train.py` | 训练循环（τ 采样在 596–623 行）、_evaluate 面板、fit_pose_stats |
| `anysole/eval.py` | 指标 + BVH 导出 + crossfade(FADE=4, 590–605 行) + continuation 分支 |

### 0.4 基线数字（F0a 协议建立后全部重测，下表为历史参照）

| 口径 | 数字 | 出处 |
|---|---|---|
| E3 满训(800ep)：VT2M MPJPE / PA-MPJPE / root_ate / contact_acc | 141.8 / 81.2 / 0.043 / 0.048 | model_fix_note.md:31-33 |
| E3 one-shot g(F)：VT / V / T | 111.4 / 119.0 / 154.0 mm | E6x_series_plan.md 表 |
| E3 50 步链：VT / V / T | 127.7 / 133.5 / 170.1 mm | 同上 |
| ridge-on-F 探针（线性读出天花板） | 85 mm | model_fix_note.md:88 |
| E3 窗口内抖动 / 拼接跳变（crossfade 前） | 116 mm/帧 / 213 mm | model_fix_note.md:34-35 |
| 接触标签 tactile_abs 假接触率（Test5） | 83.4% | Test5 记忆 |
| E6.1 pos 表示 2ep PA-MPJPE ≈ 90mm（6D 满训 97.8） | — | E6x_series_plan.md |

### 0.5 已诊断失败模式（每一步的判据设计都针对它们）

1. **欠训练混杂**：tw=20/stride=20 只有 ~5 步/epoch，200ep≈1000 步（Test12 诊断为欠训练）。
   → 短预算按**梯度步数**定（见 §2）。
2. **均值塌缩/假稳定**：head 学成 g(F)≈常量（yg01hyju）或 echo 输入噪声（E5 echo 0.293）。
   → 去扩散后该模式 = "head 读不出 F"，用 **ridge-on-F 探针每步必跑** 卡住。
3. **链负贡献**：one-shot 111.4 → 链 127.7（+16mm），抖动 ×2。→ F0b 去扩散的根因。
4. **指标涨但机制坏**（echo 0.96 的 pose head、L_con 爆炸主导总损失）。
   → 门控可解释性关卡 + 损失量级逐项记日志。
5. **接触标签缺陷**：tactile_abs 83.4% 假接触；Test5 144 session pooled（vs bvh_h）：
   tactile_abs fc=0.906、触觉自适应系 fc≈0.73–0.78/fa≈0.20–0.33、bvh_soft fa=0.177（滑步帧）；
   且 **bvh_h 自身失效**：台阶假离地（S10113）+ 低净空摆动假接触（S11，接触率 1.00）。
   → F6 标签 = 运动学状态机 + 压力证实（见 F6）。

---

## 1. 终态结构（F9 完成后）

```
V：cam3 帧 → 预训练 HMR（离线提取，GVHMR 首选）→ 逐帧 θ_hmr / 2D kp+置信度 / 图像 token / bbox+相机
     → V 流：9 部位 token + 1 全局 token          E_V (B,T,10,d)
T：左右鞋垫 4×12 @原生率 → 按脚卷积（左右共享、右脚镜像）→ 时间卷积降到 40Hz → 时间 Transformer
     → T 流：左脚 / 右脚 / 全局 token              E_T (B,T,3,d)
   两个流各自带深监督头（只读本流）和部位不确定度 σ_{m,p,t}

部位解码器 Z (B,T,9,d)，每层依次为：
   部位间自注意力 → 时间自注意力 → 局部交叉注意力(V)/(T) → 门控 {V, T, ∅} → FFN
   输出：部位 6D 旋转（BVH-23 父系局部量 + 根 tilt）/ 根轨迹 4D / 脚接触 / 融合不确定度 σ̂

精修器 R（AMASS 预训练，逐 token 噪声级）：按 σ̂ 决定每个部位加噪多深，然后少步去噪
前向模型 G：M → 压力 / 接触；用于 V2M 的压力输出、一致性损失、AMASS 伪压力
```

输出空间始终是 **BVH-23（Skeleton3）**：6D 旋转 + 轨迹，用每会话 offsets/parents 做 FK，
用 `bvh_io.write_bvh` 导出。任何 SMPL 只存在于 F1 的 HMR 特征侧，绝不进入模型输出。

---

## 2. 实验规范（所有步骤共用）

**判据阈值用种子噪声定。** F0 用 **3 个种子**测出每项指标的标准差 σ_seed。
- "显著改善"：比上一步好 2σ_seed 以上；"不劣化"：差距 ≤1σ_seed；落在 1–2σ 之间补跑一个种子。

**预算按梯度步数对齐（不按 epoch）。**
- 短预算 = **3700 步**（≈ E3 满训 800ep@stride20 的等价量），胜出配置再跑长预算（2×短预算）。
- tw=20/stride=20 时 3700 步 ≈ 800 epoch；回归训练单步更便宜，可承受。
- 若墙钟紧张，可用 `--stride 10`（密度×2，dataset.py 已支持）提速，但**必须在记录中标注口径**。

**损失与数据。**
- 每项加权损失的量级逐项记日志（wandb train/loss_* 已有）。新损失项初始权重按"首个 epoch
  加权后 ≈ 主损失 L_pose 的 10–30%"设定——吸取 E1/E2 的 L_con 主导教训。
- train/val 按受试者划分（splits.csv 已如此），并确认与 AMASS 等外部数据无受试重叠。

**单变量纪律。** 每步只动一个变量；新开关默认关闭（关闭 = 上一步行为）；失败即弃用该开关、
回退上一步配置继续，失败的 checkpoint 留档、代码不回改（开关默认关）。

### 评估协议（F0a 实现，之后固定不变）

| 类别 | 指标 | 说明 |
|---|---|---|
| 局部姿态 | MPJPE、PA-MPJPE（23 关节，骨盆对齐） | 按 9 部位分报；汇总上半身（1–14）/下半身（15–22）；踝脚（17/18/21/22）、手（10/14）单列 |
| 全局 | W-MPJPE（每 4 s 段首帧对齐）、RTE（按位移长度归一化）、朝向漂移（累计 yaw 误差） | |
| 时序 | 抖动 mm/帧（probe_jitter 口径）、加速度误差、拼接跳变 | |
| 接触 | 接触 F1、接触期脚滑 mm/帧（train._evaluate 已有同款计算，eval.py 补齐） | |
| 压力输出 | 总力 R²、CoP 误差、逐点 Pearson | 仅 V2M |
| T2M 上半身 | 照报 MPJPE，另加加速度分布误差、关节限位违规率；F8c 起加 min-of-K 与 APD | 上半身基本不可观测，不单看 MPJPE |

**实现落点**：扩展 `anysole/eval.py` 的 `_evaluate_one`（或新增 `anysole/eval_protocol.py` 复用
其 MetricSums 框架），新增 per-part 聚合与上述指标；导出协议 JSON 不变（`metrics/<split>.json`），
另存 `metrics/<split>_fseries.json` 给回填表用。
**评估方式**：val 每条序列完整推理后算指标，拼接方式随当前步骤（F3 起 = Hann 重叠混合）。
**ridge-on-F 探针每步必跑**（`z_note/` 已有探针逻辑，收敛进 `results_display/script/` 正式化）。
**鲁棒性验证集**：val 上合成两种退化——V 连续 20–40% 帧丢失、T 连续段丢失。F0 起报告，F7 起作主判据。
**参照基线（F0a 一次性建立）**：
- E3 ckpt（`results/AnySole/E3_base/checkpoints/ckpt_last.pt`；不存在则按 E6x 手册步骤 0 复跑）过新协议；
- **S2M-P**：Baselines/Step2Motion 去掉 IMU 通道重训（每脚 pressure16 + force1 + CoP2，共 38 维；
  38 维通道构建已在 `anysole/data/tactile_s2m.py --no-imu` 实现，S2M-P 是其 Step2Motion 侧对照）；
- 零训练 HMR：F1 提取特征后补上；E6.6a/b（含合成 IMU）标注"含合成 IMU"，只作上界。

---

## 3. 主线步骤

### F0 基线重建（起点 = E3 配置）

**起点**：`configs/v1.yaml` 的 E3 值（raw108、tw=20、stride=20、λ_pose=3 / λ_kp=1 / λ_traj=1 /
λ_trec=0.1 / λ_vrec=0.1 / λ_con=0、cosine-1000、noise_scaled=false、lr 恒定、joint_and 接触法）。
理由：E3 是最干净的已验证配置（无合成 IMU、接触项已关、t-token 已修好）；E4/E5 的噪声缩放与
linear schedule 已被证伪（踢腿），E6.6 含 IMU 泄漏。

**F0a 评估协议与基线。** 模型不改，按 §2 实现协议。

- 改动文件：`anysole/eval.py`（per-part 聚合 + 新指标）、新 `anysole/eval_protocol.py`（如需）、
  `results_display/script/ridge_probe.py`（正式化 ridge-on-F）。
- 验收：E3 ckpt 在新协议下出全套数字（含 3 种子 σ_seed、鲁棒性验证集、ridge 探针）。

**F0b 去扩散，改为回归**（tag `f0b_regress`，3 个种子）

- 改动：
  1. `anysole/models/pose_head.py`：加 `head_mode: "diffusion"|"regress"`。regress 模式下
     `__init__` 把 `proj_body/proj_left/proj_right` 换成可学习 query（`nn.Parameter`
     `(1, tw, n_group, dim)`，std=0.02 初始化）；`forward(self, F)` 删 `x_tau/tau` 参数与
     `t_tok` prepend（`timestep_token` 不用），token = query + time_pe + group_emb，直接过
     6 层 decoder（自注意力 / 交叉注意力到 F / FFN 不动）→ `_unembed` → 反标准化（set_stats 保留）。
  2. 新 `anysole/models/model_v2.py`：`AnySoleModelV2`（modal 名 `anysolev2`，注册进
     `models/__init__.py` MODEL_NAMES）。F0b 版 = V1 结构原样（encoders/fusion/traj/aux 同
     `model.py`），仅 pose_head 用 head_mode="regress"，forward 签名去掉 x_tau/tau。
  3. `anysole/train.py`：`head_mode=="regress"` 分支跳过 τ 采样与 q_sample（596–623 行），
     直接 `model(v_feat, t_raw, t_phys, config_id, ...)` → compute_losses（losses.py 不动，
     L_pose 直接作用在回归输出上）。`_evaluate` 同分支：删 ddim_sample_loop，单次前向。
  4. `anysole/eval.py`：`modal=="anysolev2"` 分支单次前向；E6.4 continuation 分支对回归模型
     失效，跳过（F3 用 Hann 混合替代）。crossfade（FADE=4）F0b 保留。
  5. `anysole/infer.py`：同样分支。
  6. `anysole/diffusion.py` 不删不改（旧 ckpt 评估兼容）。

- 不动：输入、融合、轨迹头、辅助头、全部损失权重、数据管线。
- 单测：新 `z_note/smoke_f0b_regress.py`——前向形状 (B,tw,138)、loss 回传、与 E3 同配置下
  `--modal anysolev1`（diffusion）行为逐参数一致（regress 关 = 旧行为）。
- 判据：
  - 通过：VT2M PA-MPJPE 与 E3（81.2mm）差距 ≤2σ_seed，与 ridge-on-F（85mm）同量级；
    抖动明显低于 E5 链的 27mm/帧。
  - 若明显差于 E3：实现问题（标准化/损失口径），先排查，不进 F1。

**F0c 去掉 V 重建头**（`--lambda-vrec 0`，CLI 已有）

- 改动：训练命令加 `--lambda-vrec 0`（aux_heads.v_rec 与 L_Vrec 自然停更）；判据通过后从
  `model_v2.py`/`aux_heads.py`/`losses.py:138-143` 永久删除 V 重建路径。
- 判据：T2M 不劣化 → 永久删除。T2M 明显变差 → 说明 V 重建确实在向 F 灌视觉信息；
  F1 起保留替代目标（重建 θ_hmr 而非 HRNet 特征），留到 F5。

### F1 视觉输入换成预训练 HMR

目的：验证"视觉读出天花板"是不是 V1 的第一瓶颈（PA-MPJPE 卡 81.2 ≈ ridge 85）。

- **候选模型**（仓库内已有）：`Baselines/Video2Motion/GVHMR`（首选）、SMPLest-X、WHAM、VIBE、ROMP。
  原计划的 CameraHMR / SAM 3D Body 不在仓库，不引入新依赖。
- **改动**：
  1. 新 `anysole/data/extract_hmr.py`（离线，参照 `extract_hrnet` 的模式）：对 cam3 每个 40Hz
     帧、同一 bbox、按所选模型预处理，保存：θ_hmr（SMPL 24 关节局部 6D + global_orient）、β_hmr、
     相机参数、2D 关键点+置信度、主干池化 token、bbox_info。缓存目录
     `AnysoleWorkspace/derived/AnySole/hmr_cache/<model>/cam3/<session>.pt`。
  2. **零训练基线（先于训练）**：新 `z_note/probe_f1_zero_hmr.py`——θ_hmr+β_hmr 经 SMPL FK 出
     世界关节，与 `kp_gt`（BVH-23）按**语义名匹配关节子集**算逐帧 PA-MPJPE。匹配表（写死在
     探针里）：Hips↔pelvis、Left/RightUpLeg↔left/right_hip、Left/RightLeg↔left/right_knee、
     Left/RightFoot↔left/right_ankle、Left/RightShoulder↔left/right_shoulder、
     Left/RightArm↔left/right_elbow、Left/RightForeArm↔left/right_wrist、Head↔head
     （共 15 对；Spine*/Neck/ToeBase/Hand 不参与比较）。只有胜出的 HMR 模型进训练。
  3. **V-Enc**（`encoders.py` 新增 `VEncHMR`）：分组 Linear（θ、关键点、图像 token、bbox+相机）
     求和 + 时间 PE + 模态嵌入；规模与 V1 相同，无流内时间模块（保单变量）。
  4. `dataset.py`：v2 modal 时读 hmr_cache 替换 hrnet_cache；V 特征维度改为运行时配置
     （`BATCH_SHAPES` 的 V_FEAT_DIM 守卫按 config 放宽，v2 专用路径）。
  5. 质量特征 q_V：每帧关键点平均置信度 + bbox 截断比例，并入输入。
- tag `f1_hmr_<model>`。判据：
  - 通过：V2M PA-MPJPE 比 F0 显著下降，且不高于零训练 HMR 的逐帧值；新 F 上 ridge 探针
    天花板同步下降。
  - V2M 仍高于零训练 HMR → 融合/读出丢信息：追加 F1b（V 可用时姿态头预测相对 θ_hmr 的残差，
    T-only 时残差基底取 0）。
  - 与 F0 持平 → 视觉读出不是主瓶颈，重心转 F3/F4；之后查 HMR 域差（服装/遮挡/鞋垫连线）。

### F2 运动表示 + FK 位置损失

**F2a 表示重构**

- 改动：
  1. **关节保持 23（BVH-23）不变**9 部位划分见 §0.2。
     可选变体（非默认）：去 10/14 手末端 → 21 关节，不推荐（会动 kp_gt/FK/导出全链）。
  2. **根拆分**。重力轴 = mocap 世界系竖直轴（BVH Y 轴向上；实现时用单测确认：GT 垂直速度
     均值≈0、水平位移 ≫ 垂直位移）。朝向角 ψ = Hips 前向量投影地面的 atan2（前向轴取
     +X/+Z 哪个由单测定：与水平速度相关性高者）。tilt = R_yaw(ψ)ᵀ·R_root，6D 表示归入姿态
     （替换 pose_gt 的根 6D）；1–22 号关节局部 6D 不动。
  3. **轨迹 4 维**：ψ̇、朝向系水平速度 2 维、根高 h。差分用窗口前一帧（复用
     `dataset.py:309-315` 的 trans_anchor 机制），修掉 `vel_gt[0]=0`；整条序列首帧复制下一帧值。
  4. 窗口内从 ψ=0、水平位置 0 积分出窗口局部轨迹；窗口锚点（ψ、水平位置、h 的世界基准）
     单独保存，供 W-MPJPE/拼接/导出。
  5. 所有维度 mean/std 标准化（复用 `fit_pose_stats` + `set_stats`，训练集拟合、冻结随 ckpt）。
  6. 姿态头输出 138 维（根 6 = tilt）、轨迹头输出 4 维。轨迹损失沿用 V1 速度 MSE + 多尺度位移
     （`losses.py:_trajectory_losses`），位移改在积分后的朝向系轨迹上。
- 数据侧：`dataset.py` 新增 `pose_gt_f2`（根 6 换 tilt）与 `traj_gt_f2 (TW,4)` 字段；
  `geometry.py` 新增 `f2_to_world(pose_f2, traj_f2, offsets, parents) -> (world_pose_6d, world_trans)`
  （先积分 ψ/traj，再 R_yaw(ψ)@tilt 恢复世界根旋转），L_kp 与导出共用。
- **单测（训练前必过）**：`z_note/smoke_f2_roundtrip.py`——GT → f2 表示 → f2_to_world →
  与 GT pose/trans 误差 ≈0（整窗累计漂移 <1mm；参照 positions_to_6d_np 的 0.00mm 口径）。
- 判据：根 RTE、朝向漂移、W-MPJPE 显著改善，至少不劣化；MPJPE 不劣化。

**F2b FK 位置损失**

- 改动：新增 L_fk = 23 关节 FK 位置与 `kp_gt` 的 MSE（`f2_to_world` 后与 kp_gt 比较，本质是
  现有 L_kp 的同一公式——**L_kp 与 L_fk 二选一保留，默认保留 L_kp 并改名/复用**，避免双份；
  实现上：L_kp 世界系监督继续存在（E7 教训：几何监督不能砍），F2b 把它的权重提到
  λ_kp=2–3 并单测其增益）。
- 注意：叶子关节（10/14/18/22 手与脚趾）旋转不影响关节位置，只能靠旋转损失监督，L_pose 必须保留。
- 判据：踝脚/腕误差显著下降，PA-MPJPE 不劣化；整体 MPJPE 反升 → 下调权重重跑。

### F3 窗口长度 + 重叠推理

- 改动：
  1. tw ∈ {80, 160} 与 20 对照（`--tw` CLI 已有；`SharedEmbeddings` 按 tw 重建 PE 已支持）。
     训练 stride = tw/2（`--stride` CLI 已有），每 epoch 随机偏移起点（可选，非必须）。
  2. 推理滑窗 stride = tw/2，Hann 窗 w(t)=sin²(π(t+½)/tw) 加权（50% 重叠权重和恒 1）：
     - 旋转：矩阵空间加权 + Gram-Schmidt 重正交化（复用 `eval.py:606-614` 的做法）；
     - 轨迹：只混合每帧 4 维速度量，整条序列一次性积分（避免各窗锚点不一致）；
     - 序列两端按权重归一化。
  3. 删除 v2 modal 的 4 帧 crossfade 分支（`eval.py:590-605`，FADE=4），由上条取代。
- 不动：位置编码（换 RoPE 后续再说）、模型结构、损失。tag `f3_tw80` / `f3_tw160`。
- 判据：抖动、加速度误差、拼接跳变显著下降（跳变应接近 GT 相邻帧差 ~8.7mm/帧）；
  T2M 下肢与根 RTE 改善；VT2M 不劣化。两窗长接近时优先 T2M 更好的那个。

### F4 触觉流独立化

**F4a 纯压力、原生率、按脚编码**

- 改动：
  1. **重采样**（新 `anysole/data/pressure_native.py`，复用 `load_pressure_csv` 读原生率）：
     左右脚按 t_us 各自插值到统一高频 f_p = r·40Hz（原生率 ≥100Hz 取 r=3，否则 r=2），
     对齐区间与视频相同。
  2. **归一化**：每 session 一个尺度 s = "有力帧"上两脚总力之和的中位数（有体重/标定则除体重）；
     逐点除以 s。
  3. **每脚特征（原生率）**：4×12 压力图；总力、CoP 2 维（复用 `cop_from_grid`）、接触面积比、
     dF/dt；质量特征 q_T（饱和点比例、死点比例）。
  4. **空间编码**（新 `anysole/models/foot_encoder.py`）：左右共享权重、右脚沿内外侧镜像。
     Conv3×3(1→32) → Conv3×3(32→64, 沿长轴 stride 2) → Conv3×3(64→64) → 展平 → Linear→128，
     与手工特征拼接 → Linear → d，加脚别嵌入。**先写单测确认网格方向**（`z_note/smoke_f4a_grid.py`）：
     脚跟着地时 CoP 应从脚跟移向脚尖（沿 `pressure.py:60` 的 heel_to_toe 定义）。
  5. **时间编码**：每脚 Conv1d(k=5, stride=r) 降到 40Hz；每帧组成 [左, 右, 全局=Linear(左‖右)]
     3 个 token，4 层时间 Transformer → E_T (B,T,3,d)。
  6. **融合接口**：每帧 3 token 拼接过 Linear → 1 个 T token (B,T,d)，替换 raw108 投影
     （`encoders.py` t_enc 换 FootConvEncoder；tactile_input 新选项 `pressure_native`）。
  7. **T 重建目标**：归一化 96 维（原生率池化到 40Hz；`aux_heads.t_rec` 维度不变）。
- 不动：V 路、融合、各头、损失权重。开关 `--tactile-input pressure_native`，tag `f4a_tstream`。
- 判据：T2M 下肢 MPJPE、根 RTE、接触 F1 优于 F3 的 raw108；下肢误差不差于 S2M-P；VT2M 不劣化。

**F4b 流内深监督**（tag `f4b_ds`）

- 改动：
  1. T 深监督头（只读 E_T）：左右脚 token → 接触 logit（2 维）；全局 token → 腿/脚 8 关节
     （15–18, 19–22）6D + tilt + 4 维轨迹。
  2. V 深监督头（只读 V token）：22 关节（1–22）6D。
  3. 损失：标准化空间 MSE；接触 BCE（本步用 f6_soft 标签，F6a 脚本可提前生成，见 F6 的标签说明）；
     λ_ds 按量级规则设（首 epoch 约主损失 30%）。
  4. 流可用即计算其深监督。
- 判据：VT2M 相对 V2M 在脚/腿/根上的增益显著扩大（抗稀释的直接证据）；T2M 不劣化；
  探针：VT 配置下 T 编码器与 V 编码器梯度范数之比相对 F4a 上升（测法同 E6.6b）。

### F5 部位查询解码器 + 门控

- 改动：
  1. **V 流部位化**（`encoders.py` VEncHMR 扩展）：θ_hmr 按 §0.2 部位分组（SMPL 侧关节按 F1
     语义匹配表归入 BVH 部位）；2D 关键点按关节归部位，躯干/根用肩髋中点；每部位 Linear → d；
     图像 token+bbox+相机组成全局 token。E_V (B,T,10,d)，加部位嵌入 + 时间 PE。
  2. **新 `anysole/models/part_decoder.py`**：`PartDecoder`（d=256、8 头、6 层、Pre-LN、GELU），
     Z 初始化为部位嵌入 + 时间 PE。每层：
     - 部位间自注意力（9 部位，运动树跳数偏置：每头每跳数 0–4 一个可学习标量；树 = JOINT_PARENTS）；
     - 每部位沿时间自注意力；
     - 局部交叉注意力：query=Z[t,p]；V 的 K/V = E_V[t−4..t+4]（90 token），T 的 K/V = E_T[t−4..t+4]
       （27 token），各带相对时间嵌入（实现：shifted gather + 掩码，非全局）；
     - 门控：g = softmax(ℓ_V, ℓ_T, ℓ_∅)，ℓ_m = MLP_m(Z[t,p]) + b_{p,m}，ℓ_∅=0，缺失模态 ℓ=−∞。
       Z += g_V·e_V + g_T·e_T。b_{p,m} 初始化：脚/腿/根的 T 取 +1，躯干/头/臂的 V 取 +1，其余 0；
     - FFN。
  3. **输出**（替换 v2 的 fusion/pose_head/traj_head/aux_heads 调用点，`model_v2.py` 重写 forward）：
     各部位线性头 → 该部位关节 6D；根 token → tilt + 4 维轨迹；脚 token → 接触 logit
     （T 可用时加 g_T×T 流接触 logit）；脚 token → 48 维压力（V2M 临时输出，F9 替换）。
  4. **增强丢弃**（新 `anysole/data/augment.py`）：VT/V/T 类别采样之外再加——模态段丢弃
     （概率 0.3，丢连续 10–40% 帧，该段门控 ℓ=−∞，流内可学习 mask token 填充）；
     V 部位遮挡（随机部位 kp 置信度置 0 + θ 加噪）。
  5. 训练不稳则 AdamW 3e-4 + warmup + grad clip 1.0，并同优化器重跑 F4b 对照。
- tags：`f5_part9`（主线）、`f5_part3`（Step2Motion 三块：左腿含脚/右腿含脚/其余含根）、
  `f5_nogate`（单一交叉注意力读拼接 E_V‖E_T）。**跑序：先 f5_nogate 验证部位 query 结构
  本身不劣化，再上 f5_part9**——part9 失败时能定位是结构还是门控。
- 判据（**先看门控探针，再看主指标**；门控不可解释时主指标提升不采信）：
  - 门控探针：T-only 时手臂 g_∅ 明显高于腿/脚；VT 时脚部 g_T 在支撑相高于摆动相。
  - f5_part9 相对 F4b：VT2M 手臂不劣化；VT2M−V2M 在脚/腿/根上的增益不小于 F4b；T2M 下肢不劣化。
  - 对照：part9 在 T2M 根轨迹或 VT2M 手臂上优于 part3；鲁棒性验证集上优于 nogate。

### F6 接触与滑步

**F6a 标签体系（零训练改动，先于一切；吸收 Test5 144 session 实测 + 用户 gif 复核）**

Test5 对比（pooled，参考 bvh_h）：tactile_abs fc=0.906；tactile_gmm/rel/pat_offset
fc≈0.73–0.78 且 fa≈0.20–0.33（GMM 谷值会切进安静站立段；pat_offset 用摆动相压力和的
**中位数**做基线，重尾残余压力压不住）；bvh_soft fc=0.000 但 fa=0.177（速度项把贴地滑步帧
判成离地）。**2026-09-18 用户复核推翻 bvh_h 本身**（gif 佐证 + 信号复算，两处独立失效面）：
- 上楼梯（seq11，S10113）：脚踩台阶 h=29–30cm 静止承重（压力和 2239/9966），bvh_h 因
  h>5cm 判离地 → **假离地**；
- 低净空受试者 S11 的走路/跑步/开合跳（S11032/53/73）：摆动相 h 仅 2–4cm，4/6cm 迟滞永不
  触发 → bvh_h 接触率 1.00（全程红）→ **假接触**；摆动相速度实为 2.2–2.6 m/s。

结论：高度判据在"非平地"与"低净空"两类场景系统性失效，**任何单一信号都不是 GT**（触觉
测承重不测触地、高度盲于台阶/低净空、速度盲于滑步）。F6 标签改为**运动学状态机为主锚 +
压力证实 + 高度仅作空中否决**。2026-09-18 预验证（4 反例 session 模拟 + 全 144 session
扫描，seq 按 session 名 `S<subject><seq2><trial1>` 解析）：
- 4 反例翻转：S10113 台阶静止承重段（h=29–30cm）→ 接触；S11032/53/73 摆动相 → 离地
  （接触率 0.61/0.44/0.57，bvh_h 为 1.00）；
- 各 seq 平均接触率与生物力学先验一致：01 站立 1.00、03 走 0.62±0.02（n=20）、05 跑
  0.47±0.03、07 开合跳 0.60±0.06、11 楼梯 0.69±0.04、06/10 1.00——同 seq 跨受试者
  std 很小，全局常数参数成立，无需逐 session 拟合。

1. **motion_f6（状态机，硬标签，40Hz，每脚独立）**：
   - 位置 3 帧中值滤波后算脚速 v（m/s）与竖直加速度 a_z（m/s²，中心差分）；
   - **接触→离地**：v > V_HI 连续 2 帧。**压力永不触发离地**——推离期压力先掉 0.4–0.7s
     是语义差；几何触地由"脚开始移动"定义，天然对齐脚尖离地；
   - **离地→接触**：(v < V_LO 且 |a_z| < A_THR 连续 2 帧) **或** (pressure_f6 承重 且
     v < V_HI)。|a_z| 门挡住跳跃顶点（v 瞬时≈0 但 a_z≈g）；压力触发把触地时刻提前到
     承重响应期（比"脚停住"早 100–150ms，步态事件检测的标准做法）；
   - **高度否决**：压力不承重 且 h ≥ 20cm → 拒绝接触（空中悬停脚/跳顶；台阶承重走压力
     分支通过，不受影响）；
   - 初值 = 接触；参数为全局常数 V_LO=0.3 / V_HI=0.6 / A_THR=3.0 / H_HIGH=0.20 m
     （非逐 session 拟合，对 held-out 受试者天然成立）。
2. **pressure_f6（承重，逐 session 标定）**：
   - 逐格基线 baseline_cell = 该格离地帧值的 90 分位（离地帧由 motion_f6 自举：先用
     v>V_HI 帧作种子标定 → 跑状态机 → 用最终离地帧重标定一次）；
   - corrected = Σ max(0, cell−baseline)；thr_lo = max(80, 0.2·stance 中位)、
     thr_hi = 1.5·thr_lo 迟滞 + 2 帧确认；
   - 离地帧 <10 的 session（拖步患者）退化为绝对阈值并记日志（该类接触率 1.0 正确，勿修）。
3. **f6_soft（融合软标签）**：硬值 = motion_f6；软值按 pressure_f6 一致与否：

   | motion_f6 | pressure_f6 | 软值 | 语义 |
   |---|---|---|---|
   | 接触 | 承重 | 0.95 | 全一致 |
   | 接触 | 不承重 | 0.70 | 卸载触地（双支撑推离期/极轻触） |
   | 离地 | 不承重 | 0.05 | 全一致 |
   | 离地 | 承重 | 0.30 | 摆动相残余压力（校准后应罕见，量化记录） |

   落盘 `contact_f6_soft.npy`（cols 6/7 软值）+ sidecar JSON（基线/阈值/一致率/事件时差）；
   `dataset.py:88-90` 机制加 `contact_method="f6_soft"`。实现为
   `results_display/script/f6_contact_labels.py`：motion_f6/pressure_f6/f6_soft 注册进
   `contact_methods.py` 的 METHODS，复用其 comparison/diff/动画框架（f6_soft 软值列绕过
   二值比较，另写软值汇总）。**bvh_h/bvh_soft 保留为诊断量**：与 motion_f6 的差异清单
   分别量化"高度判据失效帧"（台阶/低净空）与"滑步帧"。
4. **标签验收门（先于任何训练，全 144 session）**：
   - **用户 4 个反例必须翻转**：S10113 台阶静止承重段 → 接触；S11032/53/73 摆动相 → 离地
     （gif 复核）；
   - 无标签物理自检（只读 GT BVH）：接触帧脚速/竖直加速度 ≈0、离地帧 v ≫ V_HI（构造成立，
     仅查实现）；健康步行者接触率 0.55–0.7、摆动相 0.35–0.45s；
   - 压力互证（非构造成立的真实跨模态证据）：接触帧中 pressure_f6 承重占比 ≥85%
     （健康平地动作）、离地帧中不承重占比 ≥90%（校准有效性）；
   - 事件级：触地时刻（压力触发）vs 脚停时刻（速度触发）中位 |Δt| <150ms；离地时刻（速度
     触发）vs 压力卸载时刻（pressure_f6）中位滞后 0.4–0.7s 且跨受试者稳定——语义差特征，
     sidecar 报告而非硬门；
   - 按 seq×受试者分组报接触率（13 类动作全覆盖），肉眼抽查每 seq 代表 session；
   - 拖步患者（S7063/S8063/S11063）接触率保持 ≈1.0。
   - **已知边界（诚实记录，不追求零误差）**：速度型判据的固有盲区——① 快速滑步
     （v>0.6m/s 的推拉拖步）会判离地，慢速拖步不受影响；② 低悬停脚（h<20cm、v<0.3、
     无压力，如单脚站立时悬空的脚）会判接触，如 S6021 左 0.72——按 session 量化占比，
     验收门对这些帧单列豁免，不算失败。

**F6b 训练侧**（标签过验收门后）

- 改动：
  1. L_contact：BCE 直接作用在脚 token 接触 logit 上（F5 已有该输出），目标用 f6_soft 软值
     （BCE 天然接受软目标）；不再走 "FK→速度→sigmoid"路径（`losses.py:165-166` 旧路径在 v2 删除）。
  2. L_skate = mean_t c_gt·‖v_foot^xy‖²：脚部 17/18/21/22 世界水平速度，由
     `f2_to_world` + 积分轨迹得到；加权用**硬标签 motion_f6≥0.5**（防模型压接触逃逸；软值不进权重，
     避免滑步帧只受半心惩罚）。
  3. L_height = c_gt·relu(h_foot − h_ground − δ)² + relu(h_ground − h_foot)²。
     h_ground 每 session 取 GT 脚高 5% 分位，δ=2cm。
  4. L_skate/L_height 前 10% 训练步线性升权。
  5. **F6c（可选）推理后处理**：预测接触 >0.5 且持续 ≥3 帧锁定脚位，髋-膝-踝两骨 IK 修正腿
     （`geometry.py` 新增 two-bone IK）。
- tag `f6_contact` / `f6c_footlock`。判据：接触期脚滑显著下降、接触 F1 上升（V2M 尤其——
  正是 E3 设 λ_con=0 丢掉的）；**T2M 接触 F1 单列**（压力语义差注定的滞后要量化，不做硬要求）；
  MPJPE 不劣化；L_contact 量级全程稳定（不重演 E1/E2 爆炸）。

### F7 不确定度 σ

- 改动：
  1. T 流部位读出：9 个可学习 query 对 E_T[t] 交叉注意力 → 部位证据 token；V 流直接用部位 token。
  2. 流内部位头输出 (μ_{m,p,t}, log σ²_{m,p,t})，σ² 每部位每帧一个标量；取代 F4b 深监督 MSE。
  3. 融合 σ̂ 头：解码器输出后每部位每帧一个；残差用 stopgrad 主输出计算，只训 σ̂ 头本身。
  4. 损失 β-NLL（β=0.5）：stopgrad(σ^{2β})·[r/(2σ²) + ½log σ²]，r = 标准化空间该部位平均
     平方误差；log σ² 截断 [−6,3]；前 10% 步数固定 σ=1（等价 F4b）。
  5. 门控接入：ℓ_m += −α·stopgrad(log σ_{m,p,t})，α 可学习初值 1。
- tag `f7_sigma`。判据：
  - 先看诊断：val 上每部位 σ 与实际误差 Spearman ≥0.3；T 流腿/脚 σ 随步态相位变化
    （支撑低摆动高）；V 段丢失帧 σ_V 升高。
  - 主指标：鲁棒性验证集 VT2M 显著改善，干净 val 不劣化。
  - σ 塌缩成常数：先确认误差本身在变化（检查段丢弃概率与遮挡强度），再调 β。

### F8 AMASS 精修器

**F8a 预训练**（可与 F4–F7 并行）

- 改动：
  1. **数据**：AMASS（SMPL-H）→ BVH-23 转换，新 `anysole/data/amass_to_bvh23.py`。
     方法（复用已验证的 `positions_to_6d_np`，`geometry.py:190`，GT 往返 0.00mm）：
     - SMPL FK 出各帧 3D 关节位置；
     - 按语义名取 22 个匹配关节位置（pelvis→Hips、hips→UpLegs、knees→Legs、ankles→Feet、
       feet→ToeBases、spine1→Spine、spine2→Spine1、spine3→Spine2、neck→Spine3、head→Head、
       shoulders→Shoulders、elbows→Arms、wrists→ForeArms、hands→Hands）；
     - BVH Neck（索引 5，SMPL 无对应）位置 = Spine3 与 Head 位置的线性插值（系数 0.5）；
     - 23 关节位置 + BVH offsets/parents 过 `positions_to_6d_np` → BVH-23 6D。
     - **验证**：转换后 FK 位置与 SMPL 原匹配关节位置误差 ≈0（插值关节除外）；
       抽 5 条序列用现有可视化肉眼查无穿模/漂浮。
     本数据集训练 mocap 已天然是 BVH-23，直接并入。40Hz、窗长同 F3，每条序列自带
     其 offsets（BVH 每会话 offsets / AMASS 用 SMPL 均值形状+betas 骨架）——**R 只消费
     旋转+轨迹，骨架仅用于 L_fk**。
  2. **R 结构**（新 `anysole/models/refiner.py`）：9 部位时空 Transformer，与解码器同构但
     无交叉注意力，8 层。每 token 输入 = 带噪部位向量 + 该 token 噪声级嵌入
     （MLP 后接 LayerNorm——沿用 E2 修复，`embeddings.py:42-47` 同款）。R 预测 x0；
     损失 = 标准化 MSE + L_fk（各样本自己的骨架）。
  3. 扩散设置：标准化空间 N(0,1) 噪声（等价 E4 的噪声缩放）、cosine-1000（复用
     `diffusion.py:GaussianDiffusion`，schedule="cosine"）。
  4. **逐 token 噪声级采样**（Diffusion Forcing 思路）：40% 全 token 共享 τ；30% 每部位
     独立 τ（时间恒定）；20% 上半身（躯干/头/双臂）τ=T 其余 τ~U(0,0.2T)；10% 随机部位
     随机时间段 τ=T。
- 判据（R 本身）：val mocap 掩码上半身后补全合理（加速度分布、关节限位违规率）；小 τ 重建误差 ≈0。

**F8b 推理接入**（F7 模型冻结，不重训）

- 改动（`eval.py`/`infer.py` 分支）：
  1. 起始噪声级 τ*_{p,t} 满足 (1−ᾱ_{τ*})/ᾱ_{τ*} = c·σ̂²_{p,t}，c∈{0.5,1,2} 按 val 选；
     T2M 上半身 σ̂ 大 → τ* 自然接近 T。
  2. 去噪：M0 各 token 加噪到各自 τ*；全局步 s 从 max τ* 递减，每 token 当前噪声级
     min(τ*_{p,t}, s)；DDIM 步数 {5,10,20} 扫描。
  3. 精修在每个 F3 滑窗内进行，重叠混合在精修后。
- tag `f8b_refine_c{c}_s{steps}`。判据：VT2M/V2M 不劣化；抖动、加速度误差、踝/腕误差显著改善；
  T2M 上半身合理性改善。探针：按部位统计 ‖M_ref−M0‖ 与误差变化，确认修改集中在 σ̂ 高的部位。
- 失败处置一：c→0 仍变差 → R 与本数据集分布不匹配 → 用训练集 mocap 单独微调 R（只用 M）。
- 失败处置二：步数越多越差（E4/E5 现象）→ 固定少步。

**F8c T2M 多假设**：同 F8b，K=10 采样。报告上半身 min-of-K 与 APD；下肢样本间标准差应很小
（很大 = 精修在改动本有证据的部位）。

### F9 前向模型 G

**F9a 训练 G 并替换压力输出**

- 改动：
  1. **G 结构**（新 `anysole/models/forward_model.py`）：输入 F2 表示运动窗口（138+4 维）→
     4 层 9 部位时空编码器 → 脚 token；每脚 Linear → 4×6×32 → 长轴上采样 → Conv → softplus
     → 40Hz 4×12 压力 + 接触。损失：归一化压力 MSE + 总力 L1 + CoP L1 + 接触 BCE；
     训练时输入运动加小噪声（容忍预测误差）。
  2. 质量检查（`z_note/probe_f9a_g.py`）：val GT 运动输入 → 总力 R²（≥0.8 才进 F9c）、
     压力 Pearson、CoP 误差、接触 F1。
  3. V2M 压力输出改 G(M̂)，删除 F5 脚 token 压力头。
- 判据：V2M 总力/CoP/接触 F1 不差于 F5 压力头；输出压力与预测动作接触时序一致。

**F9b 分析-合成一致性损失**：L_cons = ‖G_frozen(M̂) − T_obs‖²（VT/T 配置启用，小权重逐步升权）。
判据：T2M/VT2M 脚滑、接触 F1、根 RTE 改善，MPJPE 不劣化。L_cons 降而 MPJPE 升 = 模型在骗 G
→ 降权重或加大 G 输入噪声。

**F9c AMASS 伪压力预训练**：AMASS 子集（走/跑/转身/深蹲）→ G(M) 伪压力 + 传感器增强
（增益 0.8–1.2、加性噪声、随机死点、尺度扰动）→ 伪数据 T-only 预训练 T 流+解码器 → 配对
数据微调（预算与不预训练对照一致）。判据：T2M 显著改善；另报"仅伪数据预训练未微调"的
真实 val T2M（衡量 sim-to-real 差距）。

---

## 4. 并行与依赖

- **F0 期间可同时做**：F1 HMR 特征提取、F4 原生率压力管线、S2M-P 重训、F8a 的 AMASS→BVH23
  转换脚本。
- **可并行训练**：F8a 只依赖 F2 表示 + F3 窗长（F4–F7 期间进行）；F9a 的 G 只依赖 F2。
- **严格串行**：模型主线 F0→F7；F8b 依赖 F7 的 σ̂；F9b 依赖 F9a 的 G。

## 5. 主线之外的候选

- 流间瓶颈 token（Attention Bottlenecks）：F5 门控稳定后作对照。
- 位置编码换 RoPE、V 流加时间编码器、解冻 HMR 最后几层。
- 质心动力学约束（PhysPT 思路）：鞋垫力有标定（或静止站立段标定尺度）后约束垂直总力
  ≈ m(g+z̈_com)。
- 重投影损失/测试时优化：更适合无标注数据或推理细化。
- T2M 根轨迹多假设、CFG。

---

## 6. 命令手册（env = touch_gait，全部在 /data/fangyuxuan/projects/gait 下执行）

> 通用模板（`--device` 按当时 `nvidia-smi` 空闲卡填）：
>
> ```bash
> /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
>   --config configs/v2.yaml --modal anysolev2 --contact-method joint_and \
>   --epochs <N> --out-dir results/AnySole/<STEP>/checkpoints \
>   --wandb_mode online --wandb_experiment_tag <tag> --wandb_eval_interval 10 --device <gpu>
>
> /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
>   --ckpt results/AnySole/<STEP>/checkpoints/ckpt_last.pt \
>   --modal anysolev2 --split val --write-bvh results/AnySole/<STEP>/predictions/eval_bvh --device <gpu>
>
> CUDA_VISIBLE_DEVICES=<gpu> python results_display/script/visualize_anysole.py --modal <STEP> --contact-method <method> --force
> ```

| 步 | 关键 flag（叠加到模板） | 短预算（步数） | tag |
|---|---|---|---|
| F0a | 无（E3 ckpt 过协议） | — | f0a_protocol |
| F0b | `--modal anysolev2`（回归头） | 3700×3 种子 | f0b_regress |
| F0c | `--lambda-vrec 0` | 3700 | f0c_novrec |
| F1 | `--v-input hmr_gvhmr`（待实现开关） | 3700 | f1_hmr_gvhmr |
| F2 | 无 CLI（表示随 v2 modal；F2b `--lambda-kp 3`） | 3700 | f2_rep / f2b_fk |
| F3 | `--tw 80/160 --stride 40/80` | 按步数对齐 | f3_tw80 / f3_tw160 |
| F4a | `--tactile-input pressure_native` | 3700 | f4a_tstream |
| F4b | `--lambda-ds 0.3`（待实现） | 3700 | f4b_ds |
| F5 | `--decoder part9/part3/nogate`（待实现） | 3700 | f5_nogate → f5_part9 |
| F6 | `--contact-method f6_soft` | 3700 | f6_contact |
| F7 | `--sigma on`（待实现） | 3700 | f7_sigma |
| F8a | 独立训练脚本（refiner，非主线 CLI） | — | f8a_pretrain |
| F8b | eval 侧 `--refine c1_s10`（待实现） | — | f8b_refine_c{c}_s{s} |
| F9a | G 独立训练脚本 | — | f9a_g |

**验收口径（每步同表）**：eval 全套 F 系列指标 + ridge-on-F 探针 + 训练面板
（loss 各分量、grad_norm、T/V 编码器梯度比）+ BVH 运动学（逐关节误差、脚踝相对高度、
帧间抖动——测量方式复用 `z_note/smoke_e6x_decoupling.py` 的 fk 读取）。

## 7. 实施顺序清单（AI 逐项打勾）

- [ ] F0a：评估协议 + E3 基线过协议（3 种子 σ_seed、鲁棒性验证集、ridge 探针）
- [ ] F0b：pose_head head_mode="regress" + model_v2.py + train/eval/infer 分支 + smoke 单测 → 验收
- [ ] F0c：λ_vrec=0 → 永久删除 V 重建
- [ ] F1 准备：extract_hmr.py（GVHMR）+ 零训练基线探针（15 对匹配关节）
- [ ] F1：VEncHMR + dataset v2 分支 → 验收（vs 零训练 HMR）
- [ ] F2a：表示重构 + f2_to_world + 往返单测 → 验收
- [ ] F2b：FK 位置损失权重 → 验收
- [ ] F3：tw∈{80,160} + Hann 混合推理（删 crossfade）→ 验收
- [ ] F4a：pressure_native 管线 + FootConvEncoder + 网格方向单测 → 验收
- [ ] F4b：流内深监督 → 验收（梯度比探针）
- [ ] F5：先 f5_nogate 后 f5_part9（part_decoder.py + augment.py）→ 门控探针 + 主指标
- [ ] F6：F6a f6_contact_labels.py（motion_f6 状态机 + pressure_f6 承重 + f6_soft 软标签 + 标签验收门）
      → F6b L_contact（软目标）/L_skate/L_height → 验收
- [ ] F7：σ 头 + β-NLL + 门控接入 → Spearman/鲁棒性判据
- [ ] F8a（可提前并行）：amass_to_bvh23.py + refiner.py 预训练 → R 自身判据
- [ ] F8b/F8c：σ̂→τ* 接入 + 步数扫描 → 验收
- [ ] F9a：forward_model.py + 质量门 R²≥0.8 → 替换 V2M 压力输出
- [ ] F9b/F9c：一致性损失、伪压力预训练 → 验收
