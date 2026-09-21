# AnySole 修复与演进笔记（单一历史文件）

> 2026-09-20 重写：全部历史与计划合并为本文档，**一级标题 = 时间线节点**，每节只留
> 结论与关键数值；实施细节原样归档在 `archived/`。当前基座训练命令见
> `command_manual.md`，模型结构见 `anysole/AnySole 多模态动作重建 · 模型实现说明.md`。

## 时间线（git 节点）

| 节点 | commit | 时间 | 一句话 |
|---|---|---|---|
| 0 初始模型 | `c2e71c0` | 09-15 | V1 扩散 + BVH-23/Skeleton3 协议；起点 = 均值姿态塌缩 |
| 1 E5 修改前 | `bccb60f` | 09-16 | E1–E4 完成：头修好（tau0 12mm），**扩散链是瓶颈** |
| 2 E6.x 系列 | `43f0e8c` 一带 | 09-16~17 | s2m50 触觉通道、续写式推理、位置表示对照 |
| 3 F5/BVH 修改前 | `b04a92b` | 09-18 | F0 去扩散回归突破、F1 取消、F2a/F4a 实现，F5 未开始 |
| 4 V3 计划 | — | 09-19 | 取代 v2 §F5，主线转向"触觉兑现" |
| 5 SMPL 版 + 重写 | `3a8acd8` + 工作区 | 09-19~20 | 原生 SMPL-24 迁移（进行中），旧 ckpt 全部失效 |

> 分支命名（2026-09-21）：`BVH_Motion` = 原 main@`b04a92b`（BVH 末提交）；
> `SMPL_Motion` = 当前分支@`3a8acd8`（SMPL 切换点）。Baselines 已整体出 index。

---

## 节点 0 · 初始模型（c2e71c0，09-15）

结构 = 原 V1 实现说明（HRNet 2051 + raw108 → 4 层融合 → 扩散姿态头 3 组 →
回归轨迹 → T/V 重建辅助头）。

**起点问题**（wandb yg01hyju，2000ep）：pose head 塌缩成均值姿态输出器——
tau0（喂 GT）MPJPE 145mm ≈ 均值基线 147mm，对 x_τ/τ/F 几乎无响应。
根因链：TimestepEmbedding 无归一化 → |t_tok| 训练中无界增长（0.11→37-40，50×
token 量级）→ head 学会把 τ 信号整体消掉（τ 盲）→ 唯一可行的单函数策略 =
忽略输入 → 输出 f(F)≈常量。

---

## 节点 1 · E 系列（→ bccb60f，09-16；tag e1_taumax100 / e2_tnorm / e3_nocontact / e4_normdiff / e5_sched200）

| 实验 | 改动 | 结果与结论 |
|---|---|---|
| E1 | 训练 τ~U(0,100) 低噪声带 | tau0 145→113，L_con 爆炸 2.17 → **τ 分布假说证伪**（是结构性 τ 盲） |
| E2 | t-token 尾部 LayerNorm + ×0.05 + λ_pose 1→3 | tau0 66mm，**门控恢复**；L_con 才是 total loss 大头（BCE 61-87） |
| E3 | λ_con=0 | tau0 23-31；VT2M 141.8 / PA 81.2（= ridge-on-F 85 的单视角天花板）；抖动 116mm/帧（GT 8.7）、拼接跳变 213mm；contact_acc 0.048 |
| E4 | 噪声×pose_std + cosine LR + 导出 crossfade | 单步去噪 23/57mm，但 50 步 DDIM 链 92-166 **更差**；"步数越少越好"（10 步 80.4 最优）；机制：head echo 输入（0.12-0.63），链每步注入读出误差 |
| E5 | Step2Motion 同款 linear-200 | tau0 **12mm**（历次最好）但链 114mm；"飞天脚"：误差沿运动链放大，踝 168-202mm |

对照 S2M 的代码级核验（`archived/E6_verification_against_step2motion.md`）：
**采样器/schedule 无罪**（echo 0.293 vs S2M 0.007），病根在 head 的 F 读出与
6D 表示的 FK 放大。E 系列总结论：**扩散链是瓶颈**，精修能力（tau0 12mm）无法
转化为凭空生成能力（链 114mm）。

---

## 节点 2 · E6.x 系列（09-16~17）

- **E6.1–E6.5**（`archived/E6x_series_plan.md`）：pos 位置表示（anysolev1_pos）、
  平滑/骨长损失、窗口与步长、**continuation 续写式推理**（E6.4，重叠窗只导出
  新半窗）、warm start（E6.5）。continuation/warm_start 保留为推理开关；其余
  被 F 系列路线取代。
- **E6.6a/b + E6.8**（`archived/E6x_tactile_loss_plan.md`）：`--tactile-input
  s2m50`（50 维 Step2Motion 口径，**IMU 由 GT BVH 合成 = 评估口径非部署口径**）；
  `--tactile-direct`（TactileEncoder 8 组 MLP + 直达 pose head 旁路，触觉编码器
  梯度 214.7 vs 35.8 = 6×）；`--no-imu`（38 维）。方向后由 F4a 的 foot_conv 取代。
- E6.7 损失清单（8 项权重建议）仍是损失系统参考，见归档文件。

---

## 节点 3 · F 系列（→ b04a92b，09-18；F0 起模型 = anysolev2 回归）

**F0 去扩散 → 回归 = 唯一决定性修改**（其余：warm-start 是加速手段、λ_vrec=0
被二轮归因洗清、grad-clip 是保险）：

| 指标（val 协议） | E3 扩散 800ep | F0c 回归 200ep |
|---|---|---|
| VT2M MPJPE / PA | 138.9 / 78.8 | **69.7 / 27.9** |
| T2M MPJPE / PA | 176.2 / 86.5 | 121.7 / 40.5 |
| V2M MPJPE / PA | 146.5 / 82.3 | 72.7 / 31.4 |
| 抖动 jitter | 121.1 mm/帧 | 13.0（GT 9.1） |
| 接触 F1 | 0.0 | 0.6 |

F0b_seed2（400ep，λ_vrec=0.1）VT 63.7 = 当时最强基线；偶发有限值巨梯度（ep370
爆炸）→ 自此每训必带 `--grad-clip 5.0`。回归头超过线性 ridge 天花板（27.9 vs
43.6），路线结论：扩散链是瓶颈，回归是对的路。

- **F1 取消**（GVHMR 视觉输入）：V 输入零改动下 PA 78.8→25.1 → **视觉天花板假说
  证伪**；GVHMR 代码留档（`v_input=hmr_gvhmr`）、权重不下载。
- **F2a 实现**（`--f2-repr`）：根 6D 拆 tilt/yaw + 4 维 heading-frame 轨迹，往返
  0.0001mm，前向轴 +Z（单测 5:1）。F2 结果 T2M 89.7/35.8（T2M 历史最优）。
  **F2b 弃**：λ_kp=3 同预算全面反升（VT +6.1 / T +4.1），λ_kp 保持 1.0。
- **F4a 实现**（`--t-encoder foot_conv`，数据侧零改动）：每脚 4×12 网格共享卷积
  + [左/右/全局] token。VT 配置均匀兑现（下肢 −9.1、双脚 −10、双腿 −8.3），
  **T-only 全身退化 +7.6**（退化最重在上半身：手臂 +10~15、手 +14.6）。
- **F2+4 组合**：VT 63.4/23.1（PA 三配置全表最优）、T2M 99.7/40.7（折中而非叠加）
  ——恰好钉死 F5 的位置：T-only 剩余弱点只有门控能修。
- **F5 四 run 失败**：根因 = `encode_stream()` 缺 LayerNorm（T token 量级
  1400×V → 交叉注意力 one-hot 退化 → LN 常数擦除）；part_decoder.py 留档作废。
- **触觉通路逐级 ridge 探针**（V3 的起点证据）：F 58.2 / v_tok 74.3 / t_tok1
  110.6 / 融合前拼接 73.0——融合没有毁掉触觉（F 比任何单流好 15mm），触觉单流
  全身天花板 ~110-130（增强器定位确认）；**接触/相位在训练中完全无监督**
  （λ_con=0 且标签 83.4% 假接触）——"潜能未兑现"的实锤。

---

## 节点 4 · V3 计划（09-19，取代 v2 §F5；当前主线）

**目标**：模态互补 + 模型全能（功能导向）。三个机制：

| 机制 | 结构改动 | 解决的问题 |
|---|---|---|
| A 流级胜任 | 双向流内深监督 + 接触监督（V3-1） | 流级搭便车（T 流不学全局、V 流不学接触） |
| B 信任调度 | per-part σ 软门控（V/T/先验三路，V3-3） | 缺失/不可信部位自动走条件先验 g_∅ |
| C 先验存储 | 条件先验路径；不足时接 AMASS 精修器（V3-5） | T-only 补全局、V-only 补精度的知识源 |

**步骤状态**（细节：`archived/fix_plan_v3.md` + `fix_plan_v3_impl.md`）：

| 步 | 内容 | 状态（2026-09-20） |
|---|---|---|
| V3-0 | f6_soft 接触标签（继承 v2 §F6a） | 主体已实现；全量生成 + 标签清洗未完成，**延后** |
| V3-1 | 流级深监督（机制 A） | 未开始（等标签） |
| V3-2 | pose_head 九部位最小移植（`--pose-parts 9`） | 代码完成；**两次训练同型爆炸**（xd6kcdk2 ep386、重训 1s0yeuwn ep531：traj 先行发散、硬窗口触发、有限大梯度），但健康期（1s0yeuwn ep110–510 VT 52.9–58.1 / T 115，best 52.87@ep480 优于 F4a 基线）**证明结构成立**；ckpt_best 保护已上线，待第三次重训 |
| V3-3 | σ 软门控 + 条件先验（机制 B，`--gate sigma`） | 未实现（设计定稿） |
| V3-4 | f2 表示移植（F2p4 基座） | 未实现 |
| V3-5 / V3-6 | 先验存储 / 接触滑步损失 | 触发式，未开始 |

- 执行裁决（09-19，用户）：**删除全部验收门**（"不做'还没做就先提期望'的事，
  做一步看一步"）；批次重排为 **V3-2→V3-3→V3-4 结构线先行**，V3-0/V3-1 数据线
  等标签清洗。
- F5 回滚（batch 1，09-19 完成）：三基线复测全部复现（F4a 55.49/22.96、
  F2p4 99.71/40.71、F0b 63.68）✓；encode_stream 补 LN 作为代码卫生修复（V3 不调用）。
- 已否决的设计：p_drop_both 全屏蔽（均值塌缩）、3-token 流进 V3 主线、
  门控接原始流而非融合后 F 的掩码视图。

---

## 节点 5 · SMPL-24 迁移（3a8acd8 + 09-20 工作区，进行中）

- **协议切换**：N_JOINTS=24、POSE_DIM=144；`types.py` 冻结关节名/树 + sha256
  协议锁；数据直读原生 SMPL NPZ；BVH-23 降级为 legacy（Step2Motion / 接触标签 /
  s2m50 合成 IMU）；`data/bvh_io.py` 已删。
- **数值验证**：load_smpl FK 与官方 SMPL 误差 0.0002mm；6D↔axis-angle 6.7e-8；
  floor_y 贯通 loss/eval。
- **全部 ckpt 失效并归档**：BVH 口径（138 维）与 SMPL-24 早期权重（F4a_footconv、
  V3_2_part9 等，含 group_ids 错位训练的组嵌入）全部归档
  `results/backup/AnySole_BVH_backup/`，`results/AnySole/` 已清空；修复后的代码
  需按 `command_manual.md` 的 warm-start 链（F0b→F4a/F2→F2+4→V3-2）从零重训。
- **已修 bug（09-20）**：pose_head group_ids 按关节索引打标 vs token 序
  [body,left,right] 广播 → 16/24 token 组嵌入错位；已改按 token 序打标 +
  数值验证（三种模式前向/梯度全过）。
- 待办：各变体（F0b/F2/F2+4/V3-2…）在新基座重训。

---

## 附 A · 结论数值总表（BVH-23 历史口径，仅作参照）

| 口径 | F0b_seed2 | F4a（主基座） | F2 | F2+4 |
|---|---|---|---|---|
| VT2M MPJPE / PA | 63.7 / 25.1 | **55.5 / 23.0** | 65.4 / 24.7 | 63.4 / 23.1 |
| V2M MPJPE | 66.3 | 63.5 | — | 68.7 |
| T2M MPJPE / PA | 116.2 / 37.4 | 123.8 / 40.3 | **89.7 / 35.8** | 99.7 / 40.7 |
| VT2M yaw_drift | 6.1 | 12.4 | 11.0 | **8.4** |

E 系列关键数：tau0 145（起点）→ 66（E2）→ 23-31（E3）→ 12（E5）；链 10 步
80.4（E4 最优）/ 114（E5）。

## 附 B · 归档文件索引（archived/）

| 文件 | 内容 |
|---|---|
| `fix_plan_v3.md` / `fix_plan_v3_impl.md` | V3 计划全文（机制/步骤/验收/决策树）+ 实施记录（F5 回滚表、V3-2 爆炸诊断、批次表） |
| `fix_plan_v2.md` | F 系列正式源（§F6a 标签规格全文被 V3 逐字继承、§F8 AMASS 规格、纪律与指标表） |
| `f0_command_manual.md` | F0/F4a/F1/F2/F2+4 历史命令与实现口径偏差（含 SMPL-24 迁移操作注记） |
| `fix_plan_fast.md` | F0 快速通道 + §6 验收回填（F0b 塌缩/爆炸、F0c 归因、F0b_seed2 63.7） |
| `f2_f2p4_structure.md` | F2/F2+4 分支结构说明（§7 SMPL 风险清单、关节树校验和、FORWARD_AXIS 证明） |
| `E6x_series_plan.md` / `E6x_tactile_loss_plan.md` | E6.1–6.5 计划与 E6.6/E6.7 触觉重建计划（含损失清单） |
| `E6_verification_against_step2motion.md` | E6 对照 S2M 代码级核验（echo 0.293、sampler 无罪） |
