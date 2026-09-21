# AnySole 多模态动作重建 · 模型实现说明

> 2026-09-20 重写 · 标准 SMPL-24 协议（旧 BVH-23 协议的历史文档见 git 历史与
> `model_fix_note/`）。本文描述**当前代码结构**；实验结论与演进史见
> `anysole/model_fix_note/model_fix_note.md`，当前基座训练命令见
> `anysole/model_fix_note/command_manual.md`。

约定：`B`=batch、`tw`=窗口长度（默认 20）、`d`=隐维 256、帧率 40 Hz。
三种输入配置 `VT2M / V2M / T2M` 共用同一套权重；缺失模态整路换可学习 null token。

---

## 0. 模型族总览

| 名称 | 姿态生成 | 状态 |
|---|---|---|
| `anysolev1`（AnySoleModel） | 扩散（去噪器预测 x0） | **legacy**：SMPL-24 迁移后旧 ckpt 全部被 eval/infer 拒绝；结构保留但不再训练 |
| `anysolev1_pos` | 扩散（根局部 3D 位置，69 维） | legacy（E6.1 表示对照，未成主线） |
| `anysolev1_insole_drift` | 扩散 + 鞋垫漂移补偿 | legacy 消融（raw108-only） |
| `anysolev2`（AnySoleModelV2） | **回归**（单次前向） | **主线**（F 系列起） |

路线裁决一句话：E 系列把扩散头修到 tau0 12mm 后，50 步 DDIM 链仍是瓶颈（10 步
114mm、抖动/拼接跳变失控）；F0b 去扩散改单次回归后 VT2M PA-MPJPE 78.8→27.9、
抖动 121→13 mm/帧——**扩散链是瓶颈，回归是对的路线**（过程见 model_fix_note）。

文件地图：`types.py`（协议与形状冻结）、`data/{dataset,smpl_io,tactile_s2m}.py`
（数据）、`models/{embeddings,encoders,foot_encoder,tactile_encoder,fusion,
pose_head,traj_head,aux_heads,model,model_v2}.py`（模型）、`diffusion.py`
（仅 anysolev1 用）、`losses.py`、`train/eval/infer.py`、`eval_protocol.py`。

---

## 1. 数据协议（SMPL-24，冻结）

- `N_JOINTS=24`，姿态目标 `pose_gt` **(B, tw, 144)** = 24 关节 × 6D 旋转；
  位置表示 `pose_gt_pos` **(B, tw, 69)** = 23 非根关节根局部位置（`anysolev1_pos`）。
- `types.py` 冻结 `JOINT_NAMES` / `JOINT_PARENTS`（标准 SMPL-24 树）并带 sha256
  校验和 `JOINT_PROTOCOL_CHECKSUM`——**协议锁**，防止输入适配器再次悄悄改变关节语言。
- 数据源 = 原生 SMPL NPZ（`data/smpl_io.py` 的 `load_smpl` / `resolve_smpl_path`；
  根目录 `SMPL_ROOTS` 四个 mocap 目录，可用 `ANYSOLE_SMPL_ROOTS` 覆盖）。
  **BVH 是 legacy 交换格式**：仅 Step2Motion 基线、接触标签生成、s2m50 合成 IMU
  使用（`tactile_s2m.py` 内置 23 关节 legacy 定义冻结）；`data/bvh_io.py` 已删除。
- 数值已验证的约定：`load_smpl` FK 与官方 SMPL 公式误差 0.0002mm；6D↔axis-angle
  往返 6.7e-8；`floor_y`（Y-up，每 session 地面参考）贯通损失/评估。
- 批字段（`types.BATCH_SHAPES`）：`pose_gt`、`trans_gt/vel_gt` (tw,3)、
  `traj_gt_f2` (tw,4，仅 f2)、`kp_gt` (tw,24,3)、`contact_gt` (tw,2)、
  `trans_anchor` (3,)、`psi_anchor`（仅 f2）、`root_rot_init` (3,3)、`floor_y`、
  `offsets/parents` (24 关节)、`config_id`、`session_id`。`assert_batch_shapes`
  在训练前逐项校验。
- **接触标签**（`--contact-method`，由 `results_display/script/contact_methods.py`
  生成 `contact_<method>.npy`）：`tactile_abs`（48 格压力和>100，即原 contact.npy）、
  `bvh_soft`、`bvh_h`、`joint_or`、`joint_and`、以及 F6 系列 `motion_f6` /
  `pressure_f6` / `f6_soft`（V3-0 主体已实现，标签清洗未完成）。

---

## 2. 输入通道

| 通道 | 形状 | 内容 | 开关 |
|---|---|---|---|
| V：HRNet（默认） | (B,20,2051) | 冻结 HRNet 2048 维特征 + CLIFF `bbox_info` 3 维；cam3、40Hz、crop 256 | `v_input=hrnet` |
| V：GVHMR | (B,20,1156) | rot 76（SMPL-X body_pose 63+global_orient 3+betas 10）+ kp2d 51（COCO-17）+ img 1024（HMR2 ViT）+ misc 5（bbox 3+q_V 2） | `v_input=hmr_gvhmr`（**F1 已取消，代码留档**） |
| T：raw108（默认） | (B,20,108) | `cat(T_raw 96, T_phys 12)`：左/右各 48 格压力 + 物理特征（CoP4+总力2+包络/梯度等 6） | — |
| T：s2m50 | (B,20,50) | Step2Motion 口径：每脚 25 = 压力16（4×12 池化 heel8/toes8）+ acc3 + gyro3 + 总力1 + CoP2。**IMU 由 GT BVH 合成，是评估口径、非无动捕部署口径** | `tactile_input=s2m50` |
| T：s2m50-noimu | (B,20,38) | 同上删 IMU 组（每脚 19） | `+ --no-imu` |

- **f2 表示**（`--f2-repr`）：`pose_gt` 的根 6D 变为 **tilt**（去 yaw），轨迹目标换
  4 维 heading-frame `[psi_dot, v_hx, v_hz, h]`（rad/s、heading 系水平速度、世界高度）。
- betas 用 GT（每 session 常量），只在 FK/关键点损失与导出时用。

---

## 3. 编码器（`encoders.py` / `foot_encoder.py` / `tactile_encoder.py`）

- **V 编码**：`LinearTemporalEncoder`（Linear→d + LayerNorm + 时间正弦 PE + 模态嵌入）
  或 `VEncHMR`（rot/kp/img/misc 四组投影求和，同合同）。
- **T 编码，三选一**：
  | 编码器 | 结构 | 开关 |
  |---|---|---|
  | LinearTemporalEncoder | 展平 Linear（108 或 50/38 维） | `t_encoder=linear`（默认） |
  | FootConvEncoder（F4a） | 每脚 48 格 4×12 网格共享卷积（右镜像）、手工特征（T_phys 6 + 接触面积比 + dF/dt）→ [左/右/全局] 3 token + 4 层时序 Transformer → 合并回 1 token | `t_encoder=foot_conv`（仅 raw108） |
  | TactileEncoder（E6.6b） | 8 组 MLP（左/右 × heel/toes/IMU/其他，S2M 分组）求和 + LayerNorm；`--no-imu` 时 6 组 | `tactile_input=s2m50 + --tactile-direct` |
- **模态 dropout**：`config_id`（0=VT / 1=V / 2=T）决定把整路 token 换成可学习
  null token（`null_v` / `null_t`），其余结构照常前向。

---

## 4. 融合（`fusion.py`）

- `cat([v_tok, t_tok])` → **(B, 40, d)** → 4 层 pre-LN Transformer encoder →
  fused memory `F`。
- **tactile_direct 旁路**（E6.6b）：pose head 的记忆改为 `cat([F, t_tok])`
  **(B, 60, d)**，给触觉一条绕过 V 主导稀释的直达通道；traj/aux 头仍读 40 token 的 `F`。

---

## 5. 姿态头（`pose_head.py`）

共享：每维 mean/std 标准化（`pose_mean/pose_std` 训练集拟合、随 ckpt 保存、
std 下限 1e-2）；时间 PE；分组嵌入；6 层 pre-LN TransformerDecoder
（self-attn → cross-attn(F) → FFN，GELU，ffn 1024）。

**两种模式**（`head_mode`）：

- **diffusion（V1）**：输入加噪 `x_τ` 与步数 τ。3 组分区投影
  [body 16 / left 4 / right 4]；**t-token prepend**：`TimestepEmbedding`（带
  LayerNorm，E2 修复）×0.05 + 可学习 token 作序列首 token，每层自注意力全程可见
  ——按噪声水平门控对 x_τ 的使用。输出预测 **x0**（便于在干净姿态上加 FK 损失），
  反标准化返回。`repr="pos"` 时目标为 69 维根局部位置（`anysolev1_pos`）。
- **regress（V2，主线）**：固定可学习 query `(1, tw, n_tokens, d)` + time PE +
  分组嵌入，直穿同款 6 层 decoder。`n_parts=3`（默认）＝每关节一个 query、三组
  unembed 头；`n_parts=9`（V3-2）＝9 部位查询（`PART_NAMES`/`PART_JOINTS`，root/
  torso/headneck/l_arm/r_arm/l_leg/r_leg/l_foot/r_foot）+ 9 个部位 unembed 头，
  输出按部位序拼接后经 `part_place_idx` scatter 回关节序（144 维）。
- 组划分（token 序 [body, left, right]）：`LEFT_LEG_JOINTS=(1,4,7,10)`、
  `RIGHT_LEG_JOINTS=(2,5,8,11)`、`BODY_JOINTS`=其余 16 个。

---

## 6. 轨迹头（`traj_head.py`）

- 输入 `F` 按 V/T 两路池化（reshape→mean）成 (B, tw, d) → 2 层 pre-LN encoder →
  每帧根速度 `v_hat` (B, tw, 3, m/s) → `trans_hat = cumsum(v_hat)/FPS`（窗口相对轨迹）。
- f2 表示：`out_dim=4`，`[psi_dot, v_hx, v_hz, h]` 前三维积分；世界姿态/位移由
  `geometry.f2_to_world`（tilt 姿态 + heading 轨迹 + `psi_anchor`）恢复。
- `trans_anchor`/`psi_anchor` 不参与损失，评估与长序列拼接用。

---

## 7. 辅助头与损失（`aux_heads.py` / `losses.py`）

- 辅助头（跨模态信息机制）：T 重建头读 `F` 的 T 半段 → `pressure_hat` (B,20,96)
  （归一化 T_raw）；V 重建头读 V 半段 → `vfeat_hat` (B,20,2051)。
- 损失表（`anysole/configs/v1.yaml` 默认权重）：

| 损失 | 定义 | 权重 |
|---|---|---|
| L_pose | MSE(`x0_hat`, `pose_gt`)（f2 时根=tilt） | 3.0 |
| L_traj | 速度 MSE + 多尺度位移：`mean_{Δ∈{2,4,8,19}} (1/Δ)·MSE(D_Δ(trans_hat), D_Δ(trans_gt))` | 1.0 |
| L_trec / L_vrec | 辅助重建 MSE | 0.1 / 0.1 |
| L_kp | FK(`x0_hat`) 全 24 关节 vs `kp_gt` MSE | 1.0 |
| L_con | 软接触 BCE（现关闭；软形式回放 = V3-6） | **0.0** |
| L_pose_vel / L_bone | E6.2 帧间平滑 / 骨长刚性（pos 模式） | 0.0 / 0.0 |

- f2 模式轨迹损失走 heading-frame 口径（`_f2_trajectory_losses`，per-dim 训练集
  归一化 `traj_f2_stats` 存 ckpt），FK 监督经 `f2_to_world` 恢复后比较。

---

## 8. 训练（`train.py`）

- 配置采样 `config_probs=[0.5 VT, 0.25 V, 0.25 T]`，无全空样本。
- 默认：Adam lr **1e-4**（yaml，`--lr` 不是 CLI）、batch 256、800 epoch、stride 20。
- 基建：`--init-from <ckpt>` warm-start（同结构同名参数复用、新结构按名字自动跳过
  保持新初始化）；`--grad-clip 5.0`（爆炸保险，F0 教训）；`--loss-cap`（wandb y 轴
  压制）；`--lr-warmup-frac`（V3 结构步前 5% 线性 warmup）；NaN-grad 守卫（非有限
  梯度跳步并降 lr）。
- 主要开关：`--modal`、`--contact-method`、`--tactile-input`、`--t-encoder`、
  `--tactile-direct`、`--no-imu`、`--v-input`、`--f2-repr`、`--pose-parts 9`、
  `--lambda-*`、`--epochs`、`--tw/--stride`、`--config-probs`、wandb 组
  （`--wandb_experiment_tag` 由 Wandb_Analyzer 分组）。CLI 覆盖值随 ckpt 保存，
  eval/infer 从 ckpt 读配置。
- 训练末尾自动跑一次 test eval（`--no-protocol` 关协议、`--no-write-motion` 不导出）。

---

## 9. 推理与评估（`infer.py` / `eval.py` / `eval_protocol.py`）

- **推理**：anysolev2 每窗一次回归前向；anysolev1 走 DDIM 采样链
  （`--sample-steps`；E6.4 `continuation`＝重叠窗续写链，只导出新半窗；
  E6.5 `warm_start`＝起点用姿态均值而非纯噪声）。窗口间 carry 前一窗预测的世界
  位置拼接；f2 时按 heading anchor 拼接。导出 **SMPL NPZ**（`--write-motion <dir>`，
  默认 `predictions/eval_motion`；`--no-write-motion` 关）。
- **评估指标**（`eval_protocol.py`，`--protocol-seed 0` 默认跑 1 次协议，
  `--no-robustness` 关鲁棒集）：
  - 局部：MPJPE / PA-MPJPE，按 9 个 SMPL-24 语义部位 + upper/lower/anklefoot/hands；
  - 全局：W-MPJPE（160 帧=4s 段首帧对齐）、RTE_norm（按位移长度归一化）、
    yaw_abs_deg / yaw_drift_deg（根朝向误差/累积漂移）；
  - 时序：jitter（mm/帧 @40Hz）、accel_err_ms2、seam_jump_mm（窗边界，附 GT 同缝）；
  - 接触与滑步：contact_f1、foot_slide_mm；V2M 压力重建：pressure_force_r2 /
    pressure_cop_err / pressure_pearson；T2M 上半身：accel_dist_err_upper_ms2、
    joint_limit_viol_elbow/knee；
  - 鲁棒集 robust_vdrop / robust_tdrop（VT2M 连续 20–40% 帧置零）；
  - 读出天花板：`results_display/script/ridge_probe.py` 的 ridge-on-F 表（每步必跑）。

---

## 10. 当前状态与已知问题（2026-09-20）

- **SMPL-24 迁移进行中**：全部历史 ckpt（BVH-23/138 维 + SMPL-24 早期的
  F4a_footconv / V3_2_part9，含 group_ids 错位训练的权重）已归档
  `results/backup/AnySole_BVH_backup/`，`results/AnySole/` 已清空——修复后代码
  需按 `model_fix_note/command_manual.md` 的 warm-start 链从零重训。
- **已修（2026-09-20）**：pose_head 分组嵌入 group_ids 错位——原按 SMPL-24 关节
  索引打标、但 token 序为 [body,left,right]，导致 16/24 token 拿到错误组嵌入；
  已改为按 token 序打标并数值验证（三种模式前向 + 梯度均通过）。
- **V3 主线进度**（详见 model_fix_note）：V3-2 九部位移植代码完成，两次训练同型
  爆炸（xd6kcdk2 ep386、1s0yeuwn ep531）但健康期证明结构成立，待第三次重训；
  V3-3 σ 门控、V3-4 f2 未实现；V3-0/V3-1 等接触标签清洗。
- `part_decoder.py`（F5 部位解码器+门控）已作废，仅留档。

---

## 11. 参考工作

- MMVP（视觉+压力、抗脚漂/全局平移）https://arxiv.org/abs/2403.17610
- MotionPRO（大规模压力+RGB+光学）https://arxiv.org/abs/2504.05046
- Step2Motion（鞋垫压力+IMU→locomotion；分区、时间嵌入、累积惩罚）https://github.com/JLPM22/Step2Motion
- Pressure2Motion（地面压力+文本→动作，分层扩散）https://arxiv.org/abs/2511.05038
- RoHM（TrajNet+PoseNet 解耦）https://arxiv.org/abs/2307.11692
- MDM（x0 预测、预置 token 门控）https://arxiv.org/abs/2209.14916
- 6D 连续旋转表示（Zhou et al.）https://arxiv.org/abs/1812.07035
