# F2 / F2+4 分支模型结构说明（2026-09-18，供 AI 参考）

> 本文只讲**模型结构**：F2（`--f2-repr`）与 F2+4（`--f2-repr --t-encoder foot_conv`）
> 两个分支各改了什么、在哪个文件、上下游如何衔接。命令/验收口径见
> `f0_command_manual.md`（§F2、§F2+4 组合实测），方案依据见 `fix_plan_v2.md` §F2/§F4a。
> 两者共同的底座都是 **AnySoleModelV2**（F0b 回归线，`anysole/models/model_v2.py`），
> 都不是独立模型，而是同一 V2 模型上的两个开关组合。

---

## 1. 共同底座：AnySoleModelV2（F0b）

`--modal anysolev2` → `AnySoleModelV2`（model_v2.py:33）。与 V1 唯一的结构差异：
pose head 用 `head_mode="regress"`（直接 F→pose，无 diffusion 链，forward 签名去掉
x_tau/tau）；encoders/fusion/traj/aux 与 V1 同模块。数据口径 E3：raw108、tw=20、
stride=20、40Hz。

| 组件 | 文件 | 结构（两分支共享、未动） |
|---|---|---|
| V 编码 | encoders.py:129 | `LinearTemporalEncoder(2051→d)`：Linear + LayerNorm + time PE + modality 0（V1 HRNet 特征，未动） |
| T 输入 | model_v2.py:108 | `raw108 = cat([T_raw 96, T_phys 12])`（40Hz，数据零改动） |
| T 编码 | encoders.py:131-144 | `t_encoder="linear"`（F2 用）→ `LinearTemporalEncoder(108→d)` + time PE + modality 1 |
| 配置屏蔽 | encoders.py:155-160 | config V→T 置 null token，T→V 置 null token（VT/V/T 三配置训练） |
| 融合 | fusion.py | `FusionTransformer`：V/T token 交叉注意力 → fused F (B, tw, d) |
| pose 头 | pose_head.py:210 | regress：可学习 query (1, tw, 23, d) + time PE + 部位 group emb → 6 层 decoder（自注意 + 交叉注意 F）→ 反标准化出 (B, tw, 138) 6D |
| traj 头 | traj_head.py:36 | Transformer decoder，`out_dim` 3（V1）或 4（F2） |
| aux 头 | aux_heads.py | T_rec→96 维 T_raw（L_Trec）、V_rec→HRNet 特征（L_Vrec） |

损失（v1.yaml E3 口径，losses.py:245-254）：`λ_pose=3 / λ_traj=1 / λ_trec=0.1 /
λ_vrec=0.1 / λ_kp=1 / λ_con=0`；L_kp = FK 到世界系 kp_gt（losses.py:230-233）。
warm-start：`--init-from`，新结构按参数名自动跳过。

---

## 2. F2 分支（`--f2-repr`）：heading/tilt 运动表示

**动机**：V1 的根 6D 是含朝向的世界旋转、轨迹是绝对位移——朝向与平移耦合，回归头
难学全局一致性。F2 把根的 yaw 抽到 4 维轨迹里，pose 只剩 yaw 不变的 tilt。

### 2.1 表示定义（geometry.py:244-353）

- **pose_f2 (T,138)**：关节 1–22 保持 parent-local 6D **不动**；仅根 6D 替换为
  **tilt** = R_yaw(ψ)ᵀ @ R_root_world（去掉地面朝向后的根旋转）。
- **traj_f2 (T,4)** = `[ψ̇ (rad/s), v_hx, v_hz（heading 系水平速度 m/s）, h（世界根高 m）]`。
- **锚点**（每窗口 2 个标量/向量，非模型输出）：`psi_anchor` = 窗口前一帧的 ψ；
  `trans_anchor` = 窗口前一帧的世界根位置。
- 恢复：`f2_to_world`（torch/numpy 双版本）——cumsum(ψ̇)/FPS 积出 ψ，R_yaw @ R_tilt
  还原根世界旋转，v_h 经 R_yaw 转回世界 xy 后积出位移，h 直接用。**往返实测
  0.0001mm**（smoke_f2_roundtrip.py）。
- 前向轴 `FORWARD_AXIS = +Z`（geometry.py:16，训练集验证固化）。

### 2.2 数据侧（dataset.py:315-343, 416-425，离线预处理）

逐 session：根世界旋转 → unwrap ψ → tilt 替换 pose_gt 根 6D；ψ̇/v_h 用**真前向差分**
（首帧约定为 **0**，而非计划的"复制下一帧"——0 约定与"锚点=前一帧世界状态"精确
自洽，往返无损）。每窗口取 `psi_anchor=psi[left-1]`、`trans_anchor=trans_global[left-1]`。
`pose_gt` 形状不变 (T,138)，`trans_gt` 仍为窗内相对位移（L_traj 排除锚点）。

### 2.3 模型侧改动（唯一一处）

model_v2.py:90-91：`TrajHead(out_dim=4 if f2_repr else 3)`。warm-start 时 proj 3→4
自动跳过。pose 头 138 维不变（根 6 维现在输出 tilt）。`f2_repr` 存入 ckpt config。

### 2.4 损失侧（losses.py）

- **L_pose**：MSE 不变，但目标是 tilt 根（yaw 不再进 L_pose）。
- **L_traj**（`_f2_trajectory_losses`, losses.py:101）：4 维速度 MSE，per-dim 用
  训练集拟合的 mean/std（`traj_f2_stats`，存入 ckpt config）归一化使 ψ̇/v_h/h 同尺度；
  + 多尺度位移：归一化空间 cumsum 前 3 维、`traj_deltas` 差分 MSE（尺度 1/delta）。
- **L_kp**（losses.py:223-230）：`f2_to_world(pose_hat, traj_hat, psi_anchor, trans_anchor)`
  恢复世界姿态 → FK → 与 V1 相同的世界 kp_gt 监督。**这是 F2 的世界系接地**：锚点不
  参与 L_traj，但把轨迹误差暴露给 FK 位置损失。
- eval/protocol/infer 自动走 f2→world 恢复，无需额外参数（读 ckpt config）。

### 2.5 实测（wandb f2 run，回填自手册）

VT2M 65.4/PA 24.7；**T2M 89.7/35.8、T2M yaw 22.7（T-only 大幅改善）**；VT2M yaw 11.0。
V2M PA 28.0 略劣于基线。

---

## 3. F2+4 分支（`--f2-repr --t-encoder foot_conv`）：F2 表示 + F4a 触觉编码

在 F2 的**全部结构之上**，只再换一个组件：T 编码器 `LinearTemporalEncoder` →
`FootConvEncoder`（`--t-encoder foot_conv`，foot_encoder.py:46）。数据侧零改动
（还是 raw108 108 维），下游（fusion/pose/traj/aux）零改动。

### 3.1 FootConvEncoder 结构（foot_encoder.py）

```
(B, tw, 108) = cat([T_raw 96, T_phys 12])
├─ 每脚网格：T_raw 0:48=左脚、48:96=右脚 → 4×12（行=内外侧，列=脚跟→脚尖）
│    右脚沿行轴镜像，共享卷积看到统一布局
├─ 共享卷积：Conv2d(1→32→64, stride(1,2) 脚尖轴) → 4×6×64 → 展平 1536 → Linear 128
├─ 每脚手工特征 8 维：T_phys 每脚 6（CoP2+合力+包络+空间+时间）
│    + 编码器内算的接触面积比(>0.05 阈值) + dF/dt(网格和差分×40Hz) → MLP → 128
├─ cat(卷积 128, 特征 128) → Linear 256→d，+ 左右脚身份 embedding
│    → 每帧 2 个脚 token
├─ 第 3 个 token：global = Linear(2d→d)(左⊕右)
└─ 序列 (B, tw*3, d)：frame time PE 每帧重复 3 次 + token 类型 embedding
     → 4 层 TransformerEncoder（3 token 跨窗口时间混合）
```

**两条出口**（同一权重）：
- `forward()`（v1 解码器路径，F2+4 用）：`out_merge Linear(3d→d)` + LayerNorm +
  time PE + modality embedding → **合并回单条 (B, tw, d)**，与 LinearTemporalEncoder
  同契约，fusion 无感知。
- `encode_stream()`（F5 part 解码器用，保留 (B, tw, 3, d) 的 [左/右/全局] 三 token）。

T 重建目标不变（96 维 T_raw，L_Trec 走原 aux 头）。约束：foot_conv 要求
`tactile_input=raw108`（model_v2.py:60）。

### 3.2 与 F2 的叠加关系

两开关正交：`--f2-repr` 只动表示（数据预处理 + traj 头 3→4 + f2 损失路径）；
`--t-encoder foot_conv` 只动 T 编码器。组合 run = wandb fe2s0zsz（400ep，f2_repr/
t_encoder 已核实于 ckpt config）；**warm-start 来源未留档**（train.py 的 ckpt config
白名单不含 init-from，见 train.py:539-621），按 F2a/F4a 同协议推断为 F0b_seed2
（foot_conv 与 traj proj 4 维按名字自动跳过，V 编码器/fusion 复用）。

### 3.3 实测（组合结论，回填自手册 §F2+4 组合实测）

**折中而非叠加**：VT2M 63.4/23.1（收回部分 F4a 增强），V2M PA 25.8、VT2M yaw 8.4
（全配置最优），T2M 99.7/PA **40.7（四项最差，foot_conv 的 T-only 弱点在 F2 表示下
部分存在）**，T2M yaw 21.3（最优）。无负交互崩溃、也非 1+1=2；未再调组合，直接进
F5（门控正是修 T-only 的机制）。

---

## 4. 结构 diff 总表

| 维度 | F0b 基线（V2） | F2 | F2+4 |
|---|---|---|---|
| CLI | `--modal anysolev2` | + `--f2-repr` | + `--t-encoder foot_conv` |
| pose 目标根 6D | 世界旋转（含 yaw） | **tilt**（R_yawᵀ@R_root） | 同 F2 |
| traj 头 | 3 维（绝对位移） | **4 维**（ψ̇, v_hx, v_hz, h） | 同 F2 |
| 轨迹锚点 | — | psi_anchor / trans_anchor（窗口前一帧） | 同 F2 |
| T 编码器 | LinearTemporalEncoder(108→d) | 同基线 | **FootConvEncoder**（每脚网格卷积 + 3 token 时间 Transformer → 合并 1 token） |
| L_pose | MSE 含 yaw | MSE 只含 tilt | 同 F2 |
| L_traj | 绝对位移 | 归一化 4 维速度 MSE + 多尺度位移 | 同 F2 |
| L_kp | FK(pose, trans) | **f2_to_world 恢复后 FK** | 同 F2 |
| T 数据/目标 | raw108 / 96 维 | 同基线 | 同基线（数据零改动） |
| eval 处理 | 直接 FK | f2→world 自动恢复 | 同 F2 |

---

## 5. 实现文件索引

| 文件 | 内容 |
|---|---|
| `anysole/geometry.py:16,244-353` | FORWARD_AXIS=+Z；`heading_from_root_np`、`yaw_rotmat(_np)`、`f2_to_world(_np)`（torch/numpy 双版本） |
| `anysole/data/dataset.py:315-343,416-425` | 逐 session 构造 pose_f2/traj_f2/ψ；窗口锚点 |
| `anysole/models/model_v2.py:48,90-91` | `f2_repr` 开关 → TrajHead out_dim 4；`t_encoder` 校验（foot_conv 必须 raw108） |
| `anysole/models/traj_head.py:28` | `out_dim` 参数（proj 3↔4） |
| `anysole/models/foot_encoder.py` | FootConvEncoder（§3.1 全结构） |
| `anysole/models/encoders.py:131-134` | `t_encoder="foot_conv"` 接线 |
| `anysole/losses.py:101-144,164-165,223-230` | `_f2_trajectory_losses`；f2 分支；L_kp 经 f2_to_world |
| `anysole/train.py / eval.py / eval_protocol.py / infer.py` | f2 自动处理（读 ckpt config 的 `f2_repr`/`traj_f2_stats`，warm-start 自动跳过形状变化参数） |
| `z_note/smoke_f2_roundtrip.py` / `smoke_f4a_grid.py` | F2 往返/前向轴单测；F4a 网格单测 |

## 6. 相邻分支关系（一图流）

```
F0b_seed2（V2 回归基线）
├─ F2_f2rep        = F0b + f2-repr              （本文 §2）
├─ F4a_footconv    = F0b + foot_conv            （手册 §F4a）
└─ F2p4_combo      = F0b + f2-repr + foot_conv  （本文 §3，= F2+4）
   └─ F5-B（F5B_f2p4_nogate/gated）= F2+4 + part9 解码器 + V/T/∅ 门控（手册 §F5，不属本文）

F2b（λ_kp 扫描）= 非结构分支，已实测弃用，λ_kp 保持 1.0。
```
