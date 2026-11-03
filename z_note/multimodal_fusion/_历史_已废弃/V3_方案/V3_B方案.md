# V3_Bv1 方案（2026-10-08 定稿；生产主线，PA 税源已定位，兑现账本保留）

双流融合回归式模型（`anysole/models/v3_3b.py`，核心 `common/model_v2.py`）：**编码后直接拼接进融合 Transformer、再用双流交叉注意力 + σ 门控 + soft A 逐关节混合**。三条件 = null token 替换（训练采样 0.5 VT / 0.25 V-only / 0.25 T-only）。有融合 Transformer、有门控、有 null token。

## 1. 设计主张与机制规则

B 的全部设计主张是一个赌注：**V 和 T 在融合层可以靠注意力自行学会互补，不需要显式的物理/几何先验或概率因式化**。融合分三层：早期拼接 → 融合 Transformer → pose head 内双流交叉注意力（σ 门控）+ soft A 逐关节混合。

已定位事实（范围声明：以下证伪的是 B 形态的融合，不构成对融合路线的否定）：
| 事实 | 证据 |
|---|---|
| B 形态的共享表征融合是 PA 税的来源 | V 削弱实验：7/8 配置 ΔPA +0.55~+2.64；全流程探针：F\|VT 61.4 vs F\|V 48.3（T token 混入 V 半段后 V 读出恶化） |
| T 不经过 V 通路即无税 | 对照实验 V3_Dv1（B 的残差上限实验，非完整模型：冻结 V 流 + T 只进位置头旁路）8 配置 Δ ∈ [−0.05, +0.34]；强提示——3D 同时换了输出头，非干净消融 |
| T 有真实增量信息 | T 信息量探针：GlobalMPJPE Δprobe −43/−45、traj_ATE −44（配对显著、两路径一致）；部位正信号在足部（anklefoot −1.96\*、r_foot −2.89\*）、上身无——注意这是探针线性读出弱基线上的相对量（v_tok 全局 367 vs 模型 81.1），载体能力提升后该优势需复测 |
| 学习式路由不可靠 | σ 门控不路由（全流程探针）；E v1 β-NLL σ 塌缩前科——E 起后续设计禁用门控的依据 |

## 2. 模型结构

### 2.1 数据流

```
V（HRNet 特征）→ v_tok，T（鞋垫压力 108 维）→ foot_conv → t_tok；
[v_tok; t_tok] 拼起来过一个融合 Transformer 得 F；pose head 的 9 个部位
query 对 F 做「V 半段 / T 半段」双流交叉注意力，用 σ 门控加权，9 个部位
姿态假设经 soft A 矩阵按关节混合成最终姿态。
```

### 2.2 组件

- **V 流**：输入冻结 HRNet 特征 (B, 20, 2051) = 2048 维 HRNet + CLIFF bbox 3 维（40Hz，tw=20）。`LinearTemporalEncoder`：Linear(2051→256) → LayerNorm → + 正弦时间位置编码 → + 模态嵌入 → v_tok (B,20,256)。
- **T 流**：输入 (B, 20, 108) = [T_raw 96 维压力 | T_phys 12 维物理量]。`FootConvEncoder`：每脚 48 格压力重排为 4×12 网格（行=内外侧，列=脚跟→脚尖），右脚镜像后与左脚共享同一卷积栈（Conv2d(1,32,3) → Conv2d(32,64,stride=(1,2)) → Conv2d(64,64,3) → 4×6×64=1536 → Linear 128）；每脚 8 维手工特征（T_phys 6 维 + 接触面积比 + 压力变化率）经 MLP 到 128；拼接 → Linear(256→256)。每帧产出 [左, 右, 全局] 3 个 token → 4 层时序 Transformer → 合并回 t_tok (B,20,256)。
- **模态缺失处理**：三条件由 `config_id ∈ {0=VT, 1=V-only, 2=T-only}` 控制；缺失模态的整条 token 流替换为可学习 null token `null_v / null_t`（(1,1,256) 参数，normal std=0.02 初始化）。
- **融合层（第一层）**：tokens = cat([v_tok, t_tok], 1) → 4 层 pre-LN TransformerEncoder（nhead=8、ff=1024）→ F (B, 40, 256)。前 20 个 token 为 V 源、后 20 个为 T 源（后续门控层硬校验依赖此顺序）。F 同时供 pose head、traj head、aux heads 使用。
- **Pose head（第二、三层）**：9 个部位可学习 query（root / torso / headneck / l·r_arm / l·r_leg / l·r_foot）+ 时间 PE + 组嵌入；6 层 GatedDecoderLayer：
  ```
  z ← z + SelfAttn(z)
  e_V = CrossAttn_V(z, F; 只允许看 V 半段)
  e_T = CrossAttn_T(z, F; 只允许看 T 半段)
  σ  = Linear([z, e_V, e_T] → 2)          # 每部位对两流的"信任分"
  g  = softmax([σ_V, σ_T, τ_p])           # 三路权重，和为 1
  z ← z + g_V·e_V + g_T·e_T              # 先验支 g_∅ 就是残差 z 本身
  z ← z + FFN(z)
  ```
  τ_p 是按部位的先验偏置：T 信息部位（root/legs/feet）低 τ、V 信息部位（torso/headneck/arms）高 τ，钳位 [−4,4]。soft A：part_logits (9,24) 初始化为 ±3（复现人为 9 部位划分），A = softmax(part_logits / T)；每个槽有一个全姿态头 Linear(256→144)，输出按关节混合：out = Σ_p A[p,j] · hypo_p[j]。
- **其他头**：traj head（F 按 V/T 两半池化 → 2 层 encoder → 每帧速度 → cumsum 出轨迹，out_dim=3）；aux heads（从 F 的 T/V 半段重建压力/视觉特征，跨模态正则）。

### 2.3 三条件开关

| 条件 | 机制 | 训练采样 |
|---|---|---|
| VT | 全流 | 0.5 |
| V-only | T 流替换为 null_t | 0.25 |
| T-only | V 流替换为 null_v | 0.25 |

## 3. 训练

- 从零随机初始化（09-24 独立性规则，无 warm-start）；tw=20、stride=20、lr 1e-4、warmup 0.05、grad clip 5.0、epochs 740、batch 256、seed 1。
- 损失（实际训练值）：
| 损失 | 权重 | 含义 |
|---|---|---|
| L_pose | 3.0 | 姿态 6D MSE（native 空间） |
| L_traj | 1.0 | 根轨迹（含速度/差分） |
| L_kp | 1.0 | FK 关节位置监督 |
| L_trec / L_vrec | 0.1 / 0.1 | 触觉/视觉重建 |
| L_assign | 0.05 | A 矩阵逐关节熵 + 槽负载平衡 |
| L_sigma | 0.01 | σ 门控监督（β-NLL） |

- σ 门控监督（β-NLL）：v = g_V·e^{−σ_V} + g_T·e^{−σ_T}（σ 大 = 信任高），L_sigma = β·mean(r_p²/v + log v)，β=0.5，σ 钳位 [−3,6]；误差目标与门权重、A 矩阵全部 stopgrad，σ 是唯一活参数。训练前 sigma_freeze_frac=0.1（前 10% 步 σ 冻结为 0）。

## 4. 评估

### 4.1 三条件读数（统一面板）

口径：test，ckpt_best（cam3 = 2026-10-16 12:40 重评；occ010 = `results/experiments/v3b_test/`，2026-10-18 补跑）。单位：位置类 mm、旋转类 度、Jitter m/s³；Δ = VT2M − V2M，负 = 更优。

**关节位置误差（MPJPE 家族）**
| 指标 | V2M | VT2M | Δ（VT−V） | T2M | V2M occ010 | VT2M occ010 | Δ（VT−V）occ010 | T2M occ010 |
|---|---|---|---|---|---|---|---|---|
| PA-MPJPE | 41.12 | 42.23 | +1.10 | 83.26 | 40.79 | 42.36 | +1.57 | 83.49 |
| MPJPE | 53.29 | 54.01 | +0.72 | 140.27 | 55.67 | 55.66 | −0.02 | 144.74 |
| GlobalMPJPE | 81.13 | 75.94 | −5.20 | 173.80 | 79.28 | 75.65 | −3.63 | 179.74 |
| PA-MPJPE_lower | 50.70 | 51.68 | +0.98 | 85.47 | 51.25 | 52.64 | +1.39 | 87.61 |

**关节旋转误差**
| 指标 | V2M | VT2M | Δ（VT−V） | T2M | V2M occ010 | VT2M occ010 | Δ（VT−V）occ010 | T2M occ010 |
|---|---|---|---|---|---|---|---|---|
| MPJRE | 10.81 | 11.57 | +0.75 | 18.07 | 11.10 | 11.73 | +0.63 | 17.68 |

**全局/世界系**
| 指标 | V2M | VT2M | Δ（VT−V） | T2M | V2M occ010 | VT2M occ010 | Δ（VT−V）occ010 | T2M occ010 |
|---|---|---|---|---|---|---|---|---|
| traj_ATE | 54.14 | 46.75 | −7.39 | 81.90 | 48.24 | 45.80 | −2.45 | 83.26 |
| RootOrientationDrift | 3.35 | 4.04 | +0.69 | 34.08 | 3.78 | 4.29 | +0.51 | 36.78 |
| GlobalMPJPE | 81.13 | 75.94 | −5.20 | 173.80 | 79.28 | 75.65 | −3.63 | 179.74 |

**物理合理性**
| 指标 | V2M | VT2M | Δ（VT−V） | T2M | V2M occ010 | VT2M occ010 | Δ（VT−V）occ010 | T2M occ010 |
|---|---|---|---|---|---|---|---|---|
| FootSliding | 10.53 | 9.88 | −0.65 | 14.81 | 10.03 | 9.97 | −0.05 | 15.88 |
| GroundPen | 27.45 | 21.13 | −6.32 | 41.76 | 29.65 | 21.88 | −7.77 | 41.30 |
| Jitter_pred | 727.71 | 750.51 | +22.80 | 1705.65 | 762.25 | 793.32 | +31.08 | 1857.38 |

读法：B 是账本中唯一「税 + 兑现并存」的训练通道——PA 税 +1.10 与兑现（Global −5.2 / ATE −7.4 / Pen −6.3）同时取得，税源定位见 §1。

### 4.2 跨模型定位

B 是账本中唯一「税 + 兑现并存」的训练通道（cam3：ΔPA +1.10 / Global −5.20 / ATE −7.39 / Pen −6.32；occ010 同向：ΔPA +1.57 / Pen −7.77）；税源 = B 形态融合（§1）。全表见 `V方案_性能对照.md`。

### 4.3 定位与状态

- 生产主线（结构由入口脚本硬编码并记入 ckpt config）；专才 V3_Bv1_vonly/tonly = 同架构、只改训练条件采样、独立从零——vonly 被 V3_Hv1 用作 V 特征来源。
- 谱系：V3_2（9 部位硬头，无门控）→ V3_3A（+soft A）→ V3_Bv1（+σ 门控）；F5 part_decoder 作废归档（encode_stream 缺 LayerNorm 根因）；V4A/V4B 分组学习实验结论：人为 9 部位是合格选择。
- 兑现幅度（Global −5.2 / ATE −7.4）是训练通道的参照点，PA 税 +1.10 是所有零税设计的靶子。

## 5. 落地文件

- 入口 `anysole/models/v3_3b.py`；核心 `anysole/models/common/model_v2.py`；融合层 `common/fusion.py`；部位头 `common/pose_head.py`；T 编码 `common/foot_encoder.py`。
- ckpt：`results/AnySole/V3_Bv1_joint_and/<完整字段串>/checkpoints/`（字段串 = 超参记录）。

## 6. 版本

- **V3_2**：9 部位硬头、无门控。
- **V3_3A**：+ soft A 逐关节混合（A 矩阵冻结在人为划分、未学习）。
- **V3_Bv1（现行）**：+ σ 门控（β-NLL）。
- **V3_Bv2**：压力头换查询式读出——足/时间查询 cross-attention 到 F，直出 Q-DIRECT、非负 Q-NN（训练集 cell 均值/接触率先验初始化）。
- **V3_Bv3**：+ 因式化读出——Q-FACT total×allocation 分解，零初始化起点=训练集先验。
- **V3_3B_TTA（2026-10-24）**：Bv1 + 多视角 TTA 适配（v_tok_override 注入点 + L_2D 口径/trans_anchor 修复；改动说明见 z_note/HMR2.0/hmr2_gait/notes/实验报告_V3_3B_TTA.md）。
