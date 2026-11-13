# V8_Cv5 模型结构与设计

> 轨迹生成正式方法（仅 T2M 分支）：见 `V8_BiFoot_轨迹生成正式方法.md`（2026-11-11）。

日期：2026-11-08。范围：当前 `SMPL_Motion` 分支的 V8 基座。本文只说明**结构**与结构相关的设计契约，不展开实验结果。

主要来源：

- `anysole/models/v8_cv5.py`（V8 命名空间入口）
- `anysole/models/v7_c2sd2.py`（V8_Cv5 实际结构）
- `anysole/models/v7_c2sd.py`、`v7_c2.py`、`v7_c.py`（继承链）
- `anysole/models/common/model_v7.py`（`AnySoleModelV7` / `V7MotionHead` / `_V7DecoderLayer`）
- `anysole/models/common/encoders.py`、`embeddings.py`
- `anysole/types.py`（SMPL-24 协议、`PART_NAMES` / `PART_JOINTS`）
- `anysole/utils/losses.py`（SD2 配平与锚头损失）
- `anysole/registry.py`（模型注册）

## 0. 一句话

**V8_Cv5 = V8 命名空间下注册的 V7_Cv5SD2 结构**：双独立 T 门（根速度门 + 姿态门）+ 三条件枚举训练 + 根头“水平速率 / 水平方向 / 垂直分量”四通道读出 + 平移锚头旁路。它不是新的 backbone；V8 命名是为了在同一个新基座上重建门控/条件消融，结构逐参数等于 `v7_c2sd2`。

## 1. 版本关系与“双门”的准确含义

| 版本 | 入口 | 结构增量 |
|---|---|---|
| V7_Cv5 | `v7_c2.py` | Cv5 = C 基础 + **双独立 T 门**（root/pose）+ 三条件枚举训练；根头 3 通道；无锚头 |
| V7_Cv5SD | `v7_c2sd.py` | Cv5 + `v7_root_sd`：根头四通道读出 + T 行方向掩码 |
| V7_Cv5SD2 | `v7_c2sd2.py` | SD 结构不变 + `v7_root_sd_balance` 恒等式配平损失；2026-11-05 起默认带平移锚头 |
| **V8_Cv5** | `v8_cv5.py` | `from v7_c2sd2 import CONFIG_EXTRA, STRUCTURE, build`；**结构 = V7_Cv5SD2** |

`v8_cv5.py` 全文只有这一层转发：

```python
from anysole.models.v7_c2sd2 import CONFIG_EXTRA, STRUCTURE, build  # noqa: F401
```

`v7_c2sd2.STRUCTURE` 由继承链叠加得到：

```python
# v7_c.py
STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v7": True,
    "v7_contact_root": False,
    "v7_reliability_route": True,
}
# v7_c2.py
STRUCTURE = {**BASE_STRUCTURE, "v7_task_routes": True}
# v7_c2sd2.py
STRUCTURE = {
    **BASE_STRUCTURE,
    "v7_root_sd": True,
    "v7_root_sd_balance": True,
    "v7_trans_anchor_head": True,
}
```

因此 V8_Cv5 的模型开关是：

| 开关 | 值 | 含义 |
|---|---|---|
| `reliability_route` | True | 存在 T 信任门 |
| `task_routes` | True | 门分 root / pose 两组，独立 MLP |
| `root_sd` | True | 根头四通道：速率 / 水平方向 / 垂直 |
| `root_sd_balance` | True | 损失用恒等式配平（只改损失，不改结构） |
| `trans_anchor_head` | True | 平移锚头旁路 |
| `trans_anchor_stopgrad` | True | 锚头不回传主干 |
| `v_gate` / `gate_2sigma` | False | **V8_Cv5 没有 V 侧信任门**；那是 V8_Cv5VG |
| `part_routes` / `route_temporal` | False | 门是窗口级、root/pose 两组，不做逐帧/四组 |
| `route_router` | False | 没有竞争路由（V8_Cv8） |
| `ar` / `ar2` / `ctx*` | False | 双向、非因果、非 AR/续写 |

**“双门”= 双独立 T 门**：根速度 query 一条 T 门、9 个姿态 query 一条 T 门；两条门输入相同、MLP 独立（`route_root_mlp` 在共享初始化后 `deepcopy(route_mlp)`，训练后分化）。它不是 V/T 双向门；V8_Cv5 的 V 交叉注意力没有门，只有 `v_present` 硬关闭。

## 2. 输入/输出契约

- 运动协议：SMPL-24，`POSE_DIM = 24 × 6 = 144`；关节顺序与 `PART_NAMES` / `PART_JOINTS` 由 `anysole/types.py` 唯一给出。
- 窗长 `tw = 20`（40 Hz 下 0.5 s），训练/评估的窗步长 `stride = 20`（非重叠）。
- 视觉输入 `V_feat`：HRNet 特征，2051 维/帧；`v_input="hrnet"`。
- 触觉输入：`T_raw` 96 维（左足 48 + 右足 48，归一化压力格）；`T_phys` 12 维（默认 `t_phys_v2=False`）；两者拼接成 **108 维** 进 FootConv（`tactile_input="raw108"` = 96 raw + 12 phys）。
- 输出：
  - `x0_hat`：`(B, tw, 144)` 的 6D 姿态；
  - `v_hat`：`(B, tw, 3)` 世界系根速度（m/s，y-up；由 SD 四通道组装）；
  - `trans_hat`：`(B, tw, 3)` 根轨迹（`cumsum(v_hat)/FPS`，模型内不加锚）；
  - `s_hat` / `d_hat` / `vy_hat` / `sd_dir_norm`：SD 四通道的诊断/损失读出；
  - `contact_logits`：`(B, tw, 2)` 足接触 logits；
  - `route_gate`：`(B, tw, 10, 1)` 的 root/pose T 门；
  - `z_root` / `z_parts` / `z`：pose head 前的根 token、部位 token 与拼接 token；
  - `v_tok` / `t_tok` / `m_v` / `m_t`：编码器/时序编码后的 V/T token；
  - `anchor_trans` / `anchor_scale`：平移锚头旁路输出（不进 `trans_hat`）。

## 3. 总览数据流

```text
V_feat (B,tw,2051) ── v_enc ── v_temporal(4L) ── M_V (B,tw,d)
T_raw (B,tw,108) ─┐
T_phys(B,tw,12)  ─┴─ foot_conv ── t_temporal(4L) ── M_T (B,tw,d)
                                                    │
                         ┌──────────────────────────┘
                         ▼
        motion queries: root query (1) + 9 part queries
        q = [q_root, q_parts]  → (B, tw, 10, d) → (B, tw*10, d)
                         │
                         ▼
        6 × _V7DecoderLayer:
          self-attn(q)                       # tw*10 序列，非因果
          + cross-attn(M_V) * v_present      # V 支路硬关闭
          + cross-attn(M_T) * gate * t_present  # 双 T 门
          + FFN
                         │
                         ▼
         z_root (B,tw,1,d), z_parts (B,tw,9,d)
           ├─ root_head  → [s, dx, dz, vy] → v_hat (B,tw,3)
           ├─ out_parts  → 6D/part → pose (B,tw,144)
           ├─ contact_head(t_tok[:, -tw:]) → contact_logits (B,tw,2)
           └─ anchor_trans_off(mean(M_V)) → anchor_trans (B,3)  # 旁路
```

## 4. 编码器

### 4.1 视觉分支

- `ModalEncoders` 的 V 编码器读 HRNet 2051 维/帧，投影到 `d=256`，输出 `v_tok (B,tw,256)`。
- `v_temporal = _temporal_encoder(d=256, layers=4, nhead=8, dropout=0.1)`。
- `m_v = self.v_temporal(v_tok)`，作为运动解码器的 V memory。
- V 的在场由 `v_present` 控制；V 不在场时 V 交叉注意力输出乘 0（硬关闭）。

### 4.2 触觉分支

- `T_tac = cat([T_raw (96), T_phys (12)], dim=-1)`，108 维/帧。
- `t_encoder="foot_conv"`：FootConv 编码器（左右足 + global token，内部融合后输出逐帧 token `t_tok (B,tw,256)`）。
- `t_temporal = _temporal_encoder(d=256, layers=4, nhead=8, dropout=0.1)`。
- `m_t = self.t_temporal(t_tok)`，作为运动解码器的 T memory。
- T 的在场由 `t_present` 控制；T 不在场时 T 交叉注意力输出乘 0。

### 4.3 时间嵌入

`SharedEmbeddings` 提供窗口内时间位置编码；`obs_context=tw=20`（V8_Cv5 不做长历史/分窗）。

## 5. 运动解码器（`V7MotionHead`）

### 5.1 Query 组成

- 根速度 query：`root_query`，参数形状 `(1, tw, 1, d)`；
- 9 个部位 query：`part_queries`，形状 `(1, tw, 9, d)`；
- 每帧共 10 个 query 槽：`[root_velocity] + PART_NAMES`；
- 所有 query 加时间位置编码后拼成 `(B, tw, 10, d)`，再展平为 `(B, tw*10, d)` 进入 6 层解码器。

`PART_NAMES`（同时也是部位 query 顺序）：

```text
root, torso, headneck, l_arm, r_arm, l_leg, r_leg, l_foot, r_foot
```

注意：**根速度 query（槽 0）与姿态分组 `root`（part 0）是两个不同位置**。根速度由 `root_head` 读 `z_root`；`root` 部位的 6D 旋转由 `out_parts[0]` 读 `z_parts[...,0,:]`。

### 5.2 单层解码器

`_V7DecoderLayer` 的一层：

```text
h  = LayerNorm(q + SelfAttn(q, q, q))                 # tw*10 序列，非因果
v  = CrossAttn(h, M_V, M_V)                           # V memory
h  = LayerNorm(h + v * v_present)                     # 无 V 门
t  = CrossAttn(h, M_T, M_T)                           # T memory
t  = t * gate                                         # (B,tw,10,1) 双 T 门
h  = LayerNorm(h + t * t_present)
h  = LayerNorm(h + FFN(h))
```

- 自注意力覆盖整个 `tw*10` 序列；V8_Cv5 没有因果掩码（`ar`/`ar2` 才启用）。
- V/T 交叉注意力的 key/value 是逐帧 memory `(B,tw,d)`；T 门按 `(B,tw,10,1)` 乘在每个 query 槽上。
- 6 层共享同一组 `gate`；门在解码器外计算一次，不逐层重算（`route_conditional` 变体才在层内重算）。

### 5.3 双独立 T 门

门的计算（`AnySoleModelV7._route_gate`）：

```text
x = cat(v_tok, t_tok).mean(dim=1)          # (B, 2d) 窗口级
g_root = sigmoid(route_root_mlp(x))        # (B, 1)
g_pose = sigmoid(route_mlp(x))             # (B, 1)
gates  = cat([g_root, g_pose.expand(-1, 9)], dim=-1)   # (B, 10)
gates  = _expand_gate(gates)               # (B, tw, 10, 1)
```

- `route_mlp` 是姿态门；`route_root_mlp` 在 `task_routes` 下由 `route_mlp` 深拷贝，初始化相同、训练后独立。
- 门输入是 V/T 编码 token 的窗口均值；V8_Cv5 不做逐帧门（`route_temporal=False`）也不做四组门（`part_routes=False`）。
- 门值域 `(0,1)`；V8_Cv5 没有 `gate_2sigma`（值域 `(0,2)`）和 `v_gate`（V 侧门）。
- 门只调 T 交叉注意力：`g_root` 作用在槽 0（根速度），`g_pose` 作用在槽 1–9（9 个姿态部位）。V 分支始终是 `v_present` 硬开关。

### 5.4 读出

```text
z_root  = q[:, :, 0:1, :]                # (B,tw,1,d)
z_parts = q[:, :, 1:,  :]                # (B,tw,9,d)

pose_parts = concat([out_parts[p](z_parts[...,p,:]) for p in range(9)], -1)
pose       = pose_parts[..., part_place_idx] * pose_std + pose_mean
             # (B,tw,144)，part_place_idx 按 SMPL-24 正典关节序拼装

root_out   = root_head(z_root)[:, :, 0, :]   # (B,tw,4)，SD 非 T1
contact    = contact_head(t_tok[:, -tw:])    # (B,tw,2)
```

- `out_parts[p]`：每个部位组一个 `Linear(256, 6 × len(PART_JOINTS[p]))`；
- `part_place_idx` 把 9 组 6D 放回 24 关节正典序；
- `pose_std/pose_mean` 是训练集姿态统计（`train.py` 统一拟合）；
- `contact_head` 是 `V41ContactHead(dim=d)`，输出左右足接触 logits。

## 6. 根头 SD 四通道读出

V8_Cv5 的 `root_sd=True`、`root_sd_t1=False`，根头是：

```text
root_head = Linear(d,d) → GELU → Linear(d,4)
root_out  = root_head(z_root)[:, :, 0, :]      # (B,tw,4)

s_hat  = softplus(root_out[...,0])             # 水平速率 ≥ 0
d_raw  = root_out[...,1:3]                     # 水平方向原始向量 (x,z)
d_hat  = normalize(d_raw, eps=1e-6)            # 水平单位方向
vy_hat = root_out[...,3]                       # 垂直分量
sd_dir_norm = ||d_raw||                        # 诊断量

v_hat = [s_hat * d_hat[...,0], vy_hat, s_hat * d_hat[...,1]]
        # (B,tw,3)，世界系 m/s，y-up；下游 cumsum / 锚 / FK 契约不变
```

- 轴约定：y-up，forward_axis=+Z；水平面 = (x,z) = 下标 [0,2]。
- 水平方向用单位向量 + 速率解耦，静止帧 `d_raw` 模长趋零是合法状态（只登记 `sd_dir_norm`，不报错）。
- **T 行方向掩码**：方向项只在 `config_id != T` 的行监督；T 行（压力-only）方向不可观测，不能惩罚其方向。速率与垂直分量全行监督。
- **SD2 配平损失**（`v7_root_sd_balance=True`）：利用
  `||v−v_gt||²_h = (s−s_gt)² + 2·s·s_gt·(1−cos)`
  把速度项拆成 `L_speed + L_vy + L_dir`，三项各乘 `1/3`；全权重 1.0 且 T 行不掩码时三项之和 ≡ Cv5 的 `l_vel`（浮点结合序内）。多尺度位移项对 Δ 向量用同一恒等式拆成径向 / 横向 / 垂直，避免径向重复计权。SD2 只改损失形式，不改结构。
- `v_hat` 的形状与语义与 3 通道版一致；`trans_hat = cumsum(v_hat)/FPS`（模型内不加锚）。

## 7. 平移锚头（旁路）

`trans_anchor_head=True` 时：

```text
feat = M_V.mean(dim=1)                    # (B,d)
feat = feat.detach()                      # stopgrad=True：锚头不回传主干
anchor_trans = anchor_mean + anchor_scale * anchor_trans_off(feat)
               # anchor_trans_off = Linear(d,d) → GELU → Linear(d,3)，末层零初始化
               # (B,3)，单位 m
anchor_scale = anchor_scale.view(1,3).expand(B,3)
```

- `anchor_mean` / `anchor_scale` 是按轴统计量，训练前由 `fit_anchor_stats` 在训练 loader 上拟合（`std` 下限 `1e-2 m`）。
- 零初始化 ⇒ 初始锚 = `anchor_mean`（最佳常数地板）；`stopgrad=True` ⇒ `∂L_tanchor/∂θ_主干 ≡ 0`，主干每一步与不带锚头时相同（零税契约）。
- **旁路读出**：`anchor_trans` 不进入 `trans_hat`，不改变 `trans_hat` 的形状/语义；评估主列仍吃 `batch["trans_anchor"]`（GT 锚），自锚列用该头输出。`trans_anchor_source="joint"`。
- 损失：`L_tanchor = smooth_l1((anchor_trans − trans_anchor)/scale)`，只对 V 在场行（`config_id != T`）监督；`lambda_tanchor=1.0`。
- 也支持事后挂载：`attach_trans_anchor_head(state_dict, mean, scale, source="post_hoc")`，结构与联合训练版相同，仅 provenance 不同。

## 8. 三条件枚举

V8_Cv5 的“枚举”不是额外模块，而是训练分布 + 同一前向的输入组合：

- `config_id ∈ {VT, V, T}`；
- `v_present = (config_id != T)`，`t_present = (config_id != V)`；
- 缺失模态：`condition_inputs` 把该流置零，模型内再用 `v_present/t_present` 硬关闭对应交叉注意力（输入置零与硬关闭是同一决定）；
- 训练 `train_args.condition_sampling="enum"`，`config_probs=[0.50, 0.25, 0.25]`（VT/V/T）；每个 batch 枚举三条件；
- 评估按条件导出 `*_VT2M.npz` / `*_V2M.npz` / `*_T2M.npz`，结构不变。

### 8.1 视觉失效后的 T-only 与 IMU 边界（重要）

V8_Cv5 的 T 分支是**压力-only**，不是“压力 + IMU”：

- `T_raw` = 96 维压力格（左 48 + 右 48）；`T_phys` = 12 维压力物理量；`T_tac = 108` 维进 FootConv。
- `t_encoder="foot_conv"` 在 `encoders.py` 中**要求 `tactile_input="raw108"`**；raw108 没有 IMU 通道。
- `T_s2m`（Step2Motion 口径 50 维）确实含 IMU：`S2M_IMU_COLUMNS = 16..21 + 41..46`（12 通道），`no_imu=True` 时删掉这些列得到 38 维。但它只在 `tactile_input="s2m50"` 的变体里进 T 编码器；V8_Cv5 的 forward 虽然接收 `T_s2m` 参数，**T 编码器不消费它**（V8_Cv5 没有 `route_router`；`T_s2m` 只在 V8_Cv8 的路由门特征、或 AR2 变体里使用）。

因此，“视觉失效后”要区分四条路径：

| 路径 | V 输入 | T 输入 | IMU | 对应模型/条件 |
|---|---|---|---|---|
| **VT（遮挡缓存）** | `cam3_occ_scene` 的 HRNet（被柱子污染，**不是零**） | raw108 压力 | 无 | V8_Cv5 `VT2M` |
| **T-only（主模型 T 通道，干净视觉失效）** | 置零 + `v_present=0` 硬关闭 | raw108 压力 | 无 | V8_Cv5 `T2M`（`config_id=T`）；**不是 Anysolo-TI** |
| **主模型 T 通道 + IMU（s2m50 变体）** | — | `T_s2m` 50 维（压力 + IMU 12） | 有 | `tactile_input="s2m50"` 的**主模型变体**（E6.6，T 编码器换成 50 维输入）；**不是 Anysolo-TI** |
2. “压力 + IMU 联合 / T_s2m”是哪一个？是 Anysolo-TI 吗？
不是 Anysolo-TI。 它指的是主模型的：
tactile_input="s2m50" 的 V8_Cv5 变体（E6.6）
  T 编码器输入 = T_s2m 50 维（压力 + 12 维 IMU）
它仍然是主模型（有 V/T 解码器、姿态头、根头），只是把 T 编码器从 raw108 换成了 50 维 T_s2m。Anysolo-TI 完全不经过 V8_Cv5 主干，它是独立的 contact+ESKF 轨迹链。

| **轨迹级 IMU+T（独立模型）** | — | 压力/接触 | 12 通道 | **Anysolo-TI（经典 contact+ESKF）/ GP-SSM / Hybrid（学习式）**；均不进 V8_Cv5 主干 |
Hybrid 是其中一种，但不是只有 Hybrid。 这是一个类别：
┌───────────────────┬───────────────────────────────────────┐
│ 轨迹级 IMU+T 成员 │                 类型                  │
├───────────────────┼───────────────────────────────────────┤
│ Anysolo-TI        │ 经典 TI：压力 contact/step + IMU ESKF │
├───────────────────┼───────────────────────────────────────┤
│ GP-SSM            │ 压力相位 + IMU 的小状态模型           │
├───────────────────┼───────────────────────────────────────┤
│ Hybrid            │ 学习式 44 维 IMU+压力/F2 → yaw 帧位移 │
└───────────────────┴───────────────────────────────────────┘


**命名对照（避免混淆）**：

- **T-only** = 主模型 V8_Cv5 的 `T2M` 条件，走主模型的 T 通道（pressure-only）；不是 Anysolo-TI。
- **“压力 + IMU 联合 / T_s2m”** = 主模型的 `tactile_input="s2m50"` 变体：T 编码器吃 50 维 `T_s2m`（压力 + 12 IMU）；它仍然有主模型的 V/T 解码器和姿态头。**不是 Anysolo-TI**。
- **“轨迹级 IMU+T”** 是一个类别：**Anysolo-TI** 是经典 contact+ESKF 轨迹链；**GP-SSM** 是压力相位 + IMU 的小状态模型；**Hybrid** 是学习式 44 维 yaw 帧位移模型。E9 表里的 `Anysolo-TI` 和 `Hybrid` 是两臂，均为独立轨迹模型，不经过 V8_Cv5 的 T 编码器。

需要特别写清的三点：

1. **遮挡 VT 不是 T-only**：`cam3_occ_scene` 只是把柱子合成进图像，V 分支仍然收到（被污染的）V 特征；所以它在全遮挡时的输出是 learned prior，不是“视觉关闭”。真正的视觉失效对照是 `config_id=T` / `T2M`。
2. **V8_Cv5 主干没有 IMU+T 联合**：如果要在“视觉失效后”把 IMU 与压力一起喂进主模型，现有最近路径是 `tactile_input="s2m50"`（`T_s2m` 50 维，含 12 IMU）；V8_Cv5 本身仍是 raw108。轨迹级的 IMU+T 是另一套模型（Anysolo-TI / GP-SSM / Hybrid），不经过 V8_Cv5 的 T 编码器。
3. **T-only ≠ Anysolo-TI**：T-only 是主模型的条件化前向；Anysolo-TI 是独立的经典轨迹链（contact/step + ESKF）。两者只在“都没有视觉”这一点上同类，输入/结构/输出都不同。

## 9. 损失（与结构相关的部分）

V8_Cv5 的主损失（`compute_losses_v7` + SD2）：

| 项 | 默认权重 | 作用 |
|---|---|---|
| `L_pose` | 3.0 | `x0_hat` vs `pose_gt`，全样本 |
| `L_traj` | 1.0 | SD2：`L_speed + L_vy + L_dir`（T 行方向掩码）+ 多尺度位移的径向/横向/垂直分量 |
| `L_contact` | 按配置 | f6_soft pos-weighted BCE + air 加权，仅 T 在场窗口 |
| `L_tanchor` | 1.0 | 归一化锚残差 smooth_l1，仅 V 在场行 |
| `L_route` | 0.0 | 门向 0.5 的温和正则（V8_Cv5 默认不开） |
| `L_foot` | 0.0 | 支撑帧足-地一致性（仅开 `v7_contact_root` 的变体） |

`L_pose` 3.0 / `L_traj` 1.0 / `lambda_tanchor` 1.0 与配置一致；接触项的实际权重以 `effective_config.json` 为准。

## 10. 初始化与零税契约

| 组件 | 初始化 | 目的 |
|---|---|---|
| `root_query` / `part_queries` | N(0, 0.02) | 运动 query 起点 |
| `route_mlp`（姿态 T 门） | 默认初始化 | 端到端学习 |
| `route_root_mlp`（根 T 门） | `deepcopy(route_mlp)` | 初始化等价，训练后独立 |
| `anchor_trans_off` 末层 | 零 | 初始锚 = `anchor_mean`；不扰动主干 |
| `trans_anchor_stopgrad` | True | 锚头梯度不回传主干 |
| `v_gate` / `gate_2sigma` | False | V8_Cv5 不启用 V 门/2σ 门 |
| AR/AR2/ctx 组件 | 不构造 | V8_Cv5 双向非因果 |

注意：V8_Cv5 的 `root_head` 是 4 通道，和 V7_Cv5 的 3 通道不是逐参数同构；它与 V7_Cv5SD2 逐参数相同。V8_Cv5 的“零税”只指锚头旁路（stopgrad + 零初始化）和门控的初始化等价契约，不是指与 V7_Cv5 的权重可直接互换。

## 11. 形状表

| 张量 | 形状 | 说明 |
|---|---|---|
| `V_feat` | `(B, tw, 2051)` | HRNet 逐帧特征 |
| `T_raw` | `(B, tw, 96)` | 左足 48 + 右足 48 归一化压力格 |
| `T_phys` | `(B, tw, 12)` | 压力物理量（`t_phys_v2=False`） |
| `T_tac` | `(B, tw, 108)` | `cat(T_raw, T_phys)`，FootConv 输入 |
| `v_tok` / `t_tok` | `(B, tw, 256)` | 编码器 token |
| `m_v` / `m_t` | `(B, tw, 256)` | 4 层时序编码后的 memory |
| `q_root` | `(B, tw, 1, 256)` | 根速度 query |
| `q_parts` | `(B, tw, 9, 256)` | 9 个部位 query |
| `q` | `(B, tw*10, 256)` | 展平后的解码器输入 |
| `z_root` | `(B, tw, 1, 256)` | 根 token（pose/root head 前） |
| `z_parts` | `(B, tw, 9, 256)` | 部位 token |
| `pose` | `(B, tw, 144)` | 24 关节 × 6D |
| `root_out` | `(B, tw, 4)` | SD 四通道 |
| `v_hat` | `(B, tw, 3)` | 世界系根速度 m/s |
| `trans_hat` | `(B, tw, 3)` | `cumsum(v_hat)/FPS` |
| `contact_logits` | `(B, tw, 2)` | 左右足接触 |
| `route_gate` | `(B, tw, 10, 1)` | root/pose T 门 |
| `anchor_trans` | `(B, 3)` | 平移锚旁路 |

## 12. 代码/配置索引

| 内容 | 位置 |
|---|---|
| V8 命名空间入口 | `anysole/models/v8_cv5.py` |
| 实际结构 | `anysole/models/v7_c2sd2.py` |
| 模型主体 | `anysole/models/common/model_v7.py`（`AnySoleModelV7`、`V7MotionHead`、`_V7DecoderLayer`） |
| 编码器 | `anysole/models/common/encoders.py`、`embeddings.py` |
| SMPL-24 协议 / 部位 | `anysole/types.py` |
| SD2 / 锚头损失 | `anysole/utils/losses.py` |
| 注册 | `anysole/registry.py`（`V8_Cv5` → `v8_cv5`） |
| 折配置 | `anysole/configs/v1_fold_*.yaml` |
| 典型 cell | `results/cells/V8_Cv5_<fold>_s1`（checkpoint 名 `V7_Cv5SD2_<fold>_...`） |

典型配置：`d_model=256`、`tw=20`、`dropout=0.1`、`pose_layers=6`、`nhead=8`、`dim_feedforward=1024`、`batch_size=256`、`lr=1e-4`、`stride=20`、`traj_deltas=[2,4,8,19]`、`lambda_pose=3.0`、`lambda_traj=1.0`、`lambda_tanchor=1.0`、`contact_method=f6_soft_time_accel_state_smpl24_v1`、`modal=anysolev2`、`v_input=hrnet`、`tactile_input=raw108`、`t_encoder=foot_conv`。

## 13. 与相邻变体的差异（只列结构）

| 变体 | 与 V8_Cv5 的结构差异 |
|---|---|
| V8_Cv4 | 单门（只有姿态 T 门），其余同 |
| V8_Cv2 | 双门 + 随机 dropout；训练条件分布不同 |
| V8_Cv6 | 双门 + 始终 VT（不训练缺失条件） |
| V8_Cv7 | 无门（`task_routes=False` / 无 T 门） |
| V8_Cv8 | 竞争路由门 `k=σ(s_v−s_t)` 取代 T/V 门 |

## 14. 一句话总结

V8_Cv5 的结构可以记成：

```text
HRNet V + raw108/phys T
  → 双流编码（各 4 层时序 Transformer）
  → root + 9 part queries 的 6 层共享解码器
      （自注意力 + V 交叉注意力硬开关 + T 交叉注意力 × 双独立 T 门）
  → SD 四通道根头 + 9 部位 6D 头 + 接触头
  → 根速度 cumsum 得轨迹；平移锚头只作旁路读出
```

它和 V7_Cv5SD2 是同一套结构；V8 只是当前分支给这个基座的命名空间，用于在同一个基座上重建门控/条件消融。
