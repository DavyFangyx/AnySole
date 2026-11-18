# V8_IMU 模型结构与训练配方

日期：2026-11-15。范围：当前主模型 `V8_IMU_Residual`（V2 管线）及其消融/分支的**结构**与**训练配方**。本文不展开主表结果；文末只给当前产物的口径与已知注意事项。

主要来源：

- `anysole/models/v8_cv7_physimu.py`（主模型基座入口）
- `anysole/models/v8_cv5_physimu.py`（`V8_IMU_TGate`）
- `anysole/models/v8_cv7_physimu_gate.py`（`V8_IMU_DualGate`）
- `anysole/models/v8_cv7_physimu_router.py`（`V8_IMU_Router*`）
- `anysole/models/common/model_v7.py`、`anysole/models/common/encoders.py`
- `configs/z_gen/model_registry.py`、`anysole/registry.py`
- `results/legacy_name_map.md`
- `results/diagnostics/20261112_v8_gate_frozen/finetune_router_enum.py`
- `results/diagnostics/20261112_v8_gate_frozen/router_ft_only_<fold>_v2rerun.pt` 的 `meta`
- `results/diagnostics/20261112_v8_gate_frozen/summary_v2rerun.md`
- `results/diagnostics/20261114_imu_ft_ablation_v2/experiments.md`

## 0. 一句话

**V8_IMU_Residual = V8_Cv7_PhysIMU（无门控全量 V/T 融合 + PhysIMU 两阶段触觉/IMU 融合 + SD2 根头 + 平移锚头）经 V/T/VT enum 预训练（740 ep）后，做遮挡感知 enum FT（20 ep，best-val，V2：sequential + batch val）；结构不变，只更新 T_raw/T_phys 编码、T 时序与姿态头；IMU encoder/fusion 与 V 编码/时序冻结。主模型推理用 VT。**

命名中的 **Residual** 指**无门控的全量 V/T 融合**（`gV=gT=1`）；代码中没有单独的 “residual fusion” 开关，`vt_t_residual` 也未启用。它是命名约定，不是结构开关。

## 1. 命名与角色

| 角色 | 新名 | 旧名 / 入口 | 权重 / 状态 |
|---|---|---|---|
| **主模型**（enum FT，20 ep，best-val） | `V8_IMU_Residual` | `V8_Cv7_PhysIMU_FT` / `v8_cv7_physimu` | `router_ft_only_<fold>_v2rerun.pt` |
| 未适配基座（enum base，740 ep） | `V8_IMU_Residual_woAdapt` | `V8_Cv7_PhysIMU` / `v8_cv7_physimu` | `results/AnySole/V8_Cv7_PhysIMU_<fold>_.../ckpt_last.pt` |
| 竞争式路由（正文，router-only） | `V8_IMU_Router` | `V8_Cv7_PhysIMU_Router_Frozen` / `v8_cv7_physimu_router` | v3 ckpt 已删；历史数字见 `summary_v3.md` |
| 竞争式路由（full，补充） | `V8_IMU_Router_Full` | `V8_Cv7_PhysIMU_Router` / `v8_cv7_physimu_router` | v3 ckpt 已删；v2 full rerun：`router_ft_enum_cr_<fold>_l1.0_v2rerun.pt` |
| 传感分支门控（Cv5 dual T gate） | `V8_IMU_TGate` | `V8_Cv5_PhysIMU` / `v8_cv5_physimu` | `results/AnySole/V8_Cv5_PhysIMU_<fold>_.../ckpt_last.pt` |
| 独立双分支门控（post-stage gV/gT） | `V8_IMU_DualGate` | `V8_Cv7_PhysIMU_Gate` / `v8_cv7_physimu_gate` | `gate_4fold_<fold>.pt` |
| no-enum 基座 | `V8_IMU_Base_woEnum` | `V8_Cv7_PhysIMU_NoEnum` | `results/AnySole/V8_Cv7_PhysIMU_NoEnum_<fold>_s1_.../ckpt_last.pt` |
| no-enum FT（单折试点） | `V8_IMU_Residual_woEnum` | 新建 | `results/diagnostics/20261114_imu_ft_ablation_v2/` |
| decoder-only FT（单折试点） | `V8_IMU_Residual_DecoderOnly` | 新建 | 同上 |
| clean-only FT（单折试点） | `V8_IMU_Residual_CleanOnly` | 新建 | 同上 |

口径：当前主训练管线是 **V2**（`--cond-mode sequential`、`--val-mode batch`）。v3（`joint` + `sample` val）保留为可选/历史协议，v3 的 FT 数字不进主表。融合比较正文用 router-only，full router 只作补充；主模型是 `V8_IMU_Residual`。

## 2. 主模型结构

### 2.1 基座 `V8_IMU_Residual_woAdapt`（= `V8_Cv7_PhysIMU`）

`v8_cv7_physimu.py` 的 STRUCTURE 由 `v7_a` 基座叠加：

```python
# v7_a.py
STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v7": True,
    "v7_contact_root": False,
    "v7_reliability_route": False,
}
# v8_cv7_physimu.py
STRUCTURE = {**BASE_STRUCTURE, "tactile_input": "raw96_phys_imu",
             "v7_root_sd": True, "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True}
```

模型开关：

| 开关 | 值 | 含义 |
|---|---|---|
| `tactile_input` | `raw96_phys_imu` | T_raw 96 → FootConv；T_phys 12 与 IMU 12 各走独立 MLP，再两阶段融合 |
| `reliability_route` | False | **无 T 信任门** |
| `v_gate` / `gate_2sigma` | False | **无 V 门** |
| `route_router` | False | **无竞争路由** |
| `v7_root_sd` | True | 根头四通道：水平速率 / 水平方向 / 垂直 |
| `v7_root_sd_balance` | True | SD2 恒等式配平损失（只改损失） |
| `v7_trans_anchor_head` | True | 平移锚头旁路（stopgrad 默认 True；`lambda_tanchor=1.0`） |
| `vt_t_residual` | False | 未启用零初始化触觉残差 |
| `t_phys_v2` | False | T_phys 12 维口径（不是 20 维） |
| `ar` / `ar2` / `ctx*` | False | 双向、非因果、非 AR/续写 |

### 2.2 输入/输出契约

- 运动协议：SMPL-24，`POSE_DIM = 24 × 6 = 144`；关节顺序由 `anysole/types.py` 的 `PART_NAMES` / `PART_JOINTS` 唯一给出。
- 窗长 `tw = 20`（40 Hz 下 0.5 s）；训练/评估窗步长 `stride = 20`（非重叠）。
- 视觉输入 `V_feat`：HRNet 特征，2051 维/帧。
- 触觉/IMU：`T_raw` 96 维（左足 48 + 右足 48）；`T_phys` 12 维；`T_s2m` 50 维（PhysIMU 取其中 12 个 IMU 通道）。
- 输出：`x0_hat (B,tw,144)`、`v_hat (B,tw,3)`、`trans_hat (B,tw,3)`、`contact_logits (B,tw,2)`、SD 四通道诊断读出、`anchor_trans` 旁路。

### 2.3 数据流

```text
V_feat (B,tw,2051) ── v_enc ── v_temporal(4L) ── M_V (B,tw,256)
T_raw  (B,tw,96)  ── FootConv ─┐
T_phys (B,tw,12)  ── MLP ──────┼─ PhysIMUFusion(两阶段) ── t_temporal(4L) ── M_T
T_s2m  (B,tw,50)  ── IMU MLP ──┘
                                  │
         motion queries: root(1) + 9 part queries → (B,tw,10,256)
                                  │
         6 × _V7DecoderLayer:
           self-attn(q) + cross-attn(M_V)*v_present
                       + cross-attn(M_T)*t_present + FFN
                                  │
         z_root / z_parts → root_head(SD 4通道) / out_parts(pose) / contact_head
                          → anchor_trans_off(mean(M_V))  # 旁路，不进 trans_hat
```

### 2.4 PhysIMU 两阶段融合（`encoders.py` 的 `PhysIMUFusion`）

- `raw96_phys_imu` 下 FootConv **只**看 T_raw 96；`T_phys` 走 `ChannelMLPEncoder`；`T_s2m` 的 IMU 通道走 `IMUEncoder`。
- 阶段 1：`[T_phys embedding, IMU embedding] → 2d → d`，LayerNorm；
- 阶段 2：`[T_raw FootConv token, 阶段1 physical state] → 2d → d`，LayerNorm；
- 融合后的 `t_tok` 形状不变 `(B,tw,256)`，因此下游 T 门 / 解码器契约不变。
- 与 `IMUConcatFusion` 不同，这里**没有** step-0 逐位基座契约（压力编码器从 raw108 改成 raw96），该基座是从零训练的。

### 2.5 根头 / SD2 / 锚头

- `v7_root_sd=True`：根头输出 `[s, dx, dz, vy]` 四通道，组装成世界系根速度 `v_hat`；`trans_hat = cumsum(v_hat)/FPS`（模型内不加锚）。
- `v7_root_sd_balance=True`：SD2 恒等式配平损失；只改损失，不改结构。
- `v7_trans_anchor_head=True`：`anchor_trans_off(mean(M_V))` 旁路输出；`trans_anchor_stopgrad=True`（默认）时锚头不回传主干；`lambda_tanchor=1.0`；锚头输出不进 `trans_hat`。

## 3. 主模型训练配方

### 3.1 基座 enum 预训练（`V8_IMU_Residual_woAdapt`）

- 结构：2.1 的 `V8_Cv7_PhysIMU`；从零训练。
- 数据/协议：`f6_soft_time_accel_state_smpl24_v1`、`raw96_phys_imu`、`tw20 / st20`、SMPL-24。
- 训练：740 epoch，batch 256，lr 1e-4，warmup 0.05，grad clip 5，seed 1，存 `ckpt_last`。
- 条件采样：`config_probs [0.50, 0.25, 0.25]`（VT / V / T；见 `anysole/configs/v1_fold_*.yaml:50`）。
- 损失：V8_Cv7 基座配方（`lp3/lt1/lk1` = `lambda_pose=3`、`lambda_traj=1`、`lambda_kp=1`；`lambda_contact=1.0`（配置默认）；SD2 balance；`lambda_tanchor=1.0`；其余按 V8_Cv7）。
- 权重路径：`results/AnySole/V8_Cv7_PhysIMU_<fold>_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_last.pt`。

### 3.2 遮挡感知 enum FT（主模型 `V8_IMU_Residual`）

- 初始化：3.1 的 `ckpt_last`；**结构不变**。
- 脚本：`results/diagnostics/20261112_v8_gate_frozen/finetune_router_enum.py`。
- 命令（V2）：

```bash
python results/diagnostics/20261112_v8_gate_frozen/finetune_router_enum.py \
  --gate-ckpt <V8_Cv7_PhysIMU_<fold>.../ckpt_last.pt> \
  --split-csv AnysoleWorkspace/protocol/splits/default/splits_test_<...>_val_<...>.csv \
  --clean-cache-root .../cam3 --occ020-cache-root .../cam3_occ_lower020 \
  --occ050-cache-root .../cam3_occ_lower050 \
  --degraded-cache-root .../cam3_occ_lower100 \
  --batch-size 64 --epochs 20 --lr 1e-4 --lambda-clean-rel 1.0 \
  --cond-mode sequential --val-mode batch --no-router \
  --out results/diagnostics/20261112_v8_gate_frozen/router_ft_only_<fold>_v2rerun.pt
```

- 可训参数（ckpt meta 的 `trainable_prefixes`）：`pose_head`、`t_temporal`、`encoders.t_enc`（T_raw FootConv 编码器，含其内部 temporal 层）、`encoders.t_phys`（T_phys MLP）。冻结：V 编码/时序、IMU encoder/fusion（`encoders.phys_imu_enc` / `encoders.phys_imu_fuse`）、锚头（`anchor_off`）。`pose_head` 内含 `root_head` / `out_parts`，因此**主模型**的 SD2 四通道根头（`V8_Cv7_PhysIMU` 继承的 `v7_root_sd`）随 FT 训练；SD2 balance 损失不在 FT 目标里。
- 条件：clean / occ020 / occ050 / occ100；条件权重 `0.5 / 1/6 / 1/6 / 1/6`。四档**只改 V 侧遮挡**，T/IMU 不变。
- 每步损失：同一 batch 内 V / T / VT 三路 enum 损失**等权 1/3**（`PA + MPJPE`，逐窗）；clean 额外加
  `lambda_clean_rel * relu(L_VT - L_V)`，其中 `L_V` detached，`lambda_clean_rel=1.0`。
- 优化器：单个 Adam；V2 = 逐条件顺序更新（每个 batch 一次 `opt.step()`），val = batch 均值（`tot/nb`），best-val 只看 val 列。
- 选点：best-val；四折 best epoch / val（batch-mean）：

| fold | best epoch | val |
|---|---:|---:|
| fS5S6S11 | 5 | 0.0961 |
| fS5S11S12 | 20 | 0.0850 |
| fS6S7 | 20 | 0.1003 |
| fS5S6S8 | 19 | 0.1167 |

- 主模型推理：**VT**（`summary_v2rerun.md` 的 FT@VT）；V-only 作为参照。

### 3.3 四折 split

| fold | split CSV | train | val | test |
|---|---|---|---:|---:|---:|
| fS5S6S11 | `splits_test_S5S6S11_val_S12.csv` | 62 | S12（12） | S5/S6/S11（66） |
| fS5S11S12 | `splits_test_S5S11S12_val_S13.csv` | 64 | S13（18） | S5/S11/S12（58） |
| fS6S7 | `splits_test_S6S7_val_S14.csv` | 73 | S14（18） | S6/S7（31） |
| fS5S6S8 | `splits_test_S5S6S8_val_S7.csv` | 78 | S7（11） | S5/S6/S8（51） |

### 3.4 当前产物与口径

- 主 ckpt：`router_ft_only_<fold>_v2rerun.pt`（best）；`_v2rerun_last.pt` 为同一次运行的 last，不作主模型。
- 四折 proxy（`perf_table_router.py`，PA/MPJPE mm，总体 std；**不是**标准 `anysole.eval`）：

| 行 | clean | occ020 | occ050 | occ100 |
|---|---|---|---|---|
| FT@V（参照） | 42.28±3.06/55.79±4.13 | 42.64±3.04/56.78±4.08 | 50.93±2.85/64.98±4.76 | 80.03±5.20/132.02±10.94 |
| **FT@VT（主模型）** | **41.71±2.91/54.68±4.10** | **41.83±2.80/55.11±4.06** | **44.51±2.28/59.09±3.94** | **59.89±7.09/106.19±8.52** |
| full router（补充） | 42.29±2.95/59.37±3.91 | 42.64±2.73/61.12±3.86 | 47.48±2.20/73.16±4.80 | 61.71±6.42/118.99±6.62 |

- **标准 `anysole.eval` 尚未对 V2 主模型重跑**；论文标准指标（T2M RootOrientationDrift 等）需在 V2 主 ckpt 上单独跑，不能与 v3 的 standard 表混用。
- v3 FT 的 `_v3.pt` 不是主模型；v3 数字只作历史。

## 4. 消融 / 分支

### 4.1 `V8_IMU_Base_woEnum`（no-enum 基座）

- 结构同 2.1；740 ep；`config_probs [1, 0, 0]`（只走 VT）。
- 结果口径：**V-only 行 OOD**（V-only 的 config embedding / 缺 T 路径未训练）；只比 VT 或 router，不能把 v 行当基线。
- 权重：`results/AnySole/V8_Cv7_PhysIMU_NoEnum_<fold>_s1_.../ckpt_last.pt`。

### 4.2 `V8_IMU_Residual_woEnum`（no-enum base + FT，V2 单折试点）

- 基座 4.1；FT 配方同 3.2（`--no-router`，四条件，V2 sequential + batch val，20 ep，lr 1e-4，batch 64，λ=1.0）。
- 状态：单折 fS5S6S11 试点；目录内有四折 ckpt，但下表只报 fS5S6S11。
- 单折 VT：clean 44.30/57.80、occ020 43.90/57.42、occ050 47.07/61.81、occ100 69.84/116.38；best_ep 9。
- 判读：本折 VT 优于主模型本折，但主模型 fS5S6S11 的 best-val 在 epoch 5（四折离群点）；单折不能判定 enum 不必要，需四折确认。

### 4.3 `V8_IMU_Residual_DecoderOnly`（只训 pose_head，V2 单折试点）

- 基座 `V8_IMU_Residual_woAdapt`；`--no-router --decoder-only`（`train_prefixes = pose_head`）；四条件；V2；20 ep。
- 单折 VT：clean 48.27/65.46、occ020 48.19/65.90、occ050 50.50/70.45、occ100 69.70/121.34；best_ep 17。
- 判读：clean/occ020/occ050 全面差；occ100 PA 略低但 MPJPE 更差，整体不如完整 FT。

### 4.4 `V8_IMU_Residual_CleanOnly`（只用 clean，V2 单折试点）

- 基座 `V8_IMU_Residual_woAdapt`；`--no-router --conditions clean`；train/val/选点均只用 clean；20 ep。
- 单折 VT：clean 44.25/59.44、occ020 46.20/62.01、occ050 60.59/82.07、occ100 97.21/194.47；best_ep 4。
- 判读：clean 略好，occ050/100 明显崩 → 遮挡训练必要。

### 4.5 `V8_IMU_Router`（router-only，历史）

- 入口 `v8_cv7_physimu_router.py`：`route_router=True`，`k = σ(s_v − s_t)`（逐帧逐组，4 组 → 10 槽）；解码器 V 支路乘 `(1−k)`、T 支路乘 `k`；V-only 行 `k≡0`、T-only 行 `k≡1`；init `k≈0.018`。
- 训练：冻结 enum base，只训 router；v3 协议（joint + sample val）。
- 状态：**v3 ckpt 已删除**；历史数字见 `summary_v3.md`（clean 41.93±2.41/58.17±1.57、occ020 43.87±2.76/61.72±1.45）。无 v2 rerun。
- 角色：按 `legacy_name_map.md`，融合比较正文用 router-only；主模型仍是 Residual。

### 4.6 `V8_IMU_Router_Full`（full router，补充）

- 入口同 4.5；训练 = enum base + router + 联合微调（router、pose_head、t_temporal、t_enc、t_phys）；v3 协议。
- 状态：v3 ckpt 已删除；v2 full router rerun 为 `router_ft_enum_cr_<fold>_l1.0_v2rerun.pt`（V2 sequential + batch val，20 ep，λ=1.0）。
- 结果：v3（历史）clean 39.89±2.06/53.38±3.41、occ020 40.61±1.82/54.98±3.20；v2 rerun clean 42.29±2.95/59.37±3.91、occ020 42.64±2.73/61.12±3.86。
- **注意**：旧 v2 full router 的 clean 41.97/56.08、occ020 42.38/57.49 **未复现**（MPJPE 差 3.3–3.6 mm，主要 fS6S7/fS5S6S8）；full router 只作补充，不作正文主表。

### 4.7 `V8_IMU_TGate`（= `V8_Cv5_PhysIMU`）

- 结构：V8_Cv5（`v7_c2sd2`：SD2 根头 + 平移锚头 + **双独立 T 门** root/pose）+ PhysIMU 两阶段融合；`reliability_route=True`；**无 V 门**。
- 训练：740 ep，batch 256，lr 1e-4，warmup 0.05，grad clip 5，`config_probs [0.50,0.25,0.25]`，`f6_soft_time_accel_state_smpl24_v1`，`raw96_phys_imu`，`tw20/st20`，`ckpt_last`。
- 权重：`results/AnySole/V8_Cv5_PhysIMU_<fold>_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_last.pt`。
- 角色：传感分支门控对照；不是主模型。

### 4.8 `V8_IMU_DualGate`（= `V8_Cv7_PhysIMU_Gate`）

- 结构：`V8_Cv7_PhysIMU` 基座 + `v7_external_gate=True` 的 `GateController`；四组 **root / upper / left / right**，每组独立 `g_V`、`g_T`；基座冻结，门用 VT motion loss 训练。
- 设计初值 `g0`（`v8_cv7_physimu_gate.py` docstring，来自验证固定门扫描 2026-11-12）：root `(0.75, 0.0)`、upper `(1.0, 0.5)`、left `(1.0, 1.0)`、right `(1.0, 1.0)`；边界 0/1 用内点表示，偏差记录在 `GateController` buffer。实际 ckpt 的 `v7_gate_g0` 记录为结构默认 `[[1,1],...]`，训练后门值以 ckpt buffer 为准。
- 训练：`train_gate_paired.py`，paired clean vs E5-degraded（occ020/050/100）；脚本默认 30 ep、batch 256、lr 1e-3、`lambda_rank=0.2`、`margin=0.1`、`w_clean=0.5`、`w_deg=1/6`（实际四折以训练日志为准）；inner LSO（4 train subjects，held-out train subject 作 val）。
- 权重：`results/diagnostics/20261112_v8_gate_frozen/gate_4fold_<fold>.pt`。
- 角色：独立双分支门控对照；不是主模型。

## 5. 口径与注意事项

1. **V2 是当前主训练管线**：`--cond-mode sequential` + `--val-mode batch`；v3（`joint` + `sample`）只作可选/历史协议。主模型 ckpt 是 `_v2rerun`，不是 `_v3`。
2. **配置层**：FT 的 V/T/VT 三路是**等权 1/3**；基座预训练的 `config_probs [0.50,0.25,0.25]` 是另一阶段，别混写。若论文写“FT enum 损失按 config_probs 加权”，与代码不符。
3. **标准指标**：`summary_v2rerun.md` 的四折是 `perf_table_router.py` proxy（PA/MPJPE）；标准 `anysole.eval`（T2M RootOrientationDrift、FootSliding、GroundPen 等）尚未对 V2 主模型重跑。不要把 v3 的 standard 表与 V2 proxy 表混用。
4. **v2 full router 的 MPJPE 选点敏感**：旧 56.08 未复现（rerun 59.37）；full router 只作补充。
5. **单折试点**：`woEnum` / `DecoderOnly` / `CleanOnly` 目前只报 fS5S6S11；不要写成四折结论。
6. **no-enum 的 v 行 OOD**：只比 VT / router，不能当 V 基线。
7. **router 分支**：v3 ckpt 已删除；`V8_IMU_Router` 无 v2 rerun；`V8_IMU_Router_Full` 有 v2 rerun 但只作补充。
8. **锚头 / 根头**：`trans_anchor_stopgrad=True`（默认）；锚头输出不进 `trans_hat`；FT 不训锚头（`anchor_off`）。根头（`root_head`）在 `pose_head` 内，随 FT 训练；SD2 balance 损失不在 FT 目标里。
