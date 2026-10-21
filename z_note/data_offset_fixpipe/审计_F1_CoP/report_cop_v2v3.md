# 审计面 2（补）：CoP/相位 ridge 读出 — denoised108_v2/v3 + offset（2026-10-20）

**对象**：与审计面 1（补）同条件——tactile = 交接包 v2 / v3（视频轴，
不修正），GT（相位/CoP 目标）建在修正轴（--corrected-offset）。载体 =
V3_3B_joint_and ckpt（任务指定路径），CPU。

**产物**（本目录，全部为新增，未改任何既有文件）：
- `probe_v5_phase_cop_readout_denoised108_v2_offset.json`、`probe_run_v2_offset.log`
- `probe_v5_phase_cop_readout_denoised108_v3_offset.json`、`probe_run_v3_offset.log`
- `compare_v2v3_probe.py`（对比助手）→ `compare_v2v3_summary.json`（本报告
  数字的机器可读汇总）、`compare_v2v3.log`

## 口径

- 探针：`z_note/multimodal_fusion/V5_1方案/probe_v5_phase_cop_readout.py`，
  未改动。参数：`--tactile-input denoised108_v2|v3 --corrected-offset
  --device cpu`（GT 修正轴贯穿 dataset 与 session_gt，与 raw108+offset
  审计同一脚本同一式）。λ 网格、80/20 选参、配对 bootstrap n_boot=10000
  seed=0、常数基线均未动。
- **载体 ckpt（如实登记）**：任务指定
  `results/AnySole/V3_3B_joint_and/tw20_st20_lr0.0001_lp3_lt1_lk1_la0.05_ls0.01_
  sf0.1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_best.pt`（已确认存在；
  saved config：contact_method=joint_and、tactile_input=raw108，后者被
  --tactile-input 覆盖）。对比对象 raw108+offset JSON 的 provenance.ckpt =
  `V3_3B_denoised_joint_and`（与 anchor report_cop.md 登记一致）——两者
  ckpt 不同。影响面：raw108 读出（特征 = T 拼接）与 T 可读判据
  （const−raw108）**不含编码器 → 三版直接可比**；v_tok/vt 读出含编码器
  → anchor↔v2/v3 跨 ckpt 不可比（v2↔v3 同 ckpt 可比，判读只用这一对）。
- n_valid / const 三版全等（GT 掩码只依赖修正轴 GT，不依赖触觉版本）：
  phase_L 11145/3489、phase_R 11810/3800、cop_L 33196/10687、
  cop_R 32743/10862（train/test）。
- 运行：`--device cpu` + `CUDA_VISIBLE_DEVICES=""`，全程未碰 GPU。
- 如实登记：JSON `provenance.date` 为脚本硬编码 "2026-10-19"（脚本既有
  常量未改）；provenance 增记 tactile_input 与 corrected_offset 两字段。

## 数字

### 表 1：4 通道 × 3 读出 TEST MSE（const / λ / n_valid 同列）

| 通道 | 读出 | raw108+offset（anchor） | v2+offset | v3+offset | const / λ / n |
|---|---|---|---|---|---|
| phase_L | raw108 | 0.4654 | 0.4617 | 0.4622 | 0.5000 / 1e4 / 11145/3489 |
| phase_L | v_tok | 0.4158 | 0.4156 | 0.4156 | 0.5000 / 1e2 |
| phase_L | vt | 0.3886 | 0.3854 | 0.3845 | 0.5000 / 3.2e3 |
| phase_R | raw108 | 0.4505 | 0.4540 | 0.4516 | 0.5000 / 1e4 / 11810/3800 |
| phase_R | v_tok | 0.4345 | 0.4359 | 0.4359 | 0.5000 / 3.2e2 |
| phase_R | vt | 0.3779 | 0.3803 | 0.3837 | 0.5000 / 3.2e3（anchor 1e3） |
| cop_L | raw108 | 0.006891 | 0.006313 | 0.006336 | 0.0076 / 1e4 / 33196/10687 |
| cop_L | v_tok | 0.008484 | 0.008385 | 0.008385 | 0.0076 / 3.2e3 |
| cop_L | vt | 0.006273 | 0.005956 | 0.006091 | 0.0076 / 1e4 |
| cop_R | raw108 | 0.010939 | 0.010391 | 0.010412 | 0.0109 / 3.2e3 / 32743/10862 |
| cop_R | v_tok | 0.012412 | 0.012449 | 0.012449 | 0.0109 / 1e3 |
| cop_R | vt | 0.011672 | 0.011615 | 0.010893 | 0.0109 / 3.2e2（v3 3.2e3） |

全量 12×3 单元全部 ok，无 no_data。全精度值见 JSON（`results.*.mse_test`）。

### 表 2：差表（results mse_test，全精度）

| 通道 | 读出 | v2−anchor | v3−anchor | v3−v2 |
|---|---|---|---|---|
| phase_L | raw108 | −0.003701 | −0.003238 | +0.000462 |
| phase_L | v_tok | −0.000174 | −0.000174 | +0.000000 |
| phase_L | vt | −0.003204 | −0.004110 | −0.000906 |
| phase_R | raw108 | +0.003505 | +0.001167 | −0.002338 |
| phase_R | v_tok | +0.001385 | +0.001385 | +0.000000 |
| phase_R | vt | +0.002407 | +0.005847 | +0.003440 |
| cop_L | raw108 | −0.000578 | −0.000554 | +0.000023 |
| cop_L | v_tok | −0.000099 | −0.000099 | +0.000000 |
| cop_L | vt | −0.000317 | −0.000183 | +0.000135 |
| cop_R | raw108 | −0.000548 | −0.000527 | +0.000021 |
| cop_R | v_tok | +0.000037 | +0.000037 | +0.000000 |
| cop_R | vt | −0.000057 | −0.000779 | −0.000722 |

（raw108 读出 = 纯数据版本效应，三版可比；v_tok/vt 行的 anchor 差含 ckpt
效应，v3−v2 列为同 ckpt 纯数据版本差。）

### 表 3：T 可读显著性判定（const − raw108，无编码器，三版直接可比；
粗体 = CI 不含 0；FLIP = 判定翻转）

| 通道 | anchor（raw108+offset） | v2+offset | v3+offset | 格局 |
|---|---|---|---|---|
| phase_L | −0.0017 [−0.0348, +0.0303] | +0.0065 [−0.0216, +0.0352] | +0.0064 [−0.0215, +0.0349] | 同（不显著） |
| phase_R | −0.0077 [−0.0586, +0.0406] | −0.0122 [−0.0617, +0.0365] | −0.0098 [−0.0589, +0.0384] | 同（不显著） |
| cop_L | +0.0010 [−0.0001, +0.0019] | **+0.0016 [+0.0009, +0.0022]** | **+0.0016 [+0.0009, +0.0022]** | **FLIP 不显著→显著**（v2/v3 同） |
| cop_R | +0.0002 [−0.0005, +0.0009] | +0.0007 [−0.0000, +0.0013] | +0.0006 [−0.0000, +0.0013] | 同（不显著） |

### 表 4：V 更强 / vt>const（同 ckpt，v2↔v3；anchor ckpt 不同不列）

| 通道 | v2 V更强（raw108−v_tok） | v3 V更强 | v2 vt>const | v3 vt>const |
|---|---|---|---|---|
| phase_L | +0.0215 [−0.0484, +0.0882] ns | +0.0217 [−0.0484, +0.0883] ns | **+0.0786 [+0.0207, +0.1336] SIG** | **+0.0780 [+0.0200, +0.1330] SIG** |
| phase_R | +0.0300 [−0.0466, +0.0955] ns | +0.0277 [−0.0482, +0.0922] ns | +0.0468 [−0.0257, +0.1182] ns | +0.0443 [−0.0276, +0.1146] ns |
| cop_L | **−0.0019 [−0.0027, −0.0011] SIG** | **−0.0019 [−0.0027, −0.0011] SIG** | **+0.0022 [+0.0013, +0.0031] SIG** | **+0.0020 [+0.0012, +0.0030] SIG** |
| cop_R | **−0.0016 [−0.0028, −0.0004] SIG** | **−0.0016 [−0.0028, −0.0004] SIG** | −0.0001 [−0.0012, +0.0011] ns | +0.0004 [−0.0005, +0.0014] ns |

v2 与 v3 的 8 组判定**逐组同格局**（SIG/ns 全同，均值差 ≤ 2.5e-3）。

## 内置一致性检查

1. **v_tok 读出对触觉版本不变（应然成立，实测逐位成立）**：v_tok 特征 =
   V 编码器输出（输入只有 V_feat），与 T 数据版本无关；同 ckpt 同 GT 下
   v2/v3 的 v_tok mse_test 与 λ **逐位相等**（4 通道全部 bitwise equal），
   vs_const verdict 亦逐位相等；vs_raw108 verdict 因配对项含 raw108 读出
   而微差（±1.5e-5）。anchor↔v2/v3 的 v_tok 差（−1.7e-4 / +1.4e-3 /
   −9.9e-5 / +3.7e-5）= ckpt 效应（anchor = V3_3B_denoised 编码器，
   本跑 = V3_3B_joint_and 编码器），不参与判读。
2. **n_valid / const 三版全等**（表 1 n 列）：GT 掩码只依赖修正轴 GT，
   与触觉版本无关，实测逐位相同。
3. 运行确定性：两跑同机同条件，无随机源（bootstrap seed=0 固定）。

## 判读

1. **显著性格局：T 可读判定 1/4 通道翻转（本审计核心判据）。** cop_L 的
   「raw108 显著优于常数基线」判定由 anchor 的不显著（下界 −0.0001 跨 0）
   **翻为 v2/v3 显著**（CI [0.0009, 0.0022] 不含 0），v2 与 v3 同格局；
   phase_L/phase_R/cop_R 三通道三版均不显著，无翻转。即：修正轴 GT 下，
   denoised v2/v3 的触觉 T 侧在左足 CoP 读出上相对常数基线显著可读，
   raw108 不显著——与 anchor 的修正轴结论（T 可读性削弱）相比，
   denoised 版本在 cop_L 上恢复显著性（判读只引符号与判定，不引量级）。
2. **绝对误差方向（raw108 读出，纯数据版本效应）**：v2/v3 相对 anchor
   在 phase_L（−0.0037 / −0.0032）、cop_L（−0.00058 / −0.00055）、
   cop_R（−0.00055 / −0.00053）改善，phase_R 变差（+0.0035 / +0.0012）。
   与 10-20 v3 审计无offset 的版本差方向同构（denoised 版本 phase_L/cop
   改善、phase_R 微差）。
3. **v3−v2 ≤ 2.3e-3（phase）、≤ 7.2e-4（cop）**：故障格突发清洗在
   CoP/相位读出上的效应与 F1 面同构（只动右脚、方向混合、量级小），
   v2/v3 判定格局零翻转。
4. **V 更强与 vt>const（同 ckpt，v2↔v3）格局全同**：cop 两通道 V 更强
   显著为负、phase_L vt>const 显著、cop_L vt>const 显著——V3_3B_joint_and
   ckpt 下的状态，不与 anchor（V3_3B_denoised ckpt）跨 ckpt 比较。

全精度数字：三个 JSON 的 `results`/`verdict` + `compare_v2v3_summary.json`
（差表与 verdict 三元组差机器可读）+ `compare_v2v3.log`（对比脚本输出全文）。
