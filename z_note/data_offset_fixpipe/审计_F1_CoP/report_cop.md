# 审计面 2：CoP/相位 ridge 读出（GT 修正轴，2026-10-20）

**对象**：与审计面 1 同条件——tactile = raw108（视频轴，不动），GT（相位/
CoP 目标）建在修正轴上。载体 = E（V3_Bv1_denoised ckpt，编码器冻结）。

**产物**（本目录）：
- `probe_v5_phase_cop_readout_raw108_offset.json`（修正轴跑，V3_Bv1，CPU）、
  `probe_run_offset.log`
- `probe_v5_phase_cop_readout_raw108_ctrl.json`（未修正对照，同 ckpt 同设备）、
  `probe_run_ctrl.log`
- `probe_smoke.log`（--max-sessions 2 冒烟，exit 0）
- `compare_offset_probe.py`（对比助手）→ `compare_offset_summary.json`（本报告
  数字的机器可读汇总）

## 口径

- 探针：`z_note/multimodal_fusion/V5_方案/probe_v5_phase_cop_readout.py`。
  本次唯一改动 = 新增 `--corrected-offset`（store_true，缺省 False），贯穿
  所有 GT 消费路径：① `AnySoleDataset(..., corrected_offset=args.corrected_offset)`
  （载体窗内 kp_gt/floor_y 等随修正）；② `session_gt()` 的 mocap_t 同样
  `− load_offset_correction().get(sid, 0.0)`（与 dataset 同表同式）。
  其余逻辑（λ 网格、80/20 选参、配对 bootstrap n_boot=10000 seed=0、
  常数基线）未动。
- **GT 另有来源核查（任务要求）**：探针的 phase/CoP 目标**不读** dataset 的
  `load_v5_1_gt`（`build()` 未传 `load_v5_1_gt=True`），而是 `session_gt()`
  现场派生：`load_shared_session → mocap_time_s` + `load_smpl(query_t=mocap_t)`
  （SMPL 归档在动捕时间上的采样）+ `floor_y`（取自 dataset session，
  载体 build 已按修正轴算）。即：**独立读取路径存在**（不经 dataset 内部
  GT），但其来源是 shared facts 的同一 mocap_time_s 与 SMPL 归档——不是
  预计算文件（dataset 训练用的 `v5_1_gt.load_cop` npz 才是独立文件，
  本探针未使用）。故按任务口径「贯穿所有 GT 消费路径」在 session_gt 内
  显式应用同一表同一式修正（未改变任何公式/阈值）。
- 载体 ckpt：`results/AnySole/V3_Bv1_denoised_joint_and/tw20_st20_lr0.0001_
  lp3_lt1_lk1_la0.05_ls0.01_sf0.1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/
  ckpt_best.pt`（存在，saved config：contact_method=joint_and；
  `--tactile-input raw108` 覆盖）。**与 10-19 载体不同**：10-19 JSON
  provenance = `V3_Ev1_f6_soft`（cuda:3）。因此 raw108 读出与「T 可读」判据
  （const−raw108）不含编码器，跨 ckpt 直接可比；v_tok/vt 读出跨 ckpt
  **不可比**（编码器权重不同），本审计以同 ckpt 同设备（V3_Bv1，CPU）的
  未修正对照（ctrl）作 v_tok 一致性基准。
- 运行：`--device cpu`（任务要求；本机 GPU 训练占用，全程未碰 GPU）。
- 如实登记：JSON `provenance.date` 为脚本硬编码 "2026-10-19"（脚本既有
  常量未改，v3 审计同登记）；本跑 provenance 增记 tactile_input 与
  corrected_offset 两字段（其余字段未动）。

## 数字

### 表 1：4 通道 × 3 读出 TEST MSE（const / λ / n_valid 同列）

| 通道 | 读出 | ref（10-19，未修正） | ctrl（未修正，同 ckpt） | offset（修正轴） | const / λ / n（ref→offset） |
|---|---|---|---|---|---|
| phase_L | raw108 | 0.4862 | 0.4862 | 0.4654 | 0.5000 / 1e5→1e4 / 11159/3487→11145/3489 |
| phase_L | v_tok | 0.4521 | 0.4566 | 0.4158 | 0.5000 / 1e4→1e2 |
| phase_L | vt | 0.4572 | 0.4555 | 0.3886 | 0.5000 / 1e4→3.2e3 |
| phase_R | raw108 | 0.4811 | 0.4811 | 0.4505 | 0.5000 / 1e5→1e4 / 11815/3800→11810/3800 |
| phase_R | v_tok | 0.4687 | 0.4712 | 0.4345 | 0.5000 / 3.2e4→3.2e2 |
| phase_R | vt | 0.4790 | 0.4509 | 0.3779 | 0.5000 / 1e5→1e3 |
| cop_L | raw108 | 0.0064 | 0.0064 | 0.0069 | 0.0076 / 3.2e4→1e4 / 33185/10676→33196/10687 |
| cop_L | v_tok | 0.0081 | 0.0080 | 0.0085 | 0.0076 / 3.2e4→3.2e3 |
| cop_L | vt | 0.0068 | 0.0065 | 0.0063 | 0.0076 / 3.2e4→1e4 |
| cop_R | raw108 | 0.0118 | 0.0118 | 0.0109 | 0.0109 / 1e3→3.2e3 / 32753/10850→32743/10862 |
| cop_R | v_tok | 0.0115 | 0.0116 | 0.0124 | 0.0109 / 1e4→1e3 |
| cop_R | vt | 0.0102 | 0.0104 | 0.0117 | 0.0109 / 3.2e4→3.2e2 |

全量 12×3 单元全部 ok，无 no_data。全精度值见 JSON（`results.*.mse_test`）。
n_valid 仅小幅移动（B 类 session 的 GT 掩码平移所致），const 全等。

### 表 2：差表（results mse_test，全精度）

| 通道 | raw108 | v_tok | vt |
|---|---|---|---|
| phase_L | −0.0207（offset−ref = −0.0207） | −0.0408 | −0.0670 |
| phase_R | −0.0306 | −0.0367 | −0.0730 |
| cop_L | +0.00048 | +0.00047 | −0.00020 |
| cop_R | −0.00081 | +0.00084 | +0.00128 |

（offset−ctrl = GT 轴效应，同 ckpt 同设备；offset−ref 与 offset−ctrl 在
raw108 读出上逐位相同——ctrl 与 ref 的 raw108 读出逐位全等，见下。）

### 表 3：显著性判定对照（粗体 = CI 不含 0；FLIP = 判定翻转）

T 可读（const − raw108，**无编码器，ref/ctrl/offset 直接可比**）：

| 通道 | ref = ctrl（未修正） | offset（修正轴） | 格局 |
|---|---|---|---|
| phase_L | **+0.0081 [+0.0003, +0.0163]** | −0.0017 [−0.0348, +0.0303] | **FLIP 显著→不显著** |
| phase_R | +0.0076 [−0.0095, +0.0257] | −0.0077 [−0.0586, +0.0406] | 同（不显著） |
| cop_L | **+0.0014 [+0.0009, +0.0018]** | +0.0010 [−0.0001, +0.0019] | **FLIP 显著→不显著**（下界跨 0） |
| cop_R | −0.0006 [−0.0015, +0.0004] | +0.0002 [−0.0005, +0.0009] | 同（不显著） |

V 更强（raw108 − v_tok，同 ckpt ctrl↔offset）：

| 通道 | ctrl | offset | 格局 |
|---|---|---|---|
| phase_L | +0.0264 [−0.0067, +0.0612] | +0.0305 [−0.0352, +0.0947] | 同（不显著） |
| phase_R | +0.0111 [−0.0183, +0.0404] | +0.0288 [−0.0480, +0.0947] | 同（不显著） |
| cop_L | **−0.0016 [−0.0021, −0.0011]** | **−0.0015 [−0.0025, −0.0003]** | 同（显著为负） |
| cop_R | +0.0003 [−0.0008, +0.0014] | −0.0011 [−0.0023, +0.0000] | 同（不显著） |

vt>const（同 ckpt ctrl↔offset；注意 ctrl 的 vt 是 V3_Bv1 编码器，与 ref
的 V3_Ev1 格局不同属 ckpt 效应，不作跨 ckpt 比）：

| 通道 | ctrl | offset | 格局 |
|---|---|---|---|
| phase_L | +0.0306 [−0.0160, +0.0758] | +0.0649 [−0.0001, +0.1255] | 同（不显著） |
| phase_R | **+0.0306 [+0.0014, +0.0611]** | +0.0435 [−0.0374, +0.1217] | **FLIP 显著→不显著** |
| cop_L | **+0.0014 [+0.0008, +0.0022]** | **+0.0018 [+0.0010, +0.0028]** | 同（显著） |
| cop_R | +0.0007 [−0.0001, +0.0014] | −0.0002 [−0.0012, +0.0009] | 同（不显著） |

## 内置一致性检查（v_tok）

任务预期「raw108_offset 的 v_tok 读出应与未修正版在设备漂移量级内一致」。
**如实报告：不一致，且原因明确、非缺陷。** 数字与分解：

1. **对照复现验证（先行）**：ctrl（未修正，同 ckpt V3_Bv1，CPU）vs
   ref（10-19，V3_Ev1，GPU）的 raw108 读出与 T 可读 verdict **逐位全等**
   （mse 差 = 0.000e+00 ×4 通道；verdict 三元组 == True）——管道确定性成立，
   ctrl 可作为同条件基准。
2. **ckpt 效应分解**：ctrl−ref 的 v_tok 读出差 = +0.00455 / +0.00255 /
   −4.77e-5 / +6.57e-5（phase_L/R、cop_L/R）——即 10-19 载体是
   V3_Ev1_f6_soft（JSON provenance 证据），任务口径中的 V3_Bv1 是不同编码器，
   v_tok 特征本身因权重不同而不同。设备漂移参考（v3 审计实测 CPU↔GPU
   同 ckpt ≤1.6e-9 results / ≤4.1e-9 verdict）远小于该量级。
3. **GT 轴效应分解（本检查的主体）**：offset−ctrl 的 v_tok 读出差 =
   −0.0408 / −0.0367 / +0.00047 / +0.00084。v_tok **特征**（编码器输出）
   在两次运行中逐位相同（同一 ckpt、同一 V/T 输入、valid_windows 只依赖
   fake_mask 不变），但 v_tok **读出**（ridge 在 GT 目标/掩码上的误差）
   必然随 GT 修正平移——相位目标整体平移 ~lag×dt，掩码帧集变动
   （n_valid：phase_L 3487→3489、cop_L 10676→10687 等）。故 v_tok 差
   与 raw108 差同量级（−0.041/−0.037 vs −0.021/−0.031），远非设备漂移，
   亦非任何实现不一致：它精确等于「同一读出器在修正后 GT 上的重评差」。
   结论口径：v_tok 一致性检查应以**同 GT 轴**为条件（10-19 三版对比即该
   条件，逐位全等）；跨 GT 轴时 v_tok 差 = GT 轴敏感度，不能当漂移判。

## 判读

1. **显著性格局：T 可读判定 2/4 翻转（本审计核心判据）。** 修正轴上
   phase_L、cop_L 的「raw108 显著优于常数基线」判定**由显著翻为不显著**
   （phase_L 均值甚至转负 −0.0017；cop_L 下界 −0.0001 跨 0）；
   phase_R、cop_R 原即不显著，维持不显著。即：消除视觉↔动捕时钟残差后，
   触觉 T 侧在相位/CoP 读出上相对常数基线的显著性**削弱/消失**——
   修正前 T 可读性的一部分来自「T 与 GT 之间的相位锁定」而非 T 携带
   独立信息（判读只引符号与判定，不引量级）。
2. **绝对误差方向**：4 通道 raw108 读出 mse 中 3 通道改善（phase_L
   −0.0207、phase_R −0.0306、cop_R −0.0008），cop_L 微升 +0.0005。
   λ 选择随 TRAIN GT 变化（phase 通道 1e5→1e4 等），属 GT 驱动的正常
   重选参。vt/v_tok 读出在相位通道上改善幅度更大（−0.067~−0.073），
   与 raw108 同向。
3. **V 更强格局零翻转**：4 通道 raw108−v_tok 判定 ctrl↔offset 全同号
   同判定（cop_L 显著为负维持）。vt>const 的 phase_R 翻转（显著→不显著）
   为同 ckpt 内 GT 轴效应，但该格的 ctrl 显著态本身是 V3_Bv1 ckpt 下的
   状态（10-19 V3_Ev1 下即不显著），不跨 ckpt 外推。
4. **载体 ckpt 与 10-19 不同（如实登记）**：本跑按任务口径用
   V3_Bv1_denoised_joint_and，10-19 JSON provenance 为 V3_Ev1_f6_soft。
   本审计的所有主结论（表 1 raw108 行、表 3 T 可读列）不含编码器，
   不受该差异影响；v_tok/vt 的跨 ckpt 比较不可行，已在口径与一致性
   检查中显式分解。

全精度数字：三个 JSON 的 `results`/`verdict` + `compare_offset_summary.json`
（diff 表机器可读）+ `compare_offset.log`（对比脚本输出全文）。
