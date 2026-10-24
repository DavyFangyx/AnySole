# 审计面 1（补）：接触 F1 — denoised108_v2/v3 + offset（2026-10-20）

**对象**：两个数据条件——触觉压力 = 交接包 v2（前三项，无平滑）与 v3
（v2 三项 + 故障格突发清洗）的 processed48 经 `resample_handoff` 映射到
视频帧网格（视频轴，不修正）；GT = f6_soft 标签建在**修正轴**
（`f6_soft_offfix`，B 类 session 采样时间按 `offset_correction.csv` 平移，
A/C/D 不动）。

**产物**（本目录，全部为新增，未改任何既有文件）：
- `contact_f1_audit_v2v3.py`（= 本目录 contact_f1_audit.py 的 GT 修正轴版 +
  audit_20261020_v3 build_ctxs 版本列表版，两版合并；输出文件改名
  v2v3_offfix，不覆盖既有产物）
- `contact_f1_results_v2v3_offfix.csv/.json`、`f1_run_v2v3_offfix.log`

## 口径

- 与 10-19 / v3 / raw108+offset 审计同口径：val 36 会话；tactile 预测 =
  压力派生四方法（tactile_abs / tactile_rel / tactile_gmm / pressure_f6，
  contact_labels.METHODS 生产阈值原样）在 v2/v3 交接包数据上出标签；
  GT = f6_soft_offfix 二值化（>0.5，load_contact_label 私有标签路径，
  与 raw108+offset 审计同路径）；某脚 GT 全 1 退化 → 该侧排除
  （每脚 16/36 排除，有效 20/脚——GT 与触觉版本无关，排除与
  raw108+offset 审计逐位相同）；F1 = binary pooled F1（contact=1 为正类）。
- v2/v3 ctx 只替换压力数组（left48/right48/sums_l/sums_r），BVH 关节、
  floor、air 信号等与 raw108 相同——与 10-19 / v3 审计 build_ctxs 同构；
  交接包数据与 `AnySoleDataset(tactile_input=denoised108_v2/v3)` 的 T_raw
  同源同映射（dataset.py 同款 resample_handoff 调用）。
- 运行：touch_gait 环境（本面无模型，无 GPU）；load_fail=0 / gt_missing=0 /
  gt_len_mismatch=0。

## 数字

### 表 1：主表（pooled F1，val 36；n_incl=20/脚，n_excl=16/脚）

| 方法 | 脚 | v2+offset | v3+offset | v3−v2 |
|---|---|---|---|---|
| tactile_abs | 左 | 0.8175 | 0.8175 | 0.0000 |
| tactile_abs | 右 | 0.9393 | 0.9393 | 0.0000 |
| tactile_rel | 左 | 0.8942 | 0.8942 | 0.0000 |
| tactile_rel | 右 | 0.8152 | 0.8619 | +0.0467 |
| tactile_gmm | 左 | 0.9213 | 0.9213 | 0.0000 |
| tactile_gmm | 右 | 0.9377 | 0.9383 | +0.0006 |
| pressure_f6 | 左 | 0.8836 | 0.8836 | 0.0000 |
| pressure_f6 | 右 | 0.8568 | 0.8517 | −0.0051 |

v2 与 v3 的差异只在右脚的 rel/gmm/pressure_f6 三格（v3 故障格突发清洗只
改右 cell 35/38 故障格；左足与 abs 全等）。prec/rec 明细：

| 方法 | 脚 | v2 prec→rec | v3 prec→rec |
|---|---|---|---|
| tactile_abs | 左 | 0.6921→0.9983 | 0.6921→0.9983 |
| tactile_abs | 右 | 0.9346→0.9442 | 0.9346→0.9442 |
| tactile_rel | 左 | 0.9717→0.8282 | 0.9717→0.8282 |
| tactile_rel | 右 | 0.9745→0.7007 | 0.9743→0.7728 |
| tactile_gmm | 左 | 0.9617→0.8842 | 0.9617→0.8842 |
| tactile_gmm | 右 | 0.9452→0.9304 | 0.9445→0.9322 |
| pressure_f6 | 左 | 0.9777→0.8060 | 0.9777→0.8060 |
| pressure_f6 | 右 | 0.9753→0.7640 | 0.9745→0.7563 |

### 表 2：参考列（既有产物数字，未重跑）

| 方法 | 脚 | v2 无offset（10-19） | v3 无offset（10-20 v3） | raw108+offset |
|---|---|---|---|---|
| tactile_abs | 左 | 0.8322 | 0.8322 | 0.8009 |
| tactile_abs | 右 | 0.8135 | 0.8135 | 0.9316 |
| tactile_rel | 左 | 0.7680 | 0.7680 | 0.9101 |
| tactile_rel | 右 | 0.7024 | 0.7467 | 0.8582 |
| tactile_gmm | 左 | 0.7883 | 0.7883 | 0.9413 |
| tactile_gmm | 右 | 0.8078 | 0.8086 | 0.9336 |
| pressure_f6 | 左 | 0.7907 | 0.7907 | 0.9025 |
| pressure_f6 | 右 | 0.7653 | 0.7644 | 0.8604 |

### 表 3：按 session 类分解（v2+offset − v2 无offset 逐会话 Δ，按类汇总）

有效 20 会话中 A 类 5、D 类 2（shift=0）逐格 Δ **全为 0.0000**（标签
逐位全等，F1 必等）；B 类 13 会话 × 2 脚 = 26 格承担全部变化：

| 方法 | A（10 格） | B（26 格） | D（4 格） |
|---|---|---|---|
| tactile_abs | +0.0000 | +0.0976（−0.0643 ~ +0.4685） | +0.0000 |
| tactile_rel | +0.0000 | +0.1997（+0.0412 ~ +0.4889） | +0.0000 |
| tactile_gmm | +0.0000 | +0.2138（+0.0287 ~ +0.4746） | +0.0000 |
| pressure_f6 | +0.0000 | +0.1521（+0.0157 ~ +0.3397） | +0.0000 |

（val 36 中 B 类 15，其中 S13061/S13063 全 1 退化被排除 → 13 入表；
与 raw108+offset 审计的分解同构。）

## 口径事实（一致性自检，不另写对照分析）

1. **v2+offset 与 v2 无offset 的关系**：两者只差 GT 轴（修正 vs 原生），
   tactile 数据相同。A/C/D 会话（shift=0）两轴标签逐位全等 → 逐会话 F1
   全等（表 3 实测 +0.0000）；全部差异来自 13 个 B 类会话（GT 平移
   lag_eff×0.025），方向与 raw108+offset 审计相同。pooled 8 格 Δ：
   −0.0147 / +0.1258 / +0.1262 / +0.1128 / +0.1330 / +0.1299 / +0.0929 /
   +0.0915（abs 左微降，其余 7 格上升；对照 raw108+offset 的 8 格 Δ
   −0.0148 / +0.1125 / +0.1299 / +0.1179 / +0.1397 / +0.1182 / +0.0910 /
   +0.0809，同号同构）。v3 同（−0.0147 / +0.1258 / +0.1262 / +0.1152 /
   +0.1330 / +0.1297 / +0.0929 / +0.0873）。
2. **v2+offset 与 raw108+offset 的关系**：两者只差触觉版本（v2 交接包
   vs raw108 生产值），GT 轴相同。8 格 Δ：abs 左 +0.0166、abs 右
   +0.0077、gmm 右 +0.0041（v2 更高）；rel 左 −0.0159、rel 右 −0.0430、
   gmm 左 −0.0200、pressure_f6 左 −0.0189、pressure_f6 右 −0.0036
   （v2 更低）。与无offset 基线的 v2−raw108 8 格 Δ（+0.0165 / −0.0056 /
   −0.0122 / −0.0380 / −0.0133 / −0.0075 / −0.0208 / −0.0142）6/8 同号，
   2 格（abs 右、gmm 右）由微负翻微正——方向与量级基本同构。
3. **v2+offset 与 v3+offset 的关系**：只差故障格突发清洗（右 cell 35/38）。
   左足 4 格与 tactile_abs 右逐位全等；右足 rel +0.0467（召回 +0.0721）、
   gmm +0.0006、pressure_f6 −0.0051。与 v3 审计无offset 的结论同构
   （无offset v3−v2：rel 右 +0.0443、gmm 右 +0.0008、pf6 右 −0.0009，
   其余全 0——清洗只动右脚，方向混合、量级小）。
4. 排除规则与 n_incl/n_excl 在 v2/v3/raw108 三版间逐位相同（GT 不随
   触觉版本变化，均为每脚 20/16）。

全精度数字：CSV（f1_pooled/precision/recall 4 位）+ JSON（含 per_session
与逐会话 F1，表 3 的分解数字可由 `contact_f1_results_v2v3_offfix.json`
与 10-19 JSON 逐格核对）。
