# 审计面 1：接触 F1（GT 修正轴，2026-10-20）

**对象**：`raw108 + offset` 数据条件——触觉压力 = 生产 raw108（视频轴，不动）；
GT = f6_soft 标签建在修正轴上（B 类 session 的 GT 采样时间按
`mocap_time_s − lag_eff_frames×0.025` 平移，A/C/D 不动；修正表
`z_note/data_offset_fixpipe/offset_correction.csv`，140 session，B 类 50）。

**产物**（本目录）：
- `AnysoleWorkspace/model_inputs/AnySole/adapter_v1/labels/f6_soft_offfix/`（140 npz，
  由 `contact_adapter --corrected-offset --label-method f6_soft_offfix` 生成；
  `f6_soft` 原目录未动）
- `contact_f1_audit.py`（v3 审计脚本副本 + `--gt-method` 参数；GT 加载改走
  `load_contact_label(shared, gt_method)`）
- `contact_f1_results_f6_soft{,_offfix}.csv/.json`、`f1_run_*.log`、
  `build_labels.log`、`f6_soft_offfix_selfcheck.py/.json`

## 口径

- 与 10-19 / v3 审计同口径：val 36 会话；tactile 预测 = 压力派生四方法
  （tactile_abs / tactile_rel / tactile_gmm / pressure_f6，contact_labels.METHODS
  生产阈值原样）在 raw108 视频轴上出标签；GT = f6_soft 二值化（>0.5）；
  某脚 GT 全 1 退化 → 该侧排除（两版排除相同：每脚 16/36，有效 20）；
  F1 = binary pooled F1（contact=1 为正类）。
- 口径等价性验证：`--gt-method f6_soft` 跑出的 8 格 pooled F1 与
  `审计报告.md` §1 raw108 列**逐格全等**（0.8157/0.8191/0.7802/0.7404/
  0.8016/0.8153/0.8115/0.7795）——新 GT 加载路径（私有标签 npz）与旧路径
  （contact_f6_soft.npy）等价。
- 运行：touch_gait 环境（本面无模型，无 GPU）；两次跑
  load_fail=0 / gt_missing=0 / gt_len_mismatch=0。

## 数字

### 表 1：8 行主表（pooled F1，val 36；n_incl=20/脚，n_excl=16/脚）

| 方法 | 脚 | f6_soft（原生轴） | f6_soft_offfix（修正轴） | Δ | prec（原→修） | rec（原→修） |
|---|---|---|---|---|---|---|
| tactile_abs | 左 | 0.8157 | 0.8009 | −0.0148 | 0.6890→0.6679 | 0.9994→1.0000 |
| tactile_abs | 右 | 0.8191 | 0.9316 | **+0.1125** | 0.8135→0.9218 | 0.8248→0.9417 |
| tactile_rel | 左 | 0.7802 | 0.9101 | **+0.1299** | 0.8607→0.9868 | 0.7135→0.8444 |
| tactile_rel | 右 | 0.7404 | 0.8582 | **+0.1179** | 0.8433→0.9734 | 0.6599→0.7675 |
| tactile_gmm | 左 | 0.8016 | 0.9413 | **+0.1397** | 0.8510→0.9828 | 0.7576→0.9032 |
| tactile_gmm | 右 | 0.8153 | 0.9336 | **+0.1182** | 0.8198→0.9351 | 0.8109→0.9320 |
| pressure_f6 | 左 | 0.8115 | 0.9025 | **+0.0910** | 0.9082→0.9926 | 0.7334→0.8274 |
| pressure_f6 | 右 | 0.7795 | 0.8604 | **+0.0809** | 0.8877→0.9757 | 0.6948→0.7695 |

**方向**：8 格中 7 格修正轴 F1 上升（+0.081 ~ +0.140），1 格微降
（tactile_abs 左 −0.0148，prec 降 0.021 驱动，rec 已近饱和 0.9994）。
上升格同时改善 precision 与 recall（recall 改善为主）。即：时钟残差相位
消除后，触觉派生的接触时刻与运动学 GT 的接触/离地边沿对齐显著变好。

**按 session 类分解**（佐证效应来源）：有效 20 会话中 7 个 A/C/D（shift=0）
逐格 Δ **全为 0.0000**（标签逐位全等，F1 必等）；13 个 B 会话承担全部变化，
逐会话平均 Δ：tactile_rel 左/右 +0.213/+0.204、tactile_gmm +0.229/+0.195、
pressure_f6 +0.150/+0.139、tactile_abs 左/右 −0.012/+0.185。
（val 36 中 B 类 15，其中 S13061/S13063 全 1 退化被排除 → 13 入表。）

## 自检数字（必须项）

全量 140 session 逐位比对 f6_soft_offfix vs f6_soft
（`f6_soft_offfix_selfcheck.py` → `f6_soft_offfix_selfcheck.json`）：

- **shift=0（A/C/D 共 90 session）**：90/90 **逐位全等**，无例外。
  （A 类修正量恒 0；C/D 类不在表 → 0。）
- **shift≠0（B 类 50 session）**：41 个有差异，9 个逐位全等——9 个全为
  两脚全 1 退化标签（S11061/S11063/S11101/S11102/S13061/S13063/S6103/
  S7061/S8063），常数标签平移不变，属预期非异常。

抽样明细（任务指定 3 A + 3 B）：

| session | 类 | shift_s | lag_eff（帧） | n | identical | 差异帧数 | 差异格数（左/右） | 首/末差异帧 | 平均差异帧 |
|---|---|---|---|---|---|---|---|---|---|
| S13011 | A | +0.0000 | 0.0 | 325 | **是** | 0 | 0 / 0 | — | — |
| S14022 | A | +0.0000 | 0.0 | 428 | **是** | 0 | 0 / 0 | — | — |
| S5011 | A | +0.0000 | 0.0 | 512 | **是** | 0 | 0 / 0 | — | — |
| S5091 | B | +0.2375 | +9.5 | 543 | 否 | 331 | 444（223/221） | 12 / 530 | 263.4 |
| S13112 | B | −0.2250 | −9.0 | 357 | 否 | 100 | 113（61/52） | 22 / 303 | 168.7 |
| S12042 | B | −0.0750 | −3.0 | 320 | 否 | 62 | 73（41/32） | 16 / 298 | 151.9 |

**B 类标签差 = 修正平移的性状说明**：修正轴标签 = 原标签序列按
lag_eff_frames 平移的期望方向成立（S5091 lag=+9.5 → off[i]≈orig[i−9.5]
方向逐帧一致率 0.932；S13112 −9.0 → 0.899；S12042 −3.0 → 0.959），
但不是整数帧精确滚动：① 平移量含半帧（9.5f）时插值查询天然落帧间；
② BVH frame_time 浮点表示（1/120≈0.008333330）使整数帧平移也带
~1e-5 帧插值偏差；③ F6 状态机（Schmitt 迟滞）在阈值附近放大这些
微小扰动并沿序列级联。差异集中在接触/离地边沿——与 F1 表 1 的改善
（边沿对齐变好）同源。

## 判读

- 修正轴 GT 下接触 F1 全线上升（7/8 格），上升完全来自 13 个 B 类会话
  （A/C/D 逐位不变）；方向与修正目的（消除视觉↔动捕时钟残差相位）一致。
- 唯一微降格 tactile_abs 左由 precision 驱动（绝对阈值法 100g 偏低，
  修正后 GT 左足接触边沿更密，FP 相对上升），量级 −0.015，不改变
  「修正轴对齐改善」的主结论。
- 本面为 GT 轴敏感度审计：tactile 数据未动，F1 变化全部来自 GT 轴平移。

全精度数字：CSV（f1_pooled/precision/recall 4 位）+ JSON（含 per_session
与逐会话 F1）。
