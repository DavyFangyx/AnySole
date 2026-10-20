# 接触 F1 三版审计报告（raw108 / v1 / v2）

日期：2026-10-19。执行人：Agent 1（触觉压力处理审计）。
产物：`contact_f1_audit.py`（只读脚本）、`contact_f1_results.csv`、`contact_f1_results.json`（同目录）。

## 一、口径

- **三版数据**（每版只替换 ctx 压力数组，BVH 关节 / floor / air 信号不变）：
  - `raw108`：shared facts `pressure_48.npz` 的 `left48`/`right48`（生产原值，帧对齐）；
  - `denoised108`（v1）：交接包 v1（`Tactile_pressure_handoff_20261017_v1`，含自适应平滑）的 `processed48`，经 `resample_handoff` 绝对最近行映射到生产帧网格；
  - `denoised108_v2`（v2）：交接包 v2（`Tactile_pressure_handoff_20261017_v2_no_smoothing`，仅前三项处理）的 `processed48`，同映射。
- **方法**：`tactile_abs`（48 格压力和 > 100）、`tactile_rel`（thr = min + 0.25×(max−min)）、`tactile_gmm`（log1p 双峰 GMM 谷值，单峰判全接触）、`pressure_f6`（逐格离地基线 90 分位 + corrected 和 + 迟滞阈值）。全部复用 `contact_labels.py` 生产 `m_*` 函数，公式/阈值原样；各方法的逐会话自适应量（gmm 谷值、rel 极值、f6 基线）按该方法定义在各自版本压力上重算。
- **GT**：`contact_labels.load_labels(seq_dir, "f6_soft")`（>0.5 二值化），与生产同口径。f6_soft 为 motion_f6 主导的标签（压力通道只作状态机辅助信号）。
- **会话集**：`AnysoleWorkspace/protocol/splits/default/splits.csv` 的 `val` 列，36 会话（该 csv 的 val 列与 test 列内容相同，均为 36 会话；train 104）。36 会话全部加载成功（load_fail=0、GT 缺失=0、长度不符=0）。
- **F1 定义**：逐帧二分类 F1，contact=1 为正类；报告 pooled F1（TP/FP/FN 汇总后算 P/R/F1）与 per-session mean F1 ± std。
- **排除规则**：GT 某脚全 0（无接触帧）或全 1（无离地帧）→ 该 session 该脚记 excl，不进 F1。

## 二、两包自检（前置闸门）

两包均 PASS（明细见 `00_package_selfcheck.md`）。

| 包 | 状态 | manifest 文件 | 脚文件 | 基线修正值 | 相对原始改动值 | 空文件 | 备注 |
|---|---|---|---|---|---|---|---|
| v1 | PASS | 4239 | 890 | 33,800 | 1,599,431 | 64 | 含自适应平滑 |
| v2 | PASS | 5119 | 890 | 33,800 | 34,579 | 64 | 平滑禁用（0 改动），孤立脉冲 783 |

执行环境偏差（如实登记）：`check_package.py` 使用 `Path.is_relative_to`（Python ≥3.9），任务环境 touch_gait 为 Python 3.8.20 无法执行；两包自检改用 `depthpro` env（Python 3.10.21，numpy 2.2.6）运行，未改包内任何文件。

## 三、三版数据接线（最小 diff 摘要）

- `anysole/data/handoff_pressure.py`（本为未跟踪新文件，本次修改）：
  - L29–36：新增 `HANDOFF_PACKAGES = {"v1": …, "v2": …}` 指向 `z_note/data_pressure_fixpipe/` 下两个真实包路径；`HANDOFF_PACKAGE = HANDOFF_PACKAGES["v1"]`（修复原先指向不存在的 `z_note/Tactile_pressure_handoff_20261017_v1`）；
  - L42–44：新增 `handoff_root(package)`；
  - L47–58：`_pressure_io` 改为按包缓存（`_ios` dict，模块名加包后缀），v1/v2 各自加载本包 `pipeline/pressure_io.py`（两包 `load_pressure` 签名一致，diff 仅为 v2 多一个 `impulse_only` 版本键）；
  - L80–93：`resample_handoff(..., package="v1", root=None)`——`root` 由 `None` 在调用时解析，消除函数定义时绑定默认值导致改常量不生效的坑；语义不变。
- `anysole/data/dataset.py`：
  - L196–199：`tactile_input` 合法集合加入 `denoised108_v2`；
  - L288–306：`denoised108` / `denoised108_v2` 共用分支，按名字映射 package v1/v2，调用显式传 `package=` 与 `root=handoff_root(package)`。
- raw108 分支（L283–285）与接触标签装载零改动；`denoised108` 行为与修复前等价（修复前该路径因包路径不存在必然抛 FileNotFoundError，修复后可用）。
- 说明：上述两文件在本次修改前已有他人未提交改动（V5_1 GT、cam_id 等），本任务只改了上面列出的行。

## 四、接线自测（数值验证）

对 S12042（生产目录 `…/sessions/cam3/20260808/S12/S12042`，n_frames=320，video_start=60223.736s）与 S7061（20260804/S7，n_frames=401）分别调 `resample_handoff` v1/v2，与直接 `np.load` 各包 `data/dataset/…/{left,right}.npz` 的 `processed48` 按同规则手动最近行映射对照：

| 会话 | 包 | 形状=(生产帧数,48) | 与手动映射逐值相等 | v1 vs v2 值差异 | v1 vs raw 值差异 |
|---|---|---|---|---|---|
| S12042 | v1 | (320,48) 左右均 | 是（maxdiff=0） | 左 12.3% 右 4.4% | 左 47.8% 右 43.8% |
| S12042 | v2 | (320,48) 左右均 | 是（maxdiff=0） | — | — |
| S7061 | v1 | (401,48) 左右均 | 是（maxdiff=0） | 左 31.7% 右 31.9% | 左 51.0% 右 55.1% |
| S7061 | v2 | (401,48) 左右均 | 是（maxdiff=0） | — | — |

结论：接线读到的确为各自包的 `processed48`，帧数与生产帧数一致；v1/v2 在数值上确实有差异。

## 五、接触 F1 三版表（36 会话，每脚 incl=20 / excl=16）

排除数：GT f6_soft 在 val 集每脚 16/36 会话全 1 退化（44%，与已知全量 63/140=45% 一致）；全部 32 个排除均为 all1（无 all0）。其余 20 会话两脚 GT 均含接触与离地帧。

| 方法 | 版本 | 脚 | prec | rec | F1 pooled | F1 per-session mean±std | ΔF1 pooled vs raw | ΔF1 mean vs raw |
|---|---|---|---|---|---|---|---|---|
| tactile_abs | raw108 | left | 0.6890 | 0.9994 | 0.8157 | 0.8123±0.0330 | — | — |
| tactile_abs | v1 | left | 0.7138 | 0.9979 | 0.8323 | 0.8273±0.0558 | +0.0166 | +0.0150 |
| tactile_abs | v2 | left | 0.7139 | 0.9975 | 0.8322 | 0.8273±0.0559 | +0.0165 | +0.0150 |
| tactile_abs | raw108 | right | 0.8135 | 0.8248 | 0.8191 | 0.8045±0.1472 | — | — |
| tactile_abs | v1 | right | 0.8124 | 0.8146 | 0.8135 | 0.7969±0.1499 | −0.0056 | −0.0075 |
| tactile_abs | v2 | right | 0.8124 | 0.8146 | 0.8135 | 0.7969±0.1499 | −0.0056 | −0.0075 |
| tactile_rel | raw108 | left | 0.8607 | 0.7135 | 0.7802 | 0.7677±0.1609 | — | — |
| tactile_rel | v1 | left | 0.8490 | 0.7009 | 0.7679 | 0.7566±0.1551 | −0.0123 | −0.0112 |
| tactile_rel | v2 | left | 0.8491 | 0.7011 | 0.7680 | 0.7567±0.1551 | −0.0122 | −0.0111 |
| tactile_rel | raw108 | right | 0.8433 | 0.6599 | 0.7404 | 0.7070±0.2063 | — | — |
| tactile_rel | v1 | right | 0.8433 | 0.6018 | 0.7024 | 0.6612±0.2133 | −0.0380 | −0.0458 |
| tactile_rel | v2 | right | 0.8433 | 0.6018 | 0.7024 | 0.6612±0.2133 | −0.0380 | −0.0458 |
| tactile_gmm | raw108 | left | 0.8510 | 0.7576 | 0.8016 | 0.7851±0.1589 | — | — |
| tactile_gmm | v1 | left | 0.8366 | 0.7452 | 0.7883 | 0.7722±0.1557 | −0.0133 | −0.0128 |
| tactile_gmm | v2 | left | 0.8366 | 0.7452 | 0.7883 | 0.7722±0.1557 | −0.0133 | −0.0128 |
| tactile_gmm | raw108 | right | 0.8198 | 0.8109 | 0.8153 | 0.8003±0.1514 | — | — |
| tactile_gmm | v1 | right | 0.8173 | 0.7984 | 0.8078 | 0.7918±0.1533 | −0.0076 | −0.0086 |
| tactile_gmm | v2 | right | 0.8173 | 0.7984 | 0.8078 | 0.7918±0.1533 | −0.0076 | −0.0086 |
| pressure_f6 | raw108 | left | 0.9082 | 0.7334 | 0.8115 | 0.7885±0.1695 | — | — |
| pressure_f6 | v1 | left | 0.8904 | 0.7110 | 0.7907 | 0.7683±0.1663 | −0.0208 | −0.0202 |
| pressure_f6 | v2 | left | 0.8904 | 0.7110 | 0.7907 | 0.7683±0.1663 | −0.0208 | −0.0202 |
| pressure_f6 | raw108 | right | 0.8877 | 0.6948 | 0.7795 | 0.7488±0.1880 | — | — |
| pressure_f6 | v1 | right | 0.8748 | 0.6801 | 0.7653 | 0.7334±0.1912 | −0.0142 | −0.0154 |
| pressure_f6 | v2 | right | 0.8748 | 0.6801 | 0.7653 | 0.7334±0.1912 | −0.0142 | −0.0154 |

v2 vs v1 的 F1 差：8 组中 6 组为 0，tactile_abs left −0.0001、tactile_rel left +0.0001。

标签帧级核对（36 会话两脚合计）：v1 与 raw 标签不同帧数 tactile_abs 701 / tactile_rel 1505 / tactile_gmm 882 / pressure_f6 1112；v1 与 v2 标签不同帧数仅 4 / 6 / 0 / 0。值层面：v1 vs raw 差异 44.3%，v1 vs v2 差异 22.4%。

## 六、判读

1. **v1/v2 处理相对 raw108 的作用方向：F1 略降或持平，量级小。** 4 方法 × 2 脚共 8 组中 7 组下降（pooled Δ −0.006 ~ −0.038，per-session mean Δ −0.008 ~ −0.046），唯一上升是 tactile_abs 左脚（+0.017，precision 上升驱动）。下降主要来自 recall 小幅下降（precision 基本持平或微降）：处理（前三项）删掉了部分在运动学 GT 下判为接触的压力，使压力派生方法少判了少量接触帧。所有变化 ≤0.05，属小幅，不构成"处理损坏接触信息"的证据，也不构成改善。
2. **v1 与 v2（有无自适应平滑）在接触 F1 上无差异。** 尽管值层面 v1/v2 相差 22.4%，逐帧标签几乎完全相同（0~6 帧差异），F1 差 ≤0.0001。即自适应平滑相对前三项处理对这四个方法的接触标签是中性操作。
3. **排除如实呈现：** f6_soft GT 在 val 36 会话中每脚 16 会话全 1 退化（44%，与全量 63/140 一致），仅 20 会话/脚计入 F1；32 个排除全部为 all1。样本受限，F1 数值对单个会话敏感（std 普遍 0.15~0.21）。
4. **局限（如实登记）：** GT 为 motion_f6 主导的运动学标签，压力通道仅作状态机辅助——四方法在此 GT 下的表现衡量的是"压力派生 vs 运动学"的一致性；pressure_f6 的逐格基线在各自版本压力上重标定（方法固有），非冻结阈值比较。
