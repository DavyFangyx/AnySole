# Results Display（V8 出版版，2026-11-13）

本目录只保留 **V8 四折出版结果** 与展示层脚本。旧 V7 / D_Test / R_Test /
E1–E9 早期结果已清理，不再保留。

## 当前目录

| 目录 | 内容 |
|---|---|
| `Etest_crossfit_v8/` | 主出版树：E2/E4/E5/E6/E9 的主表、主图与 `detail/` 明细。 |
| `Etest_crossfit_v8_s10/` | s10 条件版（20 帧窗、50% 重叠、三角权重），结构与主树一致。 |
| `Etest_crossfit_v8_branches/` | 仍在实验的分支：`IMU/`（IMU/PhysIMU、E6 tgate 等）、`Overlap/`（重叠窗诊断）。 |
| `script/` | 展示层脚本（唯一被 git 跟踪的目录）。 |
| `models_modes.yaml` | 模型 / 模式 / 协议能力注册表。 |
| `README_metrics.md` | 指标定义、唯一实现位置与缺失值规则。 |

## 口径入口

- 出版口径总表：`Etest_crossfit_v8/detail/Publication_Protocol_20261113.md`
- 论文实验配置与产物说明：`z_note/论文整理_V8四折_20261108.md`
- 主表最后一列 `口径` 标明锚/聚合；T 行（T+BiFoot GT-start）与 V/VT 行
  （cell self-anchor）不跨行比较绝对值。
- E5 的 WHAM 曲线已对齐 E2 四折 WHAM clean 值；E9 只报 3D。

## 生成/刷新

训练/测试走 `configs/` 的 conf 管线；本层只把 test 结果整理成表图，不进 queue。
常用脚本：

- E2/E4：`script/e2_crossfit_export.py` → `script/e2_t_bifoot_table_replace.py`
- E5：`script/v8_e5_degradation_figures.py`、`script/e5_visual_degradation_summary.py`、
  `script/e5_t_bifoot_reference_replace.py`、`script/e5_wham_align_to_e2.py`
- E9：`results/diagnostics/20261111_final_protocol/e9_tta_final.py`

环境变量 `ANYSOLE_RESULTSDISPLAY` 可覆盖本根目录。
