# 触觉压力处理审计（2026-10-19 ~ 10-20）

对象：两版交接包（原始对照 = 生产 `shared facts pressure_48.npz`）：
- v1 = `Tactile_pressure_handoff_20261017_v1`（四项处理：孤立突变噪声、悬空底值基线、持续阶跃、自适应平滑）
- v2 = `Tactile_pressure_handoff_20261017_v2_no_smoothing`（仅前三项，无平滑）

三版数据接线（`anysole/data/dataset.py` + `anysole/data/handoff_pressure.py`）：
`tactile_input ∈ {raw108, denoised108(v1), denoised108_v2(v2)}`，均映射到生产帧网格；
同时修复了 handoff 包路径（原指向不存在的 `z_note/Tactile_pressure_handoff_20261017_v1`）。

## 审计项与结论（三 agent 依次执行，主会话逐项独立复核通过）

| 项 | 产物 | 结论 |
|---|---|---|
| 0 两包自检闸门 | `contact_f1/00_package_selfcheck.md` | 两包 check_package 均 PASS（v1 改动值 1,599,431 / v2 34,579，基线修正均 33,800，空文件 64）；touch_gait 3.8 无法运行自检脚本（`is_relative_to` 需 3.9+），用 depthpro 执行已登记 |
| 1 接触 f1 | `contact_f1/report.md` | 压力派生四方法 × 三版 vs f6_soft（val 36 会话，每脚 16/36 GT 全 1 退化排除）：8 组中 7 组 F1 略降（pooled Δ −0.006~−0.038），仅 tactile_abs 左脚 +0.017；量级 ≤0.05 不构成损坏也不构成改善；v1 vs v2 无差异（标签差 0~6 帧，平滑对接触标签中性） |
| 2 COP/相位 | `cop_phase/report.md` | E 载体 ridge 读出（4 通道 × 3 读出）：v1/v2 相对 raw108 四通道方向全为正、幅度探针尺度小量；显著性格局不变（phase_L/cop_L 可读、phase_R/cop_R 不显著）；v1 vs v2 差异仅第 4~5 位小数（results mse ≤1.2e-4、verdict ≤9.7e-5），平滑无可见影响 |
| 3 探针面板 | `probe_panel/report.md` | V3_Bv1_denoised 载体补跑 v2（8 配置，`*_denoised_v2.json`）+ 三版对比图 `ProbeVAxis/figures/probe_vaxis_denoised_v1v2_compare.png`：v2 vs v1 不可见（Δprobe 最大差 GlobalMPJPE 0.061 / traj_ATE 0.054 / PA 0.011，远小于 CI 宽度，560/560 方向一致、0 显著性翻转）；raw→v1 差异含载体差+T 数据差，不可单一归因于降噪；v1→v2 为干净对比（同载体、v_tok 逐位全等，仅 T 数据版本不同） |

## 总判读

1. 前三项处理对三个下游面（接触标签、相位/CoP 可读性、探针 Δprobe）的作用方向不一、量级均小；接触 F1 略降由 recall 小幅下降驱动。
2. **v1 的自适应平滑在全部三个审计面上不可见**（接触标签差 0~6 帧、相位/CoP 判据差 ≤1e-4、探针 Δprobe 差 ≤0.06 且 0 显著性翻转）——v1/v2 选型在下游意义上无差别。
3. 局限（如实登记）：接触 f1 的 GT（f6_soft）在 val 集 44% 会话全 1 退化，有效样本仅 20 会话/脚；探针面板 raw→v1 混入载体差异；全部判读引用探针量级只引符号（弱基线相对量纪律）。

## 复核记录（主会话独立核验，非 agent 自证）

- Agent 1：接线 diff 读档核对；S12042 resample 与独立最近行映射对照（v2 maxdiff=0、v1 6e-5=float32 输出转换）；F1 12/12 抽查单元格与独立重算一致；排除规则实测（S13013 全 1 排除与 D3Test_contact 接触率 1.0 吻合）
- Agent 2：raw108 JSON 与当日 22:24 基线逐位全等（管道确定性）；v_tok results/verdict 三版逐位全等；n_valid 三版全等；报告 v1-v2 差数字两处已修正（verdict 差 ≤2e-5 → 实测 phase 9.7e-5 / cop 3.0e-6）
- Agent 3：8 JSON provenance 逐项核对；v_tok 288/288 与 v1 套件逐位全等；λ/帧数/会话数全等；Δprobe 方向 560/560 一致、0 显著性翻转；既有资产 mtime 均为 10-17 未动；新图三系列颜色像素核验；报告 GlobalMPJPE 0.048 → 0.061 已修正（路径 B 0.06065 为真最大值）
- 过程插曲：Agent 2 运行中出现的"两个同命令行进程"经查为 DataLoader num_workers=4 的 fork 子进程（非重复启动，未杀）
