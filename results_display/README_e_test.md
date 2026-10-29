# E 系列实验展示（Etest/）

每个实验一个目录（`Etest/` 下）；`script/`（`results_display/script/`）存生成
脚本，`Etest/` 只存展示产物（表 = CSV、图 = PNG）。实验报告（口径说明 + 判读）
随各自实验目录存放。本册 = 总览 `README.md` 的 E 系列逐实验说明（2026-10-29
平级拆分）。

## 命名规范（2026-10-27 用户裁定）

文件名 = `用途_图表编号_实验编号_内容名称.扩展名`，例：

```text
Main_Table1_E2_Performance.csv
Full_Table1_E2_Performance_AllMetrics.csv
Full_Table1_E2_Performance_BySession.csv
Main_Fig1_E6_Phase_walk.png
```

- **用途**：`Main_` = 正文采用；`Appendix_` = 正式附录采用；`Full_` = 组织与复核
  用的完整结果（全部十指标、全部输入模式、各模型变体、逐会话/逐帧明细）。
  正文/附录选型确定后，从 `Full_` 结果复制生成对应 `Main_`/`Appendix_` 文件。
- **图表编号**：实验内从 `Table1`/`Fig1` 起编
- **实验编号**：`E1`..`E8`（对应目录名缩写）
- 表 = CSV（utf-8-sig，Excel 直开）；图 = PNG（dpi 150）

## 实验目录

| 目录 | 对应内容 |
|---|---|
| `E1_CleanPerformance` | 同架构专家对比（未遮挡条件下的联合训练与融合收益，附录表 A 五行口径） |
| `E2_MainPerformance` | 主性能与外部基线对比（三块十指标，表 1） |
| `E3_MultiViewConsistency` | 多视角一致性（TTA 五线 × n=1..4） |
| `E4_ComponentAblation` | 组件消融（V7_C方案 §7 五模型对照） |
| `E5_OcclusionRobustness` | 遮挡退化下的触觉增益（occ 五档） |
| `E6_ContactPhaseAnalysis` | 接触阶段与门控效果（逐帧相位分析） |
| `E7_DynamicModalityInputs` | 动态输入配置下的运动重建（五阶段切换） |
| `E8_ComplementarityProbe` | 模态互补信息探针（ridge 探针，待做） |

## 运行指令（display 层：统一参数约定；不进 queue、不经过 scheduler）

统一约定：`--model` = 统计模型（cells 注册名），`--ckpt {last,best}` = ckpt 口径
（缺省 last；best 需对应口径 cells 已注册——当前仅 A/B/C 有 `*_evalbest*`，
Cv 家族暂只 last）。表格数字来自 cells（评估由 conf 管线完成），以下命令是
提取/分析层：

| 实验 | 一条命令 | 模型与口径 |
|---|---|---|
| E1 | `python results_display/script/e_results_export.py --experiment E1 --models V7_Cv5` | `--models` = 家族（Cv1–7，缺省全列）；专家口径固定：Cv1–4 = V7_SV1/ST1（A 骨架）、Cv5 = V7_Cv5_SV1_stat/ST1_stat |
| E2 | `python results_display/script/e_results_export.py --experiment E2 --models V7_Cv5 --ckpt last` | `--models` 可选（缺省 A/B/C + Cv1–7）；`--ckpt last/best`。四张表：Table1=V 块 / Table2=T 块 / Table3=VT 块 / Table4=汇总（模型类型列：U / 外部基线 / Ours） |
| E3 | 待跑（实验组五线未落地，见 `E3_MultiViewConsistency/E3_口径说明.md`） | 五线：外部标定基线 / TTA 原方法 / U@V{i}（TTA 框架）/ U@V{i}+接触约束 / 真值外参参考上限；pilot 数据不入组 |
| E4 | `python results_display/script/e_results_export.py --experiment E4 --ckpt last` | 模型固定（Cv1–7 消融组）；`--ckpt` 目前仅 last |
| E5 | `python results_display/script/e_results_export.py --experiment E5 --models V7_Cv5 --ckpt last` | `--models`/`--ckpt` 可选（缺省 = A/B/C + Cv1–7 全档）；图 1 = `Full_Fig1_E5_Degradation.png`（x=遮挡档、y=相对 occ000 退化，V 蓝实线 / VT 橙虚线，子图六指标） |
| E6 | `python anysole_tta/e6_phase_analysis.py --models V7_Cv4_stat,V7_Cv5_stat,V7_Cv7_stat --ckpt last` | 模型固定三个；`--ckpt` 目前仅 last |
| E7 | `python anysole_tta/e7_driver.py --actions walk,run,squat` | 模型固定（V7_Cv4/Cv5，TTA 阶段同步 `--which last`） |
| E8 | （待做） | `--model` 可选（读冻结 ckpt 中间量 m_v/m_t + ridge 探针） |

## 数据来源与刷新

- 模型行 = `results/cells/<model_id>/metrics/{val,test}.json`（协议三条件十指标，
  唯一实现 `anysole/utils/eval_protocol.py`）
- E3/E6/E7 = 分析驱动产物（`z_note/HMR2.0/hmr2_gait/anysole_tta/`）：
  `e6_phase_analysis.py` / `e7_driver.py` 产出 json+PNG 进各自实验目录
- 统一导出/刷新 CSV：`python results_display/script/e_results_export.py`

## 运行指令

两步：**先运行分析驱动**（选 ckpt 运行实验、产出原始结果），**再统一导出整理**
（把原始结果转成命名规范的表 CSV / 图 PNG）。驱动不进 queue、不经过 scheduler。

```bash
# ---- 第一步：运行分析驱动（选 ckpt 运行实验；E3/E6/E7 需 cwd = z_note/HMR2.0/hmr2_gait、touch_gait env）----

# E3 多视角 TTA（--which last/best 选口径；V7_Cv1..Cv5 已接入）
python anysole_tta/run_tta.py --model V7_Cv5 --split val --lr 0.01 --device cuda:7 --which last

# E6 接触相位分析（读 stat cell 逐帧预测 + 接触标签；--ckpt 选口径）
python anysole_tta/e6_phase_analysis.py --models V7_Cv4_stat,V7_Cv5_stat,V7_Cv7_stat --ckpt last

# E7 动态输入重建（Vi/ViT 阶段内部调 run_tta --which last）
python anysole_tta/e7_driver.py --actions walk,run,squat

# E8 模态互补探针（读冻结 ckpt 中间量 + ridge）
python results_display/script/e8_probe.py --model V7_Cv5 --device cuda:0

# ---- 第二步：统一导出整理（表 = CSV、图 = PNG，命名见上文）----

# 全部实验
python results_display/script/e_results_export.py

# 指定实验
python results_display/script/e_results_export.py --experiment E2 --models V7_Cv5 --ckpt last
python results_display/script/e_results_export.py --experiment E5
```

