# z_note — 项目过程文档索引

结构总则：顶层只放职责目录；时间性产物（某轮审计/某次排查的证据包）一律进
`归档/`；数据与代码不混入文档目录（活数据随 fixpipe，工具脚本在 `probes/`）；
决策类文档唯一登记处 = `Baselines/决策/`。

## 目录职责

| 目录 | 职责 |
|---|---|
| `项目概述.md` | 论文视角的项目概述（数据集/机制主张/部署场景） |
| `评估/` | 评估配置说明书（唯一现行评估说明书）、三基线模型性能与文献对照 |
| `multimodal_fusion/` | 现行融合方案与判读（V3_3–V6_Av1、失败窗/时间窗、T 信息量探针） |
| `data_pressure_fixpipe/` | 触觉压力数据管道：交接包 v1/v2/v3、审计、压力传感器协议 |
| （offset 修正表已提升） | → `AnysoleWorkspace/protocol/calibration/`（表+血缘+验证器，见 offset_README.md） |
| `data_smpl_fixpipe/` | SMPL 足部地面审计、rgb_smpl 评估 |
| `probes/` | 诊断探针/冒烟脚本（不入 git，机器本地；运行方式见其 README） |
| `外界配置说明/` | 外部数据来源与采集工具（Gait-Data-Collector / MocapVideoAligner / lizhe 数据） |
| `HMR2.0/` | HMR2.0 多视角一致性实验线文档与论文（阶段 4 搁置） |
| `归档/` | 已完成工作与被取代版本（只进不出，入档登记见其 README） |

## 活依赖（代码运行时读取，禁止移动/删除）

| 路径 | 读取方 |
|---|---|
| `AnysoleWorkspace/protocol/calibration/offset_correction.csv` | `AnysoleWorkspace/tool/build_shared.py` `corrected_lag()`（facts 构建时固化） |
| `data_pressure_fixpipe/Tactile_pressure_handoff_20261017_v1/`、`…_v2_no_smoothing/`、`…_20261020_v3/` | `anysole/data/handoff_pressure.py`（denoised108 / _v2 / _v3 三入口全在用） |

## 归档规则

- 某轮审计/排查结束 → 证据目录整体入 `归档/`（README 表头登记日期）；
- 被新版本取代的方案/文档 → 入 `归档/`（如 multimodal_fusion_backup）；
- 数据类证据（npz/csv/jpg）可整体删除，md 留档。
