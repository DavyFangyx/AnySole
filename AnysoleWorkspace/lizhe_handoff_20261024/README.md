# lizhe 交付包 2026-10-24（v3.1 修正）

对象：`/data/lizhe/projects/Tactile/tactile_pipeline/`（现有 v3 交付）。
本包只含触觉管线（用户裁定「给 lizhe 的只有触觉」），不包含 offset 管线与
消费端（hf-raw）任何内容。

## 一、触觉管线（本包唯一交付类别）

`tactile_pipeline/` 为修正版完整代码（11 py + 4 configs），可直接覆盖现有
交付同名文件。逐项：

| 文件 | 说明 | 与现有交付 |
| --- | --- | --- |
| `rebuild.py` | 端到端重建：v3 三项 + 故障格突发修复；**v3.1 改动**：写 npz 时把行级质量包 `{side}_{t_us,raw_t_us,valid,fake}` 烤进 dataset npz 本身（源 = 包内 `data/quality_rows/`，按 metadata 的 started_at_iso+subject+action+trial 推导录制名）；无 quality_rows 的录制告警跳过烤入。默认 `--package-root` 改为包根自身 | **改动** |
| `quality_rows.py` | 行级质量包读取器（包内 `data/quality_rows/`）；**改动**：去掉指向消费端 `hf-raw` 的硬编码路径，包内相对 + `root` 参数 | **改动** |
| `pressure_io.py` | 便携压力读取器；**改动**：去掉 `file_manifest.csv` 索引，dataset 树本身即目录（`{date}/{subject}/{session}/{side}.npz`） | **改动** |
| `align.py` | 质量包 → 视频帧最近邻对齐（消费端契约文档内） | 逐字节相同 |
| `baseline_core.py` | 悬空底值/持续阶跃基线估计 | 逐字节相同 |
| `check_package.py` | 包完整性自检 | 逐字节相同 |
| `detector.py` | 阶跃扫描 | 逐字节相同 |
| `fault_cell.py` | 故障格突发修复（v3 新增） | 逐字节相同 |
| `noise_model.py` | 孤立脉冲清洗 | 逐字节相同 |
| `quality_rules.py` | 行/格质量规则分析 | 逐字节相同 |
| `__init__.py` | 包入口 | 逐字节相同 |
| `configs/`（4 json） | baseline_settings / calibration_anchors（72 锚点）/ noise_settings / quality_rules | 逐字节相同 |

逐字节相同 = 与 `/data/lizhe/projects/Tactile/tactile_pipeline/` 现役文件
md5 全等（2026-10-24 实测）。

## 二、触觉质量数据（评级 + Fake 标记）

| 文件 | 说明 |
| --- | --- |
| `data/pressure_quality_manifest_144.csv` | 现行 A/B/C/D 评级（144 session：A127/B17）。唯一存活副本 = 主 manifest 的 `pressure_quality_class/pressure_quality_reason` 两列（10-24 17:0x join 版），此处为导出件；330 录音全量原始 csv（08-14 版）在生产端 `tactile_pipeline/data/pressure_quality.csv` 存档、未改动 |
| `data/quality_rows/`（245 npz + provenance.json） | 行级 Fake/valid 标记（`{left,right}_{t_us,raw_t_us,valid,fake}`，吸收自旧 mark-fake 链），管线重建时按录制名烤进 dataset npz 的源数据；provenance.json 为血缘说明 |

## 三、不交付项

- 消费端 `AnysoleWorkspace/hf-raw/`（890 份自包含 npz、rgb/bvh/smpl 农场）——按定妆照裁定属消费端资产，不交付。
- offset 修正表（`protocol/calibration/offset_correction.csv`）——用户裁定「给 lizhe 的只有触觉」，不在本包范围。

## 四、使用方式

1. 用本包 3 个改动文件覆盖现有交付同名文件。
2. 重跑 `python3 rebuild.py --package-root . --output <新目录>` 产出含质量包
   的 dataset；`--verify-reference` 需先以新产物替换 `data/dataset` 后生效
   （旧 data/ 为未烤入的 v3 参考）。
3. 消费端只消费已烤入会话（warning 列出的无 quality_rows 录制不烤入）。
4. `quality_rows.py` 的默认根 = 包内 `data/quality_rows`（代码落位包根时生效）；
   本交付包的 data/ 在上一级，重建请显式 `--package-root`（rebuild 已默认）。

