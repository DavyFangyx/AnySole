# foot_sensor_layout —— 我方 4×12 鞋垫传感器物理布局

来源：`/data/lizhe/projects/Tactile/Gait-Data-Collector_0811/foot_sensor_layout/`
（采集器仓库自带，2026-09-21 复制入 workspace）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `leftfoot_dots.csv` / `rightfoot_dots.csv` | 48 个压力传感器点的归一化坐标（`norm_x, norm_y` ∈ [0,1]，`dot_index` 1-48） |
| `leftfoot_curve.csv` / `rightfoot_curve.csv` | 足型轮廓曲线 200 点（同归一化坐标系） |

## 与 48 通道的对应（已数值验证）

- **dot_index = 通道号 + 1**：`dot_index k` ↔ 重建 CSV 第 k 列 ↔ T_raw 通道 k-1
  （验证：dots 1-12 沿脚长方向 12 个位置（y 0.904→0.091，脚尖→脚跟）、x 近似不变；
  每 12 点跳一个宽度位置（x +0.39）→ 即通道序 = 4 宽行 × 12 长列 row-major，
  与 `anysole/data/pressure.py` 冻结的 reshape(-1,4,12)、col0=脚尖 完全一致）。
- 坐标系：y = 脚长方向（y≈0.9 脚尖 → y≈0.1 脚跟）；x = 脚宽方向。
  内外侧方向（x 哪端为内侧）本文件未标注，采集器 UI 显示时做 1-x/1-y 翻转仅为显示。
- 单位：归一化 0-1，无绝对 mm 尺度；转入 SMPL 模板足底系时按足底包围盒范围对齐
  （见 results_display D_Test5 的 `our_cell_positions`）。

## 消费者

- `results_display/script/d_test5_baseline_tactile.py`（四基线触觉适配 GIF）
- 后续 pressure_tookit / VP-MoCap 适配层（Agent_06 D2 宽度方向"以标定为准"
  可改用本文件 + 左右脚各自对齐，替代整带宽占位）
