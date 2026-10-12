# 压力点 35 在 4×12 网格中的位置

**问题**：raw 压力数据 48 点中"35 号点"是哪一个物理位置？

**答案**：

- **35（1-based）= 0-based 通道 34** = 原始 CSV 第 35 列
- 4×12 排布为 **row-major：通道 k (0-based) → row k//12, col k%12**
  - 12 列沿脚长方向，**col 0 = 脚尖 → col 11 = 脚跟**
  - 4 行沿脚宽方向（内侧/外侧方向 dots 文件未标注）
- 因此 **35 = row 2（宽度第 3 行）、col 10（脚跟端第 2 列）**：脚跟附近、偏脚跟内侧第 2 个长度位置
- 它就是 **R3 裁定的右脚故障通道**（`AnysoleWorkspace/tool/pressure_washer/clean_fault_cell.py` 中 `FAULT_CHANNEL = 34  # 0-based == 1-based 35`，按 D7 判据整 session 置零）

**图示**：`z_note/pressure_cell35_position.png`（左脚/右脚物理布局 + 4×12 索引网格）

**口径来源（均已数值验证）**：

- `AnysoleWorkspace/assets/foot_sensor_layout/` 的 README 与 `{left,right}foot_dots.csv`：dot_index = CSV 列号 = 通道号+1；y 为脚长方向（0.9 脚尖 → 0.1 脚跟）
- `anysole/data/pressure.py`：`values48.reshape(-1, 4, 12)` 同口径
- `results_display/script/utils/render_common.py`：SENSOR_ROWS=4 × SENSOR_COLS=12（"4 medial-lateral rows x 12 heel-to-toe columns"）

**澄清"11×33"**：results_display / MMVP 触觉表示里的 **31×11**（341 格）不是传感器布局，而是 4×12 经过
`audited_4x12_to_31x11_nearest_cell` 映射上采样到 SMPL 足底顶点网格的表示
（`build_shared.py mmvp_map()`；掩码 `essential/pressure/insoleMask*.txt` 为 31×11，242 个有效格）。
查传感器物理位置一律以 4×12 dots 文件为准。
