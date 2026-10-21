# data_offset_fixpipe —— 视觉↔动捕逐 session 时钟修正表与使用工具

用途：后续「纯触觉测试」用修正后的动捕时间轴采样 GT/SMPL，消除每 session 的时钟残差相位差。

## 口径

**lag 来源**：S1 全库扫描终表 `z_note/对齐扫描归档/s1_scratch/lag_fleet.csv`（s80_final.py 产出；同源 `lag_fleet_final.json`）。结论与公式见 `Baselines/决策/05_R1-R3_风险裁定.md` §三（第 545–549 行）。

- **终表取列**：`lag_eff_frames` = 已裁定的有效滞后（A 类恒 0）；`conf_class` ∈ A/B/C/D。fleet 表 140 行 = 全库 session。
- **符号约定**：`UV[i] ≈ kp[i + lag]`；`lag > 0` ⟺ 图像 j 被配到"未来"的动捕姿态（图像偏晚 / GT 超前）→ 修法 `mocap_time_s[j] −= lag·dt`，等价 `offset_s_new = offset_s_old + lag·dt`。
- **dt = 0.025 s**（40 Hz）。S5091 验算：0.631612 + 9.5×0.025 = **0.869112**，与 R2 裁定文档一致。
- **四类处理规则**：
  - **A 已对齐（38）**：lag_eff 恒 0，不修，`offset_s_new = offset_s_old`。
  - **B 可平移修复（50）**：平移，`offset_s_new = offset_s_old + lag_eff_frames×0.025`（修正后的 mocap 轴 = `mocap_time_s − lag_eff_frames×0.025`）。
  - **C 修不动（4，全在 20260808/S11）**：align* ≥ 12px，2D 串人非时钟问题，不修，`offset_s_new` 留空。
  - **D 证据不足（48）**：门控帧少 / 曲线平坦 / CI 过宽（fleet flag：n_gate<100 / gate_frac<0.20 / gain<3px / lag_CI>±4f），不修，`offset_s_new` 留空。
- **注意**：修正后 offset_s 可为负（如 S13112：0.126726 − 0.225 = −0.098274），是平移参数的正常值域；fleet 表无缺失、无多出行（140 = 140 = 140，三向一致，见验证记录）。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `offset_correction.csv` | 修正表（140 行）：session_id, date, subject, conf_class, lag_eff_frames, offset_s_old, offset_s_new, remark, source |
| `build_offset_table.py` | 生成脚本（可复现）：fleet 终表 + shared facts + splits → CSV，内置一致性断言，`python3 build_offset_table.py` |
| `corrected_time.py` | 使用工具：`corrected_mocap_time()` / `class_of()` / `lag_eff_frames()` / `corrected_offset_s()`，只读 shared facts，不写任何文件；`python3 corrected_time.py` 跑自带自检 |
| `README.md` | 本文件 |

## 使用方式（纯触觉测试采 GT）

```python
import sys
sys.path.insert(0, "z_note/data_offset_fixpipe")
from corrected_time import corrected_mocap_time, class_of, lag_eff_frames

sess = "S5091"
t_mocap = corrected_mocap_time(sess)   # ndarray，B 类已平移（mocap_time_s − lag×0.025）
cls = class_of(sess)                   # 'A'/'B'/'C'/'D'
lag = lag_eff_frames(sess)             # 已裁定有效滞后（帧）

# 在修正后的动捕时间轴上采样 GT/SMPL（帧轴与视觉帧一一对应，相位差已消除）
if cls in ("C", "D"):
    # 无有效修正：按原时间轴采样，并按项目口径单独登记该 session
    pass
```

要点：

- 返回数组长度 == `session.json` 的 `frame_count`，帧序与 `frames.npz:frame_id` 对应（帧序不变，只平移时间值）。
- A 类返回原样；C/D 类**返回原样但无修正**——调用方必须用 `class_of()` 查询并自行处置（D 类的 lag_eff 仅供排序参考，**不得**当作修正量使用）。
- 本模块只读 `AnysoleWorkspace/shared/facts/sessions/` 与同目录 CSV，不写任何生产文件、不改 `session.json` / `frames.npz`。

## 验证记录（2026-10-20，touch_gait 环境）

`build_offset_table.py` 断言全过：

- 三向集合一致：fleet 140 = splits 140 = shared facts cam3 目录树 140，**无缺失、无多出**（fleet 缺失/多出行 = 0）。
- 全 140 session 恒等式 `offset_s == visual_time_s[0] − mocap_time_s[0]`（±1e-6）且 `len(mocap_time_s) == frame_count`。
- 三个已知值锚点：

| session | conf_class | lag_eff_frames | offset_s_old | offset_s_new（手算） |
| --- | --- | --- | --- | --- |
| S5091 | B | +9.5 | 0.631612 | **0.869112**（0.631612+0.2375） |
| S13112 | B | −9.0 | 0.126726 | **−0.098274**（0.126726−0.2250） |
| S12042 | B | −3.0 | 0.326787 | **0.251787**（0.326787−0.0750） |

- 类别分布：A 38 / B 50 / C 4 / D 48（与 05 决策档 §三 一致）；B 类 offset_s_new 范围 [−1.077233, +1.612958]。
- `corrected_time.py` 全 140 session 导入级校验：B 类平移量逐帧 = −lag_eff×0.025，A/C/D 类逐帧零改动，长度全等 frame_count，0 违规。

`corrected_time.py` 自带自检输出（贴录）：

```
corrected_time 自检
------------------------------------------------------------
S5091: class=B lag_eff=+9.5f 首帧差=-0.237500s len=543==frame_count(543)  [OK]
S13112: class=B lag_eff=-9.0f 首帧差=+0.225000s len=357==frame_count(357)  [OK]
S12042: class=B lag_eff=-3.0f 首帧差=+0.075000s len=320==frame_count(320)  [OK]
S5011: class=A lag_eff=+0.0f 首帧差=+0.000000s len=512==frame_count(512)  [OK]
S5013: class=D lag_eff=+27.5f 首帧差=+0.000000s len=635==frame_count(635)  [OK]
S11031: class=C lag_eff=-16.5f 首帧差=+0.000000s len=470==frame_count(470)  [OK]
------------------------------------------------------------
S5091 断言: 首帧修正前后差 == −0.2375 且长度 == frame_count → OK
```
