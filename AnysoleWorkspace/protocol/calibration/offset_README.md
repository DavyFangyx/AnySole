# offset —— 视觉↔动捕逐 session 时钟修正表（冻结时间配置）

修正表 2026-10-22 已固化进生产 facts（build_shared 构建时按表平移 B 类 session 的
mocap 轴并同步 offset_s）。本目录是表的正典位置：表 + 血缘上游 + 部署验证器。

## 口径

**lag 来源**：S1 全库扫描终表 `lag_fleet.csv`（本目录，s80_final.py 产出；同源
`lag_fleet_final.json`；扫描脚本链见 `z_note/归档/对齐扫描归档/`）。结论与公式见
`Baselines/决策/05_R1-R3_风险裁定.md` §三（第 545–549 行）。

- **终表取列**：`lag_eff_frames` = 已裁定的有效滞后（A 类恒 0）；`conf_class` ∈ A/B/C/D。fleet 表 140 行 = 全库 session。
- **符号约定**：`UV[i] ≈ kp[i + lag]`；`lag > 0` ⟺ 图像 j 被配到"未来"的动捕姿态（图像偏晚 / GT 超前）→ 修法 `mocap_time_s[j] −= lag·dt`，等价 `offset_s_new = offset_s_old + lag·dt`。
- **dt = 0.025 s**（40 Hz）。S5091 验算：0.631612 + 9.5×0.025 = **0.869112**，与 R2 裁定文档一致。
- **四类处理规则**：
  - **A 已对齐（38）**：lag_eff 恒 0，不修，`offset_s_new = offset_s_old`。
  - **B 可平移修复（50）**：平移，`offset_s_new = offset_s_old + lag_eff_frames×0.025`（修正后的 mocap 轴 = `mocap_time_s − lag_eff_frames×0.025`）。
  - **C 修不动（4，全在 20260808/S11）**：align* ≥ 12px，2D 串人非时钟问题，不修，`offset_s_new` 留空。
  - **D 证据不足（48）**：门控帧少 / 曲线平坦 / CI 过宽（fleet flag：n_gate<100 / gate_frac<0.20 / gain<3px / lag_CI>±4f），不修，`offset_s_new` 留空。
- **注意**：修正后 offset_s 可为负（如 S13112：0.126726 − 0.225 = −0.098274），是平移参数的正常值域；fleet 表无缺失、无多出行。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `offset_correction.csv` | 修正表（140 行，冻结）：session_id, date, subject, conf_class, lag_eff_frames, offset_s_old, offset_s_new, remark, source |
| `lag_fleet.csv` / `lag_fleet_final.json` | S1 全库扫描终表（s80_final.py 产出），表的血缘上游证据 |
| `offset_verify.py` | 部署验证器（只读，不写任何文件）：`python3 offset_verify.py`，核对 fleet/表/splits/facts 四向集合一致、facts 恒等式与帧数、facts offset_s == 表期望值（B 类=new / A、C、D 类=old）、表内部行级规则与三锚点 |

## 消费方

- `AnysoleWorkspace/tool/build_shared.py`（`OFFSET_TABLE_CSV` → 本目录
  `offset_correction.csv`；构建 facts 时 B 类 mocap 轴平移 + offset_s 修正，10-22 已固化）
- 模型/评估链直接读 facts 的 `mocap_time_s` / `offset_s`，不接触本表。
- 注意：修正已烘焙进 facts，任何"再按表平移 facts"的工具都会二次平移（旧
  `corrected_time.py` 即属此类，2026-10-24 已删）。

## 验证记录（2026-10-20，touch_gait 环境）

生成脚本断言全过（当时 facts 未烘焙，脚本为再生成语义；10-22 烘焙后脚本改为本
`offset_verify.py` 验证语义）：

- 三向集合一致：fleet 140 = splits 140 = shared facts cam3 目录树 140，**无缺失、无多出**。
- 全 140 session 恒等式 `offset_s == visual_time_s[0] − mocap_time_s[0]`（±1e-6）且 `len(mocap_time_s) == frame_count`。
- 三个已知值锚点：

| session | conf_class | lag_eff_frames | offset_s_old | offset_s_new（手算） |
| --- | --- | --- | --- | --- |
| S5091 | B | +9.5 | 0.631612 | **0.869112**（0.631612+0.2375） |
| S13112 | B | −9.0 | 0.126726 | **−0.098274**（0.126726−0.2250） |
| S12042 | B | −3.0 | 0.326787 | **0.251787**（0.326787−0.0750） |

- 类别分布：A 38 / B 50 / C 4 / D 48（与 05 决策档 §三 一致）；B 类 offset_s_new 范围 [−1.077233, +1.612958]。
