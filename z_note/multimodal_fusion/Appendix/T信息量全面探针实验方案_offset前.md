# T 信息量全面探针实验方案（2026-10-16）

实验目标：在官方指标集上**确认 T 含有的信息量**——零训练 ridge 探针，
读出 v_tok / t_tok / vt 三条件对每个指标的线性可读性。是
T敏感性指标分类.md（社区口径）的实现载体，探针 1 的全面化扩展。

判据：Δprobe = vt − v_tok（T 的联合增量）、t_tok 绝对可读性；对照
Δmodel = VT2M − V2M（test.json 官方输出）→ 模型兑现率。

## 1. 指标集

- **主体**：test_brief.json X2M 组 11 项（PA-MPJPE、MPJPE、GlobalMPJPE、
  MPJRE、W-MPJPE100、traj_ATE、RootOrientationDrift、FootSliding、
  GroundPen、Jitter_pred；Jitter_gt 为 GT 常数不做）；
- **PA 部位全家族展开**（全加：part_* 9 部位 + upper/lower/anklefoot/
  hands），报告聚焦 PA-MPJPE + PA-MPJPE_anklefoot；
- **排除**：接触 F1（裁定不做）、V2T 组（压力重建方向，非「T→动作信息
  量」）、WA-MPJPE100（不在 brief 11 项内）。

## 2. 读出矩阵（两条路径 × 三特征条件）

三特征条件：v_tok / t_tok / vt=concat——同一 ckpt 前向、编码后 null 替换，
与探针 1 同机制（编码器输出与输入条件无关）。

| 路径 | ridge 回归目标 Y | 指标公式 |
|---|---|---|
| 读出 A（姿态路径） | [144 维 6D 姿态 ‖ 3 维根轨迹] | rotmat / FK → MPJRE、Drift、PA 族、MPJPE、GlobalMPJPE、W-MPJPE100、滑步、穿透、Jitter_pred |
| 读出 B（位置路径） | [24×3 骨盆相对关键点 ‖ 3 维根轨迹] | 位置公式指标：PA 族、MPJPE、GlobalMPJPE、W-MPJPE100、滑步、穿透、Jitter_pred（MPJRE 除外） |

- **traj_ATE**：单列（回归根轨迹）——指标只吃根轨迹，两路径重合，
  注记「不参与双路对比」；T 在轨迹层面的作用单列可见；
- **MPJRE**：只能姿态读出（公式吃旋转矩阵，关键点无法反解关节角度）；
- **Drift**：姿态读出为主（协议口径）；位置读出为对照（特殊处理见 §5）。

## 3. 评估口径（统一、与 test.json 同口径）

- **会话级打分**：按 session 拼帧、collapse 重复时间，直接套
  `anysole/utils/metrics.py` 官方公式；
- **打分集**：test 36 会话（S13 系列）。注：splits.csv 中 val ≡ test
  （两列逐行相同，均为 S13 系列 36 会话；train 为 104 个不相交会话），
  故**不设 VAL 列**；探针 1 的 626 窗 val 读数与本次打分集同源；
- **λ 选参**：TRAIN 104 会话（与打分集零交集，防泄漏），80/20 留出，
  λ 网格 10^-3 ~ 10^5 对数 17 档，每任务各选；
- **统计**：Δprobe 逐会话配对 bootstrap 95% CI（`paired_cluster_bootstrap`）；
- **输出**：probe JSON（三条件 × 11 指标 × 两条路径 + Δprobe + CI）+
  与 test.json 的兑现率对照表（Δprobe vs Δmodel）。

## 4. 特征源（第一等配置项）

```
特征源 = { ckpt: <路径>, 层: v_tok/t_tok（编码器输出）, 条件前向: 编码后 null 替换 }
```

写入输出 JSON 的 provenance 字段。默认：V3_Bv1 生产 ckpt_best（探针 1
同源、三条件前向已验证）。**更换 ckpt → 全部读数重跑，旧读数作废**。
最终取值：待定。

## 5. 特殊处理（实现时必须遵守）

1. **Drift 位置读出**：yaw 为圆变量（±180° 缠绕），ridge 不直接回归
   1 维 yaw；回归 3 维 forward 轴向量（F(GT)）→ atan2 取 yaw → 套
   drift 公式；
2. **穿透的 floor_y**：GT 会话标定（F(GT)，非回归量）；
3. **滑步接触帧判据**：GT 位移（官方公式原样）；
4. **W-MPJPE100 / Jitter**：会话级打分下直接套官方公式（100 帧窗、
   三阶差分在会话长度内自然满足）；
5. **回归本质**：ridge 是管线描述（特征 → 底层量 → 套指标公式），
   「是否为 GT 的函数」只影响 TRAIN 时是否有监督，不构成指标分类维度。

## 6. 落地

- 脚本：`z_note/probes/probe_info_panel.py`（新脚本，探针 1 的 ridge/λ 机制 +
  会话级官方公式打分）；
- 产物：`results/AnySole/probes/probe_info_panel.json`（三条件 × 10 项 +
  PA 部位 × 两路径 + Δprobe CI + Δmodel 对照）；特征缓存
  `.info_panel_cache/feats.pt`（38420 训练帧 / 12520 测试帧）；
- 特征源：V3_Bv1 生产 ckpt_best（方案 §4 默认值）。
