# 第一部分：数据检验（D_TestN）

处理**原始数据**（对应 AnysoleWorkspace）的展示行为：原始 GT 的可视化、自检、
标签判定与协议探针。本册 = 总览 `README.md`「第一部分」的逐实验说明
（2026-10-28 拆分）。脚本位于 `script/`，产物在 `DataTest/`；编号对照见
`README.md`，指标口径见 `README_metrics.md`。

## D_Test1 原始数据可视化（BVH / SMPL）

直接渲染 AnysoleWorkspace 的原始 GT 动捕数据，不涉及任何模型输出：

| 脚本 | 作用 |
| --- | --- |
| `d_test1_data_viz.py` | 原始 GT 骨架渲染，**双协议**：默认 auto（有 SMPL-24 用 SMPL，缺失回退 BVH-23）；`--protocol smpl24/bvh23` 强制单协议；`--bvh <路径>` 单文件 BVH |

```bash
conda activate touch_gait

# 原始数据可视化（auto：SMPL-24 优先，BVH-23 回退）
python results_display/script/d_test1_data_viz.py
# 强制 BVH-23
python results_display/script/d_test1_data_viz.py --protocol bvh23
# 单文件 BVH
python results_display/script/d_test1_data_viz.py --bvh <单个.bvh>
```

## D_Test2 数据集自检（GT 自洽 + 均值姿态基线）

`d_test2_dataset_check.py` 只读数据集、不加载任何模型，回答两个问题：

- **A. GT 自洽**：FK(GT 6D, GT offsets) 必须能逐关节还原数据集内的 `kp_gt`。
  A1 numpy FK（数据集构建路径）/ A2 torch FK（train/eval 路径）交叉验证、A3 单位与几何量程、A4 SMPL axis-angle↔6D 回环、
  A5 骨骼模板一致性诊断（按受试者分组比较跨动作/采样的 rest offsets；会话 ID 为
  `S<人><动作><采样>`）。源 MoSh 文件是逐 session 独立拟合，因而 A5 不是 loader
  正确性的硬断言，而是检查"同一受试者能否视作同一 shape"的数据集设计前提。
  本管线当前使用每个 SMPL 文件自己的 betas，通过 neutral SMPL
  shapedirs/J-regressor 生成 shape-dependent rest joints 与 offsets。全量检查中 A5 实际
  FAIL：同一受试者最大 offset 分量跨度为 18.02–38.29mm。这意味着现任务是"给定每段
  GT shape 的 motion prediction"，不是同时生成 shape；不能把 A5 失败误报为 loader
  错误，也不能在未重建所有目标与基线前擅自改用平均 betas。A1–A4 仍是 FK、单位和
  旋转协议的硬验收项。
- **B. 均值姿态基线**：拿训练集均值姿态当预测算 MPJPE。B1 = 均值姿态 + GT 根轨迹（只差姿态）；
  B2 = 完全静态均值姿态。若模型 MPJPE ≈ B1 → 条件被无视；若 B1 明显低于模型 MPJPE →
  模型比"啥也不干"还差，更像 bug（脚本会自动与 `anysolev1_joint_and/metrics/test.json` 对比；该文件现在是全量明细口径，
简略口径在同目录 `test_brief.json`）。

产出（`DataTest/D2Test_dataset_check/`）：`fk_selfcheck.json`、`mean_baseline.json`、
`mean_pose.npz`、`input_means.npz`（后两者作"均值输入"基线）。

```bash
conda activate touch_gait
python results_display/script/d_test2_dataset_check.py
# 单 session 冒烟
python results_display/script/d_test2_dataset_check.py --session S10103 --limit-sessions 1 --max-windows 16
```

## D_Test3 接触检测

`d_test3_contact.py` 渲染自动识别的 GT motion 骨架 + 触觉鞋垫热力图，并叠加接触指示
（**红色=接触、绿色=无接触**）：鞋垫描边与徽章、骨架足部关节着色、底部整段接触时间轴。
每帧同步显示 48 格压力和值。

背景：原 `contact.npy` 标签口径为「48 格鞋垫 CSV 逐帧压力和 > 100」
（`prepare_sequences.py:contact_from_insoles`，阈值 `CONTACT_SUM_THRESH`），但大量鞋垫在脚
离地后压力不归零（如 S11023 左脚摆动相压力和最低 653），曾导致大量运动学离地帧被错标为接触。
为对比候选修复方案，D_Test3 改为**按方案分子目录**：`contact_labels.py` 为每个方案生成
`contact_<method>.npy`（与 `contact.npy` 同目录、同 10 列格式），`d_test3_contact.py --methods ...`
渲染判定动画。标签生成器属数据构建代码，位于 `AnysoleWorkspace/tool/contact_labels.py`。
F6 三件套（`motion_f6`/`pressure_f6`/`f6_soft`，2026-09-18 定稿，设计见 fix_plan_v2.md F6a）
是最终标签设计：`bvh_h` 在台阶承重与低净空两类反例上失效后降级为对比参考；
`f6_soft` 中间阶段按 pressure_f6 一致度计算 4 档软值，导出的 npy 已二值化
（== `motion_f6`，与其他方案统一 {0, 1}；软值不进 npy）。

| 方法 | 判据 |
| --- | --- |
| `bvh_soft` | 直接导出 BVH 运动学，与 `losses.soft_contact_from_keypoints` 同构（高<5cm ∧ 速<0.2m/s） |
| `bvh_h` | 直接导出 BVH 脚部/ToeBase 相对地面高度 h<5cm 判接触，4/6cm 滞回 |
| `tactile_abs` | 触觉绝对阈值：48 格压力和 > 100（原 contact.npy 口径） |
| `tactile_rel` | 触觉相对阈值：min + 0.25·(max−min) |
| `tactile_gmm` | 触觉自适应：log1p(压力和) 双峰 GMM 谷值阈值，单峰判全接触 |
| `pat_offset` | 患者级偏移补偿：摆动相压力和的中位数（BVH 自动选模版）+ 100 |
| `joint_or` | air union / contact intersection（任一来源判离地即离地） |
| `joint_and` | air intersection / contact union（两来源都判离地才离地） |
| `motion_f6` | F6 运动学状态机：接触→离地 v>0.6m/s 连续 2 帧（压力永不触发离地）；离地→接触 (v<0.3 且 \|az\|<3 连续 2 帧) 或 (压力承重 且 v<0.6)；h≥20cm 且无压力时拒绝接触 |
| `pressure_f6` | F6 承重：逐格基线 = 离地帧值 90 分位（自举自 motion_f6），corrected 和 + 迟滞阈值（80 或 0.2×stance 中位，1.5× 滞回） |
| `f6_soft（目前作为GT）` | F6 软标签：中间按 pressure_f6 一致度取 4 档软值 {0.95 触地承重 / 0.70 触地未承重 / 0.05 离地 / 0.30 离地残余压}，导出二值化 == motion_f6（与其他方案统一 {0, 1}） |
| `smpl_contact` | SMPL 网格足底：足部最低顶点离地 <2cm 判接触（smplx 正演 + estimate_floor_y 地面；2026-10-15 新增） |

```bash
conda activate touch_gait
# 1) 生成全部方案的标签（全部 session）+ 对比报告
python AnysoleWorkspace/tool/contact_labels.py
# 指定方案（逗号分隔，缺省 all；方案名见上方表格）
python AnysoleWorkspace/tool/contact_labels.py --methods motion_f6,pressure_f6
# 按 split 三组限定范围（train/val/test，--session 显式列表优先）
python AnysoleWorkspace/tool/contact_labels.py --split test

# 2) 渲染动画（默认全部方案 × test split；标签缺失时自动补齐）
python results_display/script/d_test3_contact.py
# 指定方案（--methods 同口径）
python results_display/script/d_test3_contact.py --methods motion_f6,pressure_f6 --split test
```

产出（`DataTest/D3Test_contact/`）：每个方案一个子目录（`<method>/`）：
`gif|mp4/<session>_contact.gif/.mp4`（触觉热力图 + GT motion + 该方案接触指示动画）、
`contact_summary.csv`（逐会话接触率/压力和统计）、`diff_vs_bvh_h.csv`
（该方案 vs 直接导出 BVH 高度参考的差异区间清单：会话/脚/起止时间s/方向/区间内高度·速度·压力和）、
`diff_by_subject.png` / `diff_by_sequence.png`（差异 × 用户/动作序列分布）。
另有与方案无关的 `threshold_analysis.png`（压力和直方图 + 接触率-阈值曲线）、
`threshold_analysis.csv`（逐帧左右脚 48 格压力和）、`threshold_analysis_hist.csv`
（64 分箱分布）与 `comparison.{csv,png}`（各方案 vs bvh_h 参考的一致率/假接触率/假离地率）。

## D_Test4 基线触觉审计（审计只读）

`d_test4_baseline_tactile.py` 是纯可视化前端，**不做任何生成**：四个面板从
`build_shared.py` 与各 adapter 的落盘产物直接读取，并逐 session 审计
（mapping 字段 / 源 hash / frame_id parity / 帧数一致），审计失败即报错。

| 工作 | 触觉输入 | 数据位置 |
| --- | --- | --- |
| AnySole（主方法） | 原始 pressure_48 → 原生 4×12 / 脚（原始传感器单位） | `AnysoleWorkspace/shared/facts/sessions/cam3/<date>/<sub>/<sid>/pressure_48.npz` |
| MotionPRO | 虚拟毯 (T,320,120)（1.25cm/px 世界系；整幅显示 + painted 像素计数） | `AnysoleWorkspace/model_inputs/MotionPRO/adapter_v1/cam3/<date>/<sub>/<sid>/pressure.npz` |
| MMVP（pressure_toolkit / VP-MoCap） | 共享唯一公共表示，每帧 (2,31,11) float32 裸数组 | `AnysoleWorkspace/shared/representations/tactile/mmvp_31x11/v1/<date>/<sub>/<sid>/insole/%06d.npy` |
| Step2Motion | 16 通道/脚：运行时 `build_gait.pool_48_to_16` 复算（无落盘 16ch 产物） | 源头 = `AnysoleWorkspace/shared/facts/.../pressure_48.npz` |

   方法                                  实际数据                            当前显示
  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   AnySole              4×12 = 48 格/脚，96 格/帧                      48 格/脚，正确
  ─────────────  ─────────────────────────────────  ──────────────────────────────────
   MotionPRO            虚拟毯 320×120（世界系锚定，                       整幅热图 + painted
                         R1 锚点偏移，大量帧毯外）                            像素计数标注
  ─────────────  ─────────────────────────────────  ──────────────────────────────────
   MMVP               31×11 = 341 个位置/脚；有效                       31×11 网格/脚
                            mask 约 242/241 个/脚
  ─────────────  ─────────────────────────────────  ──────────────────────────────────
   Step2Motion             16 通道/脚，32 通道/帧     当前显示 48 格/脚，但只有 16 个
                                                              唯一值，每个值重复 3 格

映射口径已冻结在 `build_shared.py`（`audited_4x12_to_31x11_nearest_cell`：每个 31×11
格取最近 48 格的值，脚形外置 0）；D_Test4 只审计 parity，不再持有任何映射/生成
逻辑。历史审计数字（映射冻结时实测）：MMVP mask 242/241 格/脚、最近格距离 L 均值
9.2mm / R 8.7mm。FPP weight = 静立帧总压（build_metadata 计算）。

```bash
conda activate touch_gait
# 默认全部 splits session（train ∪ val ∪ test），只读审计 + 渲染
python results_display/script/d_test4_baseline_tactile.py
# 只跑 test
python results_display/script/d_test4_baseline_tactile.py --split test
python results_display/script/d_test4_baseline_tactile.py --session S12072
```

## D_Test5 鞋垫偏移补偿（待适配）

`d_test5_insole_drift.py` 验证 insole-drift 补偿器（`anysole.ablations.insole_drift`，
tactile-in → tactile-out 纯数据模块）在真实 session 上的行为：加载训练好的补偿器权重，
渲染补偿前后触觉对比动画 + 逐 session 漂移统计（`theta_summary.csv`）。

> **待适配**：默认 checkpoint 指向 `results/AnySole/anysolev1_insole_drift_<contact>/checkpoints/ckpt_last.pt`，
> 该训练产物已不在盘上（补偿器需要训练过的权重，模板 bank 只是输入）。
> 恢复产物（重训/备份）之前本实验不可运行。

```bash
conda activate touch_gait
python results_display/script/d_test5_insole_drift.py
python results_display/script/d_test5_insole_drift.py --session S7013
```

## D_Test6 地面估计与坐标系审计（pressure_toolkit）

```bash
conda activate touch_gait
# 四日期代表 session 审计：每日期 3 帧 × 3 ROI = 9 候选，选出最优 ROI
python AnysoleWorkspace/tool/adapters/mmvp_series/pressure_tookit/floor_audit.py
# 生产 floor 查看器：读 floor_<subject>.npy + 生产深度帧
python results_display/script/d_test6_floor_view.py --date 20260808 --subject S12
```

产出（`DataTest/D6Test_floor/`）：`<date>_floor_audit.png`（四联审计图）、
`floor_candidates.csv`（36 候选残差/棋盘格偏移/法向）、`<date>_<subject>_floor_view.png`
（生产地面查看图：floor 系点云 / 侧视 / RGB+ROI / 高度直方图）。

## D_Test7 遮挡管线可视化（四相机扩展遮挡组）

`d_test7_occlusion.py` 可视化 cam3 bbox 底部遮挡实验的像素级管线：读原始帧 +
bbox 前端，用与训练提取完全相同的 `occlude_lower_strip` + `process_image` 复算
HRNet 输入 crop（256×192），并排渲染 原始帧 | 遮挡后 crop（左面板叠加 bbox 绿框 /
crop 区域黄框 / 切割线红线，副标题带 bbox_info 数值）。occ 缓存存在性作 session
闸门（无缓存即跳过，只可视化参与遮挡实验的 session）。

档位语义：`--occ 10`=足部、`20`=足+踝、`30`=到膝、`50`=下肢全失（默认 50）。

```bash
conda activate touch_gait
python results_display/script/d_test7_occlusion.py --occ 10 --session S5091
python results_display/script/d_test7_occlusion.py --occ 10 --split test
```

产出（`DataTest/D7Test_occlusion/`）：`<sid>_occ<FFF>.gif/.mp4`。

## D_Test8 GT 足底热图/CoP × 触觉热图/CoP

`d_test8_smpl_gt_cop.py` 2×2 同帧动画（facts 40Hz 帧轴，mocap 经 mocap_time_s
对齐）：左上 SMPL GT 足底热图（smplx 正演 + 离地<3cm 顶点计数，4×12 与触觉
网格同形）；右上 触觉热图（默认 shared facts raw108 0-1023，D_Test4 同口径）；
左下 SMPL GT 的 CoP、右下 触觉 CoP——两 CoP 用同一 `cop_from_grid`（GT 热图质心 /
触觉压力质心），叠加 40 渲染帧轨迹与准星，和值=0 帧不画点。

GT 侧口径（2026-10-20 GT足底×触觉冲突审计修复，只影响展示层）：地面 =
archive y=0（g0，SMPL 足部地面审计 2026-10-19 裁定；旧 estimate_floor_y 关节
p5 比物理地面高致恒接触）；接触阈值 3cm（鞋底厚度 1.4–2.0cm 稳定裕度）；
随足轴在 ‖d_xz‖<0.05m（足俯仰>50°，提踵/上楼梯）时沿用上一有效轴。标题行
附 action 编号（session_manifest.csv，无记录不显示；action 11 = 上楼梯）。
触觉源固定 = shared facts 原始 48 格/脚（旧 handoff 交接包链与
`--tactile-input` 选项已随模块删除，2026-10-24；facts 即 v3 清洗值 +
烤入质量包）。

已知局限（不建模）：上楼梯（action 11）session 台阶面高于 y=0，单平面 GT 对
台阶支撑期必然显示"悬空"——台阶支撑期 GT 热图可为空而触觉面板有载荷，正常。

```bash
conda activate touch_gait
python results_display/script/d_test8_smpl_gt_cop.py --session S5091
python results_display/script/d_test8_smpl_gt_cop.py --split test --stride 2
python results_display/script/d_test8_smpl_gt_cop.py --session S5091 --tactile-input denoised108_v2
```

产出（`DataTest/D8Test_smpl_gt_cop/`）：`<sid>_smpl_gt_cop.gif/.mp4`。

## D_Test9 BVH × SMPL 脚部对照

`d_test9_bvh_smpl_compare.py` 2×4 同帧动画，审计双源两处差异（关节定义高度偏移
~6cm / 根轨迹平移）：

```text
上排 SMPL 左足侧视 | BVH 左足侧视 | SMPL 右足侧视 | BVH 右足侧视（分足分源）
下排 BVH×SMPL 同框对照（hip 对齐） | SMPL 足底接触网格 | 高度时间线（跨2列）
```

- 上排四个侧视面板同轴同刻度：地面线 + 2cm（smpl_contact）/5cm（bvh_h）阈值线；
  每足三点对等——踝（BVH *Foot / SMPL ankle）、趾根（BVH *ToeBase / SMPL foot）、
  趾尖（BVH EndSite / SMPL 网格最靠趾端顶点）；
- 高度时间线（跨 2 列）：摆动足 max(L,R) 的 BVH 踝/足最低点、SMPL 踝/足底
  最低点四条信号 + 阈值线 + 当前帧游标 + 底部 bvh_h/smpl_contact 双方法
  接触标签带（红=接触 绿=离地；标签读 session 目录 contact_*.npy，缺失留空）；
- `--frame 316,366`：指定帧号静态渲染 PNG（<sid>_bvh_smpl_f<N>.png），
  逐帧细看用（优先于 --stride）。

```bash
conda activate touch_gait
python results_display/script/d_test9_bvh_smpl_compare.py --session S5091 --stride 1
# --stride 2 生成更简略
python results_display/script/d_test9_bvh_smpl_compare.py --session S11032,S11053,S11073,S11011,S10103,S11061,S12011 --stride 1
```

产出（`DataTest/D9Test_bvh_smpl_compare/`）：`<sid>_bvh_smpl.gif/.mp4`。


## 2026-10-28：修正版NPZ接触标签复核

D3现在支持`tafix`/`statefix`别名，分别对应`f6_soft_time_accel_v1`、`f6_soft_time_accel_state_v2`，通过标签包清单、来源哈希及帧号严格读取实际NPZ。未知方法和标签错位会报错；不再将未知新方法静默过滤为零个输出。

共享BVH/SMPL农场于2026-10-28 15:31有意重链到新派生源。旧标签包仍绑定原始BVH内容，默认读取会拒绝新的源哈希。历史复核可显式传`--source-bvh-root /data/lizhe/projects/Tactile/1_Data`：只有原始内容与固定哈希完全一致才允许显示，不改包或链接，GT显示使用原始BVH；新源候选使用独立`--candidate-label-root`。两者不能同时指定。

```bash
# 历史训练包：显式恢复内容一致的源用于显示审查
python results_display/script/d_test3_contact.py --methods tafix,statefix --session S6021,S6022,S6023 --source-bvh-root /data/lizhe/projects/Tactile/1_Data --workers 4 --out-dir results_display/DataTest/D3Test_contact/review_training_snapshot
# 新源独立候选（不注册、不覆盖训练标签）
python results_display/script/d_test3_contact.py --methods tafix,statefix --session S6021,S6022,S6023 --candidate-label-root /data/huyuxuan/05_交付/D3接触复核与论文大纲更新_20261028/candidate_labels/current_smpl24 --workers 4 --out-dir results_display/DataTest/D3Test_contact/review_current_candidates
```

`--workers`控制CPU渲染进程；`label_source.json`记录实际标签包及源模式。`contact_summary.csv`保留历史全帧接触率，并新增有效帧数和`contact_rate_valid_left/right`；全fake会话的有效率留空，不作为有效接触样本。

默认 `--methods all` 保持原有 legacy 方法范围；版本化标签请显式指定 `--methods tafix,statefix`，避免把不兼容来源静默加入旧流程。
