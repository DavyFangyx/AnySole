# Results Display

集中存放「结果可视化（Test1）」「参数对照（Test2）」「轨迹可视化（Test3）」等实验的输出。所有源码脚本位于 `script/`（唯一被 git 跟踪的目录），生成物全部落在对应实验目录下，不再散落顶层。

可用环境变量 `ANYSOLE_RESULTSDISPLAY` 覆盖本根目录。

## 目录结构

```
results_display/
├── script/                      # 源码（git 跟踪）
│   ├── bvh_aligner_pose.py      # 共享 BVH 解析库
│   ├── visualize_motionpro.py   # Test1：MotionPRO 动画
│   ├── visualize_step2motion.py # Test1：Step2Motion 动画
│   ├── visualize_anysole.py     # Test1：AnySole 动画（主模型 + 消融）
│   ├── visualize_gt_bvh.py      # Test1：GT 原始 BVH 渲染
│   ├── visualize_anysole_traj.py# Test3：AnySole 轨迹对比动画 + 静态图
│   ├── test4_insole_drift.py    # Test4：鞋垫漂移补偿器测试
│   ├── test5_contact.py         # Test5：接触标签动画 + 阈值分析（按方案分目录）
│   ├── contact_methods.py       # Test5：多方案接触标签生成 + 对比报告
│   └── evaluate_compare.py      # Test2：跨模型/参数对照评估
├── Test1_visualization/         # 实验①输出
│   ├── MotionPRO/<task>/<loss>/<lr>/<ckpt_stem>/{gif,mp4}/
│   ├── Step2Motion/gait_model/{gif,mp4}/
│   ├── AnySole/<modal>/<config>/{gif,mp4}/
│   └── gt/<session>/<bvh_stem>/skeleton_zup.{npz,gif,mp4}
├── Test2_comparison/            # 实验②输出（只出指标，不做动画）
│   ├── comparison_per_session.csv   # 逐会话明细
│   ├── comparison_summary.csv       # 精简表：模型 × 指标
│   ├── comparison_summary.png       # 精简表的表格图
│   └── evaluation.log               # 运行信息 + 各模型 checkpoint
├── Test3_trajectory/            # 实验③输出（轨迹可视化）
│   └── AnySole/<modal>/<config>/{gif,mp4,png}/
└── Test5_contact/               # 实验⑤输出：接触标签动画 + 阈值分析（按方案分目录）
    ├── <method>/                    # 每个接触判据方案一个子目录
    │   ├── gif|mp4/<session>_contact.gif/.mp4  # 触觉热力图 + GT BVH + 该方案接触指示动画
    │   ├── contact_summary.csv      # 该方案逐会话接触率/压力和统计
    │   ├── diff_vs_bvh_h.csv        # 该方案 vs bvh_h 参考的差异区间清单
    │   │                            #   (会话/脚/起止时间s/方向/区间内高度·速度·压力和)
    │   ├── diff_by_subject.png      # 该方案差异 × 用户分布（fc/fa 堆叠占比，S5~S14）
    │   └── diff_by_sequence.png     # 该方案差异 × 动作序列分布（fc/fa 堆叠占比，01~13）
    ├── threshold_analysis.png       # 压力和直方图 + 接触率-阈值曲线（与方案无关）
    ├── threshold_analysis.csv       # 逐帧左右脚 48 格压力和（直方图原始数据，t=frame/40）
    ├── threshold_analysis_hist.csv  # 与直方图一致的 64 分箱分布（side/bin_start/bin_end/count）
    └── comparison.{csv,png}         # 各方案 vs bvh_h 参考的一致率/假接触率/假离地率
```

## 时间对齐约定（动捕 ↔ 触觉/视频）

视频、触觉与动捕之间存在时间偏差与漂移。偏差来自各设备独立起录：每个 session 的
人工复核表 `AlignReviews_csv/<session>.csv` 记录常量偏移 `偏移量(s)`（= 视觉时间 − 动捕时间）；
漂移则由触觉/视频时间轴按逐帧时钟重建（触觉 `t_us` 时间戳、视频文件名时钟）吸收。
所有动捕与触觉同帧对比的脚本，GT BVH 都按 MotionPRO `prepare_sequences.py` 的补偿方式
重采样到 40 Hz 会话网格：`t_mocap = t_grid − offset_s`，其中 `t_grid = visual_start_s + n/40`

## Test1 结果可视化

| 脚本 | 作用 | 默认输出 |
| --- | --- | --- |
| `visualize_motionpro.py` | MotionPRO 触觉输入/预测/GT 三栏对比动画 | `Test1_visualization/MotionPRO/<checkpoint tag>/` |
| `visualize_step2motion.py` | Step2Motion 足底压力/生成 BVH/GT 对比动画 | `Test1_visualization/Step2Motion/gait_model/` |
| `visualize_anysole.py` | AnySole 主模型与消融（足底压力/预测 BVH/GT） | `Test1_visualization/AnySole/<modal>/<config>/` |
| `visualize_gt_bvh.py` | 原始 GT 动捕 BVH 的骨架渲染（参考动画） | `Test1_visualization/gt/` |

```bash
conda activate touch_gait

python results_display/script/visualize_motionpro.py
--session S14103 
--checkpoint results://MotionPRO/checkpoints/imagepressure2smpl/init/5e-05/imagepressure2smpl_best.pth

python results_display/script/visualize_step2motion.py
--self-test

# 默认 主模型 + 消融，全部 config，全部concat
python results_display/script/visualize_anysole.py --modal anysolev1 --gen gif
--modal anysolev1,anysolev1_insole_drift
--contact-method bvh_h,bvh_soft,tactile_abs,pat_offset,joint_and
--config-id VT2M

# BVH可视化测试脚本（纯BVH可视化）
python results_display/script/visualize_gt_bvh.py       
python results_display/script/visualize_gt_bvh.py 

--bvh <单个.bvh>
```

## Test2 参数对照

`evaluate_compare.py` 对 AnySole（modal × config 组合）、MotionPRO、Step2Motion 统一计算 MPJPE / PA-MPJPE / W-MPJPE / WAMPJPE / RTE / Accel / Jitter 指标。
产出：
`comparison_per_session.csv`（逐会话明细）、`comparison_summary.csv`（精简表：仅模型 × 指标）、
`comparison_summary.png`（精简表的表格图）、`evaluation.log`（运行信息 + 各模型 checkpoint）。

```bash
python results_display/script/evaluate_compare.py \
  --manifest AnysoleWorkspace/manifests/session_manifest.csv \
  --modal anysolev1,anysolev1_insole_drift \
  --config-id VT2M,V2M,T2M \
  --split test

# 或者完全自动扫描 results/ 下的自包含模型目录
python results_display/script/evaluate_compare.py --auto-scan
```

模型清单默认由脚本扫描 `results/` 自动构建；也可用 `--models-config <yaml|json>` 指定固定清单。

## Test3 轨迹可视化

`visualize_anysole_traj.py` 可视化 AnySole 生成的根轨迹（真实轨迹 vs 预测轨迹）：
单面板 3D 空间动画（gif/mp4）+ 每 session 一张静态图（png，3D 斜视图 / 俯视 / 高度曲线）。
动画中 GT 整段显示（橙）、预测轨迹随帧生长（蓝），窗口边界用小点标记（每窗口在 GT 锚点处重新锚定），
footer 实时显示当前帧 ATE 与整段 ATE。

**数据来源**：`anysole.eval --write-bvh` 现在会在 BVH 旁同时写出
`<session>_<config>_traj.npz`（`pred_trans_world` / `gt_trans_world`，与 `traj_ATE` 指标严格同源）。
Test3 只读这些 npz，不重新推理；请先重跑 eval 刷新产物：

```bash
conda activate touch_gait

# 重新 eval（同时刷新 BVH 与 traj npz，两者与 metrics 保持一致）
python -m anysole.eval --ckpt results/AnySole/anysolev1/checkpoints/ckpt_last.pt \
  --modal anysolev1 --config-id VT2M,V2M,T2M --split test \
  --write-bvh results/AnySole/anysolev1/predictions/eval_bvh \
  --metrics-out results/AnySole/anysolev1/metrics/test.json
python -m anysole.eval --ckpt results/AnySole/anysolev1_insole_drift/checkpoints/ckpt_last.pt \
  --modal anysolev1_insole_drift --config-id VT2M,V2M,T2M --split test \
  --write-bvh results/AnySole/anysolev1_insole_drift/predictions/eval_bvh \
  --metrics-out results/AnySole/anysolev1_insole_drift/metrics/test.json

# Test3 渲染（默认 主模型 + 消融 × 全部 config × test split）
python results_display/script/visualize_anysole_traj.py
# 单 session / 单 config
python results_display/script/visualize_anysole_traj.py --session S7013 --config-id VT2M
```

输出：`Test3_trajectory/AnySole/<modal>/<config>/{gif,mp4,png}/<session>_<config>_traj.{gif,mp4,png}`。

> 注意：eval 生成的 BVH 与 traj npz 必须与 checkpoint 配套。若重新训练了模型，
> 旧的 `eval_bvh/` 产物会与新 metrics 不一致（轨迹头更新后曾出现整窗漂移的过期 BVH），
> 需重跑上述 eval 命令再渲染 Test1/Test3。

## Test4 鞋垫偏移补偿模块

```bash
conda activate touch_gait
# 全部 test split
python results_display/script/test4_insole_drift.py
# 单 session
python results_display/script/test4_insole_drift.py --session S7013
```

## Test5 接触检测

`test5_contact.py` 渲染 GT BVH 骨架 + 触觉鞋垫热力图，并叠加接触指示
（**红色=接触、绿色=无接触**）：鞋垫描边与徽章、骨架足部关节着色、底部整段接触时间轴。
每帧同步显示 48 格压力和值。

背景：原 `contact.npy` 标签口径为「48 格鞋垫 CSV 逐帧压力和 > 100」
（`prepare_sequences.py:contact_from_insoles`，阈值 `CONTACT_SUM_THRESH`），但大量鞋垫在脚
离地后压力不归零（如 S11023 左脚摆动相压力和最低 653），导致 83.4% 的 BVH 离地帧被错标为接触。
为对比候选修复方案，Test5 改为**按方案分子目录**：`contact_methods.py` 为每个方案生成
`contact_<method>.npy`（与 `contact.npy` 同目录、同 10 列格式），`test5_contact.py --methods ...`

| 方法 | 判据 |
| --- | --- |
| `bvh_soft` | BVH 运动学，与 `losses.soft_contact_from_keypoints` 同构（高<5cm ∧ 速<0.2m/s） |
| `bvh_h` | BVH 高度 h<5cm 判接触，4/6cm 滞回 |
| `tactile_abs` | 触觉绝对阈值：48 格压力和 > 100（原 contact.npy 口径） |
| `tactile_rel` | 触觉相对阈值：min + 0.25·(max−min) |
| `tactile_gmm` | 触觉自适应：log1p(压力和) 双峰 GMM 谷值阈值，单峰判全接触 |
| `pat_offset` | 患者级偏移补偿：摆动相压力和的中位数（BVH 自动选模版）+ 100 |
| `joint_or` / `joint_and` | BVH 高度 ∨ / ∧ 触觉 GMM |

```bash
conda activate touch_gait
# 1) 生成全部方案的标签（全部 session）+ 对比报告
python results_display/script/contact_methods.py
# 2) 渲染动画（默认全部方案 × test split；标签缺失时自动补齐）
python results_display/script/test5_contact.py --methods bvh_soft --gen mp4 --session S11113
# 单方案 / 单 session 冒烟
python results_display/script/test5_contact.py --methods bvh_h --session S11023
```

## 历史迁移说明

动画统一进入 `Test1_visualization/`，对照实验进入 `Test2_comparison/`，
过时配置删除（默认自动扫描 results/）。
