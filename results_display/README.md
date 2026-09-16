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
│   ├── test6_dataset_check.py   # Test6：数据集自检（GT FK 自洽 + 均值姿态基线）
│   ├── test7_mean_pose_infer.py # Test7：均值/GT 姿态进入模型的推理 + BVH 导出
│   ├── test8_input_ablation.py  # Test8：输入-输出相关性消融（条件置换 + 梯度检查）
│   ├── test9_overfit.py         # Test9：单 batch 过拟合（关 dropout、VT-only）
│   ├── test9_1_sampler_checks.py # Test9.1：τ0 vs DDIM 差距定位（假模型/同批/τ网格/x0_hat逐元素/6D→SO3）
│   ├── test10_tgen.py           # Test10：V2T 触觉生成（GT/生成热力图 + 误差分析）
│   ├── test11_tau_regime.py     # Test11：τ 训练区间消融（τ≡0 恒等映射 / τ~U(0,100) 低噪声带 / U(0,1000) 对照）
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
├── Test6_dataset_check/         # 实验⑥输出（数据集自检，不加载模型）
│   ├── fk_selfcheck.json        # A1-A5 FK/单位/回环检查逐项最大误差
│   ├── mean_baseline.json       # B1/B2 均值姿态基线 MPJPE + 逐关节明细
│   ├── mean_pose.npz            # 训练集均值姿态（有效 6D + 相对 Hips kp），Test7 复用
│   └── input_means.npz          # 训练集 V/T 逐帧均值，Test8 复用
├── Test7_mean_pose/             # 实验⑦输出（均值姿态推理）
│   ├── <session>/<session>_<arm>.bvh   # tau0_mean/tau0_gt/ddim_noise/ddim_mean500/ddim_gt500
│   └── test7_report.json        # 各 arm MPJPE + 输出两两距离
├── Test8_input_ablation/        # 实验⑧输出（输入-输出相关性）
│   ├── <session>/<session>_<variant>_<arm>.bvh  # real/zero/mean/shuffle × ddim/tau0_gt
│   └── test8_report.json        # 各条件 MPJPE + 输出两两距离 + 梯度范数比
├── Test9_overfit/               # 实验⑨输出（单 batch 过拟合）
    ├── overfit_log.csv          # 逐步 loss / tau0 MPJPE / DDIM MPJPE
    ├── overfit_curves.png       # 曲线图
    ├── overfit_report.json      # 最终数值 + PASS/FAIL 判定
    └── overfit_ckpt.pt          # 仅 --save-ckpt 时输出
├── Test9_1_sampler/             # 实验⑨①输出（Test9 τ0-vs-DDIM 差距的五项定位）
│   ├── train_log.csv            # 单 lr 过拟合日志（与 Test9 同格式）
│   ├── check3_tau_grid.csv      # 采样 τ 网格 / ᾱ / 每步‖x_τ‖ vs 训练 q_sample 包络
│   ├── check4_trace.csv         # 每步模型输出轨迹（τ、‖x_in‖、‖x0_hat-GT‖、‖x0_hat-out0‖、漂移）
│   ├── test91_report.json       # 五项检查数值 + PASS/FAIL
│   └── overfit_ckpt_<lr>.pt     # 仅 --save-ckpt 时输出（--ckpt 复用跳过训练）
├── Test10_tgen/                 # 实验⑩输出（V2T 触觉生成）
│   ├── gif|mp4/<session>_<mode>_tgen.{gif,mp4}  # GT | 生成 | |GT-Gen| 三栏动画
│   ├── cells/<session>_<mode>_cells.{npz,png}   # 96 格逐格 MAE + 静态误差图
│   ├── tgen_summary.csv         # 逐会话 × 条件 MAE/RMSE/相关系数
│   └── tgen_report.json         # 跨会话聚合 + v2t（V2M 行）汇总
└── Test11_tau_regime/           # 实验⑪输出（τ 训练区间消融）
    ├── tau_regime_log.csv       # 逐步 arm × lr × L_pose / tau0 MPJPE（full 臂另有 DDIM）
    ├── tau_regime_curves.png    # L_pose 与 tau0 MPJPE 曲线 + 各 arm 最终 tau0 柱状图
    ├── tau_regime_report.json   # 各 arm × lr 最终数值 + PASS/FAIL 判定 + 诊断结论
    └── tau_regime_ckpt_<arm>_<lr>.pt  # 仅 --save-ckpt 时输出
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

# motionpro 可视化
python results_display/script/visualize_motionpro.py
--session S14103 
--checkpoint results://MotionPRO/checkpoints/imagepressure2smpl/init/5e-05/imagepressure2smpl_best.pth

# step2motion 可视化
python results_display/script/visualize_step2motion.py
--self-test

# 默认 主模型 + 消融，全部 config，全部concat
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal anysolev1 --gen gif --contact-method joint_and
--modal anysolev1,anysolev1_insole_drift
--contact-method bvh_h,bvh_soft,tactile_abs,pat_offset,joint_and
--config-id VT2M

--modal E3 --contact-method normdiff

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
动画中 GT 整段显示（橙）、预测轨迹随帧生长（蓝），窗口边界用小点标记（每窗口在 GT 锚点处重新锚定），footer 实时显示当前帧 ATE 与整段 ATE。

**数据来源**：`anysole.eval` 导出 BVH 时会在旁同时写出
`<session>_<config>_traj.npz`（`pred_trans_world` / `gt_trans_world`，与 `traj_ATE` 指标严格同源）。
Test3 只读这些 npz，不重新推理；请先重跑 eval 刷新产物：

```bash
conda activate touch_gait

# 重新 eval（自动推导 ckpt/BVH/metrics 路径，同时刷新 BVH 与 traj npz）
python -m anysole.eval \
  --modal anysolev1 \
  --contact-method tactile_abs

# Test3 渲染（默认 主模型 + 消融 × 全部 config × test split）
python results_display/script/visualize_anysole_traj.py --modal anysolev1 --gen gif --contact-method joint_and
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

## Test6 数据集自检（GT 自洽 + 均值姿态基线）

`test6_dataset_check.py` 只读数据集、不加载任何模型，回答两个问题：

- **A. GT 自洽**：FK(GT 6D, GT offsets) 必须能逐关节还原数据集内的 `kp_gt`。
  A1 numpy FK（数据集构建路径）/ A2 torch FK（train/eval 路径）交叉验证、A3 单位与几何量程、A4 pose↔BVH 回环（cm↔m 换算）、
  A5 骨骼模板一致性（按受试者分组：同人跨动作/采样必须一致——会话 ID 为 `S<人><动作><采样>`）。
  本管线没有 SMPL betas——"GT betas" 的对应物是每个 BVH 自带的 OFFSET 模（`offsets_m`，cm→m）， A3/A5 即其单位与一致性检查。任何系统性偏移（m/mm 混用、层级错误）都会把 MPJPE 顶到一两百且训不下来。
- **B. 均值姿态基线**：拿训练集均值姿态当预测算 MPJPE。B1 = 均值姿态 + GT 根轨迹（只差姿态）；
  B2 = 完全静态均值姿态。若模型 MPJPE ≈ B1 → 条件被无视；若 B1 明显低于模型 MPJPE →
  模型比"啥也不干"还差，更像 bug（脚本会自动与 `anysolev1_joint_and/metrics/test.json` 对比）。

产出：`fk_selfcheck.json`、`mean_baseline.json`、`mean_pose.npz`、`input_means.npz`（后两者被 Test7/Test8 复用为"均值输入"）。

```bash
conda activate touch_gait
python results_display/script/test6_dataset_check.py
# 单 session 冒烟
python results_display/script/test6_dataset_check.py --session S10103 --limit-sessions 1 --max-windows 16
```

## Test7 均值姿态推理（模型输出 vs 输入姿态）

`test7_mean_pose_infer.py` 用 `results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt`（`--ckpt` 可换）
把不同输入姿态送进模型（条件固定为真实 VT），导出 BVH：

| arm | 输入 | 说明 |
| --- | --- | --- |
| `tau0_mean` | 均值姿态，tau=0 | 干净输入直通重建 |
| `tau0_gt` | GT 姿态，tau=0 | 干净 GT 重建（应→0，否则欠训/条件弱） |
| `ddim_noise` | 纯噪声 | 标准 DDIM（与 eval VT2M 同路径，复现性检查） |
| `ddim_mean500` | 均值姿态加噪到 tau=500 再 DDIM | 均值姿态"真正进入"模型做完整推理 |
| `ddim_gt500` | GT 姿态加噪到 tau=500 再 DDIM | 推理初值上界 |

看两件事：`tau0_mean` vs `tau0_gt` 输出距离（输出对输入姿态的敏感度）；`ddim_mean500` vs `ddim_gt500` 的
MPJPE 差（初值对最终输出的影响）。BVH 写进 `Test7_mean_pose/<session>/`，可用
`visualize_gt_bvh.py --bvh <路径>` 快速查看骨架动画。

```bash
conda activate touch_gait
# 全部 test，前 4 个 session 出 BVH
python results_display/script/test7_mean_pose_infer.py
python results_display/script/test7_mean_pose_infer.py --session S10103 --export-sessions 1
```

## Test8 输入-输出相关性消融（输出是否与输入无关）

`test8_input_ablation.py` 固定模型与初始噪声，置换条件输入（`real` / `zero` / `mean` / `shuffle`，
shuffle = batch 内错位配对，真实输入、错误窗口），各跑一遍 DDIM 与 tau=0 GT 重建，
并对第一个 batch 做梯度检查（∂L/∂V、∂L/∂T vs ∂L/∂x）：

- shuffle MPJPE ≈ real MPJPE → 输入被无视；
- 四种条件输出两两距离 ≈ 0 → 输出与输入无关；
- 输入梯度范数比 ≈ 0 → 条件路径梯度死区。

```bash
conda activate touch_gait
python results_display/script/test8_input_ablation.py
python results_display/script/test8_input_ablation.py --session S10103 --export-sessions 1
```

## Test9 单 batch 过拟合（目标/管线可用性）

`test9_overfit.py` 取训练集固定一个 batch（默认 256 窗），关闭模态 dropout（固定 VT 配置，等价训练侧 `--dropoutVT 0,0`）与模型内部 dropout（`--dropout 0.0`），单 batch 反复训练。
训到 loss≈0 → 目标/管线正常，泛化差是欠训/条件弱；
训不下去 → 目标/管线有问题。

产出：`overfit_log.csv`、`overfit_curves.png`（loss 与 MPJPE 曲线）、`overfit_report.json`（PASS/FAIL 判定），
`--save-ckpt` 时额外存 `overfit_ckpt.pt`。

```bash
conda activate touch_gait
python results_display/script/test9_overfit.py
```

Test9 只留 L_pose，λ_con = λ_kp = λ_traj = λ_T = λ_V = 0，batch 降到 4–8 个窗口，lr 扫 {1e-3, 3e-4, 1e-4, 3e-5}，3000 步。CLI指令：
```bash
# 默认：batch=8 窗 × lr{1e-3,3e-4,1e-4,3e-5} × 3000 步
CUDA_VISIBLE_DEVICES=4 python results_display/script/test9_overfit.py

# 自定义扫描（batch 4 窗）
python results_display/script/test9_overfit.py --batch-size 4 --lrs 1e-3,3e-4,1e-4,3e-5 --steps 3000

# 保存每个 lr 的过拟合 checkpoint
python results_display/script/test9_overfit.py --save-ckpt
```

## Test9.1 τ0 vs DDIM 差距定位（五项检查）

Test9 同一 batch 上 τ0 ≈ 58mm 而 DDIM ≈ 220mm，且 DDIM 在训练中单调变差、τ0 同期变好——两个指标反向走，说明除了"模型只输出 g(F)"之外还存在第二个独立故障。`test9_1_sampler_checks.py` 按序跑五项检查：

1. **假模型**：DDIM 每步用 GT x0 替换模型输出，终点必须 ≈0mm（隔离采样器本身）；
2. **同批**：确认 τ0 与 DDIM 读同一 batch 张量，并测 held-out batch 的 τ0 量化"8 窗不泛化"的代价；
3. **τ 网格**：打印实际采样 τ、首末步 ᾱ、每步 ‖x_τ‖ 与训练 q_sample 包络对比；
4. **末步 x0_hat vs τ0 输出**逐元素比 + 每步模型输出轨迹（检验 g(F) 假设与 τ=0 对 x_penultimate 的敏感性）；
5. **6D→SO(3)**：两条评测路径 FK/正交化一致性 + torch/numpy 转换一致性。

检查 1–3 不需要模型（`--skip-train` 秒级）；检查 2b/4/5 默认重训一个 lr（3e-5 × 3000 步，复现 Test9）或 `--ckpt` 加载已存过拟合模型。

```bash
conda activate touch_gait
# 完整：重训 + 五项检查
CUDA_VISIBLE_DEVICES=5 python results_display/script/test9_1_sampler_checks.py --save-ckpt

# 只跑模型无关的检查 1-3
python results_display/script/test9_1_sampler_checks.py --skip-train

# 复用上次的过拟合 checkpoint，跳过训练
python results_display/script/test9_1_sampler_checks.py --ckpt results_display/Test9_1_sampler/overfit_ckpt_3e-05.pt
```

## Test10 触觉生成（V2T）

`test10_tgen.py` 用 V-only 条件（触觉输入置零）跑模型，让辅助头 `pressure_hat` 变成
**视觉→触觉（V2T）生成器**：模型仅凭 HRNet 视觉特征输出 96 格足底压力。触觉头不经过
扩散采样，每个窗口一次 tau=0 前向即可得到确定性的生成触觉，无需 DDIM。

三种条件（与 eval.py 的 `--config-id` 同名）：

| 条件 | 输入 | 意义 |
| --- | --- | --- |
| `VT2M` | 真 V + 真 T | 重建 sanity（上界参考） |
| `V2M` | 真 V + 零 T | **V2T 生成本身（主指标）** |
| `T2M` | 零 V + 真 T | 触觉自重建（输入端 sanity） |

产出：`tgen_summary.csv`（逐会话 × 条件 MAE/RMSE/相关系数）、`tgen_report.json`
（跨会话聚合，`v2t` 字段 = V2M 行汇总）、每个 session 的 96 格逐格 MAE
（`cells/*.npz` + 静态误差图 `*.png`），以及前 `--export-sessions`（默认 4）个
session 的 GT | 生成 | |GT-Gen| 三栏热力图动画。

与 `anysole.eval` 的关系：eval 在 `metrics/test.json` 每个模式行输出 `T_mae/T_rmse/T_corr`
（V2M 行即 V2T，JSON 顶层 `v2t` 字段），Test10 是该指标的逐格/逐帧可视化解剖。

```bash
conda activate touch_gait
python results_display/script/test10_tgen.py                     # 全部 test split
python results_display/script/test10_tgen.py --session S10103    # 单 session 冒烟
python results_display/script/test10_tgen.py --config-id VT2M,V2M --export-sessions 2
```

> 注意：首版仅支持主模型 `anysolev1`；`anysolev1_insole_drift` 对零触觉输入先过漂移补偿器，
> 生成口径不同，暂不支持。

## Test11 τ 训练区间消融（τ≡0 恒等映射 / 低噪声带训练）

`test11_tau_regime.py` 在训练集固定一个 batch（同 Test9 设置：关模态 dropout、VT-only、关模型 dropout、
pose-only 损失，lr 扫 {1e-3, 3e-4, 1e-4, 3e-5} × 3000 步），把训练时的 τ 采样分布切成三种区间，
定位「τ0 MPJPE 压不下去」的根因：

| arm | τ 分布 | 问题 |
| --- | --- | --- |
| `tau0` | τ ≡ 0（x_tau = x0 干净输入） | 任务退化为恒等映射 x0_hat = x_τ。L_pose 收敛不到 ~0 → x_tau → 输出**没有可用带宽**（分区 Linear 投影丢信息 / cross-attn 把残差流冲掉），门控不是主因 |
| `tau0-nocond` | τ ≡ 0 且 V/T 置零（`--no-cond` 追加） | 隔离纯 x_tau → 输出路径，排除条件 F 的补偿，进一步定位瓶颈 |
| `lowband` | τ ~ U(0, 100)（`--tau-band-max` 可调） | 低噪声区训练。τ0 MPJPE 压到**个位数 mm** → 门控 + SNR 是主因，结构改法对症；压不下去 → 同上（结构瓶颈） |
| `full` | τ ~ U(0, 1000) | 全区间对照（与 Test9 同设置），是「压下去」的参照 |

判定：`tau0` PASS = L_pose < 1e-3 且 tau0 MPJPE < 5 mm（任一 lr 达到即可）；`lowband` PASS = tau0
MPJPE < 10 mm。tau0 MPJPE = τ=0 干净 GT 姿态直通重建（GT 轨迹、23 关节 FK，mm），与训练侧看板
`val/tau0_mpjpe` 同口径。DDIM 仅在 `full` 臂采样——τ≡0 / 低噪声带训练的模型没见过高噪声区，
DDIM 无意义。

```bash
conda activate touch_gait
python results_display/script/test11_tau_regime.py                     # tau0 + lowband + full × 4 lr × 3000 步
python results_display/script/test11_tau_regime.py --arms tau0 --lrs 1e-3 --steps 1000   # 单臂冒烟
python results_display/script/test11_tau_regime.py --no-cond --save-ckpt                # 追加 tau0-nocond 臂 + 存 ckpt
```

产出：`tau_regime_log.csv`、`tau_regime_curves.png`、`tau_regime_report.json`（含诊断结论）。

**全数据集重训**：`python -m anysole.train` 新增两个旋钮（写入 checkpoint 配置），用于在完整训练
分布下验证 Test11 的结论（如 `--tau-max 100` 重训后看 `val/tau0_mpjpe` 能否到个位数）：

```bash
conda activate touch_gait
# 低噪声带全量重训
python -m anysole.train --modal anysolev1 --contact-method joint_and --tau-max 100
# τ≡0 恒等映射全量重训
python -m anysole.train --modal anysolev1 --contact-method joint_and --tau-fixed 0
```

## 训练侧 dropout 开关（两个独立旋钮）

`python -m anysole.train` 现在有两个独立 dropout 参数，都会写入 checkpoint 配置（eval/infer 自动读取）：

- `--dropout 0.1`：模型内部 nn.Dropout（fusion/pose/traj transformer）。`0.0` 关闭。
- `--dropoutVT DROP_V_PCT DROP_T_PCT`（或逗号写法 `--dropoutVT 20,30`）：**模态 dropout**，
  V 丢弃 DROP_V_PCT%、T 丢弃 DROP_T_PCT%（映射为 config_probs `[1-v-t, t, v]`，即 VT/V-only/T-only）。
  `--dropoutVT 0,0` 关闭模态 dropout。

重训实验示例（关全部 dropout + 更多轮次）：

```bash
conda activate touch_gait
python -m anysole.train --modal anysolev1 --contact-method joint_and \
  --dropoutVT 0,0 --epochs 400 \
  --out-dir results/AnySole/anysolev1_joint_and/checkpoints
```

> 注意：重训后必须重跑 `python -m anysole.eval --modal ... --contact-method ...` 刷新 BVH/npz/metrics（见 Test3 说明），
> 再重跑 Test7/Test8 才有意义（Test6 与模型无关，只需跑一次）。

## 历史迁移说明

动画统一进入 `Test1_visualization/`，对照实验进入 `Test2_comparison/`，
过时配置删除（默认自动扫描 results/）。
