# 第二部分：结果检验（R_TestN）

处理**实验结果**（对应 results）的展示行为：模型输出的可视化、参数对照、轨迹与
专项消融。本册 = 总览 `README.md`「第二部分」的逐实验说明（2026-10-28 拆分）。
脚本位于 `script/`，产物在 `ResultTest/`；编号对照与模型协议约定见 `README.md`，
模式/协议/能力注册表 = `results_display/models_modes.yaml`，指标口径见
`README_metrics.md`。

## R_Test1 模型可视化

统一入口 `r_test1_visualize.py`（2026-10-06 用户裁定：**选中 + 生成什么**）。
`--models` 选中谁出谁（`anysole` 一次出四个系列 VT2M/V2M/T2M/V2T，基线各出自己
注册的那个 mode；缺省 = 全部）；`--compare` 同 mode 模型 1×N 横排——**输入只放
一次**（VT2M/T2M 触觉热力图，V2M/V2T 视频帧），中间是各模型的预测骨架（各自
原生协议 + 逐帧 MPJPE），右侧每个出现过的协议一栏 GT。

| 产物 | 位置 |
| --- | --- |
| 单独生成（输入\|预测\|GT 三栏动画，仅 gif/mp4） | `ResultTest/R1Test_visualize/<model>/<mode>/{gif,mp4}/<session>.{ext}` |
| 对比（同 mode 1×N 横排） | `ResultTest/R1Test_visualize/compare/<mode>/` |

```bash
conda activate touch_gait
# 单独生成：--models 顶层选中（anysole + 各基线名）、--model-name 选主模型

# ① 选定主模型 + 全部基线（缺省主模型 V3_Bv1、基线全部）
python results_display/script/r_test1_visualize.py --model-name V3_4b --models anysole,motionpro,mmvp_pressure_toolkit,mmvp_vp-mocap,step2motion
# ② 只要基线（不要主模型）
python results_display/script/r_test1_visualize.py --models motionpro,mmvp_pressure_toolkit,mmvp_vp-mocap,step2motion --session S13011
# ③ 缺省 = 主模型 V3_Bv1 + 全部基线
python results_display/script/r_test1_visualize.py --session S13011

# 对比：同 mode 1×N 横排（选中口径同上）
python results_display/script/r_test1_visualize.py --compare --session S13011
```

`--variant` 缺省自动选 run（只有一个直接取、多个取最新并 log 点名）；`--variant all`
显式处理全部历史 run。缺预测的模型显示灰色占位并写明路径（不回退、不报错）。
`--split` 缺省 `val`（迭代集）；正式 36-session 集加 `--split test`。

四个分模型脚本（`r_test1_visualize_{anysole,mmvp,motionpro,step2motion}.py`）与旧
`r_test1_compare.py` 已并入本入口，移至 `script/legacy/`。

模式/协议/能力的唯一声明点：`results_display/models_modes.yaml`（AnySole 的模式 =
config-id，无需声明；基线必须声明，解析不出模式的行不进比较）。跨模式行永不共屏。

## R_Test2 参数对照

`r_test2_compare.py` = 统一数值评估入口：把选中的主模型与基线按 VT2M/V2M/T2M/V2T
四个配置各算一列指标，出一张模型 × 指标汇总表。

产出（`ResultTest/R2Test_compare/`）：
`comparison_summary.{csv,png}`（模型 × 指标汇总表，看这个就够）、
`comparison_per_session.csv`（逐会话明细）、`evaluation.log`（过程日志）。
加 `--by-mode` 额外输出：`by_mode/<mode>/` 四个模式分块表 + `mode_overview.png` 总览图。

```bash
conda activate touch_gait
# 三个选择维度：--models 顶层选中（anysole + 各基线注册名）、
# --model-name 主模型清单、--config-id 配置（缺省 VT2M,V2M,T2M,V2T）

# ① 选定主模型 + 全部基线（主模型缺省 V3_Bv1，基线缺省全部）
python results_display/script/r_test2_compare.py --model-name V3_4b --models motionpro,mmvp_pressure_toolkit,mmvp_vp-mocap,step2motion,mmvp_fpp-net --by-mode

# ② 只要基线（不要主模型）
python results_display/script/r_test2_compare.py --models motionpro,mmvp_pressure_toolkit,mmvp_vp-mocap,step2motion,mmvp_fpp-net --by-mode

# ③ 选定主模型 + 全部基线
python results_display/script/r_test2_compare.py --model-name V3_4b --models anysole,motionpro,mmvp_pressure_toolkit,mmvp_vp-mocap,step2motion,mmvp_fpp-net --by-mode

# 全部主模型 + 全部基线（--models 缺省 all = anysole + 全部 5 个基线，无需再写）
python results_display/script/r_test2_compare.py \
  --model-name F0b,F4a,F2,F2p4,V3_2,V3_3A,V3_Bv1,V3_4a,V3_4b,V3_4c,V4A --by-mode

# 正式 36-session test 集（缺省 val 迭代集）
python results_display/script/r_test2_compare.py --split test --by-mode --force
```

其他参数：`--contact-method`（缺省 joint_and）、`--variant`（缺省自动取最新 run，
`all` = 全部历史 run）、`--write-model-metrics`（指标另存一份到各模型 metrics/）、
`--surface-metrics`（额外算 PVE，慢）、`--force`（覆盖已有产物）。

表里的 `—` 表示该模型没有这项能力或该项不适用（如 Step2Motion 的 PVE）；预测文件
缺失时该行标 `missing`，格式不符标 `invalid`，原因都写在 `evaluation.log`。

## R_Test3 轨迹可视化

`r_test3_traj.py` 可视化预测根轨迹 vs 真实轨迹（**SMPL/BVH 协议自动检测**）：
单面板 3D 空间动画（gif/mp4）+ 每 session 一张静态图（png，3D 斜视图 / 俯视 / 高度曲线）。
动画中 GT 整段显示（橙）、预测轨迹随帧生长（蓝），窗口边界用小点标记（每窗口在 GT 锚点处重新锚定），footer 实时显示当前帧 ATE 与整段 ATE。

**数据来源（双协议，配对 GT）**：

- SMPL 协议模型（AnySole）：`anysole.eval` 导出的原生 SMPL motion NPZ
  （`predictions/eval_motion/<session>_<config>.npz`）内嵌轨迹字段
  `pred_pelvis_trans` / `gt_pelvis_trans`（与 `root_ate_mm` 指标严格同源）。
- BVH 协议模型（Step2Motion 等）：`predictions/<run>/<session>_gen.bvh`，
  预测轨迹取 BVH Hips 路径，GT 取该 session 的 BVH GT Hips 路径
  （两者都重采样到会话网格）——不再对 BVH 模型共用 SMPL GT。
- 统一 `eval_motion` NPZ（基线）：`joint_xyz_world` 根关节配 SMPL GT pelvis。

R_Test3 只读这些文件，不重新推理；请先重跑 eval 刷新产物：

```bash
conda activate touch_gait

# R_Test3 渲染（默认 主模型 + 全部 config × test split）
python results_display/script/r_test3_traj.py --model-name V3_4b --models motionpro,mmvp_pressure_toolkit,mmvp_vp-mocap,step2motion,mmvp_fpp-net --gen gif --contact-method joint_and
# 单 session / 单 config
python results_display/script/r_test3_traj.py --session S7013 --config-id VT2M
# 只要基线（--models 与 Test1/2 同口径）
python results_display/script/r_test3_traj.py --models motionpro,step2motion

# R_Test3 并排轨迹对比（--compare，配对协议 GT）
conda activate touch_gait
# 选定主模型 + 全部基线（默认 val，--split test 换正式 36 集）
python results_display/script/r_test3_traj.py --compare --model-name V3_4b --models anysole,motionpro,mmvp_pressure_toolkit,mmvp_vp-mocap,step2motion
```

输出：`ResultTest/R3Test_traj/AnySole/<model-name>/<config>/{gif,mp4,png}/<session>_<config>_traj.{gif,mp4,png}`；
baseline 模型输出在 `ResultTest/R3Test_traj/<model>/gen/`。

> 注意：eval 生成的 motion NPZ 必须与 checkpoint 配套。若重新训练了模型，
> 旧的 `eval_motion/` 产物会与新 metrics 不一致，
> 需重跑上述 eval 命令再渲染 R_Test1/R_Test3。

产出（`ResultTest/R3Test_traj/compare/`）：`<mode>/{gif,png}/<session>_<mode>_traj_compare.{gif,png}`。

## R_Test4 触觉生成（V2T）

`r_test4_v2t.py` 默认只读取已经导出的标准 V2T archive，不重新推理模型。它展示
**视觉→触觉（V2T）生成器**的结果；旧的即时推理诊断保留为显式 `--source infer`。

V2T 指标统一到每脚 31×11 网格（AnySole 的 4×12 在评估层重采样，FPP-Net 直接使用）。
正式指标为四层层级：表列 brief 7 键（`T_corr`、`T_rmse`、`pressure_force_rmse`、
`pressure_force_r2`、`pressure_cop_error_left/right`），其余叶 5 键只进逐会话明细
（定义见 `README_metrics.md` §4）；contact 非正式指标（仅诊断保留）。

```bash
conda activate touch_gait
# 默认 anysole(V3_Bv1) + FPP-Net，全部 test 会话；--models 选中谁评估谁
python results_display/script/r_test4_v2t.py --source archives --split test
python results_display/script/r_test4_v2t.py --source archives --session S10103
```

三种条件（与 eval.py 的 `--config-id` 同名）：

| 条件 | 输入 | 意义 |
| --- | --- | --- |
| `VT2M` | 真 V + 真 T | 重建 sanity（上界参考） |
| `V2M` | 真 V + 零 T | **V2T 生成本身（主指标）** |
| `T2M` | 零 V + 真 T | 触觉自重建（输入端 sanity） |

产出（`ResultTest/R4Test_v2t/`）：`tgen_summary.csv`、`tgen_report.json`、31×11 统一网格的
逐格误差 `cells/*.npz`/`*.png`，以及热力图动画（布局 = D4Test_baseline_tactile 口径：
1×3 面板 GT | Generated | |GT-Gen|，每面板 L/R 鞋垫块 + 分离式双行标题，0.75 缩放）。

数据源 = `anysole.eval` 与 FPP-Net 导出的标准 V2T archive；R_Test2 与 R_Test4 用同一套V2T 指标定义。
