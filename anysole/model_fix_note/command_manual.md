# AnySole 命令手册（SMPL-24 · 2026-09-20）

> env = touch_gait；根目录 `/data/fangyuxuan/projects/gait`。
> 历史与演进见 `model_fix_note.md`；模型结构见 `anysole/AnySole 多模态动作重建 · 模型实现说明.md`。

## 通用约定

- **目录编号 = `{模型版本}_{contact-method}`**（现在全部 SMPL-24，不再有任何协议
  后缀）。当前所有基座都用 `joint_and` 标签：`F0b_joint_and` / `F4a_joint_and` /
  `F2_joint_and` / `F2p4_joint_and` / `V3_2_joint_and`。
- **checkpoint 状态**：group_ids 修复（2026-09-20）后，此前全部权重已归档
  `results/backup/AnySole_BVH_backup/`。**2026-09-21 warm-start 链重训完成**：
  F0b / F4a / F2 / F2+4 / V3-2 五个基座全部训出并过 val 协议验收
  （各节已回填实测数字）。V3-3（soft A + σ 门控）代码已实现、smoke 全过，
  **尚未训练**（两阶段命令见 V3-3 节）。
- 训练顺序（warm-start 链）：**F0b（from-scratch）→ F4a / F2（←F0b）→
  F2+4（←F2）→ V3-2（←F4a）→ V3-3A（←V3-2）→ V3-3B（←V3-3A）**。
- `anysole/configs/v1.yaml` 已是 SMPL-24 训练配置：raw108、tw=20、stride=20、lr=1e-4
  （**`--lr` 不是 CLI**，要改 lr 就改 yaml）、λ_pose=3 / λ_kp=1 / λ_traj=1 /
  λ_trec=0.1 / λ_vrec=0.1 / λ_con=0、joint_and。实验差异全部由 CLI 传入，
  覆盖值随 ckpt 保存，eval/infer 从 ckpt 读配置。
- 训练前单测：`z_note/probes/smoke_f0b_regress.py`（回归头）、
  `smoke_f4a_grid.py`（foot_conv）、`smoke_f2_roundtrip.py`（f2 表示）。
- 每个基座固定四条命令：**① 训练 → ② 评估（--write-motion 刷新导出）→
  ③ R_Test1 模型可视化（骨架动画，旧称 Test1）→ ④ R_Test3 轨迹可视化（旧称
  Test3）**。可视化读 `predictions/eval_motion/<session>_<config>.npz`，因此
  评估必须先于可视化。
- results_display 编号已改 D_TestN（数据检验）/ R_TestN（结果检验，2026-09-21
  P1-4 平铺改名已落地）；手册两条可视化 = R_Test1
  （`results_display/script/r_test1_visualize_anysole.py`）与 R_Test3
  （`results_display/script/r_test3_traj.py`）。历史编号对照见
  `results_display/README.md`。
- **产物目录规范（2026-09-21 已落地）**：两级结构 `results_display/data/d_testN/`
  与 `results_display/result/r_testN/`（每个 Test 一个文件夹）；`ridge_probe.py`
  与 `z_note/probes/smoke_*.py` 保持原地；D_Test4 已撤销（SMPL 协议探针回
  `z_note/probes/`）。
- 评估默认自动跑协议 1 次（`--protocol-seed 0`）；`--no-protocol` 只出 metrics；
  `--no-robustness` 关鲁棒集。每步验收读 `metrics/<split>_fseries.json` +
  `metrics/ridge_probe_<split>.json`（ridge 探针每步必跑：
  `results_display/script/ridge_probe.py --ckpt <ckpt> --split val --device <GPU>`）。
- wandb tag 与目录同名（如 `f0b_jointand`），与 BVH 时代历史组天然区分。
- `--device` 按 `nvidia-smi` 空闲卡填（示例用 cuda:4）。
- 单会话推理（需要时）：`/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.infer --ckpt <ckpt> --modal anysolev2
  --contact-method joint_and --session <SID> --config-id VT2M --device <GPU>`。

---

## 基座 F0b · 回归本体（`--modal anysolev2`，from-scratch）

**2026-09-21 实测**（400ep from-scratch，wandb fxemwdt8；val 协议 seed0）：
VT2M MPJPE 78.1 / PA 44.8、V2M 81.4 / 46.2、T2M 154.9 / 72.4；
VT yaw_drift 4.5、jitter 12.0、contact_f1 0.43、foot_slide 2.82mm/帧。
from-scratch 400ep 是筛选预算，显著弱于 BVH 时代 warm-start 的 F0b_seed2
（63.7/25.1）；若作为最终基线，续跑至 800ep（yaml 默认）预期继续收敛。

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --epochs 400 --grad-clip 5.0 \
  --out-dir results/AnySole/F0b_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag f0b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4

# ② 评估（协议 seed0 + 导出 SMPL NPZ，R_Test1/R_Test3 依赖此步）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F0b_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/F0b_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画 → results_display/result/r_test1_visualize/AnySole/F0b/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --modal F0b --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比 → results_display/result/r_test3_traj/AnySole/F0b/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --modal F0b --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## 基座 F4a · 按脚卷积触觉（`--t-encoder foot_conv`，warm-start 自 F0b）

模型：anysolev2 + `--t-encoder foot_conv`（数据不动，只换编码器）。
历史 BVH 口径结果（仅参照）：VT 55.5/23.0、T2M 123.8/40.3；T-only 全身退化
+7.6（V3 的动机）。新 t_enc 结构 warm-start 时按名字自动跳过、保持新初始化。
**2026-09-21 实测**（400ep warm-start F0b；val 协议 seed0）：VT2M MPJPE 70.4 /
PA 37.3（较 F0b 78.1/44.8，foot_conv 兑现 −7.7/−7.5）、V2M 71.6 / 36.4、
T2M 175.2 / 69.6（**T-only 弱点依旧：yaw_drift 72.4**，即 V3 要修的对象）；
VT contact_f1 0.62（F0b 0.43）。

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F0b_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F4a_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag f4a_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F4a_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/F4a_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --modal F4a --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --modal F4a --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## 变体 F2 · f2 表示（`--f2-repr`，warm-start 自 F0b）

根 6D 拆 tilt/yaw + 4 维 heading-frame 轨迹；traj_head.proj 3→4 维自动跳过。
历史 BVH 口径结果（仅参照）：T2M 89.7/35.8（T2M 最优）、VT 65.4/24.7。
**2026-09-21 实测**（400ep warm-start F0b；val 协议 seed0）：VT2M MPJPE 79.8 /
PA 49.3、V2M 72.5 / 42.4、T2M 139.3 / 79.1；**yaw_drift 全配置大幅改善**
（T2M 30.4→3.1、VT 4.5→2.6、V 4.6→2.3，f2 表示核心收益复现）。

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F0b_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F2_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag f2_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6

# ② 评估（eval/协议/导出自动处理 f2→world 恢复）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F2_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/F2_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:6

# ③ 可视化 R_Test1（骨架动画）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --modal F2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --modal F2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## 变体 F2+4 · f2 表示 + 按脚卷积（warm-start 自 F2）

历史 BVH 口径结果（仅参照）：VT 63.4/23.1（PA 三配置最优）、T2M 99.7/40.7、
yaw_drift 三配置全最优（foot_conv 为新结构自动跳过）。
**2026-09-21 实测**（400ep warm-start F2；val 协议 seed0）：VT2M MPJPE 69.0 /
PA 39.3、V2M 64.5 / 36.2、**T2M 134.9 / 77.7（链内最优）**、yaw_drift 2.5/2.3/2.8
（三配置全优）。组合成立：f2 修 yaw、foot_conv 修 VT，T2M 较 F4a −40mm。

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr --t-encoder foot_conv \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F2_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F2p4_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag f2p4_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:7

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F2p4_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/F2p4_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:7

# ③ 可视化 R_Test1（骨架动画）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --modal F2p4 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --modal F2p4 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## V3-2 · 九部位移植（结构线主线，warm-start 自 F4a）

模型：anysolev2 + `--t-encoder foot_conv --pose-parts 9 --lr-warmup-frac 0.05`
（V3 结构步：前 5% 步数线性 warmup）。**前两次训练同型爆炸**（xd6kcdk2 ep386、
1s0yeuwn ep531：traj 先行发散、硬窗口触发、有限大梯度），但健康期（1s0yeuwn
ep110-510 VT 52.9-58.1 / T 115）证明结构成立；ckpt_best 保护已上线（ep480 最优
不再丢失）。重训命令一致，可选：v1.yaml 改 lr 3e-4 或 `--epochs 450` 缩短暴露。
**2026-09-21 实测**（740ep warm-start F4a，**全程无爆炸**——grad-clip+warmup+NaN
守卫生效）：ckpt_last@740 VT2M 72.8 / 42.2、V2M 65.7 / 35.3、T2M 183.1 / 69.9
（T-only yaw 73.6 依旧弱）；**ckpt_best@310 VT2M 67.4 / 37.9（链内最优）、
V2M 66.7 / 34.5**；contact_f1 VT 0.76（链内最优）。晚期仍有噪声漂移
（310ep 后 68→76 波动），T-only 全局弱点留给 V3-3 σ 门控。

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --init-from results/AnySole/F4a_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/V3_2_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag v32_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/V3_2_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/V3_2_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --modal V3_2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --modal V3_2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

> epoch 计数口径：每次运行从自己的 epoch 0 开始数，`--init-from` 只加载权重；
> 判断收敛/爆炸看**总步数**（3700 步短预算 ≈ 740ep batch256 @stride20）。

---

## V3-3 · soft A 矩阵 + σ 门控（两阶段单变量，warm-start 自 V3-2）

> 设计与机制关系（A 矩阵/门控/先验）见 `archived/fix_plan_v3.md` §8；
> smoke 必过：`z_note/probes/smoke_v33_soft_gate.py`（训练前跑）。
> **2026-09-21 首训审计已修两处缺陷**（§8.4）：① L_assign 负载均衡项天生为负
> 导致 loss_total 为负——已加常数 +log9 平移到非负域（**不改梯度**，在跑的
> V3-3A/B 训练轨迹不受影响，只是曲线记录值为负）；② τ_p 无界漂移（+9.8/−3.8）
> 把门控软最大化饱和成常数 [0,0,1]、σ 梯度死亡——已加前向钳位 [−4,4]。
> **两者只对新启动的 run 生效**。③ 执行顺序：**必须等 V3-3A 收敛后再启
> V3-3B**（首训时两者并跑，V3-3B warm-start 到的是 V3-3A 的 epoch≈1 权重）。
> β-NLL σ 监督未实现，σ 经软门控端到端训练（首训已实测 σ 未学习，即 plan §3
> 失败处置二，重启后优先补 β-NLL）。
> 两阶段单变量纪律（§2.1）：**V3-3A** 只加 `--soft-parts`（验收 = 不劣于 V3-2
> 且 A 保持可识别部位结构）；**V3-3B** 在 A 上叠加 `--gate sigma`（验收 = plan
> §2.3 F4：T-only 时 arm/spine g_∅>0.1、VT 时 foot g_T − arm g_T > 0.2；
> wandb 曲线 `gate/<part>_gV/gT/gE`）。
> **二轮实测结论（2026-09-21 深夜，V3-3B 740ep）**：指标链内最优但 σ 未学会
> 路由（三配置门控差 ≤0.018）、A 冻结于人工划分、F4 原判据区分不了静态/动态 →
> 判据已改差分式（§2.3 F4 修订），σ 已修（输入 [z,e_V,e_T] + β-NLL 信任监督 +
> 前 10% 冻结）。**V3-3C = V3-3B 命令 + `--lambda-sigma 0.01
> --sigma-freeze-frac 0.1`**（warm-start 自 V3_3B 终值），验收读
> `z_note/probes/probe_v33b_gates.py` 的差分判据（T vs VT Δg_T(foot)>0.2、
> T-only root g_V 下降 >0.2）。T2M 偏航乱飘已量化（probe_v33b_yaw_t2m.py：
> yaw_abs 71.5°、速率 231°/s vs GT 12.6°/s）——压力无朝向信息，结构解法 =
> V3-4 f2 移植，不是门控。

```bash
# ===== V3-3A：soft A 矩阵（warm-start 自 V3-2） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --soft-parts \
  --lambda-assign 0.05 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --init-from results/AnySole/V3_2_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/V3_3A_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag v33a_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/V3_3A_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/V3_3A_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画 → results_display/result/r_test1_visualize/AnySole/V3_3A/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --modal V3_3B --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比 → results_display/result/r_test3_traj/AnySole/V3_3A/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --modal V3_3B --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ===== V3-3B：叠加 σ 门控（warm-start 自 V3-3A） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --soft-parts --gate sigma \
  --lambda-assign 0.05 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --init-from results/AnySole/V3_3A_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/V3_3B_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag v33b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:7

# ② 评估（同 V3-3A，换目录名）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/V3_3B_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/V3_3B_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（--modal V3_3B） / ④ 可视化 R_Test3（--modal V3_3B）：同上换名


# 训练前 smoke（数值验收，§8.4；可选 --ckpt 验证 warm-start 继承）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/smoke_v33_soft_gate.py \
  --device cpu --ckpt results/AnySole/V3_2_joint_and/checkpoints/ckpt_last.pt
```

> warm-start 继承：decoder 共享部分（self-attn / V 分支交叉注意力 / norm / FFN /
> query / group_emb）按名字继承；T 分支交叉注意力 / σ MLP / τ_p / part_logits /
> 9 个全宽输出头自动新初始化（smoke 第 7 项验证）。gate 与 `--tactile-direct`
> 互斥（构造期报错）；`--soft-parts`/`--gate` 均要求 `--pose-parts 9`。

---

## 已取消 / 已作废（勿用）

- **F1**（`--v-input hmr_gvhmr`）：已取消（视觉天花板假说证伪），代码留档、权重不下载。
- **F2b**（`--lambda-kp 3`）：实测有害，弃；λ_kp 保持 1.0。
- **F5**（part_decoder 门控）：已作废并回滚（encode_stream 缺 LN 根因），
  part_decoder.py 留档；V3-3 的 `--gate sigma` 尚未实现。
- **diffusion-era 开关**（`--tau-max/--tau-fixed`、`warm_start`、`continuation` 的
  V1 用法、`--write-bvh`）：anysolev2 下被忽略并打印 WARNING；BVH 导出已废除。
