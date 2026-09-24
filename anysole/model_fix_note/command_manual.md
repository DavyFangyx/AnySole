# AnySole 命令手册（SMPL-24 · 2026-09-20）

> env = touch_gait；根目录 `/data/fangyuxuan/projects/gait`。
> 历史与演进见 `model_fix_note.md`；模型结构见 `anysole/AnySole 多模态动作重建 · 模型实现说明.md`。

## CLI 参数与支持空间（注释版）

```text
# --model-name: 支持 F0b/F4a/F2/F2p4/V3_2/V3_3A/V3_3B/V3_4a/V3_4b/V3_4c/V4A/V4B
# --contact-method: 接触标签生成/读取方案，不是输入特征 concat 开关；当前主线为 joint_and
#   因此不参与主损失，也不会被拼接进 V/T 输入。
# --variant: tw40、tw80、tw40_st20 等超参变体

# --which {last,best}: 自动选择 ckpt_last.pt 或 ckpt_best.pt，默认 last
# --ckpt FILE: 仅供调试、旧脚本和文件型探针；正式流程优先使用 model-name

# --tw INT (>=1): 窗口长度，默认 20；扫描可用 40/80/120 或其他正整数
# --stride INT (>=1): 窗口步长；扫描时必须显式设置为与 tw 相同的值
# --epochs INT、--batch-size INT、--lr FLOAT (>0)、--seed INT
# --t-encoder {linear,foot_conv}; --tactile-input {raw108,s2m50}; --no-imu
# --f2-repr; --pose-parts INT; --soft-parts; --gate {none,sigma}; --assign-cluster
#   assign-cluster 要求 pose-parts=24 + soft-parts，且不能与 gate 同用；V4A 使用它
# --split {train,val,test}; --config-id {VT2M,V2M,T2M}[,...]
# --sample-steps INT; --device {auto,cpu,cuda,cuda:N}
# 自动地址：results/AnySole/<model-name>_<contact-method>/[<variant>]/
# checkpoint：上述目录/checkpoints/ckpt_{last,best}.pt
# 例：--model-name V4A --tw 40 --stride 40 -> V4A_joint_and/tw40/
```

地址规则的单一事实源是 `anysole/registry.py`。正式命令不要手写
`results/AnySole/.../checkpoints/ckpt_last.pt`、`--out-dir` 或 `--write-motion`；
只有输入本身就是独立文件的探针才保留 `--ckpt`。

## 通用约定

- **目录编号 = `{模型版本}_{contact-method}`**（现在全部 SMPL-24，不再有任何协议
  后缀）。当前所有基座都用 `joint_and` 标签：`F0b_joint_and` / `F4a_joint_and` /
  `F2_joint_and` / `F2p4_joint_and` / `V3_2_joint_and` / `V3_3A_joint_and` /
  `V3_3B_joint_and`（V4A 聚类组目录 `V4A_joint_and`）。
- **变体命名（字段堆叠，2026-09-22 起）**：基座目录下的超参子目录，字段顺序
  固定 tw→st→lr→lp→lt→lk、默认省略（如 `V3_4a_joint_and/tw40`、
  `tw40_st20`、`t0003_tw40_lr3e4`）。**全流程零手写地址（2026-09-22 用户
  裁定）**：所有真实路径由 `anysole/registry.py` 的 `infer_anysole_paths`
  从标识推导（模型名+variant+which），warm-start 链 = registry 单一事实源；
  train 用 `--model-name`（init-from 自动=registry 父节点、out-dir 自动）；
  eval/infer/ridge 用 `--model-name [--variant] [--which]`；R_Test1/R_Test3
  用 `--model-name`；tune 用 `--model-name`。`--out-dir/--init-from/
  --ckpt/--write-motion` 一律禁止手写，仅作调试覆盖；文件型探针是例外。
- **checkpoint 状态**：group_ids 修复（2026-09-20）后，此前全部权重已归档
  `results/backup/AnySole_BVH_backup/`。**2026-09-21 warm-start 链重训完成**：
  F0b / F4a / F2 / F2+4 / V3-2 / V3-3A / V3-3B 基座全部训出并过 val 协议验收
  （各节已回填实测数字，均为**非 f2 历史口径**）。**现行链 = V3-4 f2 链
  （V3-4a/b/c）尚未训练**（F2p4 已训、为链起点；V3-4c 带 σ 填坑开关）。
- 训练顺序（warm-start 链）：
  - **历史链（非 f2 口径，已训完、不再推进）**：F0b → F4a / F2 → F2+4 →
    V3-2 → V3-3A → V3-3B（各节命令 = 产生其实测数字的原样，数字为历史记录）。
  - **现行链（f2 标准表征，见 V3-4 节）**：F2p4（已训，链起点）→
    **V3-4a → V3-4b → V3-4c**，三步串行；V3-4c = σ 门控 + β-NLL 填坑版
    （`--lambda-sigma 0.01 --sigma-freeze-frac 0.1`）。
  - **证明组（V4A 节，与主线并行）**：V4A ← F4a（24 槽梯度聚类，分组从数据
    学、K 不预设）；后续 V4B = 学到的分组硬重训对照 V3-2。
- `anysole/configs/v1.yaml` 已是 SMPL-24 训练配置：raw108、tw=20、stride=20、lr=1e-4
  （`--lr` 可作为实验覆盖值传入，默认仍取 yaml）、λ_pose=3 / λ_kp=1 / λ_traj=1 /
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
  `results_display/script/ridge_probe.py --model-name <名称> [--variant <变体>]
  --split val --device <GPU>`）。
- wandb tag 与目录同名（如 `f0b_jointand`），与 BVH 时代历史组天然区分。
- `--device` 按 `nvidia-smi` 空闲卡填（示例用 cuda:4）。
- 单会话推理（需要时）：`/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.infer
  --contact-method joint_and --model-name <名称> [--variant <变体>] --session <SID> --config-id VT2M --device <GPU>`。

---

## 基座 F0b · 回归本体（from-scratch）

> **历史口径（非 f2）**：命令 = 产生下记实测数字的原样，不再改动；f2 版 = F2 节。

**2026-09-21 实测**（400ep from-scratch，wandb fxemwdt8；val 协议 seed0）：
VT2M MPJPE 78.1 / PA 44.8、V2M 81.4 / 46.2、T2M 154.9 / 72.4；
VT yaw_drift 4.5、jitter 12.0、contact_f1 0.43、foot_slide 2.82mm/帧。
from-scratch 400ep 是筛选预算，显著弱于 BVH 时代 warm-start 的 F0b_seed2
（63.7/25.1）；若作为最终基线，续跑至 800ep（yaml 默认）预期继续收敛。

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --epochs 400 --grad-clip 5.0 \
  --model-name F0b --from-scratch \
  --wandb_mode online --wandb_experiment_tag f0b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4

# ② 评估（协议 seed0 + 导出 SMPL NPZ，R_Test1/R_Test3 依赖此步）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name F0b \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画 → results_display/result/r_test1_visualize/AnySole/F0b/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name F0b --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比 → results_display/result/r_test3_traj/AnySole/F0b/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --model-name F0b --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## 基座 F4a · 按脚卷积触觉（`--t-encoder foot_conv`，warm-start 自 F0b）

> **历史口径（非 f2）**：命令 = 产生下记实测数字的原样，不再改动；f2 版 = F2p4 节。

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
  --contact-method joint_and \
  --t-encoder foot_conv \
  --epochs 400 --grad-clip 5.0 \
  --model-name F4a \
  --wandb_mode online --wandb_experiment_tag f4a_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name F4a \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name F4a --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --model-name F4a --contact-method joint_and --split val \
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
  --contact-method joint_and \
  --f2-repr \
  --epochs 400 --grad-clip 5.0 \
  --model-name F2 \
  --wandb_mode online --wandb_experiment_tag f2_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6

# ② 评估（eval/协议/导出自动处理 f2→world 恢复）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name F2 \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:6

# ③ 可视化 R_Test1（骨架动画）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name F2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --model-name F2 --contact-method joint_and --split val \
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
  --contact-method joint_and \
  --f2-repr --t-encoder foot_conv \
  --epochs 400 --grad-clip 5.0 \
  --model-name F2p4 \
  --wandb_mode online --wandb_experiment_tag f2p4_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:7

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name F2p4 \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:7

# ③ 可视化 R_Test1（骨架动画）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name F2p4 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --model-name F2p4 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## V3-2 · 九部位移植（历史口径（非 f2），warm-start 自 F4a）

> **历史口径（非 f2）**：已训完、不再推进；f2 版 = V3-4 节的 **V3-4a**。

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
  --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_2 \
  --wandb_mode online --wandb_experiment_tag v32_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name V3_2 \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name V3_2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --model-name V3_2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

> epoch 计数口径：每次运行从自己的 epoch 0 开始数，`--init-from` 只加载权重；
> 判断收敛/爆炸看**总步数**（3700 步短预算 ≈ 740ep batch256 @stride20）。

---

## V3-3 · soft A 矩阵 + σ 门控（历史口径（非 f2），warm-start 自 V3-2）

> **历史口径（非 f2）**：A/B 已训完，数字与审计见下，不再推进；f2 版 = V3-4 节的
> **V3-4b**（soft A）与 **V3-4c**（σ 门控 + β-NLL 填坑）。
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
> 前 10% 冻结）。**σ 填坑版 = V3-4 节的 V3-4c**（`--lambda-sigma 0.01
> --sigma-freeze-frac 0.1`），验收读 `z_note/probes/probe_v33b_gates.py` 的差分
> 判据（T vs VT Δg_T(foot)>0.2、T-only root g_V 下降 >0.2）。T2M 偏航乱飘已量化
> （probe_v33b_yaw_t2m.py：yaw_abs 71.5°、速率 231°/s vs GT 12.6°/s）——压力无
> 朝向信息，结构解法 = f2 表征（V3-4 链），不是门控。

```bash
# ===== V3-3A：soft A 矩阵（warm-start 自 V3-2） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --soft-parts \
  --lambda-assign 0.05 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_3A \
  --wandb_mode online --wandb_experiment_tag v33a_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name V3_3A \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画 → results_display/result/r_test1_visualize/AnySole/V3_3A/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name V3_3B --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比 → results_display/result/r_test3_traj/AnySole/V3_3A/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --model-name V3_3B --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ===== V3-3B：叠加 σ 门控（warm-start 自 V3-3A） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --soft-parts --gate sigma \
  --lambda-assign 0.05 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_3B \
  --wandb_mode online --wandb_experiment_tag v33b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:7

# ② 评估（同 V3-3A，换目录名）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name V3_3B \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（--model-name V3_3B） / ④ 可视化 R_Test3（--model-name V3_3B）：同上换名


# 训练前 smoke（数值验收，§8.4；--model-name 解析 warm-start 检查 ckpt）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/smoke_v33_soft_gate.py \
  --device cpu --model-name V3_2
```

> warm-start 继承：decoder 共享部分（self-attn / V 分支交叉注意力 / norm / FFN /
> query / group_emb）按名字继承；T 分支交叉注意力 / σ MLP / τ_p / part_logits /
> 9 个全宽输出头自动新初始化（smoke 第 7 项验证）。gate 与 `--tactile-direct`
> 互斥（构造期报错）；`--soft-parts`/`--gate` 均要求 `--pose-parts 9`。

---

## V3-4 · f2 标准表征链（现行主线，2026-09-21 用户裁定；三步串行）

> **定位**：T-only 偏航根因 = 压力无绝对朝向（probe_v33b_yaw_t2m.py：yaw_abs
> 71.5°、速率 231°/s vs GT 12.6°/s）。f2 把整条链的预测目标从"绝对朝向"改为
> "朝向变化/相对量"（根 6D=tilt、轨迹=4 维朝向系 [psi_dot, v_hx, v_hz, h]），
> 三配置共用一套权重。F2/F2p4 = F0b/F4a 的 f2 版（已训，F2p4 为链起点）；
> V3 机制组按三步在 f2 基座上重跑，**必须等上一步收敛后再启下一步**：
>
> | 步 | 内容 | 填的坑 | 验收 |
> |---|---|---|---|
> | **V3-4a** | 9 部位（`--pose-parts 9`），warm-start 自 F2p4 | 部位结构基座 | 打平 F2p4（VT ≤69±2、T2M ≤135±3） |
> | **V3-4b** | + soft A（`--soft-parts`） | 学习分组 | 不劣于 V3-4a，A 保持可识别部位 |
> | **V3-4c** | + σ 门控 + **β-NLL 填坑**（`--gate sigma --lambda-sigma 0.01 --sigma-freeze-frac 0.1`） | **σ 无法学习的坑**：τ_p 饱和 → softmax 梯度 g(1−g)→0 → σ 梯度死亡（V3-3B 因果链实测：信号到 z 变了 79%，σ logit 只动 0.12、门控 ≤0.018）。β-NLL 给 σ 直连梯度（不经 softmax），前 10% 步 σ 冻结让 τ_p 先定初始分工 | F4 差分判据（probe_v33b_gates.py）+ T2M ≤92 且 PA ≤36、VT2M ≤65 + yaw 探针 T2M yaw_abs ≤10° |

```bash
# ===== V3-4a：9 部位（f2 基座，warm-start 自 F2p4） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --t-encoder foot_conv --f2-repr --pose-parts 9 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_4a \
  --wandb_mode online --wandb_experiment_tag v34a_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5

# ② 评估（协议 seed0 + 导出 SMPL NPZ，R_Test1/R_Test3 依赖此步）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name V3_4c \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画 → results_display/result/r_test1_visualize/AnySole/V3_4a/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name V3_4c --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比 → results_display/result/r_test3_traj/AnySole/V3_4a/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --model-name V3_4c --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ===== V3-4b：+ soft A（warm-start 自 V3-4a） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --t-encoder foot_conv --f2-repr --pose-parts 9 --soft-parts \
  --lambda-assign 0.05 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_4b \
  --wandb_mode online --wandb_experiment_tag v34b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6

# ② 评估 / ③ R_Test1 / ④ R_Test3：同上，目录与 --model-name 换 V3_4b

# ===== V3-4c：+ σ 门控（β-NLL 填坑版，warm-start 自 V3-4b） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --t-encoder foot_conv --f2-repr --pose-parts 9 --soft-parts --gate sigma \
  --lambda-assign 0.05 --lambda-sigma 0.01 --sigma-freeze-frac 0.1 \
  --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_4c \
  --wandb_mode online --wandb_experiment_tag v34c_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:7

# ② 评估 / ③ R_Test1 / ④ R_Test3：同上，目录与 --model-name 换 V3_4c
# 门控差分验收（F4 修订判据：T vs VT 的 Δg_T(foot)>0.2、T-only 时 root g_V 较 VT 下降>0.2）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/probe_v33b_gates.py \
  --model-name V3_4c --device cuda:5

# T2M 偏航验收（目标 yaw_abs ≤10°）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/probe_v33b_yaw_t2m.py \
  --model-name V3_4c
```

> warm-start 继承（每步）：共享结构按名字继承（F2p4 的 4 维 traj_head.proj、
> foot_conv、decoder 共享部分）；新结构（9 部位 query/out_parts、part_logits、
> T 分支交叉注意力/σ/τ_p）自动新初始化。每步训练前 smoke 照跑
> （smoke_v33_soft_gate.py / smoke_v32_parts.py / smoke_f2_roundtrip.py）。

---

## V4A · 24 槽梯度聚类（证明组：warm-start 自 F4a，历史口径）

> 24 槽 + 均匀初始化 A + 退火熵/聚集奖励/死槽税 + 软 EM（A 独立 lr、70% 锁定）。
> 与 V3_2 同起点同预算，唯一差异 = 分组来源（学习 vs 人为）。
> 训练前 smoke：`smoke_v4a_cluster24.py`。实测与结论见 `model_fix_note.md` 节点 6。

```bash
# ① 训练（740ep warm-start 自 F4a；预算与 V3_2/V3_3A/B 对齐）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 24 --soft-parts --assign-cluster \
  --lambda-assign 1.0 --lambda-assign-ent 0.05 --lambda-assign-conc 0.01 \
  --lambda-assign-dead 0.02 --assign-dead-beta 2.0 \
  --assign-temp-init 1.0 --assign-temp-final 0.2 \
  --assign-anneal-frac 0.7 --assign-lock-frac 0.9 --assign-lr-mult 5.0 \
  --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V4A \
  --wandb_mode online --wandb_experiment_tag v4a_jointand \
  --wandb_eval_interval 10 --loss-cap 10.0 --device cuda:7

# ② 评估（常规三配置；分组本身是产物，指标供 V4B 对照参考）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name V4A \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 读出学到的分组（A 矩阵 / 有效 K / ARI·NMI / 左右对称；落盘 partition_v4a_learned.json）
# 注意：该文件型探针当前只接受 --ckpt，因此此处保留显式 checkpoint 地址。
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/probe_v4a_readout.py \
  --ckpt results/AnySole/V4A_joint_and/checkpoints/ckpt_last.pt --device cuda:5

  
```

> `--assign-cluster` 要求 `--pose-parts 24 --soft-parts`，与 `--gate` 互斥；
> `--lambda-assign` 固定 1.0（三个聚类项自带 `--lambda-assign-*` 权重）。

---

## V4B · 方案 B 表征聚类 → 学到的分组硬训练（warm-start 自 F4a，对照 V3_2）

> 用户裁定（2026-09-22）：V4A 零启动后，学习分组改走表征聚类。F4a 源 per-joint
> 表征 → Ward 聚类 + silhouette 定 K → `--part-json` 硬训练，与 V3_2 同预算
> 对比 = 人为划分终审。实测见 `model_fix_note.md` 节点 6；候选：
> `results/AnySole/F4a_joint_and/metrics/partitions_cluster_b_K{9,12,13,14}.json`。
> **实测（740ep，K9）**：test T 198.9 / V 88.3 / VT 92.0 vs V3_2 205.4 / 88.1 /
> 93.2（打平偏优）；val 协议 T 173.3 / V 65.7 / VT 70.5（F4a 级，不敌 V3_3B）。

```bash
# ① 表征聚类（F4a 源；该文件型探针当前只接受 --ckpt）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/probe_part_cluster_b.py \
  --ckpt results/AnySole/F4a_joint_and/checkpoints/ckpt_last.pt --device cuda:5

# ② 硬训练学到的分组（示例 = K9 候选）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --lr-warmup-frac 0.05 \
  --part-json results/AnySole/F4a_joint_and/metrics/partitions_cluster_b_K9.json \
  --epochs 740 --grad-clip 5.0 \
  --model-name V4B \
  --wandb_mode online --wandb_experiment_tag v4b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6

# ③ 评估（与 V3_2 对比 = 终审）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name V4B \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ④ 读出分组（落盘 metrics/partition_v4b_learned.json，格式同 V4A 组；文件型探针）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/probe_v4a_readout.py \
  --ckpt results/AnySole/V4B_joint_and/checkpoints/ckpt_last.pt --device cuda:5



```

> 验收：不劣于 V3_2（其 val 协议为过期口径，先重跑 V3_2 ② 评估刷新，或直接比
> test.json）。K9 若劣，再试 K=12。`--part-json` 与 `--soft-parts`/`--gate`
> 互斥；分组须覆盖 24 关节恰好一次。

---

## 窗口扫描（tw sweep，基模 V3_4a）

> **设计（2026-09-22 用户批准，2026-09-22 修订 stride 与目录规则）**：
> 基线 tw=20 = V3_4a 现成数字（VT 72.1/41.2、V 64.5/35.2、T 121.2/69.1），
> 不重跑。扫描点 **40 / 80 / 120**（宽间距先定趋势；若 40 附近有拐点再补
> 30/50）。单变量锁定：**`--stride` 显式传 `<tw>`**（v1.yaml 固定 stride=20，
> 不传会变成 50% 重叠窗——TW40 首跑即因此污染，见 tw40_st20 记录）、
> `traj_deltas` 保持 [2,4,8,19]、epochs 740、一律 warm-start 自 V3_4a
> （**已键级验证**：330/336 键继承，仅 time_pe/query 按新 tw 初始化）。
> 三个点互不依赖，可三卡并行。
> **目录规则（字段堆叠命名，2026-09-22 用户裁定）**：扫描点 = 基座目录下的
> 超参子目录 `V3_4a_joint_and/tw40|tw80|tw120`；字段顺序 tw→st→lr→lp→lt→lk，
> 默认省略——**模型+接触+超参 = checkpoint 地址**，eval 用 `--variant`、可视化用
> `--variant`、ridge 用 `--model-name + --variant`；只接受文件输入的探针才用 `--ckpt`。
> **TW40 首跑记录**：`V3_4a_joint_and/tw40_st20` = stride=20 重叠窗污染版
> （VT 90.5 / V 81.5 / T 148.6，三配置较基线 +17~+27mm），不可判读，干净
> 口径以 tw40（stride=40）为准。
> 判读：seam_jump 随 tw 变大而样本数锐减（tw=80/120 时 val 每会话仅 2–3
> 条缝），勿过度解读。成本估算：740ep ≈ 1.5h / 3.5h / 5h。

```bash
# ① 训练（地址全推断：--model-name + 字段自动拼目录，禁止写 --out-dir；
#    tw80/120 只需换 --tw/--stride，目录自动落到 tw80/tw120，不可能互相覆盖）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and --model-name V3_4a \
  --t-encoder foot_conv --f2-repr --pose-parts 9 --lr-warmup-frac 0.05 \
  --tw 40 --stride 40 \
  --epochs 740 --grad-clip 5.0 \
  --wandb_mode online --wandb_experiment_tag tw40_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:2
#   → out-dir 自动 = results/AnySole/V3_4a_joint_and/tw40/checkpoints

# ② 评估（--model-name + --variant → ckpt 地址自动解析；导出默认落
#    <模型目录>/predictions/eval_motion，--ckpt/--write-motion 都不写）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --contact-method joint_and --model-name V3_4a --variant tw40 \
  --split val --protocol-seed 0 --no-robustness --device cuda:7

# ③④ 可视化（R_Test1/R_Test3 用 --variant 定位，输出自动落到变体名下）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name V3_4a --contact-method joint_and --variant tw40 --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ridge 探针每步必跑（使用 model-name + variant 自动定位）：
# results_display/script/ridge_probe.py --model-name V3_4a --variant tw40 --split val --device <GPU>
```

> **碰撞事故（2026-09-22，教训已入代码）**：三个 tw run 复制粘贴了显式
> `--out-dir .../tw40/checkpoints`，全部互相覆盖——幸存的 ckpt 实为
> tw=80/stride=40，已重贴标签为 `tw80_st40`；tw40 干净口径与 tw120 已丢失，
> 需重跑。**此后所有命令禁止手写 --out-dir / --ckpt / --write-motion**：
> 目录由 --model-name + 字段自动推断（train），或由 --model-name +
> --variant 解析（eval），或由 ckpt 路径推导（探针）——地址只从字段来。

---

## Optuna 调参（`python -m anysole.tune`，基模 V3_4a）

> **定位（2026-09-22 用户批准）**：V3_4a = 最简 V3 结构 + 链内最优 V2M +
> T2M 121.2，且窗口扫描证明它还有成长空间 → 作为调参对象。模式沿用
> SurvPGC 的 optuna_utils（TPE/random sampler + sqlite study 存储 +
> 子进程逐 trial 训练/评估 + trials.csv/best_trial.txt 产物）。
> **v1 无剪枝**（objective 每 trial 结束才报一次，MedianPruner 无中间值可用）
> → 预算控制靠短 `--epochs`。study 存 sqlite，中断后同 `--study-name` 续跑。
> objective = 三配置 val MPJPE 加权均值（默认 1:1:1，最小化）；
> 每个 trial warm-start 自 V3_4a（tw 形张量按采到的 tw 重新初始化）。

| 搜索维度 | 取值 | 说明 |
|---|---|---|
| `tw` | {30, 40, 50, 80} | 上下文长度（主要成长轴） |
| `lr` | {3e-5, 1e-4, 3e-4, 1e-3} | 经新加的 `--lr` CLI（原 yaml 默认 1e-4） |
| `lambda_pose` | {2, 3, 5} | 主损失权重 |
| `lambda_traj` | {0.5, 1, 2} | 轨迹损失 |
| `lambda_kp` | {0.5, 1, 2} | FK 关键点损失 |
| `stride` | {tw, tw//2} | 重叠窗（half） |

固定不变：f2 / foot_conv / 9 部位 / λ_con=0 / λ_assign、λ_sigma 无关（V3_4a
无 A/门控） / grad-clip 5.0 / warmup 5% / 结构开关全锁。

```bash
# 先 dry-run 抽 2 个 trial 看拼出的命令（不训练）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.tune \
  --study-name v34a_dry --n-trials 2 --dry-run --seed 0 --device cuda:7

# 正式搜索（150ep/trial ≈ 15–45min 依 tw；30 trials ≈ 8–20 GPU 小时）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.tune \
  --study-name v34a_sweep1 --n-trials 30 --epochs 150 \
  --sampler tpe --seed 0 --device cuda:4

# 产物：results/AnySole/optuna/<study>.db（study 本体，可续跑）
#       results/AnySole/optuna/<study>/optuna_trials.csv + best_trial.txt
# 每个 trial：results/AnySole/V3_4a_joint_and/<t{NNNN}_字段堆叠>/（ckpt + metrics + trial.log；
# 字段堆叠规则同窗口扫描节——模型+接触+超参 = ckpt 地址，eval/可视化/探针三种定位等价）
```

> 依赖：optuna 已装入 touch_gait（2026-09-22）。调参后若发现更优超参组合，
> 按单变量纪律回灌到对应基座命令再正式重训。

---

## 已取消 / 已作废（勿用）

- **F1**（`--v-input hmr_gvhmr`）：已取消（视觉天花板假说证伪），代码留档、权重不下载。
- **F2b**（`--lambda-kp 3`）：实测有害，弃；λ_kp 保持 1.0。
- **F5**（part_decoder 门控）：已作废并回滚（encode_stream 缺 LN 根因），
  part_decoder.py 留档；V3-3 的 `--gate sigma` 尚未实现。
- **diffusion-era 开关**（`--tau-max/--tau-fixed`、`warm_start`、`continuation` 的
  V1 用法、`--write-bvh`）：anysolev2 下被忽略并打印 WARNING；BVH 导出已废除。
