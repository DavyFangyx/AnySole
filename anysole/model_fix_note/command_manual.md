# AnySole 命令手册（SMPL-24 · 2026-09-20）

> env = touch_gait；根目录 `/data/fangyuxuan/projects/gait`。
> 历史与演进见 `model_fix_note.md`；模型结构见 `anysole/AnySole 多模态动作重建 · 模型实现说明.md`。

## 通用约定

- **目录编号 = `{模型版本}_{contact-method}`**（现在全部 SMPL-24，不再有任何协议
  后缀）。当前所有基座都用 `joint_and` 标签：`F0b_joint_and` / `F4a_joint_and` /
  `F2_joint_and` / `F2p4_joint_and` / `V3_2_joint_and`。
- **当前无有效 checkpoint**：group_ids 修复（2026-09-20）后，此前全部权重（含
  BVH 时代与 SMPL-24 早期的 F4a_footconv / V3_2_part9）已归档
  `results/backup/AnySole_BVH_backup/`，必须按下面的链条从零重训。
- 训练顺序（warm-start 链）：**F0b（from-scratch）→ F4a / F2（←F0b）→
  F2+4（←F2）→ V3-2（←F4a）**。
- `anysole/configs/v1.yaml` 已是 SMPL-24 训练配置：raw108、tw=20、stride=20、lr=1e-4
  （**`--lr` 不是 CLI**，要改 lr 就改 yaml）、λ_pose=3 / λ_kp=1 / λ_traj=1 /
  λ_trec=0.1 / λ_vrec=0.1 / λ_con=0、joint_and。实验差异全部由 CLI 传入，
  覆盖值随 ckpt 保存，eval/infer 从 ckpt 读配置。
- 训练前单测：`z_note/probes/smoke_f0b_regress.py`（回归头）、
  `smoke_f4a_grid.py`（foot_conv）、`smoke_f2_roundtrip.py`（f2 表示）。
- 每个基座固定四条命令：**① 训练 → ② 评估（--write-motion 刷新导出）→
  ③ Test1 可视化（骨架动画）→ ④ Test3 可视化（轨迹对比）**。Test1/Test3 读
  `predictions/eval_motion/<session>_<config>.npz`，因此评估必须先于可视化。
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

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --epochs 400 --grad-clip 5.0 \
  --out-dir results/AnySole/F0b_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag f0b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4

# ② 评估（协议 seed0 + 导出 SMPL NPZ，Test1/Test3 依赖此步）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F0b_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/F0b_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 Test1（骨架动画 → results_display/Test1_visualization/AnySole/F0b/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole.py \
  --modal F0b --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 Test3（轨迹对比 → results_display/Test3_trajectory/AnySole/F0b/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole_traj.py \
  --modal F0b --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## 基座 F4a · 按脚卷积触觉（`--t-encoder foot_conv`，warm-start 自 F0b）

模型：anysolev2 + `--t-encoder foot_conv`（数据不动，只换编码器）。
历史 BVH 口径结果（仅参照）：VT 55.5/23.0、T2M 123.8/40.3；T-only 全身退化
+7.6（V3 的动机）。新 t_enc 结构 warm-start 时按名字自动跳过、保持新初始化。

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F0b_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F4a_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag f4a_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F4a_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/F4a_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 Test1
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole.py \
  --modal F4a --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 Test3
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole_traj.py \
  --modal F4a --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## 变体 F2 · f2 表示（`--f2-repr`，warm-start 自 F0b）

根 6D 拆 tilt/yaw + 4 维 heading-frame 轨迹；traj_head.proj 3→4 维自动跳过。
历史 BVH 口径结果（仅参照）：T2M 89.7/35.8（T2M 最优）、VT 65.4/24.7。

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F0b_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F2_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag f2_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4

# ② 评估（eval/协议/导出自动处理 f2→world 恢复）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F2_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/F2_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 Test1
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole.py \
  --modal F2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 Test3
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole_traj.py \
  --modal F2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## 变体 F2+4 · f2 表示 + 按脚卷积（warm-start 自 F2）

历史 BVH 口径结果（仅参照）：VT 63.4/23.1（PA 三配置最优）、T2M 99.7/40.7、
yaw_drift 三配置全最优（foot_conv 为新结构自动跳过）。

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr --t-encoder foot_conv \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F2_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F2p4_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag f2p4_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F2p4_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/F2p4_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 Test1
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole.py \
  --modal F2p4 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 Test3
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole_traj.py \
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

```bash
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --init-from results/AnySole/F4a_joint_and/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/V3_2_joint_and/checkpoints \
  --wandb_mode online --wandb_experiment_tag v32_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4

# ② 评估
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/V3_2_joint_and/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-motion results/AnySole/V3_2_joint_and/predictions/eval_motion \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 Test1
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole.py \
  --modal V3_2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 Test3
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/visualize_anysole_traj.py \
  --modal V3_2 --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

> epoch 计数口径：每次运行从自己的 epoch 0 开始数，`--init-from` 只加载权重；
> 判断收敛/爆炸看**总步数**（3700 步短预算 ≈ 740ep batch256 @stride20）。

---

## 已取消 / 已作废（勿用）

- **F1**（`--v-input hmr_gvhmr`）：已取消（视觉天花板假说证伪），代码留档、权重不下载。
- **F2b**（`--lambda-kp 3`）：实测有害，弃；λ_kp 保持 1.0。
- **F5**（part_decoder 门控）：已作废并回滚（encode_stream 缺 LN 根因），
  part_decoder.py 留档；V3-3 的 `--gate sigma` 尚未实现。
- **diffusion-era 开关**（`--tau-max/--tau-fixed`、`warm_start`、`continuation` 的
  V1 用法、`--write-bvh`）：anysolev2 下被忽略并打印 WARNING；BVH 导出已废除。
