> **已归档（2026-09-20）**：细节留档，不再维护。当前状态与结论见
> ../model_fix_note.md，当前基座命令见 ../command_manual.md。

# V3 结构主线命令手册

> env = touch_gait；主线 V3-2 → V3-3 → V3-4（结构线先行），V3-0/V3-1 数据线延后。
> 每步三条命令：训练 / 验证 / 可视化。跑完对照基线看走势，不设门槛。

---

## V3-2 九部位最小移植（代码已实现）

- **模型**：anysolev2 回归 + `--t-encoder foot_conv`
- **基线**：F4a_footconv（VT 55.5 / V 63.5 / T 123.8）
- **改动**：pose_head 的 23 关节查询（3 组）→ 9 部位查询 + 9 个部位 unembed 头；
  其余（融合、traj/aux 头、损失、数据）不动。warm-start 复用 decoder/编码器/fusion。

**训练**

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --init-from results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/V3_2_part9/checkpoints \
  --wandb_mode online --wandb_experiment_tag v32_part9min \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5
```

**验证**

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/V3_2_part9/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/V3_2_part9/predictions/eval_bvh \
  --protocol-seed 0 --no-robustness --device cuda:5
```

**可视化**

```bash
CUDA_VISIBLE_DEVICES=5 python results_display/script/visualize_anysole.py \
  --modal V3_2 --contact-method part9 --force
```

---

## V3-3 σ 软门控（`--gate sigma` 待实现）

- **模型**：V3-2 结构 + σ 门控
- **基线**：V3_2_part9
- **改动**：pose_head 交叉注意力改为对 fused F 的双模态掩码视图（前 20 token =
  V 源、后 20 = T 源）+ 每部位 σ 路由 + 条件先验路径（g_∅ 不注入保留 z）；
  σ 监督 = β-NLL（β=0.5，r_p = 部位标准化空间误差 stopgrad）。

**训练**

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv --pose-parts 9 --gate sigma --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --init-from results/AnySole/V3_2_part9/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/V3_3_gate/checkpoints \
  --wandb_mode online --wandb_experiment_tag v33_sigma_gate \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6
```

**验证**

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/V3_3_gate/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/V3_3_gate/predictions/eval_bvh \
  --protocol-seed 0 --no-robustness --device cuda:7
```

**可视化**

```bash
CUDA_VISIBLE_DEVICES=7 python results_display/script/visualize_anysole.py \
  --modal V3_3 --contact-method gate --force
```

---

## V3-4 f2 表示移植（V3-3 后）

- **模型**：V3-3 结构 + f2 表示
- **基线**：F2p4_combo（VT 63.4 / V 68.7 / T 99.7）
- **改动**：`--f2-repr`（根 6D 换 tilt + 4 维朝向系轨迹），门控/深监督机制代码
  零改动，按 f2 口径适配输出维度。

**训练**

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr --t-encoder foot_conv --pose-parts 9 --gate sigma --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --init-from results/AnySole/F2p4_combo/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/V3_4_f2/checkpoints \
  --wandb_mode online --wandb_experiment_tag v34_f2 \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5
```

**验证**

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/V3_4_f2/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/V3_4_f2/predictions/eval_bvh \
  --protocol-seed 0 --no-robustness --device cuda:7
```

**可视化**

```bash
CUDA_VISIBLE_DEVICES=7 python results_display/script/visualize_anysole.py \
  --modal V3_4 --contact-method f2 --force
```
