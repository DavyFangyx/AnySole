# Agent 05 · AnySole 缺陷修复任务书

> 由排查结论（见对话记录与本文"验收清单"）转化而来。共 4 项任务 + 1 条重要提醒 + 验收清单。
> 环境：`/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python`（Python 3.8，Torch 2.4）。新文件保留 `from __future__ import annotations`，改动风格与周围代码一致。

---

## Task 1【必做】修复可视化 0.4s trim 错位（骨架面板 vs 压力面板差 16 帧）

**文件**：`results_display/script/r_test1_visualize_anysole.py`

**背景（已数值验证）**：`parse_bvh_aligner` 默认 `trim_leading_seconds=0.40`，会裁掉 BVH 开头 0.4s：
- GT BVH（120Hz）被裁 48 帧 → `load_gt` 里 `interp_joints` 把裁剪后数组当 time 0 处理，导致 viz GT[t] 实际显示的是会话帧 t+16（开头 ~14 帧还被 clamp 在 0.4s 的固定姿态）；
- 预测 BVH（40Hz）被裁 16 帧 → viz pred[t] 显示的是写入帧 t+16。
- 两个骨架面板互相是对齐的，但都比压力/时间轴晚 0.4s。AnySole 的预测 BVH 帧 0 本身就是对齐后的 `visual_start_s`，不应再裁。

**改法**：仅修改 `r_test1_visualize_anysole.py` 内的两个调用点，传入 `trim_leading_seconds=0.0`：

- `load_gt`（约第 66 行）：
  ```python
  parsed = parse_bvh_aligner(bvh_path, trim_leading_seconds=0.0)
  ```
- `load_pred`（约第 74 行）：
  ```python
  parsed = parse_bvh_aligner(pred_bvh, trim_leading_seconds=0.0)
  ```

**不要改** `bvh_aligner_pose.py` 的默认值——`d_test1_data_viz.py`、`r_test1_visualize_motionpro.py`、`r_test1_visualize_step2motion.py` 渲染原始/基线 BVH 仍依赖默认 trim。

**注意**：改完后 `load_gt` 的 `t_mocap = t_grid - offset_s` 直接对应未裁剪 BVH 时间轴（与 `anysole/data/dataset.py` 的映射完全一致），无需其他改动；个别会话 `t_mocap[0]` 可能为负，`interp_joints` 的 clamp 行为与 dataset 的 `np.interp` 一致，可接受。

**已知残留（不改）**：预测 BVH 只有 `n_frames//20*20` 帧（dataset 只取整窗），`n = min(...)` 会让动画尾部最多少 19 帧。属既有行为，不在本次范围。

---

## Task 2【致命·必做】`T_phys` 归一化（模型学不动的根本原因）

**文件**：`anysole/data/pressure.py`，函数 `load_session_pressure`（约第 87-102 行）

**背景（已数值验证）**：
- `T_raw` 在 `dataset.py` 里经 `normalize_raw` 归一化到 0..1，但 `T_phys` 由**原始压力值**（0..1023）计算：训练 batch 实测 force 项均值 **6255**、最大 **20362**（`dataset.py:192-193`：`"T_raw": normalize_raw(...)`、`"T_phys": pressure["T_phys"]`）。
- 4 个数量级的尺度差被直接 concat 进 T-Enc 的 `Linear(108→256)`：fresh 模型实测 `t_tok` std≈**1051** → 融合输出 `F` std≈**744** → `v_hat` 初始 ~**300 m/s**（对应训练日志 epoch 1 的 `L_pose=24601 / L_traj=11598 / L_Trec=209962`）。`norm_first` 的 LayerNorm 挡不住——残差流把输入尺度一路带到各头。
- 对照实验：把 T_phys 换成归一化后重算（与 `anysole/ablations/insole_drift/physical_tokens.py` 的做法一致），同样 300 步单 batch 过拟合：`L_pose` 1.53→**0.086**，`L_kp` 0.14→**0.048**，`L_traj` 0.77→**0.009**。
- 这同时解释了 `anysolev1_insole_drift`（MPJPE 205mm）优于主模型（331mm）：`anysole/models/model.py:47-50` 用归一化后的 T_raw 重算 physical tokens。

**改法**：在 `load_session_pressure` 内对归一化副本计算物理特征，保持 `physical_tokens` 函数本身不变：

```python
left48 = interpolate_values(left["t_us"], left["values"], t_grid_us)
right48 = interpolate_values(right["t_us"], right["values"], t_grid_us)
t_raw = np.concatenate([left48, right48], axis=1)
if t_raw.shape[1] != T_RAW_DIM:
    raise ValueError("T_raw dim %d != %d" % (t_raw.shape[1], T_RAW_DIM))
# T_phys 与 T_raw 保持一致量级：用归一化后的压力（0..1）计算物理特征。
# 与 dataset.normalize_raw 的公式完全一致（clip 到 PRESSURE_CLIP 后除以 PRESSURE_CLIP）。
norm_left = np.clip(left48, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP
norm_right = np.clip(right48, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP
return {
    "T_raw": t_raw.astype(np.float32),
    "T_phys": physical_tokens(norm_left, norm_right),
    "left48": left48,
    "right48": right48,
}
```

要点：
- 归一化公式必须与 `anysole/data/pressure.py:105-106` 的 `normalize_raw` **逐字一致**（`np.clip(x, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP`）。
- 返回的 `left48`/`right48` 保持原始值不变（如有外部消费者依赖原始值）。
- 消费者只有 `anysole/data/dataset.py:151` 与 `anysole/infer.py:117` 两处，改这一个函数即可覆盖 train/eval/infer 三链路，且与 drift 消融在模型内重算的 physical tokens 同量级。
- **输入尺度变化后，旧 checkpoint 全部作废，必须重训**（用户已计划重训，此处仅提醒）。

**可选加固（建议顺手做）**：在 `anysole/models/encoders.py` 的 `LinearTemporalEncoder.forward` 末尾加一层 `nn.LayerNorm`（对投影输出做归一化，再叠加 time PE 与模态嵌入），防止未来任何输入尺度问题再次沿残差流扩散。若做，务必同时检查 `assert_batch_shapes` 之外的任何依赖（无）；若不做，写明原因。

---

## Task 3【必做】训练期 val 指标关节口径统一为 23 关节

**文件**：`anysole/train.py`，函数 `_evaluate`（约第 150 行）

**背景**：`_evaluate` 里 MPJPE 用 `pred_kp[:, :, :22]` 与 `kp_gt[:, :, :22]`（前 22 关节），而 `anysole/eval.py:214` 用全部 23 关节（`N_JOINTS=23`，即真实 BVH 的 Hips..RightToeBase）。口径不统一导致 val 与 test 的 MPJPE 不可比。

**改法**：去掉 `[:, :, :22]` 切片，与 `eval.py`、`fk_pose6d`（输出 23 关节）统一：

```python
totals["mpjpe"] += float(torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1).mean().item()) * n * 1000.0
```

仅此一处。`root_ate` / `contact_f1` / `foot_slide` 口径不动。

---

## Task 4【必做】补充 wandb 监控（3c"训练不充分"判定的参数）

**文件**：`anysole/train.py`

**背景**：训练不充分判定依据 = ① 每 epoch 仅 ~5 步（1167 窗口 / batch 256）、200 epoch ≈ 1000 优化步；② grad_norm≈3426；③ lr=1e-3；④ 各损失曲线；⑤ tau=0 干净输入重建 MPJPE（判定模型连训练集都拟合不了的关键实验：主模型 246mm / drift 127mm）。

**已有监控（勿重复添加）**：`train/lr`、`train/grad_norm`、`train/loss_*`（含 loss_Trec_Tmissing / loss_Vrec_Vmissing）、`val/mpjpe/{V,VT,T}`、`val/root_ate/*`、`val/contact_f1/*`、`val/foot_slide/*`、`val/gap_mpjpe_*`。

**需新增三项**：

1. **`train/global_step`** 与 **`train/steps_per_epoch`**：在 epoch 循环开头记录 `epoch_start_step = global_step`，epoch 结束时：
   ```python
   log["train/global_step"] = global_step
   log["train/steps_per_epoch"] = global_step - epoch_start_step
   ```
2. **`val/tau0_mpjpe/{V,VT,T}`**（tau=0 干净输入重建诊断）：在 `_evaluate` 里对每个 config 的**第一个 batch**（只算一个 batch，不跑扩散采样，开销可忽略）额外做：
   ```python
   zero = torch.zeros(bsz, device=device, dtype=torch.long)
   out0 = model(v_feat, t_raw, t_phys, batch["pose_gt"], zero, cid, batch.get("session_id"))
   kp0 = fk_pose6d(out0["x0_hat"], batch["trans_gt"] + anchor, batch["offsets"], batch["parents"])
   tau0_mpjpe = torch.linalg.vector_norm(kp0 - batch["kp_gt"], dim=-1).mean().item() * 1000.0
   ```
   计入该 config 的 result（键名 `val/tau0_mpjpe/{config_name}`，单位 mm）。注意 `_evaluate` 现有循环里已有 `pred_trans`/`anchor`/`fk_pose6d` 等变量，复用即可；注意 `model.eval()` 与 `torch.inference_mode()` 上下文（`_evaluate` 已有）。
3. 顺带：epoch 结束时若发生 non-finite 跳批（现有 `train.py:291-297` 分支），可加一个 `train/nonfinite_skips` 计数（可选，非必须）。

`wandb_run.log` 的 x 轴仍用 `step=epoch + 1`，不变。

---

## 明确不做（out of scope，验收时勿视为遗漏）

1. **1a 旧 BVH / 重训流程**——用户自行重训。**但见下方重要提醒**。
2. **3b contact 标签阈值**——用户负责（含新增 Test5 可视化验证）。本次**不要**动 `Baselines/MotionPRO/data_prep/prepare_sequences.py` 与 `anysole/types.py` 的 `CONTACT_SUM_THRESH`。
3. **3c 超参调节**——用户负责（batch/lr/grad clip 等）。Task 4 只加监控，不改超参。
4. **`r_test2_compare.py`（Test2）GT 未对齐问题**——本次不动，待后续任务。
5. **dataset 尾窗丢弃 vs infer 尾窗填充的不一致**——本次不动。
6. 设计文档偏差（`N_JOINTS=23` vs 文档 24、global_orient 并入 Hips）——非 bug，不改代码；文档更新另行安排。

## 重要提醒（写给实施者与使用者）

`anysole/train.py:349-350` 训练结束的自动 eval **没有传 `--write-bvh`**，只写 metrics。因此"重新训练"本身**不会**刷新 `results/AnySole/<modal>/predictions/eval_bvh/`（当前里面的 BVH 是旧 checkpoint 的产物，时间戳早于 ckpt_last.pt 40 分钟）。重训完成后必须手动执行（或自行在 train.py 自动 eval 里补上 `--write-bvh` 一行）：

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --config anysole/configs/v1.yaml \
  --ckpt results/AnySole/anysolev1_tactile_abs/checkpoints/ckpt_last.pt \
  --split test --write-bvh results/AnySole/anysolev1_tactile_abs/predictions/eval_bvh
```

---

## 验收清单（实施完成后按此验收）

### Task 1（可视化对齐）
运行以下数值对照（环境 touch_gait，任选一个 test 会话，如 S10103）：
- `r_test1_visualize_anysole.py` 的 `load_gt` 输出的 `GT[0]` Hips 关节应与 `anysole` dataset 管线 `kp_gt[0]` 的 Hips 一致（仅差 z-up 显示变换 `(x, -z, y)`：viz_y = -native_z、viz_z = native_y，Hips 高度不变）。
- `load_pred` 输出帧数应等于写入 BVH 帧数（不再少 16 帧）；viz pred[t] 与压力帧 t 同帧。
- 重跑 `python results_display/script/r_test1_visualize_anysole.py --force --session <S>`，确认 GIF/MP4 中踩地相位与压力面板同步。

### Task 2（T_phys 归一化）
- 数值检查：`T_phys` 各维均值应在 0..~100 量级（force 项 ≈6，cop ≈0.5-0.8，不再是 ~6255）。可用 16 窗 batch 打印 `raw['T_phys'].mean(dim=(0,1))` 对比改动前后。
- fresh 模型量级检查：`t_tok` std 应从 ~1051 降到 O(1)；`v_hat` 初始 std 应 ~0.1（不再 ~119）；初始 `L_pose` 应 ~1（不再 ~24601）。可用 epoch-1 训练日志对照。
- `python -m anysole.data.extract_hrnet` 不涉及；train/eval/infer 三链路仍可正常加载数据（跑 `--limit-sessions 1` 的 eval 冒烟即可）。

### Task 3（口径统一）
- `grep -n ":22" anysole/train.py` 不再命中 `_evaluate` 的 MPJPE 行；train 期 val MPJPE 与 eval.py 的 test MPJPE 同一关节口径。

### Task 4（wandb 监控）
- 重训（哪怕 1 个 epoch，`wandb_mode: offline`）后，run 里应出现 `train/global_step`、`train/steps_per_epoch`、`val/tau0_mpjpe/{V,VT,T}` 三个键，且 `train/lr`、`train/grad_norm` 仍在。

### 回归确认
- `python -m anysole.eval --ckpt <旧ckpt>` 仍能正常加载旧 checkpoint（不因 Task 2 报错；注意其输入尺度与重训后不一致属预期，指标无参考意义）。
- 不改动范围外的文件：`bvh_aligner_pose.py`、`prepare_sequences.py`、`r_test2_compare.py`、`types.py` 的 `CONTACT_SUM_THRESH` 保持原样。

---

## 参考证据（排查时实测，供实施者对照）

| 指标 | 实测值 | 来源 |
|---|---|---|
| 主模型 test MPJPE（VT2M） | 331.5 mm | `results/AnySole/anysolev1_tactile_abs/metrics/test.json` |
| 旧 eval BVH（07:55）同会话 MPJPE | 754 mm | 本地重算（ckpt_last.pt 为 08:35） |
| tau=0 干净输入重建 MPJPE | 主模型 246 mm / drift 127 mm | 本地重算 |
| T_phys force 项（原始尺度） | 均值 6255、最大 20362 | 训练 batch 统计 |
| contact_gt 均值（3b，勿动） | 0.9703 | 训练 batch 统计 |
| fresh 模型 t_tok / v_hat std | 1051 / 119 | 本地重算 |
| 归一化 T_phys 后 300 步过拟合 | L_pose 0.086 / L_kp 0.048 / L_traj 0.009 | 对照实验 |
| viz GT[0] 对应 BVH 时间 | 0.4s（trim 引入）vs dataset GT[0] 0.05s | S10103 实测 |
