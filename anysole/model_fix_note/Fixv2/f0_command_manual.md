# F0 系列命令手册（F0a 评估协议 / F0b 去扩散回归 / F0c 去 V 重建）

> **依据**：`fix_plan_v2.md` §2（实验规范）与 §3 F0 节。本文是 F0 三小步的**照抄级命令手册**，
> 每步：训练 / 评估 / 协议 / 探针 / 可视化 + 验收判据。
>
> **代码状态（2026-09-17）**：F0a/F0b 全部代码已落地、尚未运行测试。训练前先跑
> `z_note/smoke_f0b_regress.py` 单测。实现文件清单见文末。
>
> **env = touch_gait，全部在 `/data/fangyuxuan/projects/gait` 下执行**；`--device` 为示例值，
> 执行前用 `nvidia-smi` 确认空闲卡，勿与在跑实验撞卡。
>
> **v1.yaml 全程不动**（其值即 E3 口径：raw108、tw=20、stride=20、λ_pose=3/λ_kp=1/λ_traj=1/
> λ_trec=0.1/λ_vrec=0.1/λ_con=0、noise_scaled=false、lr 恒定、joint_and）。所有实验差异由
> CLI 传入，CLI 覆盖值随 checkpoint 保存（现有机制），eval/infer 从 ckpt 读配置、无需重复传参。
> （train/eval/infer 的 `--config` 默认路径已修复为仓库根 `configs/v1.yaml`，原默认
> `anysole/configs/v1.yaml` 不存在。）

---

## 工作流：叠加 + 单步回退

1. **叠加链**（每条命令都是"到当前步为止的全量 flags"，直接照抄）：

   ```
   F0a（E3 ckpt 过协议，模型不改）→ F0b（--modal anysolev2）→ F0c（再 + --lambda-vrec 0）
   ```

2. **预算按梯度步数对齐**：短预算 = **3700 步** ≈ E3 满训 800 ep @ tw=20/stride=20
   （v1.yaml 已含 `stride: 20`）。墙钟紧张可用 `--stride 10`（密度 ×2，400 ep ≈ 3700 步），
   但**必须在实验记录中标注口径**。
3. **3 种子**：train.py 无 `--seed` 开关，3 个训练种子 = 3 次独立运行，各自独立 out-dir
   （随机初始化/shuffle 天然不同）。评估侧种子用 `--protocol-seed 0/1/2`（见步骤 1）。
4. **判据阈值用种子噪声定**：F0a 用 E3 ckpt 跑 3 个协议种子，测出每项指标的 σ_seed。
   "显著改善" = 比上一步好 2σ_seed 以上；"不劣化" = 差距 ≤1σ_seed；落在 1–2σ 之间补跑
   一个种子。**回归模型无采样随机性，3 种子 σ_seed 只在 diffusion 基线上做。**
5. **回退规则**：某步验收劣于上一步 → 弃用该开关、用上一步命令继续；失败 ckpt 留档不删、
   代码不回改（新开关默认关闭）。
6. **单变量纪律**：每步只动一个变量；F0b 判据不过时**先排查实现**（标准化/损失口径），
   不进 F1。

## 开关一览（CLI）

| 开关 | CLI | 默认 | 状态 |
|---|---|---|---|
| 回归模态（F0b） | `--modal anysolev2` | anysolev1 | **已实现**（2026-09-17，未测试） |
| 评估协议（F0a） | eval 时默认自动跑；`--no-protocol` 关；`--protocol-seed N`；`--no-robustness` 关鲁棒集 | 协议开、鲁棒开、seed 0 | **已实现**（2026-09-17，未测试） |
| 去 V 重建（F0c） | `--lambda-vrec 0` | 0.1（yaml） | 已有 CLI（E6.7） |
| 训练预算 | `--epochs 800`（=3700 步 @stride20） | 800（yaml） | 已有 |
| 窗口/步长 | `--tw` / `--stride` | 20 / 20（yaml） | 已有（E6.3） |

**anysolev2 下的行为提示**：`pose_repr` 强制 6d；`--tau-max/--tau-fixed/warm_start/
continuation` 为 diffusion-era 开关，v2 下被忽略并打印 WARNING（正常）。

---

## 步骤 0：E3 基线复跑（仅当 E3 ckpt 不存在时）

计划指定的 E3 ckpt 是 `results/AnySole/E3_base/checkpoints/ckpt_last.pt`，当前不存在。
不想复跑可先用现成的 `results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt`（E3 满训同配置），
但实验记录中要写明用的是哪个 ckpt、contact_method 以 ckpt config 为准。

```bash
# 训练（v1.yaml 即 E3 值；modal 缺省 anysolev1）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --epochs 800 \
  --out-dir results/AnySole/E3_base/checkpoints \
  --wandb_mode online --wandb_experiment_tag e3_base \
  --wandb_eval_interval 10 \
  --device cuda:4

# 评估（协议默认自动跑）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E3_base/checkpoints/ckpt_last.pt \
  --modal anysolev1 --contact-method joint_and \
  --split val --write-bvh results/AnySole/E3_base/predictions/eval_bvh \
  --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal E3 --contact-method base --force
```

---

## 步骤 1：F0a — E3 基线过协议（模型不改）

产出：全套 F 系列数字 + **3 种子 σ_seed** + 鲁棒性验证集 + ridge 探针。

```bash
# 协议评估 ×3（协议默认自动跑、鲁棒集默认开；diffusion 每次 3 口径 + 2 鲁棒 = 5 次
# DDIM×val，墙钟较长，属预期）。--no-write-bvh 可省 BVH 导出，不影响协议。
for seed in 0 1 2; do
  /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
    --ckpt results/AnySole/E3_base/checkpoints/ckpt_last.pt \
    --modal anysolev1 --contact-method joint_and \
    --split val --protocol-seed $seed \
    --write-bvh results/AnySole/E3_base/predictions/eval_bvh \
    --device cuda:5
done
# 输出：metrics/val.json（经典，不变）+ metrics/val_fseries.json（最新一次）
#       + metrics/val_fseries_seed{0,1,2}.json（分种子归档，σ_seed 从这里算）

# ridge-on-F 探针（E3 参照 ≈ 85mm MPJPE；train 拟合、val 评估）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/E3_base/checkpoints/ckpt_last.pt --split val --device cuda:7
# 输出：metrics/ridge_probe_val.json
```

**验收（F0a）**：E3 ckpt 在新协议下出全套数字——9 部位 MPJPE/PA-MPJPE、W-MPJPE、
RTE_norm、yaw、抖动/加速度/拼接跳变、接触 F1 + 脚滑、V2M 压力三项、T2M 上半身两项、
robust_vdrop/robust_tdrop 两行；每项指标有 3 种子 σ_seed；ridge 探针 ≈ 85mm 量级
（历史参照 81.2/85 见 fix_plan_v2.md §0.4）。协议数字明显离谱（如 W-MPJPE 异常大）
→ 先查协议实现，不进 F0b。

---

## 步骤 2：F0b — 去扩散回归（+ `--modal anysolev2`）

```bash
# 单测（训练前必跑：前向形状 / loss 回传 / regress 关=旧行为 / E3 ckpt 严格加载守卫）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/smoke_f0b_regress.py --device cuda:7

# 训练 ×3 种子（独立 out-dir；3700 步 ≈ 800 ep @ tw=20/stride=20）
for s in 1 2 3; do
  /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
    --modal anysolev2 \
    --contact-method joint_and \
    --epochs 800 \
    --out-dir results/AnySole/F0b_regress_s$s/checkpoints \
    --wandb_mode online --wandb_experiment_tag f0b_regress \
    --wandb_eval_interval 10 \
    --device cuda:4
done

# 评估（协议默认自动跑；回归无采样随机性，--protocol-seed 0 一次即可，无需 3 种子）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F0b_regress_s1/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/F0b_regress_s1/predictions/eval_bvh \
  --device cuda:7

# ridge-on-F 探针（每步必跑；若回归头健康，协议 MPJPE 应接近此天花板）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/F0b_regress_s1/checkpoints/ckpt_last.pt --split val --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal F0b --contact-method regress_s1 --force
```

**验收（F0b，种子 1/2/3 各跑一套）**：
- 通过：VT2M PA-MPJPE 与 E3（81.2mm）差距 ≤2σ_seed，与 ridge-on-F（85mm）同量级；
  抖动明显低于 E5 链的 27mm/帧。
- **若明显差于 E3 → 实现问题**（标准化/损失口径），先排查，不进 F1；排查时对照训练面板
  tau0（回归下 tau0 = 直接输出 MPJPE）与 loss 各分量量级。
- 3 种子间差异 >2σ_seed → 训练不稳定，查 lr/梯度（wandb grad_norm）。

---

## 步骤 3：F0c — 去 V 重建（再 + `--lambda-vrec 0`，F0b 通过后）

```bash
# 训练（V 重建头与 L_Vrec 自然停更；3700 步）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 \
  --contact-method joint_and \
  --lambda-vrec 0 \
  --epochs 800 \
  --out-dir results/AnySole/F0c_novrec/checkpoints \
  --wandb_mode online --wandb_experiment_tag f0c_novrec \
  --wandb_eval_interval 10 \
  --device cuda:4

# 评估（同 F0b；wandb 面板 loss_Vrec_Vmissing 应不再下降）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F0c_novrec/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/F0c_novrec/predictions/eval_bvh \
  --device cuda:7

# ridge 探针
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/F0c_novrec/checkpoints/ckpt_last.pt --split val --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal F0c --contact-method novrec --force
```

**验收（F0c）**：T2M 不劣化 → 永久删除 V 重建路径（model_v2.py / aux_heads.py /
losses.py:138-143，联系 AI 会话实施）。T2M 明显变差 → 说明 V 重建确实在向 F 灌视觉信息，
F1 起保留替代目标（重建 θ_hmr 而非 HRNet 特征），留到 F5。

---

## 单会话推理（任一步 checkpoint，导出可交互 BVH）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.infer \
  --ckpt results/AnySole/F0b_regress_s1/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --session S7013 --config-id VT2M \
  --device cuda:4
```

---

## 验收口径（每步必须同表）

| 类别 | 指标 | 出处 |
|---|---|---|
| 局部姿态 | MPJPE / PA-MPJPE：9 部位（root/torso/headneck/l_arm/r_arm/l_leg/r_leg/l_foot/r_foot）+ upper(1–14)/lower(15–22)/踝脚(17,18,21,22)/手(10,14) | `metrics/<split>_fseries.json` |
| 全局 | W-MPJPE（4s 段首帧对齐）、RTE_norm（按位移长度归一化）、yaw_abs_deg / yaw_drift_deg | 同上（forward_axis 由数据自选，写进 JSON） |
| 时序 | jitter_mm（probe_jitter 口径 mm/帧）、accel_err_ms2、seam_jump_mm（附 GT 同缝参照） | 同上 |
| 接触 | contact_f1、foot_slide_mm / foot_slide_gt_mm（mm/帧） | 同上 |
| 压力输出（V2M） | pressure_force_r2、pressure_cop_err、pressure_pearson | 同上 |
| T2M 上半身 | accel_dist_err_upper_ms2、joint_limit_viol_elbow / knee（<15° 违规率） | 同上 |
| 鲁棒集 | robust_vdrop / robust_tdrop（VT2M，连续 20–40% 帧置零） | 同上；F0 起报告、F7 起主判据 |
| 经典 | VT2M/V2M/T2M MPJPE、PA-MPJPE、MPJRE、traj_ATE、contact_acc、T_mae/T_rmse/T_corr | `metrics/<split>.json`（口径不变） |
| 读出天花板 | ridge-on-F MPJPE / PA-MPJPE / 分部位 / L_pose | `metrics/ridge_probe_<split>.json`，**每步必跑** |
| 训练面板 | tau0、loss 各分量量级、grad_norm、steps_per_epoch | wandb（`--loss-cap 1.0` 可压 y 轴） |
| BVH 运动学 | 逐关节误差、脚踝相对高度（踢腿判据）、帧间抖动（GT 参照 8.7mm/帧） | 导出 BVH；fk 读取复用 `z_note/smoke_e6x_decoupling.py` 的方式 |

**每步回填**：把上表数字回填到 `fix_plan_v2.md` 对应小节末尾（未回填 = 未完成）。

---

## 实现文件清单（已落地 2026-09-17，未运行测试）

| 文件 | 改动 |
|---|---|
| `anysole/models/pose_head.py` | `head_mode="diffusion"\|"regress"`；diffusion 布局逐字节保留（旧 ckpt strict 加载）；regress = 可学习 query (1,tw,23,dim) + 同 6 层 decoder，forward 只收 F（位置参数守卫） |
| `anysole/models/model_v2.py`（新） | `AnySoleModelV2`（modal `anysolev2`）：V1 结构原样，仅 pose_head 用 regress，forward 去 x_tau/tau |
| `anysole/models/model.py` / `__init__.py` | `anysolev2` 注册进 MODEL_NAMES；AnySoleModel 构造 v2 时报错指向 V2 |
| `anysole/train.py` | regress 分支跳过 τ/q_sample；_evaluate 同分支；v2 强制 pose_repr=6d；diffusion-era 开关打警告；自动 eval 加 `--no-protocol` |
| `anysole/eval.py` | `_load_model` 建 V2；6D 分支 regress 单次前向（continuation 不适用、crossfade 保留）；`--no-protocol/--protocol-seed/--no-robustness` |
| `anysole/eval_protocol.py`（新） | F0a 协议全部指标 + 鲁棒集；写 `metrics/<split>_fseries.json` + `_seed<N>.json`；continuation 重叠窗自动跳过 |
| `anysole/infer.py` | regress 分支；`--config` 默认路径修复 |
| `results_display/script/ridge_probe.py`（新） | 正式化 ridge-on-F（512 维/帧口径，v1/v2 通用） |
| `z_note/smoke_f0b_regress.py`（新） | F0b 单测（形状/loss 回传/regress 关=旧行为/E3 ckpt 守卫） |
| `anysole/diffusion.py` | 未动（旧 ckpt 评估兼容） |

## 风险与诚实声明

1. **协议对 diffusion 模型贵**：一次 eval 附带的协议 = 3 口径 + 2 鲁棒 = 5 次 DDIM×val。
   `--no-robustness` 可省 2 次；`--no-protocol` 可完全关掉（训练自动 eval 已默认关）。
   回归模型协议很便宜（单次前向）。
2. **σ_seed 口径**：只对 E3 diffusion 基线有意义（采样随机性）；F0b 回归模型的 3 种子
   是 3 次独立训练，评估侧只跑一次即可。
3. **train.py 无 `--seed`**：3 个训练种子靠 3 次独立运行 + 独立 out-dir 保证；若日后要
   精确复现某个失败 run，需给 train.py 补 `--seed`（当前不在计划内）。
4. **F0b 失败先查实现**：regress 与 diffusion 共用 pose_mean/pose_std 反标准化与损失
   口径（losses.py 未动），明显差于 E3 时优先排查这两处而非改架构。
5. **目录命名约定**：可视化按 `<modal>_<contact-method>` 找目录（
   `visualize_anysole.py --modal F0b --contact-method regress_s1` → `results/AnySole/F0b_regress_s1`）；
   命令里 `--device` 为示例值，执行前 `nvidia-smi` 确认。
