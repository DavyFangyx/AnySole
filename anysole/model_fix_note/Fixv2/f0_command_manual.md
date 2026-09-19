# F0 系列命令手册 · 快速通道版（A 单测 / B F0a′ 协议 / C F0b′ 回归 warm-start）

> **依据**：`fix_plan_fast.md`（2026-09-18 快速通道，取代本手册旧版步骤 0/1/2/3）。
> 本文是 F0 的**照抄级命令手册**：每步 训练 / 评估 / 协议 / 探针 / 可视化 + 验收判据。
> `fix_plan_v2.md` 仍为 F1–F9 方案内容、单变量纪律、协议指标表的正式源。
>
> **代码状态（2026-09-18）**：F0a/F0b 代码已落地；`--init-from/--init-drop` warm-start
> 已实现并实测（E3 ckpt → AnySoleModelV2 复制 254/255 key，query 梯度正常）。
> 训练前先跑步骤 A 单测。
>
> **env = touch_gait，全部在 `/data/fangyuxuan/projects/gait` 下执行**；GPU 分配（2026-09-18
> 实测 cuda:4 轻载、cuda:6/cuda:7 空闲）：
> A 单测与 B 协议串行用 **cuda:6**，C 训练用 **cuda:4**（与 B 并行），D 验收用 **cuda:7**。
> 执行前用 `nvidia-smi` 复核。
>
> **v1.yaml 全程不动**（其值即 E3 口径：raw108、tw=20、stride=20、λ_pose=3/λ_kp=1/λ_traj=1/
> λ_trec=0.1/λ_vrec=0.1/λ_con=0、noise_scaled=false、lr 恒定、joint_and）。所有实验差异由
> CLI 传入，CLI 覆盖值随 checkpoint 保存（现有机制），eval/infer 从 ckpt 读配置、无需重复传参。

---

## 工作流：叠加 + 单步回退 + 提前终止

1. **叠加链**（每条命令都是"到当前步为止的全量 flags"，直接照抄）：

   ```
   A 单测 → B F0a′（E3 ckpt 过协议一次，模型不改）
   C F0b′（--modal anysolev2 + --init-from E3 ckpt）→ E F0c′（再 + --lambda-vrec 0）
   ```

2. **1 种子 + 提前终止**（替换旧版"3700 步 ×3 种子"）：每候选 1 种子、400ep 短预算，
   训练中观察 wandb 曲线按观察规则 kill 或续跑（见步骤 C）。幸存者补 1 个确认种子。
3. **判据用固定阈值**（σ_seed 已废弃——n=3 无统计效力且对回归模型无含义）：
   - VT2M PA-MPJPE 改善 **≥5mm** 或抖动/脚滑下降 **≥20%** → 值得追；
   - 改善 **<2mm** → 噪声，弃；介于两者之间 → 幸存者补 1 个确认种子。
   - F0b′ 对比基准固定：E3 VT2M PA-MPJPE **81.2**、ridge-on-F **85**、E5 链抖动 27mm/帧。
4. **warm-start 默认开**：结构允许时 `--init-from <上一步 ckpt>`；数值与 from-scratch 可能有
   小幅偏差（编码器已收敛），对筛选无影响，**最终链条收尾前 from-scratch 全预算补跑 1 次**。
5. **回退规则**：某步验收劣于上一步 → 弃用该开关、用上一步命令继续；失败 ckpt 留档不删、
   代码不回改（新开关默认关闭）。
6. **单变量纪律**：每步只动一个变量；C 判据不过时**先排查实现**（标准化/损失口径），不进 F1。

## 开关一览（CLI）

| 开关 | CLI | 默认 | 状态 |
|---|---|---|---|
| 回归模态（F0b′） | `--modal anysolev2` | anysolev1 | **已实现**（2026-09-17，未训练） |
| warm-start（F0b′） | `--init-from <ckpt>` / `--init-drop <前缀,...>` | 关 | **已实现并实测**（2026-09-18） |
| 评估协议（F0a′） | eval 时默认自动跑；`--no-protocol` 关；`--protocol-seed N`；`--no-robustness` 关鲁棒集 | 协议开、鲁棒开、seed 0 | **已实现**（2026-09-17，未运行） |
| 去 V 重建（F0c′） | `--lambda-vrec 0` | 0.1（yaml） | 已有 CLI（E6.7） |
| 训练预算 | `--epochs 400`（快速版短预算） | 800（yaml） | 已有 |
| 梯度裁剪 | `--grad-clip <max-norm>`（2026-09-18 新增，爆炸保险；建议 5.0） | 关 | **已实现**（2026-09-18，未训练验证） |
| 触觉编码（F4a） | `--t-encoder foot_conv`（数据不动，只换编码器） | linear | **已实现并单测**（2026-09-18，未训练） |
| 视觉输入（F1） | `--v-input hmr_gvhmr`（需先跑 extract_hmr） | hrnet | **已实现并单测**（2026-09-18，未训练；依赖权重下载） |
| 窗口/步长 | `--tw` / `--stride` | 20 / 20（yaml） | 已有（E6.3） |

**anysolev2 下的行为提示**：`pose_repr` 强制 6d；`--tau-max/--tau-fixed/warm_start/
continuation` 为 diffusion-era 开关，v2 下被忽略并打印 WARNING（正常）。

---

## 步骤 A：单测（GPU 6，约 5 分钟，训练前必跑）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/smoke_f0b_regress.py --device cuda:6
```

**验收（A）**：5 项检查全过——① AnySoleModelV2 前向形状（x0_hat (B,tw,138)、F (B,40,256) 等）；
② E3 权重下 compute_losses 有限且 query 收到梯度；③ regress 模式下 state dict 无
proj_*/timestep_token；④ diffusion forward 仍出 (B,tw,138) 且 regress 拒绝位置参数误调；
⑤ E3-era ckpt 对 anysolev1 strict 加载通过（V1 布局未被 F0b 改动破坏）。

warm-start 通路已另行实测（2026-09-18）：E3 ckpt → AnySoleModelV2 复制 254/255 个 key，
仅 V1 diffusion 投影被跳过，前向/反向正常。

---

## 步骤 B：F0a′ — E3 ckpt 过协议一次（GPU 6，与步骤 C 并行）

模型不改。产出：E3 val 全套 F 系列数字 + ridge 天花板（参照 ~85mm）。
**这是唯一一次"E3 过协议"**：1 个协议种子、不写 BVH、不开鲁棒集（vdrop/tdrop 到 F5 起
才作主判据）。

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt \
  --modal anysolev1 --contact-method joint_and \
  --split val --protocol-seed 0 --no-robustness \
  --device cuda:6

/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt --split val --device cuda:6
```

**验收（B）**：`metrics/val_fseries.json` + `metrics/ridge_probe_val.json` 生成；
数字与历史参照同量级（PA-MPJPE 81.2、ridge ~85，见 fix_plan_v2.md §0.4）。
协议数字明显离谱（如 W-MPJPE 异常大）→ 先查协议实现；C 可以继续并行跑，不影响。

> 说明：E3_nocontact 只有 test.json（训练末尾自动 eval 写的是 test split），val 全套数字
> 由本步骤第一次补齐。E3_base ckpt 不存在、**不复跑**——E3_nocontact 的 config 已逐项核对
> 与 v1.yaml E3 口径一致（joint_and、λ_con=0、tw=20、800ep）。

---

## 步骤 C：F0b′ — 回归 warm-start，1 种子（GPU 4，主线）

从 E3 ckpt 热启动：encoders/fusion/traj/aux 逐参数相同直接复用，pose_head 的 diffusion
投影自动跳过（query 新初始化）。1 种子、400ep（≈1850 步 @stride20）。

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 \
  --contact-method joint_and \
  --epochs 400 \
  --init-from results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F0b_warm/checkpoints \
  --wandb_mode online --wandb_experiment_tag f0b_warm \
  --wandb_eval_interval 10 --loss-cap 1.0 \
  --device cuda:4
```

**观察规则**（wandb `val/tau0_mpjpe/{VT,V,T}`——regress 模式下 tau0 就是直接输出 MPJPE，
每 10ep 一个点）：

| 时点 | 状态 | 动作 |
|---|---|---|
| ep100 | >110mm 且最近两档不降 | **kill**：查实现（标准化/损失口径，见风险 4） |
| ep200 | ≤95mm 且仍下行 | 继续到 400ep |
| ep400 | 未达标但 ≤95 且仍下行 | 续跑到 800ep（唯一允许的续跑情形） |

**验收（C 训练面板）**：tau0(VT) 曲线按上表；loss 各分量量级稳定（无 L_con 爆炸——
λ_con=0 本无此项）；grad_norm 正常。3 种子对比 → **1 种子即可**：回归无采样随机性，
"多种子"在判据框架里没有统计含义。

---

## 步骤 D：F0b′ 验收（GPU 7）

```bash
# 评估（协议默认自动跑 1 次：--protocol-seed 0；回归单次前向，协议很便宜，不开 --no-robustness 也可）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F0b_seed2/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/F0b_seed2/predictions/eval_bvh \
  --protocol-seed 2 --no-robustness \
  --device cuda:7

# ridge-on-F 探针（每步必跑；若回归头健康，协议 PA-MPJPE 应接近此天花板）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/F0b_seed2/checkpoints/ckpt_last.pt --split val --device cuda:7

# 可视化（目录约定 <modal>_<contact-method>：F0b_seed2 = modal F0b + contact-method seed2）
CUDA_VISIBLE_DEVICES=6 python results_display/script/visualize_anysole.py \
  --modal F0b --contact-method seed2 --force
```

**验收（D，固定阈值，不再用 2σ_seed）**：

- 通过：VT2M PA-MPJPE **≤86mm**（E3 81.2 + 5mm 容差）且抖动明显低于 E5 链 27mm/帧；
  T2M 相对 E3 参照 154.0 不劣化 >5mm。
- **塌缩门（F0b_warm 教训：塌缩模型能骗过 PA-MPJPE 与抖动）**：ridge L_pose ≤ 2× E3 基线
  （≈0.019）**且** MPJPE − PA-MPJPE ≤ 80mm（健康参照：E3 60、F0c 42；F0b_warm 塌缩 = 110）。
  任一不满足 → 按塌缩处理，kill。
- 明显差于 E3（>92mm）→ 实现问题（标准化/损失口径），先排查，不进 F1；排查时对照训练
  面板 tau0 与 loss 各分量量级。
- 怀疑 warm-start 是失败原因 → **from-scratch 1 种子重跑**（本版唯一允许的重跑）。

---

## 步骤 E：F0c′ — 去 V 重建（`--lambda-vrec 0`，移出主线）

不挡任何后续方案判定；推迟到 F4 前（T2M 判据真正用得上时），或后台顺手验证：

```bash
# 训练（warm-start 自 F0b′；V 重建头与 L_Vrec 自然停更；200ep 短预算）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 \
  --contact-method joint_and \
  --lambda-vrec 0 \
  --epochs 200 \
  --init-from results/AnySole/F0b_warm/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F0c_novrec/checkpoints \
  --wandb_mode online --wandb_experiment_tag f0c_novrec \
  --wandb_eval_interval 10 \
  --device cuda:6

# 评估（同 D；wandb 面板 loss_Vrec_Vmissing 应不再下降）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F0c_ext400/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/F0c_ext400/predictions/eval_bvh \
  --protocol-seed 0 --no-robustness \
  --device cuda:7

# ridge 探针
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/F0c_ext400/checkpoints/ckpt_last.pt --split val --device cuda:7

# 可视化
CUDA_VISIBLE_DEVICES=6 python results_display/script/visualize_anysole.py \
  --modal F0c --contact-method ext400 --force
```

**验收（E）**：T2M 不劣化 → 永久删除 V 重建路径（model_v2.py / aux_heads.py /
losses.py:138-143，联系 AI 会话实施）。T2M 明显变差 → 说明 V 重建确实在向 F 灌视觉信息，
F1 起保留替代目标（重建 θ_hmr 而非 HRNet 特征），留到 F5。

> **epoch 计数口径（重要）**：train.py 无续训开关，每次运行都从自己的 epoch 0 开始数；
> `--init-from` 加载的是上一段训练的**权重**。所以"F0c 续跑到 400ep 总量"的命令写
> `--epochs 200`（= 在 F0c 已训 200ep 的权重上再训 200ep），wandb 曲线显示 1–200，
> 但权重走过的总步数 = 1000 + 1000 = 2000 步 ≈ 400ep@stride20。判断爆炸/收敛看
> **总步数**（~1800–2000 步 = 本次运行自己的 epoch ~160–200）。
> **二轮实验实测口径（2026-09-18）**：F0c_ext400 实际以 `--lambda-vrec 0.1` 跑
> （wandb run 5knmn5n1 config 为证，init-from F0c_novrec 权重）——即"在 λ_vrec=0 训了
> 200ep 的权重上重开 V-recon 再训 200ep"，结果健康（VT 64.5）。λ_vrec=0 的 400ep 版本
> 未跑、且已无必要（见 fix_plan_fast.md §6 二轮归因）。

---

## 单会话推理（任一步 checkpoint，导出可交互 BVH）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.infer \
  --ckpt results/AnySole/F0b_warm/checkpoints/ckpt_last.pt \
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
| 鲁棒集 | robust_vdrop / robust_tdrop（VT2M，连续 20–40% 帧置零） | **F5 起才开**（此前 `--no-robustness`） |
| 经典 | VT2M/V2M/T2M MPJPE、PA-MPJPE、MPJRE、traj_ATE、contact_acc、T_mae/T_rmse/T_corr | `metrics/<split>.json`（口径不变） |
| 读出天花板 | ridge-on-F MPJPE / PA-MPJPE / 分部位 / L_pose | `metrics/ridge_probe_<split>.json`，**每步必跑** |
| 训练面板 | tau0、loss 各分量量级、grad_norm、steps_per_epoch | wandb（`--loss-cap 1.0` 压 y 轴） |
| BVH 运动学 | 逐关节误差、脚踝相对高度（踢腿判据）、帧间抖动（GT 参照 8.7mm/帧） | 导出 BVH；fk 读取复用 `z_note/smoke_e6x_decoupling.py` 的方式 |

**每步回填**：把上表数字回填到 `fix_plan_fast.md` 对应小节末尾（未回填 = 未完成）。

---

## 实现文件清单（已落地）

| 文件 | 改动 |
|---|---|
| `anysole/models/pose_head.py` | `head_mode="diffusion"\|"regress"`；diffusion 布局逐字节保留（旧 ckpt strict 加载）；regress = 可学习 query (1,tw,23,dim) + 同 6 层 decoder，forward 只收 F（位置参数守卫） |
| `anysole/models/model_v2.py`（新） | `AnySoleModelV2`（modal `anysolev2`）：V1 结构原样，仅 pose_head 用 regress，forward 去 x_tau/tau |
| `anysole/models/model.py` / `__init__.py` | `anysolev2` 注册进 MODEL_NAMES；AnySoleModel 构造 v2 时报错指向 V2 |
| `anysole/train.py` | regress 分支跳过 τ/q_sample；_evaluate 同分支；v2 强制 pose_repr=6d；diffusion-era 开关打警告；自动 eval 加 `--no-protocol`；**2026-09-18 追加 `--init-from/--init-drop` + `init_from_checkpoint()`（warm-start，已实测）** |
| `anysole/eval.py` | `_load_model` 建 V2；6D 分支 regress 单次前向（continuation 不适用、crossfade 保留）；`--no-protocol/--protocol-seed/--no-robustness` |
| `anysole/eval_protocol.py`（新） | F0a 协议全部指标 + 鲁棒集；写 `metrics/<split>_fseries.json`；continuation 重叠窗自动跳过 |
| `anysole/infer.py` | regress 分支；`--config` 默认路径修复 |
| `results_display/script/ridge_probe.py`（新） | 正式化 ridge-on-F（512 维/帧口径，v1/v2 通用） |
| `z_note/smoke_f0b_regress.py`（新） | F0b 单测（形状/loss 回传/regress 关=旧行为/E3 ckpt 守卫） |
| `anysole/diffusion.py` | 未动（旧 ckpt 评估兼容） |

## 风险与诚实声明

1. **warm-start 口径**：F0b′/F0c′ 数值与 from-scratch 可能有小幅偏差（编码器已收敛）。
   对筛选无影响；正式基准在最终链条收尾时 from-scratch 全预算补跑 1 次。
2. **1 种子**：会漏掉恰好落在种子噪声里的真效果——但 <2mm 的效果本来就不追（固定阈值），
   ≥5mm 的效果 1 个种子足够看见。幸存者进主线前补 1 个确认种子。
3. **400ep 短预算**：观察规则要求 tau0 仍在下行才续跑、曲线平台/上行才 kill，
   不会造成"预算不足却当失败"的误判。
4. **协议对 diffusion 模型仍贵**：步骤 B 一次协议 = 3 口径 DDIM×val（`--no-robustness`
   已省 2 次），墙钟较长属预期；回归模型协议是单次前向、很便宜。
5. **F0b′ 失败先查实现**：regress 与 diffusion 共用 pose_mean/pose_std 反标准化与损失
   口径（losses.py 未动），明显差于 E3 时优先排查这两处而非改架构。
6. **train.py 无 `--seed`**：1 种子 = 1 次独立运行；若日后要精确复现某个失败 run，
   需给 train.py 补 `--seed`（当前不在计划内）。
7. **目录命名约定**：可视化按 `<modal>_<contact-method>` 找目录——
   `--modal F0b --contact-method warm` → `results/AnySole/F0b_warm`；
   真实接触口径仍以 ckpt config 为准（joint_and），visualize 的 contact-method 只是路径拼写。

---

# F4a / F1 双方向验证（2026-09-18 实现，与 F0 同层级：输入表征改动）

> 依据 fix_plan_fast.md §3 跑序：F0b_seed2 为基线（VT 63.7 训练面板，协议验收待跑），
> **F4a（触觉编码）与 F1（视觉 HMR）并行筛选**，各 1 种子 400ep warm-start。
> 两个方向互不依赖：F4a 只动 T 编码器（`--t-encoder foot_conv`），F1 只动 V 编码器
> （`--v-input hmr_gvhmr`）——可各占一张卡同时训练。

## 实现口径（与 fix_plan_v2.md §F4a/F1 的偏差，如实记录）

- **F4a 数据侧零改动**（用户裁定 2026-09-18）：实测鞋垫 CSV 原生率 = 40Hz（无更高
  采样源），计划第 1-2 项（重采样 r·40Hz、每会话归一化尺度）取消；继续使用现有
  T_raw(96)+T_phys(12) 108 维输入。实现 = 计划第 3-6 项：每脚 4×12 网格共享卷积
  （右脚镜像）、每脚手工特征（T_phys 6 维 + 编码器内算的接触面积比、dF/dt；质量
  特征 q_T 需 per-session 统计，暂缓）、[左/右/全局] 三 token + 4 层时间
  Transformer → 合并 1 token 进融合。T 重建目标维持现有 96 维不变（计划第 7 项 =
  现有 L_Trec）。
- **F1 = GVHMR 离线提取 + 零训练基线 + VEncHMR**：权重需下载（extract_hmr.py 头部
  列了 4 个必需文件，Google Drive 链接见 GVHMR docs/INSTALL.md）；GVHMR 仓库需
  `pip install -e .`。零训练基线用提取时保存的 joints_hmr（22 SMPL-X 关节，
  fk_v2 免 LBS）对 kp_gt 做 15 对语义匹配（Hips↔pelvis、UpLeg↔hip、Leg↔knee、
  Foot↔ankle、Shoulder↔shoulder、Arm↔elbow、ForeArm↔wrist、Head↔head、Neck↔neck）。
- **F4b 备注（用户裁定）**：流内深监督的接触 logit（2 维）已有对应接触量
  （--contact-method），F4b 实施时默认用 **joint_and** 标签（基线 ckpt 已带）。

## F4a：触觉流按脚编码（`--t-encoder foot_conv`，GPU 4，tag f4a_footconv）

```bash
# 单测（训练前必跑）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/smoke_f4a_grid.py --device cuda:6

# 训练（warm-start 自 F0b_seed2；新 t_enc 结构按名字自动跳过、保持新初始化）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F0b_seed2/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F4a_footconv/checkpoints \
  --wandb_mode online --wandb_experiment_tag f4a_footconv \
  --wandb_eval_interval 10 --loss-cap 1.0 \
  --device cuda:4

# 验收（同 D 步骤口径）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/F4a_footconv/predictions/eval_bvh \
  --protocol-seed 0 --no-robustness --device cuda:7

/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt --split val --device cuda:7

CUDA_VISIBLE_DEVICES=6 python results_display/script/visualize_anysole.py \
  --modal F4a --contact-method footconv --force
```

**验收（F4a，对照 = F0b_seed2 基线，固定阈值）**：T2M 下肢 PA-MPJPE、根 RTE、
接触 F1 **改善 ≥5mm / F1 ≥0.05** 且 VT2M PA-MPJPE 不劣化（≤+5mm）；塌缩门同 D
（ridge L_pose ≤0.019 且 MPJPE−PA ≤80mm）。任一失败 → 弃开关回基线，试 F1。

## F1：视觉输入换 GVHMR（`--v-input hmr_gvhmr`）—— ⚠️ 已取消（2026-09-18，代码留档）

> 取消裁定：F0 回归在 V 输入零改动下 VT2M PA 78.8→25.1（F0b_seed2），旧 81.2 平台是
> 扩散链+head 读出问题、不是视觉信息天花板；V2M−VT2M 仅 1.8mm 说明视觉已独木支撑、
> 剩余缺口在触觉侧（T2M−VT2M = 12.3mm）。GVHMR 权重不下载、以下命令**不执行**；
> F1 的结构性价值（F5 部位化需要 per-part 视觉特征）留到 F5 再议。

```bash
# 0) 下载 GVHMR 权重（4 个文件，见 anysole/data/extract_hmr.py 头部清单）+
#    cd Baselines/Video2Motion/GVHMR && pip install -e . （touch_gait env）
# 1) 离线提取（先单条验证，再训练集批量；GPU 6）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.data.extract_hmr \
  --cam-id 3 --session S7011
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.data.extract_hmr \
  --cam-id 3 --skip-existing
# 2) 零训练基线探针（免费证据：逐帧 PA-MPJPE，15 对匹配关节）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probe_f1_zero_hmr.py \
  --split val --device cuda:6
# 3) 单测
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/smoke_f1_hmr.py --device cuda:6

# 4) 训练（warm-start 自 F0b_seed2；新 v_enc 结构按名字自动跳过）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --v-input hmr_gvhmr \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F0b_seed2/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F1_hmr/checkpoints \
  --wandb_mode online --wandb_experiment_tag f1_hmr_gvhmr \
  --wandb_eval_interval 10 --loss-cap 1.0 \
  --device cuda:4

# 5) 验收（同 F4a 换路径；可视化 --modal F1 --contact-method hmr → F1_hmr）
```

**验收（F1，对照 = F0b_seed2 基线）**：V2M PA-MPJPE **改善 ≥5mm** 且不高于零训练
HMR 逐帧值（探针数字）→ 视觉读出升级成立；与基线持平（<2mm）→ 视觉读出非第一
瓶颈，弃方向、重心转 F3/F5；塌缩门同 D。注意：L_Vrec 目标仍是 HRNet 特征
（λ_vrec 现状不变，替代重建目标留 F5 再议——二轮归因已证 λ_vrec 非关键）。

---

# F2：运动表示重构 + FK 位置损失（2026-09-18 实现）

> 依据 fix_plan_v2.md §F2。F2a 表示重构（`--f2-repr`），F2b = λ_kp 扫描（CLI 已有，无新代码）。
> 跑序：F4a 与 F2a 可并行（各自对照 F0b_seed2 基线，1 种子 400ep warm-start），两者完成后进 F5。

## 实现口径（与计划 §F2a 的偏差，如实记录）

- 关节保持 23 不变；根拆分、tilt/yaw、4 维轨迹按计划实现（geometry.py：
  `heading_from_root_np` / `f2_to_world[_np]`，torch/numpy 双版本）。
- **前向轴单测 = +Z**（有运动信息的训练集 session，速度相关性 +Z 胜 5:1；
  smoke_f2_roundtrip 检查 1 固化此结论）。
- **首帧零而非"复制下一帧"**：计划原文"序列首帧复制下一帧值"，实现改为 ψ̇[0]=0、
  v[0]=0——复制约定会让第 0 帧积分位置偏移一帧位移（≈9mm@40Hz），零约定与
  "窗口锚点 = 前一帧世界状态"精确自洽，**往返实测 0.0001mm**（计划标准 <1mm）。
- **traj 4 维 per-dim 归一化**：训练集拟合 mean/std（`fit_traj_stats`）存入 ckpt
  config `traj_f2_stats`，损失在归一化空间算速度 MSE + 多尺度位移（ψ̇/v_h/h 尺度平衡）。
- 轨迹头 3→4 维（warm-start 时 proj 按形状自动跳过）；pose head 138 维不变（根 6 = tilt）。

## F2a 命令（GPU 4，tag f2_f2rep）

```bash
# 单测（训练前必跑：往返 + 前向轴 + 集成）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/smoke_f2_roundtrip.py --device cuda:6

# 训练（warm-start 自 F0b_seed2；traj_head.proj 3→4 维自动跳过）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F0b_seed2/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F2_f2rep/checkpoints \
  --wandb_mode online --wandb_experiment_tag f2_f2rep \
  --wandb_eval_interval 10 --loss-cap 1.0 \
  --device cuda:5

# 验收（eval/协议/导出自动处理 f2→world 恢复）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F2p4_combo/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/F2p4_combo/predictions/eval_bvh \
  --protocol-seed 0 --no-robustness --device cuda:7

/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/F2_f2rep/checkpoints/ckpt_last.pt --split val --device cuda:7

CUDA_VISIBLE_DEVICES=6 python results_display/script/visualize_anysole.py \
  --modal F2p4 --contact-method combo --force
```

**验收（F2a，对照 = F0b_seed2，计划 §F2a 判据）**：根 RTE、yaw 漂移、W-MPJPE
**显著改善且 MPJPE 不劣化**（≤+5mm）；塌缩门同 D。失败 → 弃开关回基线，不挡 F5。

## F2b：FK 位置损失权重（无新代码，λ_kp 扫描）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr --lambda-kp 3 \
  --epochs 200 --grad-clip 5.0 \
  --init-from results/AnySole/F0b_seed2/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F2b_kp3/checkpoints \
  --wandb_mode online --wandb_experiment_tag f2b_kp3 \
  --wandb_eval_interval 10 --device cuda:5
```

**验收（F2b）**：踝脚/腕 PA-MPJPE 相对 F0b_seed2 显著下降且整体 PA 不劣化；
MPJPE 反升 → 降 λ_kp 重跑（L_pose 必须保留：叶子关节只能靠旋转损失监督）。

```bash
# 验收（eval/协议/导出自动处理 f2→world 恢复）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F2b_kp3/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/F2b_kp3/predictions/eval_bvh \
  --protocol-seed 0 --no-robustness --device cuda:7

CUDA_VISIBLE_DEVICES=6 python results_display/script/visualize_anysole.py \
  --modal F2b --contact-method kp3 --force
```

---

# F5：部位查询解码器 + 门控（2026-09-18 实现）

> 依据 fix_plan_v2.md §F5 + 用户裁定（2026-09-18）：**F4a 的 foot_conv 保留**——触觉
> 在小数据集上是"增强器"而非"载体"（VT −8.2mm 证明增强有效；T-only +7.6mm 由 F5
> 门控正面处理）。F5 的组合 = foot_conv 触觉流 + 部位门控解码器。

## 实现口径（与计划 §F5 的偏差，如实记录）

- **V 流部位化推迟**：F1（HMR）已取消，V 流仍是 HRNet 平铺特征（每帧 1 token）；
  part 解码器的 V 交叉注意力读局部时间窗（t−4..t+4），部位化留待 F1 重启或跳过。
- **aux 头（T_rec/V_rec）删除**：F5 结构变更的一部分（计划 §F5 输出本就替换
  aux 调用点）；L_Trec/L_Vrec 归零，eval/protocol 相应行跳过。
- **接触 logit 头**：双脚 token 各出 2 维 logit（本步不监督，F6 起用 joint_and
  标签 BCE 监督）；L_con 仍走 FK 软接触路径（λ_con=0 不变）。
- **augment.py 段丢弃未实现**（推迟）：模态缺失训练信号目前来自 config 采样
  （VT/V/T），门控掩码已支持逐帧 mask（接口 ready，段丢弃后补）。
- 门控可解释性探针 = wandb `val/gate_{V,T,E}_part{0-8}/{config}`（每 10ep）。

## F5 两条基线命令（各自独立，先 nogate 后 gated）

> 单测（训练前必跑）：`python z_note/smoke_f5_part.py --device cuda:6`

### F5-A：基于 F4a（v1 表示 + foot_conv + part9）

warm-start 自 `F4a_footconv`（foot_conv 在 v1 表示下训了 400ep，全复用）；
表示保持 v1（**不加 --f2-repr**）。对照 = F4a_footconv。

```bash
# F5-A nogate（结构对照）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv --decoder part9 --gate-mode nogate \
  --batch-size 64 \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F5A_f4a_nogate/checkpoints \
  --wandb_mode online --wandb_experiment_tag f5a_f4a_nogate \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4

# F5-A gated（主线变体；仅 --gate-mode 与目录/tag 不同）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --t-encoder foot_conv --decoder part9 --gate-mode gated \
  --batch-size 64 \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F5A_f4a_gated/checkpoints \
  --wandb_mode online --wandb_experiment_tag f5a_f4a_gated \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5
```

### F5-B：基于 F2+4（f2 表示 + foot_conv + part9，主线）

warm-start 自 `F2p4_combo`（encoders 在 f2 表示下训了 400ep，全复用；part
decoder 新建）。对照 = F2p4_combo。

```bash
# F5-B nogate（结构对照）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr --t-encoder foot_conv --decoder part9 --gate-mode nogate \
  --batch-size 64 \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F2p4_combo/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F5B_f2p4_nogate/checkpoints \
  --wandb_mode online --wandb_experiment_tag f5b_f2p4_nogate \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6

# F5-B gated（主线；仅 --gate-mode 与目录/tag 不同）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr --t-encoder foot_conv --decoder part9 --gate-mode gated \
  --batch-size 64 \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F2p4_combo/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F5B_f2p4_gated/checkpoints \
  --wandb_mode online --wandb_experiment_tag f5b_f2p4_gated \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:7
```

**跑序**：B nogate 先跑（主线），A nogate 可并行；各自通过后再跑 gated。
每条训练后验收同口径：eval + ridge + 可视化（路径换对应 out-dir；可视化
`--modal F5B --contact-method f2p4_nogate` → `results/AnySole/F5B_f2p4_nogate`）。

> **内存口径（F5 必须）**：part9 比 v1 解码器重——交叉注意力共享修复后实测
> batch 64 峰值 4.2GiB / 128 8.3GiB；**训练用 --batch-size 64**（共享卡更紧时
> 降 32）。**验收 eval 命令同样要加 `--batch-size 64`**（eval 默认读 ckpt 的
> 256 会再次 OOM；协议按 session 组批不受该 flag 影响、体量相近安全）。
> 首次 F5 的 OOM 根因 = 交叉注意力曾把窗口 token 为每个部位查询重复 9 倍
> （part_decoder.py 已修，2026-09-18）。
> 第二次 F5 报错（BCE 范围断言）= λ_con=0 时 BCE 仍被计算且 part9 冷启动
> 极端姿态产生 NaN 接触——losses.py 已修：λ_con=0 完全跳过 BCE + nan_to_num/
> clamp 保险（2026-09-18）。冷启动提醒：part decoder 全新建，epoch1 MPJPE
> ~859mm（v1 warm-start 起点 ~150mm），前几十 epoch 快速下降属正常；400ep
> 可能偏紧，ep200 观察走势，仍下行则续 800ep。

**验收（F5，先看门控探针再看主指标；计划 §F5 判据）**：
- 门控探针：T-only 时手臂 g_∅ 明显高于腿/脚；VT 时脚部 g_T 支撑相高于摆动相
  （wandb gate 曲线 + 导出 BVH 核对相位）。
- 主指标（A 对照 F4a_footconv；B 对照 F2p4_combo）：VT2M 手臂不劣化（≤+5mm）；
  **T2M MPJPE 不劣于各自对照**（A ≤123.8、B ≤99.7）**且 B 的 T2M PA ≤38**
  （门控修好 foot_conv 的 T-only = 核心判据）；nogate 与 gated 对照：gated 在
  T2M 根轨迹或 VT2M 手臂上优于 nogate。
- 塌缩门同 D。失败 → 弃开关回对照基线。

> B 通过后即"F2 表示 + F4 触觉流 + F5 门控"完整候选；F6（接触/滑步）在其上。

---

# F2+4 组合实测（2026-09-18，wandb fe2s0zsz，回填）

**结果 = 折中而非叠加**（val 协议，对照 F0b / F4a / F2 / F2+4）：

| | F0b 基线 | F4a | F2 | **F2+4** |
|---|---|---|---|---|
| VT2M MPJPE / PA | 63.7 / 25.1 | 55.5 / 23.0 | 65.4 / 24.7 | **63.4 / 23.1** |
| VT2M yaw_drift | 6.1 | 12.4 | 11.0 | **8.4（最优）** |
| V2M PA | 26.9 | 26.1 | 28.0 | **25.8（最优）** |
| T2M MPJPE / PA | 116.2 / 37.4 | 123.8 / 40.3 | **89.7 / 35.8** | 99.7 / 40.7 |
| T2M yaw_drift | 29.5 | 31.5 | 22.7 | **21.3（最优）** |

判定：VT/V 侧收回部分 F4a 增强（MPJPE −2mm、PA 全配置最优），T2M 侧保住部分 F2
收益（−16.5/26.5）但 **T2M PA 恶化到 40.7（四项最差）**——foot_conv 的 T-only 弱点
在 F2 表示下部分存在。**组合无负交互崩溃、但也非 1+1=2**；不必再调组合，直接进 F5：
门控正是修 T-only 的机制，且 part9 结构本就必须带 foot_conv。

## F5 在 F2 表示上（先 nogate 后 part9；warm-start 自 F2p4_combo，encoders 全复用）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --f2-repr --t-encoder foot_conv --decoder part9 --gate-mode nogate \
  --batch-size 64 \
  --epochs 400 --grad-clip 5.0 \
  --init-from results/AnySole/F2p4_combo/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F5_nogate/checkpoints \
  --wandb_mode online --wandb_experiment_tag f5_nogate \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4
```

**F5 判据（关键）**：T2M MPJPE ≤ F2+4 的 99.7（门控生效 = 修好 foot_conv 的 T-only，
PA 不再恶化），VT2M 不劣于 F2+4；先看 wandb 门控曲线（T-only 手臂 g_∅ 高、
VT 脚部 g_T 支撑相高）。通过后 part9 替换 nogate 同条件复跑。

---

# F2b 实测（2026-09-18，wandb h7jes2ij，回填）→ 弃

λ_kp=3、200ep、warm-start F0b_seed2：VT 72.5/PA 25.7、V 74.6/28.4、T 96.5/38.1。
**判据未达标**：MPJPE 全面反升（对照 F2 同预算 λ_kp=1 @200ep 训练面板 VT 66.4 /
T 92.4 → λ_kp=3 同预算下 VT +6.1、T +4.1，混在 200ep vs 400ep 里的不是预算问题）。
**λ_kp 保持 1.0**，不再扫描；F5 继续（F2 表示 + foot_conv + part9）。
