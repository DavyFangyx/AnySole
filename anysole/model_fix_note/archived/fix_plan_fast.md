> **已归档（2026-09-20）**：细节留档，不再维护。当前状态与结论见
> ../model_fix_note.md，当前基座命令见 ../command_manual.md。

# AnySole Fixv2 快速通道（F0 修订版）

> **本版是对 `fix_plan_v2.md` 的修订**：只改 §2 实验规范（判据/预算/种子）、§3 的 F0 一节
> 与主线跑序。F1–F9 的方案内容、单变量纪律、协议指标表**全部沿用 v2 原文**。
> `f0_command_manual.md` 的步骤 0/1/2/3 被本文件取代（其余口径说明仍有效）。
>
> **修订动机**：原方案在基线重建（E3 复跑 + F0a 三种子协议 + F0b 三种子回归 + F0c）
> 上要花约 7–9 小时算力才开始测第一个真方案，且其中多项是过度设计（见 §0）。
> 本版把 F0 关键路径压到约 1 小时，且不牺牲每个后续步骤的判据质量。

---

## 0. 修订理由（原方案的过度设计点）

| # | 原设计 | 问题 | 修订 |
|---|---|---|---|
| 1 | 步骤 0 复跑 E3（800ep ≈ 1.7h 墙钟） | `results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt` 已存在，config 已逐项核对 = E3 口径（joint_and、λ_con=0、tw=20、lr 恒定、800ep） | **删除**，直接用 E3_nocontact ckpt |
| 2 | F0a 协议 ×3 种子 ×5 口径（3 配置 + 2 鲁棒） | ① σ_seed 由 n=3 估计只有 2 个自由度，且对 diffusion 基线主要测的是 **DDIM 采样噪声**而非训练种子差异；② 鲁棒集（vdrop/tdrop）在 F0–F6 只是"报告"、F7 起才作主判据（v2 原文自述）——成为判据前跑它是纯成本；③ F0b 是回归模型、无采样随机性，manual 自己写明"3 种子 σ_seed 只在 diffusion 基线上做"——3 次独立训练在判据框架里没有统计含义，就是 3 份算力 | **1 次协议跑 + `--no-robustness`**，后台执行不挡主线；鲁棒集 F5 起再开 |
| 3 | F0b 3 种子 × 3700 步 from-scratch | ① 同上，多种子无统计意义；② V1/V2 的 encoders/fusion/traj/aux 模块逐参数相同，只有 pose_head 的 `proj_*/timestep_token`（V1）↔ `query`（V2）不同——**可以热启动**，从零重学已收敛的编码器是浪费 | **1 种子 + warm-start + 提前终止**（见 §2 步骤 C） |
| 4 | F0c 挡主线（再 3700 步） | 删 V 重建辅助头不挡任何后续方案的判定，T2M 判据到 F4 才真正用得上 | **移出主线**：推迟到 F4 前，或后台 200ep warm-start 顺手验证 |

**判定标准替换 σ_seed（改自 v2 §2）**：n=3 的 σ 既无统计效力、又被采样噪声主导，
用它当 2σ 显著性门槛要么什么都"显著"要么什么都不通过。快速筛选改用**固定实用阈值**：

- VT2M PA-MPJPE 改善 **≥5mm**，或抖动/脚滑下降 **≥20%** → 值得追；
- 改善 **<2mm** → 噪声，弃；
- 介于两者之间 → 幸存者补 1 个确认种子。
- 对比基准用固定历史数字：E3 VT2M PA-MPJPE **81.2**、ridge-on-F **85**、
  E3 one-shot VT 111.4 / 链 127.7、T2M 154.0、E5 链抖动 27mm/帧（v2 §0.4 表）。

---

## 1. 快速测试规范（替换 v2 §2 的预算与种子部分）

1. **每候选 1 种子**、短预算（默认 400ep ≈ 1850 步 @stride20），训练中观察 wandb
   曲线、按提前终止规则 kill 或续跑。**幸存者补 1 个确认种子**，不跑 ×3。
2. **warm-start 默认开**：结构允许时用 `--init-from <上一步 ckpt>` 热启动（已实现，
   见 §4）。表示/输入维度变化的步骤（F2a、F4a、F5）只有变动模块无法热启动，其余照常；
   记录口径。warm-start 数值与 from-scratch 可能有小幅偏差，对筛方案无影响；
   **最终链条收尾前用 from-scratch 全预算跑 1 次做正式基准**。
3. **协议每步跑 1 次**：`--protocol-seed 0 --no-robustness`；F5 起去掉 `--no-robustness`
   （鲁棒集成为判据后再开）。ridge-on-F 探针每步必跑（便宜且是核心判据，保留）。
4. 失败处置与 v2 一致：弃开关、回退上一步配置、失败 ckpt 留档、代码不回改（开关默认关）。

---

## 2. F0 快速版命令手册（照抄级）

> env = touch_gait，全部在 `/data/fangyuxuan/projects/gait` 下执行。
> GPU 现状（2026-09-18）：cuda:4 轻载、cuda:6/cuda:7 空闲，用它们并行。

### 步骤 A：单测（约 5 分钟，训练前必跑）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/smoke_f0b_regress.py --device cuda:6
```

warm-start 通路已另行验证（2026-09-18）：E3 ckpt → AnySoleModelV2 复制 254/255 个
key，仅 V1 diffusion 投影被跳过，query 梯度正常回传。

### 步骤 B：F0a′ — E3 ckpt 过协议一次（后台，与步骤 C 并行，GPU 6）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt \
  --modal anysolev1 --contact-method joint_and --split val \
  --protocol-seed 0 --no-robustness --device cuda:6

/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt --split val --device cuda:6
```

产出：E3 val 全套 fseries 数字 + ridge 天花板（参照 ~85mm）。**这是唯一一次
"E3 过协议"**；不写 BVH、不跑鲁棒集。

### 步骤 C：F0b′ — 回归 warm-start，1 种子（主线，GPU 4）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and \
  --epochs 400 \
  --init-from results/AnySole/E3_nocontact/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F0b_warm/checkpoints \
  --wandb_mode online --wandb_experiment_tag f0b_warm \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4
```

**观察规则**（wandb `val/tau0_mpjpe/VT`——regress 模式下 tau0 就是直接输出 MPJPE，
每 10ep 一个点；V/T 两行同看）：

- ep100 时 >110mm 且最近两档不降 → **kill**，查实现（标准化/损失口径，同原判据）；
- ep200 时 ≤95mm 且仍下行 → 继续到 400ep；
- ep400 未达标但 ≤95 且仍下行 → 续跑到 800ep（唯一允许的续跑情形）。

### 步骤 D：F0b′ 验收（GPU 7）

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/F0b_warm/checkpoints/ckpt_last.pt \
  --modal anysolev2 --contact-method joint_and \
  --split val --write-bvh results/AnySole/F0b_warm/predictions/eval_bvh \
  --device cuda:7

/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/ridge_probe.py \
  --ckpt results/AnySole/F0b_warm/checkpoints/ckpt_last.pt --split val --device cuda:7

CUDA_VISIBLE_DEVICES=6 python results_display/script/visualize_anysole.py \
  --modal F0b --contact-method warm --force
```

**判据（固定阈值，不再用 2σ_seed）**：

- 通过：VT2M PA-MPJPE **≤86mm**（E3 81.2 + 5mm 容差）且抖动明显低于 E5 链 27mm/帧；
  T2M 相对 E3 参照 154.0 不劣化 >5mm。
- **塌缩门（2026-09-18 增补，F0b 教训：塌缩模型能骗过 PA-MPJPE 与抖动，只有 ridge 探针
  抓得住）**：ridge L_pose ≤ 2× E3 基线（≈0.019）**且** MPJPE − PA-MPJPE ≤ 80mm
  （健康参照：E3 60、F0c 42；F0b 塌缩 = 110）。任一不满足 → 按塌缩处理，kill。
- 明显差于 E3（>92mm）→ 实现问题，先排查，不进 F1。
- 怀疑 warm-start 是失败原因 → **from-scratch 1 种子重跑**（本版唯一允许的重跑）。

### 步骤 E：F0c′ — 移出主线

推迟到 F4 前（T2M 判据真正用得上时），或顺手后台验证：

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --modal anysolev2 --contact-method joint_and --lambda-vrec 0 \
  --epochs 200 --init-from results/AnySole/F0b_warm/checkpoints/ckpt_last.pt \
  --out-dir results/AnySole/F0c_novrec/checkpoints \
  --wandb_mode online --wandb_experiment_tag f0c_novrec \
  --wandb_eval_interval 10 --device cuda:6
```

---

## 3. 主线跑序（F1–F9 内容不变，只调顺序/并行）

| 阶段 | 内容 | 说明 |
|---|---|---|
| 立即（无训练/后台） | F2a 表示往返单测（`z_note/probes/smoke_f2_roundtrip.py`）、F1 GVHMR 离线提取 + 零训练基线探针、S2M-P 重训、步骤 B 协议 | 全部不占主线 GPU；先把"免费信息"拿到手 |
| 主线串行 | F0b′（步骤 C/D）→ F1 与 F4a 并行筛选 → F2a/F2b → F3 → F5（先 nogate 后 part9）→ F6 → F7（起开鲁棒集）→ F8 → F9 | F1/F4a 都是输入侧独立改动，各自对照 F0b′ 基线即可并行；F5 依赖 F1+F4 的流结构，串行；F8a/F9a 仍按 v2 §4 提前并行 |
| 收尾 | 最终链条 from-scratch 全预算 1 次 | 正式基准，替换 warm-start 口径 |

每步命令模板不变（v2 §6），叠加：`--init-from <上一步 ckpt>`、`--epochs 400`、协议
`--no-robustness`（F5 前）。每步验收回填同 v2：fseries JSON + ridge 探针 + 训练面板
（tau0、loss 分量、grad_norm）。

---

## 4. 时间预算对照

| 项 | 原方案 | 快速版 |
|---|---|---|
| E3 复跑（步骤 0） | ~1.7h（800ep 墙钟实测 102min） | **0**（ckpt 现成，config 已核对） |
| F0a 协议 | 3 种子 × 5 口径 ≈ 0.5–1h，挡主线 | 1 次 + no-robustness ≈ 15min，**后台** |
| F0b | 3 × 800ep from-scratch ≈ 4–5h | 1 × 400ep warm-start ≈ 30–45min |
| F0c | 1 × 800ep ≈ 1h，挡主线 | 0（推迟） |
| **F0 关键路径合计** | **~7–9h 后才有第一个判据** | **~1h 内拿到 F0b′ 判据**，F1/F4a 筛选随后 30–45min/个 |

E3 墙钟参照：800ep@stride20 = 102min，其中 DDIM 评估开销 ~64%（train.py
`--wandb_eval_interval` 注释）。regress 模式的评估是单次前向，该开销基本消失。

---

## 5. 诚实声明（本版引入的风险）

1. **warm-start 口径**：F0b′/后续步的数值与 from-scratch 可能有小幅偏差（编码器已
   收敛）。对筛选无影响；正式基准在收尾时 from-scratch 补跑。
2. **1 种子**：会漏掉恰好落在种子噪声里的真效果——但 <2mm 的效果本来就不追，
   ≥5mm 的效果 1 个种子足够看见。
3. **400ep 短预算**：不会造成"预算不足却当失败"的误判——观察规则要求 tau0 仍在下行
   才续跑，曲线平台/上行才 kill。
4. `--init-from` 已实现（2026-09-18）：train.py 新 flag + `init_from_checkpoint()`，
   按名字/形状匹配复制、其余保持新初始化；`--init-drop` 可显式跳过前缀
   （如 `pose_head.`）。已实测 E3→V2 复制 254/255 key、前向/反向正常。

---

## 6. F0 验收回填（2026-09-18 实测）

**F0a′（B）**：E3_nocontact 过协议一次（seed0、无鲁棒）：VT2M MPJPE 138.9 / PA-MPJPE
78.8 / W-MPJPE 157.9 / jitter 121.1（GT 9.1）/ seam 180.5（GT 9.9）/ contact_f1 0.0；
ridge：L_pose 0.0093 / MPJPE 66.8 / PA-MPJPE 42.5。历史参照一致（81.2/85 同量级），协议可信。

**F0b′（C/D，400ep，λ_vrec=0.1）→ 塌缩，弃**。VT2M MPJPE 192.6 / PA-MPJPE 82.2 /
ridge L_pose 0.77（F 被掏空）/ jitter 15.0（常量输出所以"平滑"）/ contact_f1 0.2。
可视化 = 单一动作（用户确认）。**塌缩机制（wandb 时间线）**：ep330 前健康（最好 64.9），
ep360-380 一次性梯度爆炸（loss_Vrec 0.0078→0.86、val 67.8→510、nonfinite_skips 全程 0 =
有限值巨梯度，守卫未触发），ep390 起待在常量盆地（loss_total 0.45 = 健康的 30×）。
两个判据（PA≤86、抖动≤27）都被塌缩模型"通过"——**本教训已增补为 §2 判据的塌缩门**。

**F0c′（200ep，λ_vrec=0）→ 全指标突破，暂定为新基线候选**。VT2M MPJPE 69.7 /
PA-MPJPE **27.9** / W-MPJPE 76.7 / jitter 13.0 / accel_err 20.7 / seam 106.5 /
contact_f1 0.6 / foot_slide 4.0；V2M 72.7/31.4；T2M 121.7/40.5；ridge L_pose 0.0097。
可视化 ≈ GT（用户确认）。PA-MPJPE 低于线性 ridge 43.6 = 非线性+时间上下文 decoder 的
合法增益。**重要归因修正**：ep200 前 F0b≈F0c≈70mm，两曲线重合——F0c 的好不是 λ_vrec=0
的直接功劳；F0c 停在 1000 步，F0b 的爆炸点在 ~1850 步。**λ_vrec 是引爆剂还是无辜
旁观者未定，唯一决定性实验 = F0c 续跑到 400ep（f0c_ext400）看是否重演爆炸。**
不爆炸 → λ_vrec=0.1 是引爆剂，永久删除 V 重建路径；爆炸 → 结构性问题（lr=1e-3 恒定
+warm-start 过热），修 lr/梯度裁剪。seam 仍 106-172mm（GT 9.9）→ F3 重叠混合价值仍在。

**二轮归因实验（2026-09-18，2×2 表格补全）→ 两个假说都被排除，爆炸是偶发事件**：

| | λ_vrec=0 | λ_vrec=0.1 |
|---|---|---|
| 200ep | F0c ✓ 69.7（健康） | F0b_warm@200 ≈70（健康，原 run 曲线自身） |
| 400ep | 未跑（已无必要） | F0b_warm ✗ ep370 爆炸；**F0b_seed2 ✓ 63.7（健康）** |

- F0b_seed2（wandb 7u641tmu，λ_vrec=0.1，400ep，无 loss-cap 裸奔）走过 ep370 爆炸点、
  全程健康（grad_norm 全程 ≤1.3），最终 VT MPJPE **63.7**（当前最优，优于 F0c@200 的
  69.7）→ 爆炸不重现，λ_vrec 洗清嫌疑，400ep 回归本身健康。
- F0c_ext400（wandb 5knmn5n1，**实际 λ_vrec=0.1**、init-from F0c_novrec 权重，即
  "λ_vrec=0 训 200ep 后重开 V-recon 再训 200ep"）健康，VT 64.5 → 中途切换 λ_vrec 也安全。
- 结论：原 F0b_warm 的爆炸 = **偶发有限值巨梯度**（概率性失稳，nonfinite 守卫拦不住）。
  防御 = `--grad-clip 5.0`（2026-09-18 已实现）+ 每 10ep 曲线监控。
- **基线决策**：F 系列基线用 **400ep 口径**（F0b_seed2 为当前最强：VT 63.7）；λ_vrec 的
  去留不再是必须项，删除 V 重建降级为可选简化（F1 后按"重建式辅助头对回归是否有害"
  重新评估，勿沿用旧判据）。
- **epoch 计数口径**：train.py 无续训开关，`--init-from` + `--epochs 200` = 权重再走
  200ep，总步数 = 前后两段之和（wandb 曲线显示的是本段 epoch，判爆炸/收敛看总步数）。
