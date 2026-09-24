# AnySole 命令手册（SMPL-24 · 2026-09-24 独立性重构）

> env = touch_gait；根目录 `/data/fangyuxuan/projects/gait`。
> 历史与演进见 `model_fix_note.md`；模型结构见 `anysole/AnySole 多模态动作重建 · 模型实现说明.md`。

> **⚠️ 2026-09-24 独立性重构（本手册已按新口径改写）**：
> ① **无 warm-start 血缘**。每个模型 = `anysole/models/` 下一个独立入口脚本
> （`f0b.py` … `v4b.py`），结构由入口硬编码；任何一次训练（基座或超参变体）
> 都从随机初始化开始。**模型 + 超参数 = 一次独立训练**，没有父/子、没有
> "在默认参数结果上后训练"。
> ② **结构不再是 CLI 选项**。`--t-encoder / --f2-repr / --pose-parts /
> --soft-parts / --gate / --tactile-input / --no-imu / --v-input /
> --tactile-direct / --from-scratch` 已删除；`--init-from` 仅为调试覆盖，
> 永不自动推断。
> ③ **旧结果已全部删除**（2026-09-24 用户裁定）：本手册各节保留的数字
> 一律为 **warm-start 链口径的历史记录**，仅作参照；所有基座待从零重训，
> 重训后以新数字为准。

## CLI 参数与支持空间（注释版）

```text
# --model-name: F0b/F4a/F2/F2p4/V3_2/V3_3A/V3_3B/V3_4a/V3_4b/V3_4c/V4A/V4B
#   每个名字对应 anysole/models/<name>.py 入口脚本，结构由入口固定
# --contact-method: 接触标签生成/读取方案，不是输入特征 concat 开关；当前主线为 joint_and
# --variant: 显式覆盖变体目录名（仅特殊用途）；省略时由 train 自动生成
#   —— 目录名 = 本次 run 的完整超参记录（见"通用约定 · 变体命名"）

# --which {last,best}: 自动选择 ckpt_last.pt 或 ckpt_best.pt，默认 last
# --ckpt FILE: 仅供调试、旧脚本和文件型探针

# --tw INT (>=1): 窗口长度，默认 20；扫描可用 40/80/120 或其他正整数
# --stride INT (>=1): 窗口步长；扫描时必须显式设置为与 tw 相同的值
# --epochs INT、--batch-size INT、--lr FLOAT (>0)、--seed INT
# --part-json FILE: 仅 V4B —— 学到的硬分组（结构数据，其他模型拒绝）
# --assign-cluster + --lambda-assign-*/--assign-*: 仅 V4A 训练协议超参
# --lambda-assign/--lambda-sigma/--sigma-freeze-frac/--lr-warmup-frac: 超参
# --init-from FILE: 仅调试覆盖（独立性规则：永不自动推断，默认随机初始化）
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

- **模型目录 = `{模型版本}_{contact-method}`**（全部 SMPL-24）：
  `F0b_joint_and` / `F4a_joint_and` / `F2_joint_and` / `F2p4_joint_and` /
  `V3_2_joint_and` / `V3_3A_joint_and` / `V3_3B_joint_and` /
  `V3_4a_joint_and` / `V3_4b_joint_and` / `V3_4c_joint_and` /
  `V4A_joint_and` / `V4B_joint_and`。
- **独立性（2026-09-24）**：每个模型的**结构**由 `anysole/models/`
  下同名入口脚本固定（`STRUCTURE` 字典 + `build(config, part_joints)`），
  训练时把 `STRUCTURE`/`CONFIG_EXTRA` 记入 ckpt config，eval/infer 按记录的
  `model_name` 找入口重建并校验结构一致。**任何一次训练都从随机初始化
  开始**——基座、tw 变体、optuna trial 一视同仁，互相之间没有任何
  初始化继承关系。
- **公共层**：编码器/融合/头等共享模块在 `anysole/models/common/`
  （embeddings / encoders / foot_encoder / tactile_encoder / fusion /
  pose_head / traj_head / aux_heads + V1/V2 装配类 + 归档的 part_decoder）。
- **变体命名（完整超参记录，2026-09-24 用户裁定）**：任何 run 都落
  `<模型>_joint_and/{完整字段堆叠}/`，字段顺序固定
  `tw→st→lr→lp→lt→lk→la→lae→lac→lad→adb→ati→atf→aaf→alf→alm→ls→sf→wu→gc→lcp→ep→bs→sd→pk`，
  **配置里出现的字段全部写出、无默认省略**——目录名本身就是该 run 的超参
  记录（实现 = `types.variant_from_config`，train/tune/调度器共用）。例：
  V3_4a tw40 = `V3_4a_joint_and/tw40_st40_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_lcp1_ep740_bs256_sd1/`；
  全默认超参 run 同样有完整目录（不再落裸基座目录，基座目录只作容器）。
  两个 run 只要任一被记录的超参不同，目录就不同、不可能互相覆盖。
  **全流程零手写地址**：真实路径由
  `anysole/registry.py` 的 `infer_anysole_paths` 推导；train 自动生成
  variant（`--variant` 仅特殊覆盖）；eval/infer/ridge 用
  `--model-name [--variant 完整字段串] [--which]`；
  `--out-dir/--init-from/--ckpt/--write-motion` 一律禁止手写，仅作调试覆盖；
  文件型探针是例外。
- **checkpoint 状态（2026-09-24）**：旧 warm-start 链的全部结果（含 ckpt、
  metrics、tw 扫描产物、V4B 分组文件）已删除；所有 12 个基座待从零重训。
- `anysole/configs/v1.yaml` 是 SMPL-24 训练配置：raw108、tw=20、stride=20、
  lr=1e-4、λ_pose=3 / λ_kp=1 / λ_traj=1 / λ_trec=0.1 / λ_vrec=0.1 / λ_con=0、
  joint_and。实验差异全部由 CLI 传入，覆盖值随 ckpt 保存，eval/infer 从
  ckpt 读配置。
- 训练前单测（`z_note/probes/`）：`smoke_f0b_regress.py`（回归头）、
  `smoke_f4a_grid.py`（foot_conv）、`smoke_f2_roundtrip.py`（f2 表示）、
  `smoke_v32_parts.py`（9 部位）、`smoke_v33_soft_gate.py`（soft A + σ 门控）、
  `smoke_v4a_cluster24.py`（24 槽聚类）。warm-start 检查项已随独立性重构移除；
  `smoke_f5_part.py` 为 F5 归档残留（138D 口径，勿用）。
- 每个基座固定四条命令：**① 训练 → ② 评估（--write-motion 刷新导出）→
  ③ R_Test1 模型可视化（骨架动画）→ ④ R_Test3 轨迹可视化**。可视化读
  `predictions/eval_motion/<session>_<config>.npz`，因此评估必须先于可视化。
- results_display 编号：D_TestN（数据检验）/ R_TestN（结果检验）；手册两条
  可视化 = R_Test1（`results_display/script/r_test1_visualize_anysole.py`）
  与 R_Test3（`results_display/script/r_test3_traj.py`）。
- **产物目录规范（2026-09-24 改组）**：四组两级结构
  `results_display/{DataTest,ResultTest,ATest,BTest}/`；`ridge_probe.py` 与
  `z_note/probes/smoke_*.py` 保持原地。
- 评估统一执行 canonical session metrics（`--protocol-seed 0`）；
  `--no-robustness` 关鲁棒集。每步验收读 `metrics/<split>_fseries.json` +
  `metrics/ridge_probe_<split>.json`（ridge 探针每步必跑：
  `results_display/script/ridge_probe.py --model-name <名称> [--variant <变体>]
  --split val --device <GPU>`）。
- wandb tag 与目录同名（如 `v34a_jointand`）。
- `--device` 按 `nvidia-smi` 空闲卡填。
- 单会话推理（需要时）：`python -m anysole.infer --contact-method joint_and
  --model-name <名称> [--variant <变体>] --session <SID> --config-id VT2M --device <GPU>`。

---

## 基座总表（12 模型 · 各自独立入口 · 一律从零训练）

| 模型 | 入口脚本 | 结构（入口固定） | 关键训练超参 |
|---|---|---|---|
| F0b | `anysole/models/f0b.py` | linear 触觉编码、3 部位回归头 | grad-clip 5.0 |
| F4a | `f4a.py` | foot_conv 触觉编码、3 部位 | grad-clip 5.0 |
| F2 | `f2.py` | f2 表示（tilt/yaw + 4 维朝向系轨迹） | grad-clip 5.0 |
| F2p4 | `f2p4.py` | f2 + foot_conv | grad-clip 5.0 |
| V3_2 | `v3_2.py` | foot_conv + 9 部位硬分组 | lr-warmup 0.05、grad-clip 5.0 |
| V3_3A | `v3_3a.py` | + soft A 矩阵 | + λ_assign 0.05 |
| V3_3B | `v3_3b.py` | + σ 门控 | + λ_assign 0.05 |
| V3_4a | `v3_4a.py` | f2 + foot_conv + 9 部位 | lr-warmup 0.05、grad-clip 5.0 |
| V3_4b | `v3_4b.py` | + soft A | + λ_assign 0.05 |
| V3_4c | `v3_4c.py` | + σ 门控 | + λ_assign 0.05、λ_sigma 0.01、σ_freeze 0.1 |
| V4A | `v4a.py` | foot_conv + 24 槽 soft（聚类在训练中学习） | `--assign-cluster` + assign-* 系 |
| V4B | `v4b.py` | foot_conv + `--part-json` 学到的硬分组（K 任意） | 需先导出分组文件 |

---

## 基座 F0b · 回归本体（from-scratch，历史上 400ep 从零训练）

> **历史记录（warm-start 链口径，2026-09-24 前，结果已删除）**：
> 400ep from-scratch：VT2M MPJPE 78.1 / PA 44.8、V2M 81.4 / 46.2、
> T2M 154.9 / 72.4；VT yaw_drift 4.5、jitter 12.0、contact_f1 0.43、
> foot_slide 2.82mm/帧（wandb fxemwdt8）。from-scratch 400ep 是筛选预算，
> 显著弱于 BVH 时代 warm-start 的 F0b_seed2（63.7/25.1）。
> 独立性重训建议：800ep（yaml 默认）从零收敛。

```bash
# ① 训练（从零，结构 = f0b 入口）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --epochs 800 --grad-clip 5.0 \
  --model-name F0b \
  --wandb_mode online --wandb_experiment_tag f0b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:4

# ② 评估（协议 seed0 + 导出 SMPL NPZ，R_Test1/R_Test3 依赖此步）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name F0b \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1（骨架动画 → results_display/ResultTest/R1Test_visualize/AnySole/F0b/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name F0b --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force

# ④ 可视化 R_Test3（轨迹对比 → results_display/ResultTest/R3Test_traj/AnySole/F0b/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test3_traj.py \
  --model-name F0b --contact-method joint_and --split val \
  --config-id VT2M,V2M,T2M --gen gif --force
```

---

## 基座 F4a · 按脚卷积触觉（`f4a.py` 入口）

> **历史记录（warm-start 链口径，结果已删除）**：
> 400ep warm-start F0b：VT2M 70.4 / 37.3、V2M 71.6 / 36.4、T2M 175.2 / 69.6
> （T-only yaw_drift 72.4）；VT contact_f1 0.62。BVH 历史口径（仅参照）：
> VT 55.5/23.0、T2M 123.8/40.3。

```bash
# ① 训练（从零）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --epochs 800 --grad-clip 5.0 \
  --model-name F4a \
  --wandb_mode online --wandb_experiment_tag f4a_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5

# ② 评估 / ③ R_Test1 / ④ R_Test3：同 F0b，--model-name 换 F4a
```

---

## 基座 F2 · f2 表示（`f2.py` 入口）

> **历史记录（warm-start 链口径，结果已删除）**：
> 400ep warm-start F0b：VT2M 79.8 / 49.3、V2M 72.5 / 42.4、T2M 139.3 / 79.1；
> yaw_drift 全配置大幅改善（T2M 30.4→3.1、VT 4.5→2.6、V 4.6→2.3）。
> BVH 历史口径（仅参照）：T2M 89.7/35.8、VT 65.4/24.7。

```bash
# ① 训练（从零）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --epochs 800 --grad-clip 5.0 \
  --model-name F2 \
  --wandb_mode online --wandb_experiment_tag f2_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6

# ② 评估（eval/协议/导出自动处理 f2→world 恢复）/ ③④ 可视化：--model-name 换 F2
```

---

## 基座 F2p4 · f2 表示 + 按脚卷积（`f2p4.py` 入口）

> **历史记录（warm-start 链口径，结果已删除）**：
> 400ep warm-start F2：VT2M 69.0 / 39.3、V2M 64.5 / 36.2、T2M 134.9 / 77.7、
> yaw_drift 2.5/2.3/2.8。BVH 历史口径（仅参照）：VT 63.4/23.1、T2M 99.7/40.7。

```bash
# ① 训练（从零）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --epochs 800 --grad-clip 5.0 \
  --model-name F2p4 \
  --wandb_mode online --wandb_experiment_tag f2p4_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:7

# ② 评估 / ③④ 可视化：--model-name 换 F2p4
```

---

## V3-2 · 九部位（`v3_2.py` 入口）

> **历史记录（warm-start 链口径，结果已删除）**：
> 740ep warm-start F4a：ckpt_last@740 VT2M 72.8 / 42.2、V2M 65.7 / 35.3、
> T2M 183.1 / 69.9；ckpt_best@310 VT2M 67.4 / 37.9、V2M 66.7 / 34.5；
> contact_f1 VT 0.76。历史爆炸教训（grad-clip + warmup + NaN 守卫已生效）。

```bash
# ① 训练（从零；前 5% 步数线性 lr warmup 防爆炸）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_2 \
  --wandb_mode online --wandb_experiment_tag v32_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5

# ② 评估 / ③④ 可视化：--model-name 换 V3_2
```

> epoch 计数口径：每次运行从自己的 epoch 0 开始数；判断收敛/爆炸看
> **总步数**（3700 步短预算 ≈ 740ep batch256 @stride20）。

---

## V3-3 · soft A 矩阵 + σ 门控（`v3_3a.py` / `v3_3b.py` 入口）

> 设计与机制关系（A 矩阵/门控/先验）见 `archived/fix_plan_v3.md` §8；
> smoke 必过：`z_note/probes/smoke_v33_soft_gate.py`（训练前跑）。
> **历史记录（warm-start 链口径，结果已删除）**：V3-3A/B 曾以 warm-start
> 链训练（含 §8.4 两处修复：L_assign 平移 +log9、τ_p 前向钳位 [−4,4]——
> 这些修复仍在代码中生效）；二轮实测 σ 未学会路由（门控差 ≤0.018）、A 冻结
> 于人工划分；判据已改差分式（§2.3 F4 修订），σ 填坑 = V3_4c 的
> `--lambda-sigma 0.01 --sigma-freeze-frac 0.1`。T2M 偏航乱飘已量化
> （yaw_abs 71.5°、速率 231°/s vs GT 12.6°/s）——压力无朝向信息，
> 结构解法 = f2 表征（V3-4 系），不是门控。

```bash
# ===== V3_3A：soft A 矩阵（从零） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --lambda-assign 0.05 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_3A \
  --wandb_mode online --wandb_experiment_tag v33a_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6

# ② 评估 / ③④ 可视化：--model-name 换 V3_3A

# ===== V3_3B：叠加 σ 门控（从零） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --lambda-assign 0.05 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_3B \
  --wandb_mode online --wandb_experiment_tag v33b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:7

# ② 评估 / ③④ 可视化：--model-name 换 V3_3B

# 训练前 smoke（数值验收，§8.4）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/smoke_v33_soft_gate.py \
  --device cpu
```

> 入口校验：A 矩阵/σ/τ_p/part_logits 的结构约束由 `pose_head.py` 构造器
> 检查（gate 与 soft 组合、n_parts 合法性），入口脚本保证每个模型的
> 组合固定——不再依赖命令行传对开关。

---

## V3-4 · f2 标准表征系（`v3_4a/b/c.py` 三个独立入口）

> **定位**：T-only 偏航根因 = 压力无绝对朝向（历史探针：yaw_abs 71.5°、
> 速率 231°/s vs GT 12.6°/s）。f2 把整条链的预测目标从"绝对朝向"改为
> "朝向变化/相对量"（根 6D=tilt、轨迹=4 维朝向系 [psi_dot, v_hx, v_hz, h]），
> 三配置共用一套权重。
> **独立性（2026-09-24）**：V3-4a/b/c 是三个结构递进但**训练互相独立**的
> 模型（软 A、σ 门控由各自入口固定），**没有先后顺序依赖**——三个可以
> 并行训练；每个都从随机初始化开始。
>
> | 模型 | 结构（入口固定） | 验收 |
> |---|---|---|
> | **V3-4a** | 9 部位 | — |
> | **V3-4b** | + soft A | 不劣于 V3-4a，A 保持可识别部位 |
> | **V3-4c** | + σ 门控（β-NLL 填坑超参：`--lambda-sigma 0.01 --sigma-freeze-frac 0.1`） | F4 差分判据（probe_v33b_gates.py）+ T2M ≤92 且 PA ≤36、VT2M ≤65 + yaw 探针 T2M yaw_abs ≤10° |

```bash
# ===== V3-4a：9 部位（f2 基座，从零） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_4a \
  --wandb_mode online --wandb_experiment_tag v34a_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:5

# ② 评估（协议 seed0 + 导出 SMPL NPZ，R_Test1/R_Test3 依赖此步）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --model-name V3_4a \
  --contact-method joint_and \
  --split val \
  --protocol-seed 0 --no-robustness --device cuda:4

# ③ 可视化 R_Test1 / ④ 可视化 R_Test3：--model-name V3_4a

# ===== V3-4b：+ soft A（从零） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --lambda-assign 0.05 --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_4b \
  --wandb_mode online --wandb_experiment_tag v34b_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:6

# ② 评估 / ③ R_Test1 / ④ R_Test3：同上，--model-name 换 V3_4b

# ===== V3-4c：+ σ 门控（β-NLL 填坑超参，从零） =====
# ① 训练
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --lambda-assign 0.05 --lambda-sigma 0.01 --sigma-freeze-frac 0.1 \
  --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V3_4c \
  --wandb_mode online --wandb_experiment_tag v34c_jointand \
  --wandb_eval_interval 10 --loss-cap 1.0 --device cuda:7

# ② 评估 / ③ R_Test1 / ④ R_Test3：同上，--model-name 换 V3_4c
# 门控差分验收（F4 修订判据：T vs VT 的 Δg_T(foot)>0.2、T-only 时 root g_V 较 VT 下降>0.2）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/probe_v33b_gates.py \
  --model-name V3_4c --device cuda:5

# T2M 偏航验收（目标 yaw_abs ≤10°）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/probe_v33b_yaw_t2m.py \
  --model-name V3_4c
```

> 每个模型训练前 smoke 照跑（smoke_v33_soft_gate.py / smoke_v32_parts.py /
> smoke_f2_roundtrip.py）。
> 注：V3-4a 基座命令的超参与窗口扫描节的基线 tw=20 完全相同 → 两者是
> **同一个 run、同一个目录**（`…/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_lcp1_ep740_bs256_sd1/`），
> 跑一次即可。

---

## V4A · 24 槽梯度聚类（`v4a.py` 入口，从零）

> 24 槽 + 均匀初始化 A + 退火熵/聚集奖励/死槽税 + 软 EM（A 独立 lr、70% 锁定）。
> 分组从数据学、K 不预设。`assign_cluster` 是 V4A 入口的固有部分
> （CONFIG_EXTRA 记入 ckpt），`--assign-cluster` CLI 只负责传递运行时
> 退火/损失超参。训练前 smoke：`smoke_v4a_cluster24.py`。

```bash
# ① 训练（740ep 从零；assign-* 为运行时超参）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --assign-cluster \
  --lambda-assign 1.0 --lambda-assign-ent 0.05 --lambda-assign-conc 0.01 \
  --lambda-assign-dead 0.02 --assign-dead-beta 2.0 \
  --assign-temp-init 1.0 --assign-temp-final 0.2 \
  --assign-anneal-frac 0.7 --assign-lock-frac 0.7 --assign-lr-mult 5.0 \
  --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 \
  --model-name V4A \
  --wandb_mode online --wandb_experiment_tag v4a_jointand \
  --wandb_eval_interval 10 --loss-cap 10.0 --device cuda:7

# ② 评估（常规三配置；分组本身是产物）
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

---

## V4B · 方案 B 运动学聚类 → 学到的分组硬训练（`v4b.py` 入口，从零）

> 用户裁定（2026-09-24 修订）：V4A 零启动后，学习分组改走方案 B——
> **数据版**：分组直接从**原始训练数据的关节运动学**导出（速度/加速度
> 统计 + 关节间速度相关矩阵 → Ward 聚类 + silhouette 定 K），**不经过任何
> 训练模型、无 F4a 依赖**（旧表征版 `probe_part_cluster_b.py` 已作废留档）。
> V4B 训练仍从零、分组冻结。
> 产出（已生成）：`results/AnySole/partitions/partitions_kinematic_b_K{9,10,11,14}.json`
> —— K10 = silhouette 最优（0.936）；K9 = 与 V3_2 的对位比较（默认）。
> **⚠️ 文件缺失时**：train 与调度器响亮报错并给出重新导出步骤
> （重导出纯数据、无模型，约 26 秒）。
> **历史记录（warm-start 链口径，结果已删除）**：旧表征版 740ep K9 打平偏优 V3_2。

```bash
# ① 导出分组（数据版；已生成过则跳过；重新生成无需任何模型 ckpt）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python z_note/probes/probe_part_cluster_b_data.py

# ② 硬训练学到的分组（示例 = K9 对位 V3_2；从零）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --lr-warmup-frac 0.05 \
  --part-json results/AnySole/partitions/partitions_kinematic_b_K9.json \
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

> 验收：不劣于 V3_2。K9 若劣，再试 K10/K14。`--part-json` 仅 V4B 接受；
> 分组须覆盖 24 关节恰好一次（pose_head 构造器校验）。

---

## 窗口扫描（tw sweep，基模 V3_4a · 从零独立版）

> **设计（2026-09-24 独立性修订）**：四个扫描点 = 同一 V3_4a 结构在
> 四个窗口参数下**各自从零独立训练**——基线 tw=20 与扫描点 40 / 80 / 120
> 地位完全平等（宽间距先定趋势；若 40 附近有拐点再补 30/50）。**四点之间
> 无任何初始化继承关系**（模型 + 参数独立）。
> 单变量锁定：`--stride` 显式传 `<tw>`（v1.yaml 固定 stride=20，不传会变成
> 50% 重叠窗——TW40 首跑即因此污染，见 tw40_st20 历史记录）、
> `traj_deltas` 保持 [2,4,8,19]、epochs 740。四个点互不依赖，可四卡并行。
> **目录规则（完整超参记录，2026-09-24 用户裁定）**：四个点全部落
> `V3_4a_joint_and/{完整字段堆叠}/`，目录名 = 该 run 的全部超参
> （tw/stride/lr/lp/lt/lk/wu/gc/lcp/ep/bs/sd），无默认省略：
>
> | 点 | 目录 |
> |---|---|
> | 基线 tw=20 | `V3_4a_joint_and/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_lcp1_ep740_bs256_sd1/` |
> | tw=40 | `…/tw40_st40_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_lcp1_ep740_bs256_sd1/` |
> | tw=80 / 120 | 同理换 tw/st 字段 |
>
> **模型+接触+超参 = checkpoint 地址**，eval 用 `--variant <完整字段串>`、
> 可视化用 `--variant`、ridge 用 `--model-name + --variant`；只接受文件
> 输入的探针才用 `--ckpt`。**历史教训（2026-09-22 碰撞事故）**：曾因手写
> `--out-dir` 三 run 互相覆盖——所有命令禁止手写 --out-dir / --ckpt /
> --write-motion。
> 判读：seam_jump 随 tw 变大而样本数锐减（tw=80/120 时 val 每会话仅 2–3
> 条缝），勿过度解读。成本估算：740ep ≈ 1–2h（tw20）/ 1.5h（40）/ 3.5h（80）
> / 5h（120）。

```bash
# ===== 基线 tw=20（从零；目录 = 下方 V3-4 节 V3_4a 基座命令的同一目录：
#       两者超参完全相同 → 是同一个 run，跑一次即可）=====
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and --model-name V3_4a \
  --lr-warmup-frac 0.05 \
  --epochs 740 --grad-clip 5.0 --loss-cap 1.0 \
  --wandb_mode online --wandb_experiment_tag tw20_jointand \
  --wandb_eval_interval 10 --device cuda:2
#   → out-dir 自动 = results/AnySole/V3_4a_joint_and/
#     tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_lcp1_ep740_bs256_sd1/checkpoints

# ===== 扫描点 tw=40（从零；--model-name + 字段自动拼目录，禁止写 --out-dir）=====
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and --model-name V3_4a \
  --lr-warmup-frac 0.05 \
  --tw 40 --stride 40 \
  --epochs 740 --grad-clip 5.0 --loss-cap 1.0 \
  --wandb_mode online --wandb_experiment_tag tw40_jointand \
  --wandb_eval_interval 10 --device cuda:3
#   → out-dir 自动 = results/AnySole/V3_4a_joint_and/
#     tw40_st40_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_lcp1_ep740_bs256_sd1/checkpoints
# tw80/120 只需换 --tw/--stride（与 --device），目录自动落到对应完整字段串

# ② 评估（--model-name + --variant 完整字段串 → ckpt 地址自动解析；
#    导出默认落 <模型目录>/predictions/eval_motion，--ckpt/--write-motion 都不写）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --contact-method joint_and --model-name V3_4a \
  --variant tw40_st40_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_lcp1_ep740_bs256_sd1 \
  --split val --protocol-seed 0 --no-robustness --device cuda:7

# ③④ 可视化（R_Test1/R_Test3 用 --variant 定位，输出自动落到变体名下）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test1_visualize_anysole.py \
  --model-name V3_4a --contact-method joint_and \
  --variant tw40_st40_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_lcp1_ep740_bs256_sd1 \
  --split val --config-id VT2M,V2M,T2M --gen gif --force

# ridge 探针每步必跑（使用 model-name + variant 自动定位）：
# results_display/script/ridge_probe.py --model-name V3_4a \
#   --variant tw40_st40_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_lcp1_ep740_bs256_sd1 \
#   --split val --device <GPU>
```

---

## Optuna 调参（`python -m anysole.tune`，基模 V3_4a）—— 待重设计

> **独立性影响（2026-09-24）**：旧设计"每 trial warm-start 自 V3_4a 基座 +
> 150ep 短预算"在从零口径下不再成立——150ep 从零远不够收敛，且 trial 之间
> 也必须互相独立（这恰恰是 Optuna 的正确用法）。代码层面已改：trial 一律
> `--model-name V3_4a` + 各自超参从零训练，不再传 `--init-from`。
> **预算需重设计**（例如每 trial ≥400ep 或缩搜索空间），由用户裁定后再用；
> 在此之前本工具保持可用但数字不可与旧口径比较。
> objective = 三配置 val MPJPE 加权均值（1:1:1 最小化）；study 存 sqlite、
> 中断续跑；产物 `results/AnySole/optuna/<study>.db` + `optuna_trials.csv` +
> `best_trial.txt`；trial 目录 `V3_4a_joint_and/t{NNNN}_完整字段堆叠/`
> （tune 与 train 共用 `variant_from_config`，目录名 = trial 的超参记录）。

```bash
# 先 dry-run 抽 2 个 trial 看拼出的命令（不训练）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.tune \
  --study-name v34a_dry --n-trials 2 --dry-run --seed 0 --device cuda:7

# 正式搜索（预算待用户重定；示例 400ep/trial）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.tune \
  --study-name v34a_sweep1 --n-trials 30 --epochs 400 \
  --sampler tpe --seed 0 --device cuda:4
```

---

## 已取消 / 已作废（勿用）

- **warm-start 血缘链（2026-09-24 作废）**：全部模型改为独立入口 + 从零
  训练；registry 不再有 parent 字段，`--init-from` 永不自动推断
  （仅调试覆盖）。旧链上全部 ckpt/数字已删除，仅作历史记录保留在上文。
- **F1**（`--v-input hmr_gvhmr`）：已取消（视觉天花板假说证伪），代码留档、权重不下载。
- **F2b**（`--lambda-kp 3`）：实测有害，弃；λ_kp 保持 1.0。
- **F5**（part_decoder 门控）：已作废并回滚，part_decoder.py 留档
  （`models/common/`）；smoke_f5_part.py 为 138D 口径残留，勿跑。
- **diffusion-era 开关**（`--tau-max/--tau-fixed`、`warm_start`、`continuation` 的
  V1 用法、`--write-bvh`）：anysolev2 下被忽略并打印 WARNING；BVH 导出已废除。
