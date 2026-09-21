> **已归档（2026-09-20）**：细节留档，不再维护。当前状态与结论见
> ../model_fix_note.md，当前基座命令见 ../command_manual.md。

# E3 增强清单（E6.x 系列，2026-09-16）

> **基础 = E3**。实测证据：E3 导出 BVH 脚踝误差 174mm、脚踝相对高度均值
> -0.606m（贴地）；E4 起脚踝误差 742mm、脚踝平均悬空 +0.013m（"向天空踢腿"）。
> 结论：E3→E4 的"信噪比失配"假设错误——E3 在未缩放噪声（8.5× 失配）下输出良好；
> E4 的噪声缩放 + cosine LR 与 E5 的 linear-200 引入了踢腿，**本系列不启用**。
> 基础口径：tw=20、batch 256、lr=1e-3 恒定、cosine-1000、noise_scaled=false、
> sample 50 步、λ_pose=3.0 / λ_traj=1.0 / λ_kp=1.0 / λ_con=0.0、joint_and。
>
> 所有增强为独立开关（默认关闭 = 纯 E3 行为），解耦已验证：
> `z_note/probes/smoke_e6x_decoupling.py`。E6.4/E6.5 为零重训的评估侧机制，
> 可直接作用于 E3 模型（E6.5 实测对 E3 无收益，保留为可选工具）。

'''
┌──────┬─────────────────────┬───────────────┬─────────────────────────────┐
│ 口径 │ one-shot g(F) MPJPE │ one-shot 抖动 │       50 步链（对照）         │
├──────┼─────────────────────┼───────────────┼─────────────────────────────┤
│ VT2M │ 111.4 mm            │ 57.7 mm/帧    │ 127.7（test）/ 131.8（val） │
├──────┼─────────────────────┼───────────────┼─────────────────────────────┤
│ V2M  │ 119.0 mm            │ 55.4 mm/帧    │ 133.5（val）                │
├──────┼─────────────────────┼───────────────┼─────────────────────────────┤
│ T2M  │ 154.0 mm            │ 59.8 mm/帧    │ 170.1（val）                │
└──────┴─────────────────────┴───────────────┴─────────────────────────────┘

逐关节 VT：髋 48 → 踝 135-166 → 脚趾 213mm；
抖动髋 6.7 → 脚 120-145mm/帧。FK 放大签名清晰。

配合 wandb：tau0 = 22.99mm（头在干净输入上是好的）、训练总量 3950 步（800 ep × 5 步/ep）、链 131.8。三个数拼起来的图景：头读 F 的能力 111.4（离 ridge 85mm 不远）→ 链把它做到 127.7（+16mm，链是负贡献）→ 抖动 57.7 被链放大到 ~130（~2×）。
'''

---

## E6.1 表示切换：6D 旋转 → 根局部 3D 位置（66 维）

- **改进思路**：E3 的 6D 空间里旋转误差沿 FK 链放大（实测踝 174mm、手 336mm，
  且随肢体变长单调增长）。把扩散目标换成 22 个非根关节的根局部位置（会话第 0 帧
  朝向系，Step2Motion initial_global_rot 约定），输出空间=指标空间，放大结构消失。
  解码器/编码器/融合结构完全不动，只改 head 每关节 I/O 宽度（6→3 维）；
  导出时经逆 FK（`positions_to_6d_np`，GT 往返 0.00mm）恢复旋转。
  **λ_kp=1.0 保留**（pos 模式用 R_init 转世界系做 FK 关键点监督）——E7 的教训：
  几何监督不能砍。
- **预期效果**：同等训练量下远端关节（踝/手）误差显著下降；表示红利已实测
  （2-epoch PA-MPJPE ≈ 90mm ≈ 6D 版训练满的 97.8mm）。训练 100-200 epoch 后
  与 E3 基线同表对比 MPJPE/脚踝高度。
- **验证状态**：开关单点冒烟通过（前向/损失/λ_kp 监督）。

## E6.2 平滑与刚性监督（pos 模式的两项损失）

- **改进思路**：位置版每帧独立预测，无任何时序/几何约束时，位置噪声经
  方向→旋转换算按骨长倒数放大（脚趾/手等短骨放大 10–20 倍）→ 旋转混乱
  （E7 失败的直接机制）。加两项：`lambda_pose_vel`（预测位置一阶差分 vs GT 差分的
  MSE，traj 头 velocity 思路的延伸）压帧间抖动；`lambda_bone`（预测骨长 vs 静止
  骨长的 MSE）保骨架刚性。默认权重 0 = 关闭。
- **预期效果**：30–60 epoch 内帧间抖动显著下降、旋转混乱消失。若摆动腿被抹平
  （过度平滑），权重 10/1 → 3/0.3。
- **验证状态**：两项损失激活 + 反向通过（单点冒烟）。

## E6.3 窗口 20→100 + 训练 stride-1

- **改进思路**：20 帧（0.5s）不含完整步态周期，条件上下文太短；且每 epoch 仅
  5 个优化步（1167 窗 ÷ 256），887 epoch 才 ~4.4k 步（E5 的欠训练根源）。tw=100
  + stride-1 使窗口 ~17k/epoch、优化步数 ~270/epoch，模型见到全部帧对齐方式
  （Step2Motion 同款机制，保守起见 stride 亦可试 2/4/5）。
- **预期效果**：40–100 epoch（9–25k 步 ≈ Step2Motion 预算）即饱和；同等墙钟内
  训练充分度大幅提升。tw=100 时 batch_size 建议 64（显存）。
- **验证状态**：17220 窗、训练步通过（单点冒烟）。

## E6.4 续写式推理（eval/infer，零重训）

- **改进思路**：窗口独立生成导致接缝跳变 213mm（E4 实测）。续写机制
  （Step2Motion prev_x_t）：窗 stride=tw/2，每窗前半帧由上一窗同噪声级隐变量接续，
  相邻窗口共享同一条扩散链；窗 0 全帧 + 后续窗新半帧参与指标与导出。
  已验证同款机制在 6D 模型上把 S7013 的 MPJPE 161.6→83.6mm。
- **预期效果**：接缝跳变结构性消失；生成序列的帧间连续性接近 GT。
  直接对 E3 或任一 E6.x 已训模型启用（`continuation: true`），无需重训。
- **验证状态**：eval 续写接线跑通（效果待真实训练模型上评估）。

## E6.5 暖启动（采样起点对齐训练边缘分布，零重训）

- **改进思路**：链从纯噪声起步 = 训练分布外输入（训练 τ 顶格有 36% 信号）；
  x_T = √ᾱ_top·均值姿态 + √(1-ᾱ_top)·噪声把起点搬回分布内。
- **预期效果**：E5 上实测 -18%（197→161mm）。**但在 E3 上实测无收益
  （85.1→88.1mm）**——E3 的 head 冷启动本来就很好（这正说明 E3 基础是对的）。
  保留为可选工具，不建议对 E3 使用。
- **验证状态**：E3/E5 双模型实测完成。

---

# CLI 指令

> 全部命令在 `/data/fangyuxuan/projects/gait` 下执行，env = touch_gait。
> `--device` 按空闲 GPU 填（当前 4/7 空闲）；GPU 6 上可能仍有旧训练进程。
> 每步实验的 yaml 改动执行后**评估完记得改回**（或直接复制一份 yaml 用 `--config` 传入）。

## 训练
conda activate touch_gait

```bash
# ===== E3 基线复跑（v1.yaml 已是 E3 值，无任何开关）=====
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --epochs 800 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E3_base/checkpoints \
  --wandb_mode online --wandb_experiment_tag e3_base \
  --wandb_eval_interval 10 \
  --device cuda:4

# ===== E6.1 表示切换（其余全为 E3 口径）=====
# v1.yaml 无需改动；只加 --modal anysolev1_pos
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --modal anysolev1_pos \
  --epochs 200 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E6.1_pos/checkpoints \
  --wandb_mode online --wandb_experiment_tag e6_1_pos \
  --wandb_eval_interval 10 \
  --device cuda:4

# ===== E6.2 平滑+刚性（在 E6.1 之上，仅动两个权重）=====
# v1.yaml 改动：lambda_pose_vel: 0.0 -> 10.0；lambda_bone: 0.0 -> 1.0
# （摆动腿被抹平则降为 3.0 / 0.3）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --modal anysolev1_pos \
  --epochs 200 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E6.2_smooth/checkpoints \
  --wandb_mode online --wandb_experiment_tag e6_2_smooth \
  --wandb_eval_interval 10 \
  --device cuda:5

# ===== E6.3 长窗口 + stride-1（在 E6.1+E6.2 之上）=====
# v1.yaml 改动：stride: 20 -> 1（tw 用 CLI 传，不污染 yaml）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.train \
  --contact-method joint_and \
  --modal anysolev1_pos \
  --tw 100 \
  --batch-size 64 \
  --epochs 200 \
  --out-dir /data/fangyuxuan/projects/gait/results/AnySole/E6.3_longwin/checkpoints \
  --wandb_mode online --wandb_experiment_tag e6_3_longwin \
  --wandb_eval_interval 10 \
  --device cuda:6
```

## 评估

```bash
# ===== E6.1 / E6.2 / E6.3（pos 模型，命令同形；注意目录名=训练 out-dir）=====
# modal 不用传：eval 从 ckpt 内置 config 读取
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.1_pos/checkpoints/ckpt_last.pt \
  --write-bvh results/AnySole/E6.1_pos/predictions/eval_bvh \
  --device cuda:7

/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.2_smooth/checkpoints/ckpt_last.pt \
  --write-bvh results/AnySole/E6.2_smooth/predictions/eval_bvh \
  --device cuda:7

/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.3_longwin/checkpoints/ckpt_last.pt \
  --write-bvh results/AnySole/E6.3_longwin/predictions/eval_bvh \
  --device cuda:7

# ===== E3 基线（对照基准；与 E3_nocontact 存档 141.8mm 同口径）=====
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E3_base/checkpoints/ckpt_last.pt \
  --modal anysolev1 --contact-method joint_and \
  --write-bvh results/AnySole/E3_base/predictions/eval_bvh \
  --device cuda:4

# ===== E6.4 续写式评估（零重训；对任意 pos 模型）=====
# v1.yaml 改动：continuation: false -> true（评估完改回）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E6.1_pos/checkpoints/ckpt_last.pt \
  --modal anysolev1_pos --contact-method joint_and \
  --write-bvh results/AnySole/E6.1_pos/predictions/eval_bvh_cont \
  --device cuda:5

# ===== E6.5 暖启动评估（零重训；对 E3 已实测无收益，仅备查）=====
# v1.yaml 改动：warm_start: false -> true（评估完改回）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.eval \
  --ckpt results/AnySole/E3_base/checkpoints/ckpt_last.pt \
  --modal anysolev1 --contact-method joint_and \
  --write-bvh results/AnySole/E3_base/predictions/eval_bvh_warm \
  --device cuda:6

# ===== 单会话推理（导出可交互观看的 BVH）=====
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.infer \
  --ckpt results/AnySole/E6.1_pos/checkpoints/ckpt_last.pt \
  --modal anysolev1_pos --contact-method joint_and \
  --session S7013 --config-id VT2M \
  --device cuda:4
```

## 可视化

```bash
# 目录约定：<modal>_<contact-method> 纯字符串拼接到 results/AnySole/<dir>
# 各实验：E3_base -> --modal E3 --contact-method base
#         E6.1_pos -> --modal E6.1 --contact-method pos
#         E6.2_smooth -> --modal E6.2 --contact-method smooth
#         E6.3_longwin -> --modal E6.3 --contact-method longwin
# （E6.4/E6.5 的续写/暖启动导出在 eval_bvh_cont / eval_bvh_warm 子目录，
#   可视化脚本固定读 eval_bvh，可把对应 BVH 拷入 eval_bvh 再渲染，或按需扩展脚本）

# E3 基线
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal E3 --contact-method base

# E6.1（E6.2/E6.3 同理，换 --modal/--contact-method）
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal E6.1 --contact-method pos --force
CUDA_VISIBLE_DEVICES=5 python results_display/script/visualize_anysole.py --modal E6.2 --contact-method smooth --force
CUDA_VISIBLE_DEVICES=6 python results_display/script/visualize_anysole.py --modal E6.3 --contact-method longwin --force

# 指定单会话（渲染更快）
CUDA_VISIBLE_DEVICES=4 python results_display/script/visualize_anysole.py --modal E6.1 --contact-method pos --session S7013
```

## 验收口径（每步实验必须同表对比）

1. **数字**：eval 的 VT2M MPJPE / PA-MPJPE / traj_ATE + 训练面板的 tau0（head 健康度）；
2. **运动学（关键）**：导出 BVH 的逐关节误差、**脚踝相对高度均值/最大值**
   （踢腿判据：E3 为 -0.606/0.286，E4 为 +0.013/0.627）、帧间位移；
   测量脚本可复用 `z_note/probes/smoke_e6x_decoupling.py` 的 fk 读取方式。
