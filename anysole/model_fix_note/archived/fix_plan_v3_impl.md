> **已归档（2026-09-20）**：细节留档，不再维护。当前状态与结论见
> ../model_fix_note.md，当前基座命令见 ../command_manual.md。

# fix_plan_v3 实施修改方案（F5 回退 → 起点确认 → V3 分支实现）

> 2026-09-19。本文是 fix_plan_v3.md 的**代码级实施手册**：先外科手术式回退 F5
> （v2 §F5 作废、实现留档），再正式化 F4a / F2p4 两个起点，最后按 v3 §3 实现
> V3-1 ~ V3-3。行号以 2026-09-19 HEAD（b04a92b + 工作区）为准，实施后回填。
>
> **回退不能用 git revert**：F5 实现与 F4a/F2a/F0b 等混在同一提交 b04a92b
> （"F5修改前"），必须按触点清单逐文件移除。
>
> **用户裁定（2026-09-19，对三个决策点的答复）**：
> 1. V3-3 σ logit 方向——按方案 A 字面实现即可（分支型实验，初期不重要，
>    不设对照开关，跑完看结果）；
> 2. encode_stream LN——按实际情况修改（新增 stream_norm + strict-load 允许清单）；
> 3. **验收门槛全部删除**——不做"还没做就先提期望"的事，做一步看一步；探针/
>    wandb 面板仅作诊断观察。§2.7 起点一致性核对保留（逐步验证原则）。
>
> **批次重排（2026-09-19 用户裁定）**：f6_soft 标签已在
> `results_display/Test5_contact/f6_soft/` 生成（summary + gif），但**仍需清洗
> （数据质量问题，短期内无法完成）**——训练异常可能来自数据而非训练本身。
> 因此主线重排为 **V3-2 → V3-3 → V3-4（结构线：多头融合 + 门控）先行**；
> **V3-0/V3-1（数据线：标签 + 深监督）延后**。V3-3 不再等待 V3-1
> （β-NLL 自带监督信号、门控不依赖接触标签）；V3-2/V3-3 训练均用
> joint_and（λ_con=0，无接触监督）。已知风险：推迟 V3-1 可能让 T 流
> 欠训练（搭便车），门控价值被低估——跑完看结果，不作预判。
>
> **V3-2 首训爆炸诊断（2026-09-19，wandb xd6kcdk2，用户 kill 于 ep386）**：
> 健康期 ep70-250 **VT 54.0-54.7（优于 F4a 基线 55.5）、T 118.8-120.4（优于
> 123.8）——9 部位结构本身成立**。ep271 起轨迹损失先行（traj 0.027 vs 健康
> ~0.002）→ pose/kp 级联 → val VT 55.2→580.7；全程 nonfinite_skips=0 =
> **有限大梯度事件**（与 F0b_warm ep370 同型，NaN 守卫不触发，clip 5.0 压
> 不住）。kill 时参数已"愈合"（无 NaN、无异常权重）但功能停在更差盆地
> （VT 85.4）。回归家族 7 次长跑 2 次爆炸（F0b_warm、V3_2）→ 偏低概率训练
> 动力学事件，非 9 部位固有缺陷；重训判定随机/固有。**已加保护**：train.py
> 按 val/tau0 VT 保存 ckpt_best.pt（纯快照不动训练轨迹）。重训建议：命令与
> 首训一致；再炸则从 ckpt_best 续跑 + 试 clip 1.0（v2 §F5 处方）。

---

## 1. 现状盘点（已核实）

### 1.1 F5 集成触点全清单（回退对象）

| 文件:行 | 内容 | 处置 |
|---|---|---|
| models/model_v2.py:24 | `from ...part_decoder import PartDecoder` | 删 |
| model_v2.py:37-38 | `decoder="v1", gate_mode="gated"` 构造参数 | 删 |
| model_v2.py:49-53 | decoder 校验（part9 要求 foot_conv） | 删 |
| model_v2.py:72-83 | part9 分支（fusion=None / PartDecoder / 各头置 None） | 删，else 变唯一路径 |
| model_v2.py:111-131 | forward part9 分支（encode_stream 3-token 路径） | 删 |
| train.py:320-327 | `--decoder` arg | 删 |
| train.py:329-334 | `--gate-mode` arg | 删 |
| train.py:614-617 | config 传播 decoder/gate_mode | 删 |
| train.py:727-728 | 模型构造 gate_mode kwarg | 删 |
| train.py:429-430, 535-539 | _evaluate gates_sum 累积与 gate 日志 | 删 |
| eval.py:245-246, 254 | _load_model 读 decoder/gate_mode 并传构造 | 删 + F5 ckpt 显式拒绝（§2.3） |
| infer.py:215-216 | 同上 | 删 + 拒绝 |
| results_display/script/ridge_probe.py:86 | part9 分支 | 删（回退后 model 无 decoder 属性，死代码） |
| z_note/probes/smoke_f5_part.py | 检查 4/5 构造 AnySoleModelV2(decoder=part9) | 检查 1-3 保留（PartDecoder 独立构造仍可跑），4/5 删 + 标注归档 |
| losses.py:235 注释 | 提到 F5 part9 冷启动 | 注释改写（守卫本身保留，见 1.2） |

### 1.2 明确保留（不回退）

- **part_decoder.py**：v3 计划明确"实现留档、保留不删"→ 保留，顶部 docstring 加
  "已作废（fix_plan_v3.md 取代 v2 §F5），无调用方，仅留档"。
- **losses.py 工作区改动**（λ_con=0 完全跳过 BCE + nan_to_num/clamp 保险）：保留。
  V3 全部步骤同样受益，防 F5 式 NaN 重演。
- part_decoder.py 工作区改动（交叉注意力共享 k/v 修复）：随文件保留。
- f0_command_manual.md / model_fix_note.md 工作区改动、fix_plan_v3.md、
  f2_f2p4_structure.md、z_note/probes/probe_f4_tactile_path.py：全部保留。
- F5A/F5B 四个 ckpt：留档不删；回退后对其 eval 报明确错误（§2.3）。

### 1.3 起点核实（ckpt config 已 dump 验证）

| | F4a_footconv（主基座） | F2p4_combo（V3-4 移植基座） |
|---|---|---|
| 结构开关 | t_encoder=foot_conv, f2_repr=False, decoder 未设（=v1） | t_encoder=foot_conv, f2_repr=True |
| 训练口径 | 400ep、batch 256、lr 1e-3 cosine、grad_clip 5.0、config_probs [0.5,0.25,0.25]、joint_and、λ pose3/kp1/traj1/trec0.1/vrec0.1/con0 | 同左 |
| 参照数字（v3 §0 表） | VT2M 55.5/PA 23.0；V2M 63.5、踝脚 80.0、contact_f1 0.681；T2M 123.8/PA 40.3 | VT2M 63.4/23.1；T2M 99.7/PA 40.7 |

### 1.4 已有资产（比 v3 清单预期的少做）

- **f6 标签（V3-0 主体已实现）**：results_display/script/contact_methods.py 已注册
  motion_f6 / pressure_f6 / f6_soft（METHODS 表 + _f6_pipeline 自举标定），部分
  session 已落盘 contact_f6_soft.npy。**剩：全量 144 session 生成 + 验收门回填。**
- 通路探针：z_note/probes/probe_f4_tactile_path.py（正式化即可，§4.1）。
- dataset.py 已支持 `--contact-method f6_soft`（读 contact_f6_soft.npy 列 6:8，
  软值直通 contact_gt；缺文件时报错提示生成命令）。
- train.py 已有 loss 级 nonfinite 守卫（L863）；**缺 per-param 梯度检查（§3.3）**。

---

## 2. 阶段一：F5 回退（恢复 F4a/F2p4 干净起点）

按 §1.1 表逐项删改。要点：

**2.1** model_v2.py 回退后 `__init__` 签名 = pre-F5（d, tw, nhead, dropout,
pose_layers, tactile_input, tactile_direct, no_imu, v_input, t_encoder,
f2_repr），forward 恢复唯一路径（encoders → fusion → pose/traj/aux）。行为与
F4a/F2p4 训练时代码一致。

**2.2** train.py：`if model.pose_head is not None:` 拟合 pose stats 的守卫**保留**
（防御性无害）。`--decoder`/`--gate-mode` 删除后旧 F5 命令行 argparse 直接报错（预期）。

**2.3** eval.py / infer.py `_load_model`：删除 decoder/gate_mode 读取与构造参数，
构造前加显式检查：

```python
if str(saved_config.get("decoder", "v1")) == "part9":
    raise ValueError(
        "F5 part9 checkpoint is archived (v2 §F5 作废，fix_plan_v3 取代)；"
        "V3 代码不回载。基线用 F4a_footconv / F2p4_combo。")
```

（strict load 本身也会因 state_dict 不匹配失败；显式报错信息更清晰。）

**2.4** ridge_probe.py:86 删分支。**2.5** smoke_f5_part.py 按 §1.1 处理。
**2.6** part_decoder.py 顶部标注留档。

### 2.7 回退验收门（不通过不许进 V3）

1. smoke 回归全过：smoke_f4a_grid.py、smoke_f2_roundtrip.py（F4a/F2 路径不受影响）。
2. **F4a 复测**：回退后代码 eval + eval_protocol + ridge_probe 跑 F4a_footconv
   ckpt，VT2M 55.5±1.5 / V2M 63.5±1.5 / T2M 123.8±2（对齐 v3 §0 表）；F2p4 同法
   复测 63.4/23.1。任何漂移 → 先查回退遗漏，不进 V3。
3. eval 导出刷新（BVH/npz/metrics；重训前必须重跑 eval，既有经验）。
4. `--decoder part9` 报 argparse 错误；F5 ckpt 加载报 §2.3 明确错误。

---

## 3. 阶段二：代码卫生修复（与回退同批；v3 §4 修复项）

### 3.1 encode_stream 补 LayerNorm

- foot_encoder.py：新增 `self.stream_norm = nn.LayerNorm(dim)`；encode_stream
  返回 `self.stream_norm(seq.reshape(batch, tw, 3, self.dim))`。forward() 路径
  **一行不动**（out_merge → norm 契约不变，F4a 前向数值零影响）。
- **关键副作用**：新增 2 个 state_dict key → F4a/F2p4 ckpt 的 strict load 失败。
  处置（eval.py `_load_model` 与 infer.py 同步改）：

```python
_MISSING_OK = {"encoders.t_enc.stream_norm.weight",
               "encoders.t_enc.stream_norm.bias"}
missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
if unexpected or not set(missing) <= _MISSING_OK:
    raise RuntimeError("state_dict mismatch: missing=%s unexpected=%s" % (missing, unexpected))
```

  保持"准 strict"语义，唯一豁免 = 本次卫生新增 key。

> **实施记录（2026-09-19，逐步验证抓到的 bug）**：第一版实现把 `stream_norm`
> 直接加进 `encode_stream()` 返回，而 `forward()` 内部调用 `encode_stream()`
> → out_merge 输入被归一化，前向数值改变（F0b 精确复现、F4a/F2p4 的 T2M
> 漂移 +3.6/+8.8mm，V2M 零漂移——正好是 null-token 替换绕开编码器的配置）。
> 修复：共享主体拆为 `_stream_raw()`（forward 用，逐字节还原旧行为），
> `encode_stream()` = `stream_norm(_stream_raw())`（归档接口归一化）。修复后
> F2p4 协议复测 T2M 99.7 复现。

### 3.2 token 量级 smoke 断言

z_note/probes/smoke_token_scale.py（新）：F4a ckpt 前向取 |v_tok|/|t_tok| 均值断言
∈ [1,100]；断言 encode_stream 输出（修复后）与 forward 输出同量级（比值 ∈
[0.5,50]）。防"形状对量级错"再过检（F5 根因）。

### 3.3 NaN-grad 守卫（train.py，全步骤生效）

现有 loss 级守卫之外，backward 后、clip 前：

```python
grads_ok = all(p.grad is None or bool(torch.isfinite(p.grad).all())
               for p in model.parameters())
if not grads_ok:
    for group in optimizer.param_groups:
        group["lr"] = 1.0e-4
    optimizer.zero_grad(set_to_none=True)
    nonfinite_skips += 1
    global_step += 1
    continue
```

日志沿用 train/nonfinite_skips（或新增 nonfinite_grad_skips 单列）。防 F5B_gated
式"单 NaN 梯度经 clip_grad_norm_ 传染全模型"。

---

## 4. 阶段三：起点正式化

**4.1 通路探针正式化**：复制 z_note/probes/probe_f4_tactile_path.py →
results_display/script/tactile_path_probe.py（docstring 指向 z_note 为草稿），
输出口径保持 v3 §2.3 的 F / v_tok / t_tok1 三行天花板表；加 `--f2-repr` 适配
（V3-4 起在 F2p4 基座使用；F4a 基座用默认 False）。

**4.2 基线复测回填**：与 §2.7-2 同一次跑，F4a / F2p4 的 fseries + ridge + 探针
数字回填 fix_plan_v3.md §0 表。

**4.3 T/V 梯度比**：训练循环加 grad-ratio 统计（t_enc/v_enc 参数梯度平方和之比，
随 §5.1 同时落地），**仅作诊断观察，不设验收门槛**（用户裁定：做一步看一步）。

---

## 5. 阶段四：V3 分支实现（基座 = F4a）

### 5.0 V3-0 补完（零训练）

- 全量生成：`python results_display/script/contact_methods.py --methods f6_soft`
  （144 session，train+val 全量落盘 contact_f6_soft.npy + sidecar JSON）。
- 验收门 = v2 §F6a 四项一字不改（4 反例翻转 / 接触率先验 / 压力互证 / 事件级时差）
  + 拖步患者豁免，结果回填 fix_plan_v3.md §V3-0。
- 数据集冒烟：train/val 各抽 3 session 确认 contact_gt 软值 ∈ {0.05,0.3,0.7,0.95}。

### 5.1 V3-1 流级深监督（tag `v31_ds`；结构零改动）

新文件 `anysole/models/ds_heads.py`：

```python
class StreamDSHeads(nn.Module):
    # t_enc: 2 层 TransformerEncoder，输入 t_tok (B,tw,d) = foot_conv forward() 输出
    # t_pose:   Linear(dim, 48)  # 关节 15-18,19-22 的 6D（标准化空间，set_stats 注入）
    # t_contact: Linear(dim, 2)   # BCE 目标 = f6_soft 软值
    # t_traj:   Linear(dim, 3)    # v1 口径 vel_gt
    # v_enc: 2 层 TransformerEncoder，输入 v_tok
    # v_contact: Linear(dim, 2)
    # v_pose:   Linear(dim, 48)（对照开关默认关，构造参数传入）
    # set_stats(mean, std)：收 138 维全量 stats，内部取 8 关节子集
```

- model_v2.py：`ds=True`（CLI `--ds-heads`）时构造 ds 头；ds 启用时 forward 附加
  `"ds": {...}` 输出（v_tok/t_tok 由 forward 内部传入，不改主路径）。
- losses.py：新增 L_ds_t_pose（标准化 MSE）、L_ds_t_contact（BCE f6_soft）、
  L_ds_t_traj（MSE vs vel_gt）、L_ds_v_contact（BCE）；**流存在性掩码**：T 头项
  仅对 config ∈ {VT,T} 行计算，V 头项仅对 {VT,V} 行；权重 `--lambda-ds-t 0.3
  --lambda-ds-v 0.1`（默认 0 = 上一步行为）。
- train.py：新 flags + wandb 分量日志（train/loss_ds_t_pose 等）+ **T/V 梯度比日志**
  （train/grad_ratio_t_v：backward 后 t_enc/v_enc 参数 grad 平方和之比）。
- 量级规则：首 epoch 核对 train/loss_ds_t_* vs train/loss_pose，加权后落入
  L_pose 的 10–30% 带（§2.2），不符则调 λ 重跑。
- 命令（基座 F4a；batch 256 × 5 步/epoch × 740 = 3700 步）：

```bash
python -m anysole.train --modal anysolev2 --contact-method f6_soft \
  --t-encoder foot_conv --ds-heads --lambda-ds-t 0.3 --lambda-ds-v 0.1 \
  --epochs 740 --init-from results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt \
  --grad-clip 5.0 --out-dir results/AnySole/V3_1_ds/checkpoints \
  --wandb_experiment_tag v31_ds --wandb_eval_interval 10 --loss-cap 1.0 --device <gpu>
```

- 跑完看结果（做一步看一步）：观察 T2M / V2M / VT2M / contact_f1 与 ridge 探针
  t_tok1 下肢天花板（防标签泄漏）的走向，再决定下一步；不预设门槛。

### 5.2 V3-2 9 部位最小移植（tag `v32_part9min`；可与 V3-1 并行）

> **代码已实现（2026-09-19）**：`types.py` 冻结 PART_NAMES/PART_JOINTS
> （eval_protocol/part_decoder 改 import，单一真源）；`pose_head.py` 加
> `n_parts` 参数（9 = 9 部位查询 + 9 个 unembed 头 + `part_place_idx` 散射
> buffer 拼回关节序，r_leg/r_foot 换序已覆盖；n_parts=3 路径逐字节不动，
> F4a 复测 55.494/63.539/123.817 与重构前完全一致）；`--pose-parts` /
> `--lr-warmup-frac` flags；eval/infer 读 ckpt config 的 pose_parts。
> smoke_v32_parts.py 四查全过（warm-start 312/336 键复制，跳过键 = 预期
> 9 个形状变化键）；1-epoch 训练冒烟通过（epoch1 冷启动 VT2M ~207mm，
> 部位头全新属预期，对照 F5 全新建 decoder 的 859mm）。命令手册 =
> `archived/v3_command_manual.md` §1。

- types.py：新增冻结 `PART_NAMES` / `PART_JOINTS`（内容 = eval_protocol.py:69-77
  现定义）；eval_protocol.py 与 part_decoder.py（归档）改 import 自 types.py，
  消除三分重复。
- pose_head.py：新增 `n_parts=3` 参数。n_parts=9 时：
  - query (1,tw,9,dim)；group_emb Embedding(9)；part_ids = arange(9)；
  - 输出：9 个 per-part unembed 头（宽度 = len(PART_JOINTS[p])×6），按关节序散射
    拼回 138——预注册 `part_scatter_idx` buffer（(138,) 长整型，位置 j×6..j×6+5 ←
    part p 的第 k 个关节块）；
  - decoder 层、pose stats、标准化路径、时间 PE **全部复用**；n_parts=3 路径
    逐字节不动（默认关 = 上一步行为）。
- model_v2.py：`pose_parts` 参数；train.py `--pose-parts`（choices 3/9，默认 3）；
  ckpt config 存 `pose_parts`。
- warm-start 自 F4a：query (1,20,3,256)→(1,20,9,256)、group_emb 3→9、
  out_body/left/right→out_part_* 按名字+形状自动跳过（init_from_checkpoint 既有
  机制，无需改）；decoder 6 层 + pose stats 全复用。
- 命令 = V3-1 命令去 ds 开关、加 `--pose-parts 9`（contact-method 可回 joint_and，
  本步不动接触监督）。跑完对照 F4a 看结果再定；显著劣化则弃 9 组粒度、V3-3 降级
  在 3 组上做。
- smoke：z_note/probes/smoke_v32_parts.py——n_parts=3 vs 9 前向形状、散射拼回顺序正确性
  （恒等输入验证）、warm-start 加载 key 报告。

### 5.3 V3-3 σ 软门控（tag `v33_sigma_gate`；依赖 V3-1、V3-2 都过门）

pose_head.py 新增 `gate="none"|"sigma"`（默认 none = V3-2 行为）+ 自定义
`_GatedLayer`（仅 gate="sigma" 时替换 decoder 层）：

```
z  ← self-attn(9×tw 查询)                    # 部位+时间上下文 = 条件先验宿主（机制 C）
e_V ← cross-attn(z, F, mask=T 源 token)      # F 前 20 token 可见
e_T ← cross-attn(z, F, mask=V 源 token)      # F 后 20 token 可见
σ_V, σ_T ← MLP_σ(z) + b_p（9×2 部位偏置）     # b_p 初始化沿 gate_bias 思路：
                                             # 根/腿/脚 b_T=+1，躯干/头/臂 b_V=+1
g ← softmax([σ_V, σ_T, τ_p])                 # τ_p (9,1) 可学习，init 0（先验路径初始关闭）
z ← z + g_V·e_V + g_T·e_T                    # g_∅：不注入模态信息，保留 z = 条件先验
FFN
```

- F 的模态掩码视图：fusion.forward 拼接顺序 = cat([v_tok, t_tok]) → 前 tw=20 为
  V 源、后 20 为 T 源；key_padding_mask 实现（cross_v 掩 20:40，cross_t 掩 0:20）。
  **fusion 本身不动**（v3 §2.4：门控必须作用在融合后 F 上，不可绕过融合）。
- σ 监督（β-NLL，β=0.5，显式不依赖掩码）：r_p = 该部位标准化空间误差（主输出
  stopgrad）。**按方案 A（字面）实现**：σ_V/σ_T 分别对 r_p 做 β-NLL，各自按流
  存在性掩码；语义 = "σ 标定到残余量级，softmax 里 σ 大 = 该流证据强"。方向对错
  属分支实验，跑完看结果再定（用户裁定，不设对照开关）。
- log σ 截断 [−6,3]（softmax 前 clamp）；前 10% 步 σ MLP 输出固定 = 1（等价纯
  bias 路由）。
- train.py：`--gate sigma`；结构步骤 warmup 前 5% 步（§2.2）；日志 per-part
  per-config 门控（恢复 _evaluate 的 gates 统计，作诊断观察）。
- 跑完看结果：重点看 g_∅/g_T 行为与三配置指标走向（v3 §V3-3 的失败处置思路留作
  参考：g_∅ 不激活→提前 V3-5；σ 塌缩→查 r_p / 调 β）。
- smoke：z_note/probes/smoke_v33_gate.py——掩码方向正确（V 掩后 e_V 读不出 T 侧信息）、
  g 行和 = 1、β-NLL 梯度回传到 σ MLP、σ 固定期行为。

### 5.4 鲁棒性验证集（V3-0 起可用）

- eval_protocol.py 加 `--degrade {none,v_drop}`（v_drop_frac ∈ {0.2,0.4}）：val
  每序列随机取连续 20–40% 帧，V_feat 置零（v1 路径无逐帧 mask，置零是受支持
  代理；T 丢段同构实现，后续步再开）。
- 报告：V config 接触 F1 掉幅（vs 无退化）、VT2M 掉幅单列，作诊断观察（v2 §2
  定义口径保留，不设门槛）。

### 5.5 后续（触发式，不进本轮首批）

- V3-4：V3-1~3 开关全量 + `--f2-repr` + warm-start 自 F2p4_combo；机制代码零
  改动（深监督头/门控天然支持 4 维轨迹与 tilt，按 f2 口径适配输出维度）。
- V3-5：按 V3-3 门控结果触发（g_∅ 健康→正式 F8a；不激活→轻量 AMASS finetune）。
- V3-6：F6b 三损失（L_contact/L_skate/L_height，v2 §F6b 全文不变，前 10% 步
  线性升权）。
- F8a AMASS→BVH23 转换 + refiner 预训练：与 V3-1~3 全程并行（只依赖数据转换
  脚本，不依赖主模型）。

---

## 6. 实施顺序（批次）

| 批次 | 内容 | 产出 |
|---|---|---|
| 1 | 阶段一 F5 回退 + 阶段二卫生修复 + §2.7 回退验收（F4a/F2p4 复测） | **已完成（2026-09-19）**，复测表见 §8 |
| 2 | **V3-2 9 部位最小移植（结构主线，先行）** | **代码完成（2026-09-19）**，训练待启动（命令见 v3_command_manual.md §1） |
| 3 | **V3-3 σ 软门控（不再等待 V3-1）** | v33_sigma_gate |
| 4 | **V3-4 f2 移植（F2p4 基座重跑）** | v34_f2 |
| 5 | V3-0 全量标签 + 验收门 + **标签清洗**（数据线，延后） | contact_f6_soft 清洗版 |
| 6 | V3-1 流级深监督（数据线，标签清洗后） | v31_ds |
| 7 | V3-5 / V3-6 按门控与标签状态触发 | — |

## 7. 纪律检查表（v3 §2.1/2.2 逐项映射）

- 新开关默认关（--ds-heads / --pose-parts 9 / --gate sigma 默认 = 上一步行为）✓
- 每步只动一个组件；失败弃开关、ckpt 留档、代码不回改 ✓
- 短预算 3700 步 = batch 256 × 740 epoch；结构步骤 warmup 前 5% + NaN-grad 守卫 ✓
- 新损失首 epoch 加权 ≈ L_pose 的 10–30%，wandb 分量核对 ✓
- 每步跑完同口径看结果：eval fseries + ridge 探针表 + 训练面板 + 门控日志（诊断
  观察，不设门槛）✓

## 8. 批次 1 回填（2026-09-19，回退后代码复测）

smoke：smoke_f4a_grid / smoke_f2_roundtrip / smoke_f5_part（检查 1-3）/
smoke_token_scale 全过。`--decoder part9` argparse 报错；F5 ckpt 加载报
"F5 part9 checkpoint is archived" 明确错误。

| 基线 | 配置 | §0 记录 MPJPE/PA | 复测 MPJPE/PA | 结论 |
|---|---|---|---|---|
| F4a_footconv | VT2M | 55.5 / 23.0 | 55.49 / 22.96 | ✓ |
| | V2M | 63.5 / contact 0.681 | 63.54 / 0.681 | ✓ |
| | T2M | 123.8 / 40.3 | 123.82 / 40.31 | ✓ |
| F2p4_combo | VT2M | 63.4 / 23.1 | 63.45 / 23.12 | ✓ |
| | V2M | 68.7 | 68.68 | ✓ |
| | T2M | 99.7 / 40.7 | 99.71 / 40.71 | ✓ |
| F0b_seed2 | VT/V/T | 63.7 / 66.3 / 116.2 | 63.68 / 66.26 / 116.20 | ✓ |

BVH/npz/metrics 已按记忆要求全部刷新（eval 重跑过）。起点 = 回退后代码 + 上表数字；
F4a 为 V3-1/V3-2 基座、F2p4 为 V3-4 移植基座，warm-start 命令见 §5.1/§5.2。
