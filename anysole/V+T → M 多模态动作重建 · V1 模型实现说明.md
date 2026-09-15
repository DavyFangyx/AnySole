# V+T → M 多模态动作重建 · V1 模型实现说明

约定:`B`=batch,`Tw`=窗口长度=20,`d`=模型隐维(默认 256),公共帧率 40 Hz。
三条推理支路共用同一套权重:`{V,T}→M`、`{V}→M+T`、`{T}→M`。
姿态用**扩散**、根位移用**回归**,两头解耦。

---

## 0. 总体结构

```
                 ┌── V-Enc ──┐
JPG→HRNet(2048)──┤           ├─► 融合 Transformer ─► fused memory F ─┬─► 姿态扩散头 ─► SMPL 旋转 (6D)
                 │  (冻结)   │        (自注意力)                     ├─► 轨迹回归头 ─► 根位移 transl
raw 48×2 + 物理──┤── T-Enc ──┘                                        ├─► T 重建头  ─► pressure_hat / contact
                                                                     └─► V 重建头  ─► HRNet 特征_hat
缺失模态 → 该路 token 全部换成可学习 null token
```

- V、T 各自编码成 token 序列,拼在一起过融合 Transformer 得到 `F`。T 路先在特征维拼接 `T_raw` 与 `T_phys`,只生成一组 T token。
- `F` 分出四个头:两个是真正输出(姿态扩散、轨迹回归),两个是辅助监督(T 重建、V 重建),辅助头是让缺模态支路拿到跨模态信息的机制。
- 训练时按概率把某一路输入置空(null token),这就是缺失模态鲁棒性的来源。

---

## 1. 数据与输入维度(参考：Baselines/MotionPRO)

真正进训练的是**对齐 + 重采样到 40 Hz** 之后的窗口序列,不是原始 MP4/BVH/CSV。三模态对齐后按同一帧索引取用:`feature[t] ↔ pressure[t] ↔ SMPL[t] ↔ contact[t]`。质量标为 C/D 的 session 跳过;窗口内含 fake 帧则整窗丢弃。

### 视频 V
- 四机位(top/left/front/right),**只用 front(相机 3)**。
- 原始 1624×1240、H.264、实测 ~38.1–38.6 fps(非固定 40)。
- 训练**不读 MP4**:取该机位 JPG,按对齐区间最近邻抽到 40 Hz → crop 256×256 → 冻结 HRNet 提特征。
- **训练输入 = HRNet 特征 2048 维 + CLIFF 归一化 `bbox_info` 3 维,每帧共 2051 维**。
- 窗口张量:`V_feat` **(B, 20, 2051)**。
- 参考:HRNet https://arxiv.org/abs/1902.09212
{
  代码：one/ReferenceWorks/deep-high-resolution-net.pytorch
}

### 触觉 T
- 鞋垫 4×12=48 点/脚,左右各一路 CSV(`frame_idx, t_us, 1..48`),左右脚分开、原始帧率不齐。
- **决定:直接用 48 点原始向量,不做高斯平滑升采样成稠密图**(插值不增信息)。
- 原始压力(插到 40 Hz):左 48 + 右 48 → 每帧 96 维 → `T_raw` **(B, 20, 96)**。
- 附加物理特征(从原始压力导出,与 raw 在特征维拼接后共同编码):
  - CoP 左/右各 (x,y) → 4 维
  - 总力 左/右 → 2 维
  - 接触包络、压力梯度 → 若干维(窗口/邻域大小当超参,先随手定)
  - 汇总 `T_phys` **(B, 20, Kp)**,`Kp` 由上面几项拼出。
- 接触 GT:`contact.npy` (T,10),取左右脚两列 → `contact_gt` **(B, 20, 2)**。
- `fake_mask` 只用于筛窗,不进网络。

### 动捕 M(目标,非输入)
- 预处理已从 BVH(23 关节,120 Hz)裁切插值到 40 Hz 并**转成 SMPL**:
  - `smpl.npy`:`betas`(10)、`global_orient`(T,3)、`body_pose`(T,69)、`transl`(T,3)
  - `keypoints.npy`(T,24,3),**只用前 22 个**
- **扩散目标(姿态头)**:把 `global_orient`(1)+`body_pose`(23)=24 个关节的 axis-angle → **6D 旋转** → `pose_gt` **(B, 20, 144)**(24×6)。
  - 必须转 6D:axis-angle 在 ±π 附近不连续,直接当扩散目标会学崩。参考 Zhou et al. 2019 https://arxiv.org/abs/1812.07035
{
方法： 采用6D连续表示替代axis-angle

**原理：**根据本文4.2节提出的Case 3，SO(3)可通过6D连续表示实现无拓扑间断。具体方法是：
**编码：**将3×3旋转矩阵的前两列（6个元素）作为表示向量
**解码：**通过Gram-Schmidt正交化过程恢复第三列，公式如下：
\mathbf{b}_1 = \mathcal{N}(\mathbf{a}_1),\quad \mathbf{b}_2 = \mathcal{N}(\mathbf{a}_2 - (\mathbf{b}_1 \cdot \mathbf{a}_2)\mathbf{b}_1),\quad \mathbf{b}_3 = \mathbf{b}_1 \times \mathbf{b}_2
其中 \mathcal{N}(\cdot) 为归一化函数。

**目标空间转换：**将扩散模型的目标从axis-angle向量改为6D向量
**损失函数设计：**
直接在6D表示空间计算L2损失（本文5.1节验证有效）
或转换为旋转矩阵后计算 geodesic 损失（式12）：
L_{\text{angle}} = \cos^{-1}\left(\frac{\text{Tr}(\mathbf{M}\mathbf{M}'^{-1}) - 1}{2}\right)
**采样后处理：**生成6D向量后通过解码公式转换为旋转矩阵，确保物理有效性
}
- **回归目标(轨迹头)**:根位移前向差分 → `vel_gt` **(B,20,3, m/s)**，并积分为窗口相对轨迹 `trans_gt` **(B,20,3)**；窗口 anchor 单独保存。
- `betas`:v1 用 GT(每人常量),不逐帧扩散,只在算 FK/关键点损失时用。
- 关键点 GT:`keypoints.npy[:, :22]` → `kp_gt` **(B, 20, 22, 3)**。

> global_orient 归到姿态头(它是旋转)、transl 归到轨迹头(它是位移),这正好落在"旋转→扩散、位移→回归"的解耦上。global_orient 也可以改挂轨迹头,列入测试项。

---

## 2. 编码器

| 模块 | 输入 | 结构 | 输出 |
|---|---|---|---|
| V-Enc | `V_feat` (B,20,2051) | Linear(2051→d) + 时间正弦位置编码 | V token (B,20,d) |
| T-Enc | `cat(T_raw,T_phys)` (B,20,96+Kp) | Linear((96+Kp)→d) + 时间 PE | T token (B,20,d) |

- 每路 token 加一个**模态类型嵌入**(V / T 两种)。
- HRNet 冻结,V-Enc 只是投影 + 位置编码(v1)。解冻/换 SAM 3D Body 列入测试项。
- T-Enc v1 用 flat Linear;按脚 4×12 上小卷积列入测试项。

---

## 3. 融合

- 把当前**可用**模态的 token 沿序列拼接:`[V(20), T(20)]`(缺失路整段换成该路的**可学习 null token**)。
- 过 N 层融合 Transformer(自注意力,N=4 起步),token 带时间 PE + 模态嵌入。
- 输出 fused memory `F`:**(B, 40, d)**(两路各 20)。给下游各头做 cross-attn 的 key/value。

---

## 4. 生成头

### 4.1 姿态扩散头(去噪器,预测 x0)
- 输入:加噪姿态 `x_τ`(B,20,144)、扩散步 `τ`。
- **主体分区编码**:24 关节按 {左腿 / 右腿 / 其余} 分组,每组各自 Linear 投影,加时间正弦 PE + 关节/组嵌入。
- **Mean/Std 标准化**:6D 姿态按维做 mean/std 标准化(训练集一次性拟合、冻结,随 checkpoint 保存;std 下限 1e-2)。参考 MDM/RoHM 对 motion 表征的标准化。
- **Timestep 门控(prepend token)**:`t_emb = MLP(τ)` 作为序列第一个 token 拼在 `[t_emb; x_tokens]` 前面,每层自注意力全程可见——按噪声水平门控对 `x_τ` 的使用(低 τ 复制 x_τ、高 τ 忽略 x_τ 只依赖 F)。参考 MDM/RoHM 的 prepend 方式。
- **多层 decoder(6 层起步)**:每层 = pre-LayerNorm 自注意力 → pre-LayerNorm 交叉注意力 → pre-LayerNorm FFN,激活 GELU。
- **交叉注意力(跨层重复)**:每层 Query=姿态 token(+t token),Key/Value=`F`(缺失模态那段是 null token)。
- 去掉 t token → 投影回姿态空间 → 反标准化 → `x0_hat`(B,20,144)。
- **预测 x0 而非 ε**:方便在预测的干净姿态上直接加 FK 类几何损失(接触、关键点)。参考 MDM https://arxiv.org/abs/2209.14916
{
  代码：one/ReferenceWorks/motion-diffusion-model
}
- 扩散:DDPM,训练 1000 步、cosine β 调度;采样 DDIM ~50 步 https://arxiv.org/abs/2010.02502
{
  最常用的是 Hugging Face diffusers
  DDIM 已经并入 diffusers，通过 DDIMPipeline 就能调用：
  python
  from diffusers import DDIMPipeline

  ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32")
  image = ddim(num_inference_steps=50).images[0]
  image.save("ddim_generated_image.png")

  如果只是想把别的模型的采样器换成 DDIM，用 DDIMScheduler 替换即可：
  python
  from diffusers import StableDiffusionPipeline, DDIMScheduler

  ddim = DDIMScheduler.from_pretrained("模型ID", subfolder="scheduler")
  pipeline = StableDiffusionPipeline.from_pretrained("模型ID", scheduler=ddim)

  另外 diffusers 里还有 DDIMInverseScheduler，是 DDIMScheduler 的反向版本，实现主要参照 Null-text Inversion 那篇里的 DDIM inversion 定义，做图像编辑、反演回噪声隐变量时会用到。 
}
- 分区编码 + 时间嵌入的必要性:上下半身对触觉依赖不同、重建难度不同;时序波形能压缩 T→M 的病态(跑步 vs 深蹲的时域曲线不同)。参考 Step2Motion https://github.com/JLPM22/Step2Motion
{
  代码：Baselines/Step2Motion
}

### 4.2 轨迹回归头
- 结构:2 层 Transformer encoder,输入 `F`(池化或直接序列)→ 每帧根速度 `v_hat`(B,20,3, m/s)→ `trans_hat = cumsum(v_hat) / FPS` (米)。
- 预测速度再积分,比直接回归绝对 transl 更稳,也天然对上累积位移惩罚。
- 窗口内 `vel_gt[0]=0`、`vel_gt[t]=(trans[t]-trans[t-1])*FPS`; `trans_gt=cumsum(vel_gt)/FPS` 是相对窗口轨迹。`trans_anchor` 不进入轨迹损失；评估/长序列拼接及世界坐标 FK 辅助项使用它恢复全局坐标。
- 不同输入配置下根位移的最佳来源不同(V 有深度漂移、IMU/压力抗漂移),条件集随可用模态变化。config 自适应条件列入测试项;v1 先统一吃 `F`。
- 解耦参考:RoHM(CVPR 2024,TrajNet+PoseNet)。
{
  代码：one/ReferenceWorks/RoHM
}

---

## 5. 辅助头与损失

| 头 | 输入→输出 | 作用 |
|---|---|---|
| T 重建头 | `F` → `pressure_hat`(B,20,96) | 重建归一化后的原始触觉压力 `T_raw`(左48+右48);强迫共享隐变量在只看 V 时也编码接触结构 |
| V 重建头 | `F` → `Vfeat_hat`(B,20,2051) | 重建 HRNet 特征与 `bbox_info`;强迫隐变量在只看 T 时携带视觉全局信息 |

### 损失表(v1)

| 损失 | 定义 | 激活配置 | 权重 |
|---|---|---|---|
| L_pose | MSE(`x0_hat`, `pose_gt`) | 全部 | 1.0 |
| L_traj | `λ_v·MSE(v_hat,vel_gt) + λ_d·mean_{Δ∈{2,4,8,19}}[(1/Δ)·MSE(D_Δ(trans_hat),D_Δ(trans_gt))]` | 全部 | λ_traj |
| L_Trec | MSE(`pressure_hat`,`T_raw`) | 全部 | λ_T |
| L_Vrec | MSE(`Vfeat_hat`,`V_feat`) | 仅 `{T}→M` | λ_V |
| L_con | FK(`x0_hat`,GT betas)→脚部速度/高度→软接触 → 与 `contact_gt` 的 BCE | 全部 | λ_con |
| L_kp | FK(`x0_hat`)→关节位置(22) 与 `kp_gt` 的 MSE | 全部 | λ_kp |

- 多尺度位移项覆盖 2/4/8/19 帧，`Δ=19` 约束整窗端点位移；`1/Δ` 做尺度归一化，避免长跨度项主导。
- L_con 是把"压力→接触→抗漂移"落到损失上的地方,v1 就带简单版,别等 v2,否则看不出压力的贡献。`contact_gt` 直接用 `contact.npy`,不用自己伪标注。
- T 重建在 T 是输入时是平凡拷贝(损失自然很低),在 T 缺失时才是真正的跨模态预测,不用额外开关。重建目标始终是 `T_raw` 的 96 维归一化值,不重建 `T_phys`。

---

## 6. 模态 dropout 与训练

- **按 config 类别采样**(比独立 Bernoulli + 拒绝更干净),默认比例可由训练配置 `config_probs=[VT,V,T]` 调节:
  - `{V,T}` : 0.50 (默认)
  - `{V}` (丢 T) : 0.25 (默认)
  - `{T}` (丢 V) : 0.25 (默认)
  - **v1 不留全空样本**(至少保一路输入)。全空样本 + CFG 引导采样列入测试项。
- 丢弃 = 该路 token 换 null token,其余不变。
- 优化器 Adam,batch 256,lr 1e-3(融合 Transformer 若不稳降到 1e-4)。
- **联合训练**所有头(不再像 Step2Motion 那样两网分开训);500/200 epoch 只是起点参考。
- 无需全模态 teacher:dropout + T/V 两个重建损失已在灌跨模态信息。teacher 蒸馏列入测试项,只在 T→M 明显塌或打不过纯 P2M 时再加。

---

## 7. 推理

- 按可用模态置 null,其余照常前向。
- 姿态:从噪声 DDIM ~50 步条件采样。
- 轨迹:直接回归 + 积分。
- 接触/压力:`{V}→M+T` 用 T 重建头输出;有 T 输入时直接用输入。
- 长序列:滑动窗口拼接。v1 先简单拼(非重叠或朴素衔接);重叠区当 inpainting 已知条件硬约束列入测试项。

---

## n. 改进与测试项(v2+)

跑通 v1 后按需开启,每项都是独立分支:

1. **CFG 引导采样** — 训练留 ~5% 全空样本学无条件先验,推理按模态分别调 guidance scale(如 V+T 时调大 T 引导强化接触)。Ho & Salimans 2022 https://arxiv.org/abs/2207.12598
2. **全模态 teacher 蒸馏** — 独立预训 `{V,T}→M` 冻结当锚点,隐层蒸馏,防 T→M 塌向单模态解。
3. **T→M 轨迹也上扩散** — 只在这条最病态的支路试根轨迹用扩散(多假设),对比轨迹回归。RoHM 式双扩散。
4. **触觉分区 vs 全局输入** — 对比 {纯物理特征 / 分区原始 / 分区原始+物理},只在 T→M 支路比,看下肢 MPJPE 和根轨迹误差。
5. **窗口重叠 inpainting** — 相邻窗重叠区当扩散已知条件,消拼接处轨迹跳变。
6. **T-Enc 形态** — flat Linear vs 按脚 4×12 小卷积;以及是否值得升采样稠密图(默认判为不值)。
7. **V 编码器** — HRNet 冻结 → 解冻微调 → 换 SAM 3D Body / CameraHMR。SAM 3D Body https://arxiv.org/abs/2602.15989
8. **生成器风格** — 姿态/轨迹各自 扩散 vs 回归 vs 隐空间扩散(先过 VAE) 的组合对比。
9. **global_orient 归属** — 挂姿态头 vs 挂轨迹头。
10. **config 自适应轨迹条件** — 轨迹头按可用模态换条件集,而非统一吃 F。
11. **高频接触抗混叠核查** — 若 T→M/抗漂移不及预期,回查 40 Hz 降采样是否把 heel-strike/toe-off 抽没了(窗口内 max/能量聚合再降采样)。

### 参考工作（这些你不用找）
- MMVP(视觉+压力,抗脚漂/全局平移)https://arxiv.org/abs/2403.17610 · https://metaverse-ai-lab-thu.github.io/MMVP-Dataset/
- MotionPRO(大规模压力+RGB+光学;压力分支小核解码+长短时注意力)https://arxiv.org/abs/2504.05046
- Step2Motion(鞋垫压力+IMU→locomotion;分区、时间嵌入、IMU-only 位移、累积惩罚)https://github.com/JLPM22/Step2Motion
- Pressure2Motion(地面压力+文本→动作,分层扩散;确认 T→M 病态)https://arxiv.org/abs/2511.05038
