# Agent E 任务书：FPP-Net 与 PoseTransOpt 语义保真适配

> 前置：Agent A 已冻结 shared facts/representations/frontend 契约。  
> 目标：恢复 FPP-Net 原生顶点级接触语义，并使 FPP-Net → PoseTransOpt 全链用 frame id 显式对齐。

## 1. 语义红线

- FPP-Net 是 V→T/顶点接触模型，不使用 AnySole 脚级 f6 标签。
- PoseTransOpt 原生消费 FPP-Net 顶点接触并分 toe/heel 四区，不得退化为整脚开关。
- 不修改 FPP 网络结构、loss 权重和 PoseTransOpt 优化目标。
- 跨模态对齐发生在 adapter/workspace 层，不在 dataset 内按数组位置猜测。

## 2. F0/V5：恢复原生 contact GT

1. FPP dataset 恢复通过 `InsoleModule.press2Cont` 生成格级二值接触。
2. 恢复 `getVertsPress(soft=False)` 生成 192 维顶点级二值 GT。
3. 删除 dataset 对 `contact_f6_soft.npy` 的强制依赖。
4. 不再把脚级标量广播到 31×22 网格。
5. 训练 target shape 和接触头结构不变。

验收必须统计：

- 左 toe/heel、右 toe/heel 四区的独立变化率；
- 同脚内存在不同顶点值的帧比例；
- 不得出现所有脚内顶点均由同一脚级值广播的数据链。

## 3. V3：pixel_weight

FPP-Net 的原始物理触觉来源与私有转换链必须明示为：

```text
raw pressure_left/right.csv (48/脚)
→ shared/facts pressure_48.npz (2×4×12)
→ shared/representations/tactile/mmvp_31x11/<version>
→ FPP-Net 只读消费
```

不得重新生成第二份私有 31×11 树，也不得经旧 MotionPRO `pressure.npz` 回推 4×12。D_Test4 已确认 4×12→31×11 几何转换正确，本 Agent 只恢复 FPP 原生归一化/contact 语义。

- `pixel_weight` 必须与 FPP 实际消费的 31×11 insole mask 和 `pixel_num` 定义一致。
- 不得直接用原始 96 格和代替映射后 mask 总量。
- `sub_info` 记录 weight 的单位、计算域、standing frame id 和源 hash。
- 生成前后报告 sigmoid 输入、饱和率和动态范围。

## 4. V1：frame id join

所有链路文件必须保留 shared frame id：

- FPP temporal-5 中心帧；
- RTMPose keypoints；
- CLIFF pose/betas/bbox；
- FPP `pred_contact_smpl`；
- PoseTransOpt 输入与输出。

PoseTransOpt adapter 以 frame id 做 inner join，并输出：

- 每路总帧数；
- join 后帧数；
- 缺帧列表及原因；
- 被 fake/invalid 剔除的帧；
- 连续片段边界。

禁止使用 `[2:-2]` 后直接按位置 zip 不同模态。

## 5. V2：基于唯一受试者 mask 的 CLIFF 单人契约

Agent E 不再实施额外多人 track 算法，直接消费 Agent A 的 SAM3.1 唯一受试者 frontend：

1. 每个 canonical frame 取 human mask 重算 bbox；
2. CLIFF HPS 只处理该 bbox，不保存其他检测人；
3. NPZ 必须保存 frame_id、mask source/time error/bbox provenance；
4. pose/betas/bbox 行数必须与 canonical frame 数相等；
5. betas 平均和平滑只在这个唯一受试者序列内进行；
6. 现有 133 份行数异常 NPZ 必须重建，不得因文件存在而跳过。

如任一帧无有效 mask 或超过 20ms 容差，该帧标记 invalid，不回退到多人 YOLO 行。

## 6. V4/V6/V7 处置

- 先用上游 VP-MoCap 代码/本地原始备份证明行为归属。
- 若为上游原生 flip、主点或 confidence 行为，正式 baseline 保留并记录。
- 若为本项目更换 HALPE/标定格式后造成的排列错配，只在 adapter 中转回上游期望排列，不修改 loss。
- 不得为了提高结果而直接关闭上游原生 augmentation。

## 7. 路径和输出

```text
AnysoleWorkspace/model_inputs/FPP-Net/<adapter_version>/...
AnysoleWorkspace/model_inputs/PoseTransOpt/<adapter_version>/...
AnysoleWorkspace/work/VP-MoCap/<run_id>/
├── fpp_predictions/
├── pose_optimization/
└── artifact.json
```

最终 V2T/V2M prediction 仍分别导出到：

```text
results/baselines/FPP-Net/predictions/
results/baselines/VP-MoCap/predictions/
```

## 8. 写入范围

允许：

- `Baselines/VP-MoCap/**`；
- `AnysoleWorkspace/tool/adapters/FPP-Net/**`；
- `AnysoleWorkspace/tool/adapters/PoseTransOpt/**`；
- FPP/Pose 专用测试。

共享 CLIFF/RTMPose frontend 的变更如需修改 Agent A 所属文件，不直接改；提交明确的接口需求或候选 patch 说明。

## 9. 必做测试

- FPP dataset one-batch，GT shape `(B,192)` 且为原生顶点接触。
- 代表 session 的 FPP 31×11 insole 与 D_Test4 已验收映射数值 parity。
- FPP 只读公共 MMVP 31×11 representation，不存在第二份私有副本。
- FPP contact loss one-step finite。
- 19 个有 fake 缺口 session 的 FPP/keypoint frame id 报告零静默错位；如 CLIFF V2 导致 PoseTransOpt 不可运行，按暂缓阻断交付，不要求伪通过。
- 140 个 canonical CLIFF NPZ 行数等于各自 frame count，无重复 frame id，无跨人 betas/平滑。
- FPP V2T one-session smoke 与标准 archive validator。
- PoseTransOpt/V2M one-session smoke，不再因 CLIFF 行数错位崩溃。

## 10. 交付报告额外要求

列出：

- f6 依赖移除证据；
- native contact 脚内差异统计；
- human-mask 匹配覆盖、时间误差、bbox 重算与 133 份旧异常 NPZ 重建统计；
- frame join 缺失报告；
- V4/V6/V7 分别被裁定为“上游保留”还是“adapter 错配修复”的证据。
