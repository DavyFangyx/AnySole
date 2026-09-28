# Agent C 任务书：MotionPRO-InsoleAdapted 复现适配

> 前置：Agent A 已冻结 shared facts session/path API。  
> 目标：保留 MotionPRO 模型架构，明确将压力地毯输入适配为鞋垫 dual-sole raster，正式名称为 `MotionPRO-InsoleAdapted`。

## 1. Contact 的最终裁定

`contact.npy` 有两个完全不同的用途，必须分开记录：

### 1.1 训练用途：必须保留

`app/train_frappe.py` 将 contact 传入 `loss_foot`，且默认 `lamda_foot=5`。本适配采用官方训练所需的十列 contact 接口，但鞋垫只能提供左/右脚权重，因此仅第 6/7 列非零。

- 不得删除 contact 数据；
- MotionPRO 私有 adapter 使用真正四档软 `f6_soft={0.05,0.30,0.70,0.95}` 生成左/右脚权重；
- 不读取或覆盖 AnySole 当前已二值化的 `f6_soft.npy`；MotionPRO 私有软标签必须单独生成和记录；
- 在 artifact 中将它的语义记录为 `foot_loss_weight`，不是公共 contact GT；
- 物理文件名继续叫 `contact.npy`，以保持上游 loader 契约。

### 1.2 原生 Contact IoU：非正式指标

- `val/contact_iou` 只是训练日志。
- 它不参与 scheduler；scheduler 使用 `loss_eval['loss']`。
- 它不参与 best checkpoint 选择；best 仍使用 eval loss。
- 它不进入 `Baselines/utils/evaluate.py` 的正式公共指标。
- 可保留日志兼容，但必须标记 `diagnostic_only`；也可删除该日志计算，前提是训练数值和 checkpoint 选择不变。

## 2. M1 处置：修复本地适配的索引混用

官方备份的训练逻辑用 `contactIds=[1,2,4,5,7,8,10,11,20,21]` 对齐十列 contact 和关节。本地适配后新增 `footContactIds=[6,7]`，却把 contact 列位置 6/7 错当成 SMPL joint id。

正式修复固定为：

- contact 权重取十列中的第 `[6,7]` 列；
- 预测/GT 关节取 SMPL `[10,11]`；
- 其余 contact 列在鞋垫步态适配中为 0；
- 这是修复本地适配回归，不建立 `MotionPRO-corrected`。

## 3. M2 处置：MotionPRO 私有四档 soft-f6

当前整脚压力和 `>100` 产生的二值 contact 作废。MotionPRO adapter 复用 F6 中间软判据，但不读已二值化的 AnySole NPY：

- 触地且承重：0.95；
- 触地未承重：0.70；
- 离地无载：0.05；
- 离地但有残余压：0.30。

输出仍为 `(T,10)`，仅列 6/7 写入软值。这些值直接作为 `loss_foot` 权重，不作为正式公共 contact GT。

## 4. M3/M4/M5 处置

### M3 时间轴

- 直接读取 shared facts session 已对齐的 frame id 与 pressure_48。
- MotionPRO adapter 不得再解释 PressureWasher `t_us`、visual offset 或另建 40 Hz 网格。
- 所有 MotionPRO 私有文件保存 shared frame id。

### M4 split

- MotionPRO 只读 canonical split。
- `make_splits.py` 不得再写入 canonical split；将其改为验证命令或明确退出。
- README/帮助中删除自行划分引导。共享 README 改动建议交给 Agent A。

### M5 padding

- 在 dataset/adapter 索引阶段只生成完整且全 valid 的窗口。
- 不修改 loss 公式，不通过新 loss mask 改变有效帧权重。
- 尾部不足窗长的帧不进入训练，但可在完整 session 推理导出时用 valid mask 明确标记。

## 5. 目标路径

MotionPRO 触觉输入链必须是：

```text
raw pressure_left/right.csv (48/脚)
→ shared/facts pressure_48.npz (2×4×12)
→ MotionPRO adapter rasterize_feet (160×120)
→ MotionPRO model input bilinear (96×96) / 255
```

禁止从旧 MotionPRO `pressure.npz` 反向 crop 出 4×12 再称为公共原始数据。当前固定左右块和 bilinear 口径保留，但语义改为鞋垫 `dual-sole raster`，明确不是 pressure carpet。

### 两阶段强制暂停门

**C1：只生成审计产物，禁止训练**

- 用当前固定左右块生成代表 session 的 dual-sole raster；
- 生成 MotionPRO 私有四档 soft-f6；
- 可视化原始 4×12、raster、soft-f6 时序和值分布；
- 提交给用户确认。

**C2：用户确认 C1 后**

- 才允许全量生成 model input；
- 才允许训练、推理和公共评估。

```text
AnysoleWorkspace/model_inputs/MotionPRO/<adapter_version>/cam3/<date>/<subject>/<session>/
├── pressure.npz
├── smpl.npy
├── keypoints.npy
├── bbox.npy
├── feature_hrnet.pth
├── contact.npy
├── frame_id.npy
└── artifact.json
```

`artifact.json` 必须声明：

```json
{
  "contact": {
    "physical_file": "contact.npy",
    "semantic_role": "motionpro_private_soft_f6_foot_loss_weight",
    "value_set": [0.05, 0.30, 0.70, 0.95],
    "formal_metric_input": false,
    "native_diagnostic_only": true
  }
}
```

## 6. 写入范围

允许：

- `Baselines/MotionPRO/**`；
- `AnysoleWorkspace/tool/adapters/MotionPRO/**`；
- MotionPRO 专用测试和小样本 model input。

禁止：

- 公共 manifest/split/path 实现；
- `anysole/**`、其他 baseline；
- 公共 metric 公式；
- 将 MotionPRO contact 加入正式公共 contact metric。

## 7. 必做测试

- contact 确实进入 `loss_foot`，删除原生 IoU 日志时训练 loss 不变。
- scheduler 和 best checkpoint 选择不依赖 IoU。
- 公共 evaluator 不读取 MotionPRO contact。
- MotionPRO 私有 contact 列值只能来自 `{0.05,0.30,0.70,0.95}`，其他八列为 0。
- `loss_foot` 的 contact 列 `[6,7]` 与 SMPL joint `[10,11]` 分开索引。
- shared frame id 与 feature/pressure/SMPL/keypoints/contact 逐帧一致。
- 代表 session 的新 raster 与 D_Test4 已验收旧转换数值 parity；若不一致，不得擅自重定义转换。
- 压力与足高互相关不再出现中位约 11 帧的系统时移。
- train one-batch，val one-batch，test one-session，公共 prediction validator。

## 8. 交付报告额外要求

必须单独列出：

- M1 本地适配回归的修复证据；
- contact 在训练中的调用链；
- 原生 IoU 最终是保留诊断还是删除计算；
- C1 用户确认记录、四档 soft-f6 计数和每脚时序统计；
- 窗口总数、因 fake 剔除数、因尾部不足剔除数。
