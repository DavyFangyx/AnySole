# Agent D 任务书：Step2Motion 语义保真适配

> 前置：Agent A 已冻结 shared facts session/path API。  
> 目标：用 canonical split 和 shared frame 事实重建 Step2Motion gait 数据，保留其 BVH-23、压力/IMU 和扩散模型语义。

## 1. 语义红线

- 不将 Step2Motion 映射成 SMPL-24 训练。
- 不为了和 AnySole 数值一致而修改压力归一化、IMU 通道、输入维度或 normalizer 定义。
- 不修改扩散模型、translation head、loss 或上游 metrics 实现。
- 公共指标可以按 BVH-23 关节名称重新计算，但不得反向改变模型训练。

## 2. S1：canonical split

1. 删除/禁用 gait 转换中的自行划分逻辑。
2. train/val/test 只读 `protocol/splits/default/splits.csv`。
3. 重建 gait 和 gait_noimu 两套 `.pt`。
4. 每套数据保存 `split_sessions.json` 和 split/manifest hash。
5. gait normalizer 只从 canonical train `.pt` 生成，不使用 val/test。

目标路径：

```text
AnysoleWorkspace/model_inputs/Step2Motion/<adapter_version>/
├── gait/
│   ├── gait_train.pt
│   ├── gait_val.pt
│   ├── gait_test.pt
│   ├── normalizer_gait.pth
│   └── split_sessions.json
├── gait_noimu/
└── artifact.json
```

上游下载的 default/dancing/step2motion normalizer 属于 `assets/upstream_reference_data`，本项目 gait normalizer 属于 model input，两者不得混放。

## 3. 数据构建

- 从 shared 取得 frame id、valid/fake 和 left/right 48 格压力。
- BVH 源路径从 shared metadata 取得，保留 Step2Motion 当前 Y-X-Z、BVH-23、root translation 和合成 IMU 逻辑。
- 时间采样必须使用 shared 提供的 mocap time，不再独立构造 visual/pressure offset。
- 任何输入帧必须保留对应 shared frame id；fake 帧不进入有效 clip。
- 48→16 池化保持 Step2Motion 原生 heel/toe 顺序。

触觉来源链必须在 artifact 中写明：

```text
raw pressure_left/right.csv (48/脚)
→ shared/facts pressure_48.npz (2×4×12)
→ Step2Motion pool_48_to_16
→ left16/right16 (heel[0:8] + toe[8:16])
→ Step2Motion pressure/force/CoP/(IMU) 原生输入
```

D_Test4 的 48→16 结果已经用户肉眼验收通过，不重新设计池化、heel/toe 排列或可视化展开方式。

## 4. S2：指标关节组

不修改上游 `src/metrics.py` 来伪装原生重现。处置方式：

- 上游 native metrics 可保留作为上游调试产物；
- 项目正式比较必须由公共 evaluator 根据 BVH-23 joint names 选择腿、脚、趾；
- 不得将旧硬编码索引的数字标成新数据的 legs/toes；
- 如公共 evaluator 已完整覆盖，本 Agent 只需增加验证，不重写公式。

## 5. S3：压力增益

- 保留 Step2Motion 原生输入与 normalizer 契约。
- 在 artifact/report 中记录每个 session 的压力范围、日期和增益差异。
- 不将 AnySole `[0,1]` 归一化方法移入 Step2Motion。
- 任何新的跨 session 标定只能作为单独 ablation，不属于本任务。

## 6. 写入范围

允许：

- `Baselines/Step2Motion/**`；
- `AnysoleWorkspace/tool/adapters/Step2Motion/**`；
- Step2Motion 专用测试和小样本 model input。

禁止：

- 公共 split/manifest/path 实现；
- AnySole/MotionPRO 目录；
- 公共 metric 公式；
- 把 gait normalizer 写回静态 dependencies 或上游 normalizer 目录。

## 7. 必做测试

- `split_sessions.json` 与 canonical train/val/test 逐项一致；计数 92/12/36。
- train/val/test 无重复，subject-disjoint。
- normalizer provenance 中只有 train hash/session。
- gait/gait_noimu 的输入维度、BVH-23 关节序和 translation head 数据符合当前正式配置。
- 代表 session 的新 16 通道产物与 D_Test4 已验收旧产物数值 parity。
- pred=GT 时 BVH-23 公共指标归零；腿/脚组通过关节名称校验。
- one-batch train smoke，one-session test/export smoke，公共 evaluator 可读取预测。

## 8. 交付报告额外要求

列出：

- 旧 gait.pt 与新 gait.pt 的 session 差异；
- normalizer 重建源；
- 保留的原生输入语义；
- native metrics 与 formal public metrics 的分界；
- 压力增益异常报告位置。
