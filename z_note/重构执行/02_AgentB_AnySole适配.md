# Agent B 任务书：AnySole 私有适配

> 前置：Agent A 已交付并冻结 shared facts session/path API。  
> 目标：AnySole 直接消费公共采集事实，但其标签、特征和消融输入保持私有。

## 1. 语义红线

- AnySole 仍使用原生 SMPL-24、axis-angle/6D 旋转、米和当前世界轨迹契约。
- 不修改模型结构、loss、解码器、训练策略和评估公式。
- `f6_soft`、`joint_and` 等 contact 方法是 AnySole 实验资产，不是 shared GT，不得推送给基线。
- 启发式 contact 不得被宣称为正式跨模型 contact 指标。
- AnySole 当前二值化 `f6_soft.npy` 本轮不恢复为四档软值；四档 soft-f6 仅属于 MotionPRO-InsoleAdapted 任务。
- 正式 AnySole 配置显式写 `contact_method: f6_soft`；代码不保留 `tactile_abs` 隐式 fallback，配置缺失时直接报错。

## 2. 目标路径

```text
AnysoleWorkspace/model_inputs/AnySole/<adapter_version>/
├── labels/<contact_method>/<session>.npz
├── visual/hrnet/cam3/<session>.pt
├── visual/hmr/<model>/cam3/<session>.pt
├── tactile/<optional_repr>/...
└── artifact.json
```

不再从 `derived/MotionPRO/sequences` 获取 `align_meta.json`、`fake_mask.npy` 或 contact 文件。

## 3. 实施步骤

1. 将 AnySole dataset 的 session、frame/time、valid/fake、raw pressure 读取切到 Agent A 提供的 shared facts API。
   - 触觉物理源是每个 session 左/右 `pressure_*.csv` 的 48 数值列；
   - AnySole 必须直接读 `shared/facts/.../pressure_48.npz`，不得经 MotionPRO raster 回推 4×12；
   - `T_raw` 的左右顺序仍是 `left48 | right48`。
2. SMPL 源路径从 shared facts session metadata 取得，保留当前 AnySole 的 SMPL 重采样和运动表示。
3. HRNet/HMR cache 改为 `model_inputs/AnySole` 私有产物，不存放在 shared。
4. contact builder 只写 AnySole labels 目录，每份标签带 session/frame id、method、参数和源 hash。
5. 默认配置使用新 URI，可显式指定旧路径做迁移期回归比较，但旧路径不得继续是默认。
6. 更新 AnySole 训练/评估/推理中的数据 provenance，checkpoint 必须记录 adapter/schema/split hash。

## 4. 允许写入

- `anysole/**`；
- `AnysoleWorkspace/tool/adapters/AnySole/**`；
- AnySole 专用测试；
- `model_inputs/AnySole` 下的小样本验证产物。

不得修改：

- `AnysoleWorkspace/README.md` 和公共 path/manifest/split builder；
- `Baselines/**`；
- 根公共指标公式；
- `results_display/**`。

## 5. 测试

- 同一 session 在新老 loader 中的 frame/time/fake 完全一致。
- 对代表 session 核对 shared left/right 48 格与原始左/右 CSV 经时间对齐后逐值一致。
- raw pressure 96 维、SMPL pose/trans 和 HRNet cache shape 一致。
- 无 fake 的固定窗口上，新老 dataset tensor 逐值相等或在原有浮点容差内。
- 含 fake 的窗口不进入训练。
- contact method A 的文件不会被 method B 或基线读取。
- 训练 one-batch smoke；评估 one-session smoke；标准 prediction 可被公共 evaluator 消费。

## 6. 交付要求

报告中列出：

- 新老路径映射；
- 数值 parity session 和比较结果；
- 已私有化的 label/cache 列表；
- checkpoint 中新增的 provenance 字段；
- 仍需 Agent A 更新的共享文档内容。
