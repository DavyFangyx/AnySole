# Agent F 任务书：pressure_toolkit 语义保真适配

> 前置：Agent A 已冻结 shared facts/representations/frontend/path API。  
> 目标：恢复 pressure_toolkit 的人体深度、地面、触觉和拟合契约，分离模型输入与 fitting 中间态。

## 1. 语义红线

- 不修改 pressure_toolkit 拟合目标的数学定义或权重。
- 修复必须恢复上游期望的输入语义，不使用 AnySole 或 FPP 标签代替。
- 人体深度 mask、行走地面和 insole 都是 pressure_toolkit 模型输入，不是“展示产物”。
- 过程 mesh、gt depth、yaml、temp、log 不属于正式 results。

## 2. P1：地面契约

当前适配器把标定棋盘格平面当成真实行走地面，与数据实测相差约 1.45m。必须：

1. 回归上游深度地面估计路径，复用上游 `calculateFloorNormal` 等实现和坐标契约。
2. 棋盘格标定仅用于相机坐标变换，不直接定义 floor y=0。
3. floor artifact 记录选择帧、深度源、算法、normal/trans/matrix 和质量残差。
4. 不将 PoseTransOpt 的单帧场景深度地面直接作为 pressure_toolkit 的地面；两者分别保持原生流程。

本项为本轮必做，并新增 `D_Test6 地面估计与坐标系审计`：

- 每个采集日期生成候选地面帧/ROI；
- 输出深度点云、平面、法向、坐标轴和人体/脚底叠加可视化；
- 由用户对四个日期各确认一次；
- 落点为 `results_display/DataTest/D6Test_floor/`；
- Agent F 允许仅为 D_Test6 新建对应 `results_display` 脚本/文档，不修改其他展示任务。

## 3. P2：基于 SAM3.1 唯一受试者的深度 mask

Agent F 不运行新的分割模型，直接消费 Agent A 索引的外部 SAM3.1 单受试者 mask。每帧 pressure_toolkit depth mask 固定为：

```text
human_mask_sam31
∩ finite_depth
∩ depth_in_[0.4m,5m]
```

原 loader 的 mask 膘胀可保留，但膘胀必须作用于交集后的人体 mask，不能重新引入全景地面/背景。产物保存 frame_id、human-mask path/hash、匹配误差、交集前后面积和空 mask 检查。

## 4. P3/P4：采样与触觉输入

### P3

- 撤销本地适配新增的默认 `depth_sample_stride=2`，恢复上游全采样口径。
- 若需 stride=2 的性能模式，只能显式命名为近似/加速模式，不得作正式 baseline 默认。

### P4

- 31×11 insole 作为公共版本化 MMVP representation，pressure_toolkit 只读消费。
- 不得从 `results_display` 或“仅展示”缓存读取。
- fake/invalid 帧不生成有效优化任务，不得用补帧压力驱动拟合。
- 每帧 insole 保存 shared frame id，不仅依赖排序文件名。

完整来源链必须是：

```text
raw pressure_left/right.csv (48/脚)
→ shared/facts pressure_48.npz (2×4×12)
→ shared/representations/tactile/mmvp_31x11/<version>
→ pressure_toolkit 只读消费
```

不得重新生成第二份私有 31×11 树，也不得经 MotionPRO raster 回推 4×12。D_Test4 的几何映射已人工验收通过。

## 5. P5/P6/P7/P8

### P5 CLIFF 单受试者初始化

pressure_toolkit 只消费 Agent E 基于 human-mask bbox 重建的单受试者 CLIFF 结果。初始帧/frame id/bbox/mask provenance 必须可追溯，不得从旧多人 NPZ 默认取第一行。

### P6 NaN

- 对照上游/本地备份，恢复 non-finite 时回滚后继续而不是终止整个序列。
- 这是恢复上游控制流，不是重新设计优化器。

### P7 前帧接触

- 在进入 optimizer 前验证所需前帧接触状态。
- 缺失时明确跳过该帧/片段或返回契约错误，不得用负索引回绕。

### P8 HALPE

- 如本项目将上游关键点替换为 HALPE，adapter 必须映射成上游 loss 期望的顺序。
- 不直接改 loss 中的数学逻辑；保留上游输入契约。

## 6. 目标路径

```text
AnysoleWorkspace/model_inputs/pressure_toolkit/<adapter_version>/...
AnysoleWorkspace/work/pressure_toolkit/<run_id>/
├── fitting/
├── meshes/
├── gt_depths/
├── yamls/
├── temp/
├── logs/
└── artifact.json
```

最终标准 prediction/metric 导出至：

```text
results/baselines/pressure_toolkit/predictions/
results/baselines/pressure_toolkit/metrics/
```

## 7. 写入范围

允许：

- `Baselines/pressure_tookit/**`；
- `AnysoleWorkspace/tool/adapters/pressure_toolkit/**`；
- pressure_toolkit 专用测试。

如需修改 Agent A 所属的 DepthPro/CLIFF/shared frontend，必须交付明确需求，不自行改公共契约。

## 8. 必做测试

- floor transform 在四个日期的代表 session 上检查，脚底与估计地面无约 1.45m 偏置。
- depth cloud 组成报告显示观测被限定在唯一受试者人体前景，地面/背景不再主导。
- 默认 stride 与上游一致；近似模式不被正式配置使用。
- 代表 session 的 pressure_toolkit 31×11 insole 与 D_Test4 已验收映射数值 parity。
- pressure_toolkit 与 FPP-Net 读取同一份公共 31×11 representation，不存在重复树。
- 含 fake 的 session 报告实际跳过帧数；全 fake session 不得产生伪正式结果。
- NaN 回滚单元测试，前帧接触缺失测试。
- one-frame fitting smoke、one-session resume smoke、标准 prediction export/validator。

## 9. 交付报告额外要求

列出：

- floor 方法及四日期残差；
- SAM3.1 human-mask 匹配覆盖、时间误差、交集前后面积及人体/地面/背景占比；
- D_Test6 四日期候选可视化和用户确认记录；
- fake 帧剔除统计；
- 正式与近似采样模式的分界；
- fitting 过程产物和正式 results 的路径。
