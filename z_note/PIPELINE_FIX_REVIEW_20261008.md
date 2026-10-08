# 管道复核与本地修复记录（2026-10-08）

## 范围与 Git

复核基于原始提交 `e9accee`，涉及原始压力对齐、共享事实、AnySole 数据集、模型输入、普通评估、缺失率网格、导出和实验调度。仓库已有 Git 历史，保留该历史；原始状态保存为 `backup/pipeline-before-fixes`，修复位于 `fix/pipeline-contracts`。提交身份仅在本仓库配置为胡宇轩 <28601919@qq.com>。未执行 push、创建远程仓库或 PR。

## 已确认并修复

| 连接位置 | 原有问题与证据 | 修复 |
| --- | --- | --- |
| 压力质量标记 → manifest | `pressure_flags` 按 CSV 行数比例映射，shared facts 按时间映射。同一合成记录的 fake 行在两条流程中落到不同帧。 | 两者调用同一个绝对时间重采样入口 `recording_pressure`。 |
| 缺失压力 → 编码器 | mask 仅替换编码后的 token；FootConvEncoder 的时间注意力已读到缺失帧。改变被遮蔽帧的压力会改变保留帧 token。预先计算的包络和差分也携带邻帧信息。 | 编码前擦除观测，重算压力包络和差分（包括窗口边缘），编码后保留 null token 替换。完整输入路径保持原计算。 |
| 跳过的窗口 → 时序指标 | 将有效窗口直接拼接，再用连续时间轴计算导数，会在缺口处产生虚假的加速度和 jerk。合成匀速轨迹可复现。 | 保存实际 mocap 时间和原始帧号，按连续帧区间计算时序、接触和轨迹指标；导数按有效样本数汇总。 |
| 网格预测 → motion/repr 文件 | rho 导出直接展平窗口，丢失原始位置、缺口和尾帧，且会跨缺口 crossfade。 | motion 恢复完整帧网格，保存 valid_mask、frame_indices、真实时间；repr 保存 window_starts 和 window_length。 |
| 普通评估 → 指标/导出 | 导出进行 crossfade，正式指标却重新推理并计算未处理预测，两者不是同一份运动。 | 普通评估和 rho 共用后处理、指标汇总与 motion writer；仅相邻窗口 crossfade 4 帧，统一正交化旋转。 |
| 配置 → 结果目录/队列 | 原目录名遗漏 dropout、config_probs 等字段，浮点格式截断精度，分区只记组数。不同实验可以写同一路径。调度计算也未读取实际 YAML。 | 目录增加精确配置指纹，含分区内容和 split 文件哈希；调度读取实际 YAML、覆盖参数并统一 CLI 数值类型，显式 variant 同步传入训练。 |
| 指标 JSON → 调参/角点校验 | 调参读取 `mpjpe_mm`，JSON 已改为 `MPJPE`；角点校验也漏读重命名指标。 | 调参兼容新旧字段；角点先还原内部指标名称。 |
| 本地项目 → 默认结果根 | `GAIT_ROOT` 指向作者机器上的固定绝对路径。 | 改为当前仓库根目录。外部原始数据和 SMPL 路径仍按项目配置提供。 |

## 复核后保留的设计

- 当前默认 train 与 test 不重叠；val 与 test 相同是已有明确协议，未擅自重新划分。它不能提供独立于调参的测试证据，解读结果时需考虑这一点。
- 每个窗口使用前一帧 GT 平移作为轨迹锚点；F2 还使用已有 GT heading 锚点。这是现有评估口径，未改成自由运行推理。结果不应解释为完全无 GT 初始化的连续轨迹生成。
- manifest 中检查的 144 条记录均满足 `visual_start_s - offset_s = mocap_start_s`，未发现这一方向接反。
- 没有改变模型参数结构、损失设计、冻结划分或生成数据文件。

## 验证与限制

执行方式（仓库根目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B tests/test_pipeline_contracts.py
python -B tests/test_build_shared_resample.py
git diff --check
```

新增 11 项验证涵盖：遮蔽帧不影响保留帧、完整输入行为一致、压力标记同源、缺口与尾帧、跨缺口导数、导出与指标使用相同运动、rho 帧网格、配置冲突、调度参数/命名分区、字段兼容，以及真实 AnySoleModelV2 在 VT/V/T 三个角点的评估与 rho 指标一致性。共享压力重采样原有 7 项测试通过。合成测试中仅 SMPL 资源相关转换/表面计算使用替代实现；实际网络、FK、旋转、时间指标和压力计算参与验证。

本克隆不含实际原始数据、shared/model_inputs、真实训练权重、完整 `Baselines/` 和 SMPL 资源；`results_display/models_modes.yaml` 也缺失。因此未执行真实数据训练、完整基线/展示测试或真实 SMPL 表面评估。通过合成验证不等于真实数据全链路已经运行。

## 结果迁移

- 新命名带 `_cfg<指纹>`，旧模型不会自动移动；需要复用历史目录时明确提供其 `--variant` 或 checkpoint 路径。
- 指标与 rho 缓存使用 `evaluation_pipeline=session_frames_v2`。旧版 rho 缓存不复用；角点比较要求普通评估先按新版重算。该版本号仅标识计算口径，运行时仍需分开不同 checkpoint、split 和 session 子集的输出目录。
- crossfade 现在纳入正式指标，缺口按连续区间评价；旧指标不能直接与新版混合比较。Root drift/RTE 等路径指标也按连续区间起点计算。
- 新版缺失率实验擦除输入观测；历史仅替换 token 的部分缺失率结果需要重跑。完整输入和 0/100 角点验证通过，模型参数结构未变。
- 若重新生成 manifest，应从原始资源重建对应 shared/model_inputs，核查有效帧数量和构建 provenance 后重新评估；本次没有数据资源，未执行重建。
