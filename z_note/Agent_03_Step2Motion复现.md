# Agent 03：Step2Motion 统一协议复现任务

## 目标

在不改变 Step2Motion 核心网络结构的前提下，使其能够使用项目统一数据底座 `AnysoleWorkspace` 的序列，并在冻结的 session/split/40 Hz 协议下完成训练、测试和统一格式导出。Step2Motion 可以保留自身的输入工程（48 点压力池化为 16 通道、由 BVH 足部运动合成 IMU），但这些变换必须可追溯、可测试。

工作范围：`Baselines/Step2Motion` 及其对应的 `AnysoleWorkspace/derived/Step2Motion/` 输出。不得修改中央 split 或其他模型代码。

## 统一输入协议

- 序列根目录默认使用 `AnysoleWorkspace/derived/MotionPRO/sequences/cam3`；相机固定为 `cam3`，目标频率固定为 40 Hz。
- session、动作和 train/val/test 列表来自 `AnysoleWorkspace/splits/default/splits.csv`（或 Agent 01 冻结的等价 manifest），不得在本任务中重新划分。
- 每个 session 读取 `align_meta.json`、`fake_mask.npy`、对应 fake-marked pressure CSV 和 `bvh_path`。时间网格必须按 `visual_start_s + i / 40`，动捕查询时间为该网格减 `offset_s`。
- 动捕输入为 23 关节、72 通道 BVH：根节点平移 + 23 个关节的 YXZ Euler 旋转；长度单位从厘米转换为米。关节名称和 parent 必须与 `process_gait.py` 的 `SRC_JOINTS/SRC_PARENTS` 一致，不能替换为 AMASS/SMPL 24 关节模板。
- 压力原始值为左右脚各 48 点（4×12），保留原始量程和时间戳，插值到统一 40 Hz 网格后再做 Step2Motion 特有变换。左脚必须先于右脚拼接成 insole 特征。
- 质量过滤沿用 `process_gait.py`：仅允许质量等级 A/B；缺少质量等级或 C/D 的 session 记录为跳过原因，不得静默纳入训练。

## 必须核对和固定的处理逻辑

### 48→16 压力池化

`pool_48_to_16()` 的输入 reshape 为 `(T, 4, 12)`：4 为脚宽方向，12 为脚长方向；列 0–5 是 toe 半区，列 6–11 是 heel 半区。每个半区按 4 行、两组各 3 列取均值，得到 8 个通道；最终顺序为 `heel[8] + toe[8]`。不得在此处额外翻转左右、行列或 toe/heel。

需要报告：输入量程、是否归一化、每个输出通道对应的原始索引、池化前后总力比例。若发现上游压力 CSV 的布局与上述约定冲突，应先记录证据并停止转换，不能凭视觉效果猜测方向。

### 合成 IMU

沿用 `synthesize_imu()`：由 23 关节 BVH FK 得到左右足位置和旋转，在 `dt=1/40` 下计算二阶差分加重力得到加速度，并由相对旋转计算角速度。左脚局部 IMU 的 Y 轴执行一次反向，右脚不反向；输出单位和坐标系必须在日志中注明。不得把合成 IMU 宣称为真实传感器观测。

每脚特征固定为：`pressure16(16) + acceleration(3) + angular_velocity(3) + total_force(1) + CoP(2) = 25`，左右拼接后 `insole.shape[-1] == 50`。`MotionDataset` 会再把加速度转换到世界系并丢弃每个 clip 的首帧，需在帧数报告中明确这一行为。

### 23 关节和输出

- 保留 Skeleton3 的 23 关节顺序、parents 和 offsets；root translation 单位为米。
- `pose` 为去 root 的局部位置（22×3），`displacements` 为 root 相邻帧位移，`quats` 为 23 个关节四元数（wxyz）。
- 预测结果必须导出为统一 evaluator 可读格式：`session_id`、`frame_index`、`valid_mask`、`joint_xyz_world[T,23,3]`、`root_xyz_world[T,3]`，可选 `pose/rotation/contact`。同时保留原生 BVH 导出用于可视化。

### 窗口和 fake-pressure

- `split_fake_clips()` 将连续非 fake 区间切成 clip；默认 `min_frames=101`，以满足 `input_T=100` 且首帧会被 `MotionDataset` 丢弃。
- fake 帧不得进入训练或指标计算。包含 fake 帧的区间应被切开，而不是用零压力填充后继续训练。
- 转换日志必须列出每个 session 的总帧、fake 帧、保留 clip 起止、丢弃首帧数和最终有效帧数；若与中央 manifest 不一致，任务标记为失败。

## 推荐执行步骤

1. 在仓库根目录检查路径和依赖：

   ```bash
   cd /data/fangyuxuan/projects/gait
   python -m py_compile Baselines/Step2Motion/src/process_gait.py
   ```

2. 运行内置确定性检查（不得产生模型或数据修改）：

   ```bash
   cd Baselines/Step2Motion
   python src/process_gait.py --self-test
   ```

3. 先对单个 session 做 dry-run，确认 manifest、质量过滤、BVH 帧数和压力列数：

   ```bash
   python src/process_gait.py --session <SESSION_ID> --dry-run
   ```

4. 生成统一 Step2Motion gait tensor（输出目录默认 `AnysoleWorkspace/derived/Step2Motion/gait`），记录命令、git 状态和汇总统计：

   ```bash
   python src/process_gait.py
   python src/normalizer.py gait AnysoleWorkspace/derived/Step2Motion/gait/gait_train.pt
   ```

5. 使用 `configs/config_gait.json` 做最小训练 smoke test（固定随机种子，短 epoch/少 batch；不得覆盖已有正式 checkpoint），随后执行测试并导出 BVH/统一预测文件：

   ```bash
   python src/train.py --config configs/config_gait.json
   python src/test_model.py <SMOKE_MODEL_DIR> AnysoleWorkspace/derived/Step2Motion/gait/gait_test.pt --only_test
   ```

   若当前分支没有 `train.py` 或配置字段不同，记录实际入口和阻塞原因，不要复制其他数据集的 checkpoint 冒充复现。

## 必须完成的测试与验收

### 数据和变换测试

- `--self-test` 通过；`pool_48_to_16()` 的单位脉冲测试覆盖左/右脚、toe/heel、四行和两组三列；左右交换后输出严格交换。
- 常数压力输入满足池化后各通道相等且总力比例可解释；随机输入池化结果与手工索引均值一致。
- CoP 测试覆盖全零、单 toe、单 heel；全零输出必须为有限值且按约定为零。
- 23 个 joint names/parents/offsets、BVH 72 通道、厘米→米转换通过检查；FK 后 root 轨迹与输入平移一致。
- 40 Hz 查询时间单调、无越界；压力/BVH/`fake_mask` 的帧数和时间范围有报告。
- 合成 IMU 在静止构造序列上加速度/角速度符合预期，左脚 Y 翻转只发生一次。

### 数据集和模型 smoke test

- 单 session dry-run 与正式转换的 clip 数、有效帧数和 insole 维度（50）一致。
- `MotionDataset` 首帧丢弃后，输入、pose、quats、displacements 长度一致；不存在跨 fake 区间窗口。
- 最小训练能完成至少一个 forward/backward 和 checkpoint 保存；测试能生成预测而无 NaN/Inf。
- 每个预测文件的 23 关节顺序、40 Hz 帧间隔和 session/frame 索引与 manifest 一致，可被统一 evaluator 读取。

### 验收标准

只有同时满足以下条件才算完成：

1. 使用中央 manifest 和同一 IID/OOD split，不生成私有划分；
2. 原始输入确实来自统一 40 Hz BVH/压力序列，且 fake-mask 规则可复核；
3. 48→16、CoP 和合成 IMU 均有确定性测试及日志；
4. 23 关节 Skeleton3 输出可转换到统一 evaluator；
5. 训练、测试 smoke test 和至少一个短序列端到端导出成功；
6. 回传材料包含运行命令、统计、输出路径、失败项和与 AnySole/MotionPRO 的特有输入差异。

## 禁止事项

- 不得修改 Step2Motion 核心网络、loss 或采样策略来掩盖数据接口问题。
- 不得把 16 通道池化结果描述成统一的 48 点原始压力协议。
- 不得使用 3D GT 关键点、GT 投影或其他模型预测作为视频/IMU 观测；合成 IMU 必须明确标为由 BVH 派生。
- 不得更改 `AnysoleWorkspace/splits`、删除原始数据、覆盖正式 checkpoint 或静默跳过失败 session。
- 不得将 Procrustes/pelvis 对齐指标当作唯一结果；导出必须保留世界坐标。

## 回传格式

完成后在消息中按以下顺序回传：

1. **改动文件**：相对路径及每项改动目的（若只改文档/测试也要说明）。
2. **输入与划分**：实际使用的 seq root、split manifest、session 数、总/有效/fake 帧数。
3. **运行命令**：逐条列出 self-test、dry-run、转换、训练、测试和导出命令。
4. **测试结果**：通过/失败、关键数值（insole 维度、23 joints、40 Hz、clip 数、NaN 检查）。
5. **输出位置**：tensor、checkpoint、预测 BVH、统一 evaluator 文件和日志路径。
6. **未解决问题**：按阻塞级别列出，并说明需要主 agent 决策或其他 agent 提供的接口。

