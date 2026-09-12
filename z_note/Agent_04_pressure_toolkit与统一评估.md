# Agent 04：pressure toolkit 与统一评估

## 任务目标

将统一工作区序列适配到 `Baselines/pressure_tookit` 的 MMVP 输入协议，并建立可比较的统一评估入口。不得把 3D GT 关键点伪装成观测输入。

## 输入与约束

- 输入根：`AnysoleWorkspace/derived/MotionPRO/sequences/cam3/<date>/<subject>/<session>/`。
- 必须读取 `align_meta.json` 的 `visual_start_s`、`offset_s`、`target_fps` 和 `n_frames`。
- 所有命令默认从仓库根目录 `/data/fangyuxuan/projects/gait` 执行；统一路径通过
  `workspace://`、`results://` 和 `display://` 解析，也可分别用
  `ANYSOLE_WORKSPACE`、`ANYSOLE_RESULTS`、`ANYSOLE_RESULTSDISPLAY` 覆盖。
- 统一相机：cam3；统一目标帧率：40 Hz；统一 GT 动捕：23 关节 Skeleton3 BVH。
- toolkit 配置入口：`Baselines/pressure_tookit/configs/fit_smpl_rgbd.yaml`。
- 输出不得覆盖原始数据；适配数据放在 `AnysoleWorkspace/derived/pressure_tookit/`。

### toolkit 适配后的目录协议

`basdir` 必须指向 `AnysoleWorkspace/derived/pressure_tookit/`，单个序列使用以下
目录和文件名；`color`、`depth`、`depth_mask`、`insole` 必须按同一帧号配对：

```text
derived/pressure_tookit/
├── images/<dataset>/<subject>/<sequence>/
│   ├── color/<frame>.png
│   ├── depth/<frame>.png          # uint16，毫米
│   ├── depth_mask/<frame>.png
│   ├── insole/<frame>.npy
│   └── calibration.npy
├── annotations/<dataset>/floor_info/floor_<subject>.npy
└── input/<subject>/<sequence>/keypoints/<frame>.npy
```

RTM-pose 文件必须包含 `keypoints` 和 `keypoint_scores` 字段，格式须能直接通过
`lib.dataextra.data_loader.read_rtm_kpts()` 读取。注意：当前 loader 对 keypoints
使用相对当前工作目录的 `input/...` 路径；在修复前必须从
`Baselines/pressure_tookit/` 启动，修复后则必须改为相对于 `basdir` 或显式配置根，
并用测试覆盖两种启动目录。

## 执行步骤

1. 检查 `AnysoleWorkspace/dependencies/pressure_tookit/essential/` 是否包含 body model、pose prior、左右脚压力区域映射、foot surface ids 和 hand ids；缺失时记录阻塞，不下载或伪造资产。
2. 设计 session 转换器，生成 toolkit 所需的 `images/<dataset>/<subject>/<sequence>/`、`annotations/.../floor_info/` 和 `input/.../keypoints/` 层级，并写出 source-to-target 映射表。
3. 从统一 `color/*.jpg` 生成 PNG/RGB 输入；按统一时间轴生成 metric depth 和 `depth_mask`。记录 DepthPro 版本、模型路径和失败帧。
4. 从真实 RGB 帧运行 RTM-pose，输出 toolkit `read_rtm_kpts()` 可读取的逐帧 `.npy`。禁止使用 `keypoints.npy` 的 3D GT 投影代替观测；如需 oracle 上限，必须单独命名并单独统计。
5. 将压力 CSV/`pressure.npz` 转成逐帧 `insole/*.npy`，验证左右脚顺序、4×12 方向、toe/heel 区域和 fake/valid 标记。
6. 从真实标定资料生成 `calibration.npy` 与 `floor_<subject>.npy`；若没有可信标定或地面参数，停止端到端拟合并在回传中标红。
7. 修复 toolkit 数据 loader 的路径健壮性：keypoint 路径必须相对于 `basdir` 或显式配置根解析，不能依赖当前 shell 工作目录。
8. 为 toolkit 单帧执行 `init_shape -> init_pose -> tracking` smoke test；确认结果文件、mesh 和日志可生成。
9. 建立统一导出格式：每个模型输出 `session_id/frame_index/valid_mask/joint_xyz_world(23,3)/root_xyz_world`，可选附带 rotations、vertices、contact。
10. 实现统一 evaluator，明确 world MPJPE、pelvis-aligned MPJPE、Procrustes MPJPE、轨迹误差、Accel、Jitter、接触 Accuracy/F1/IoU 的单位、FPS、mask 和加权方式。

适配命令应显式指定输入和输出，示例：

```bash
python Baselines/pressure_tookit/data_prep/rgb2depth.py \
  workspace://derived/MotionPRO/sequences/cam3/20260810/S13/S13013/color \
  workspace://derived/MotionPRO/sequences/cam3/20260810/S13/S13013/depth \
  --skip-existing
python Baselines/pressure_tookit/main_singleview.py \
  --config Baselines/pressure_tookit/configs/fit_smpl_rgbd.yaml
```

配置中的 `dataset`、`sub_ids`、`seq_name`、`start_idx`、`end_idx` 必须与 manifest
中的单个 session 一致；禁止用默认示例序列覆盖实际 session。

## 必测命令与测试

```bash
python -m py_compile Baselines/pressure_tookit/main_singleview.py
python Baselines/pressure_tookit/data_prep/rgb2depth.py --help
python -m pytest -q Baselines/pressure_tookit/tests  # 若测试目录存在且依赖可用
```

另需运行：

- essential 完整性检查；
- 转换前后帧数/时间轴/压力 checksum 检查；
- calibration/floor 变换可逆性检查；
- toolkit 单帧 smoke test；
- evaluator 的 GT→GT 检查：所有误差为 0，接触指标为 1；
- 构造样例验证 world、pelvis、Procrustes 三种指标确实去除不同误差。
- `read_rtm_kpts()` 的字段/shape 检查，以及从仓库根目录和 toolkit 目录启动时的路径检查；
- DepthPro 输出 `uint16` 毫米编码、`meta.json` 的 scale/unit 与失败帧检查。

## 验收标准

- 无 GT 3D keypoint 泄漏；
- 每一帧能追溯到统一 session 和 frame index；
- toolkit 观测输入与 AnySole/MotionPRO 使用同一 RGB/压力来源；
- 缺失 essential/calibration/floor 时明确失败，不产生伪结果；
- 四类模型输出都能进入 evaluator；
- evaluator 报告 FPS=40、valid/fake mask、单位及 session/frame 加权规则。
- depth 输出明确标注单位为毫米 PNG（`scale=1000`），不得把 uint16 数值直接当作米；
- toolkit loader 不依赖 shell 当前目录，且单帧配置中的 frame range 与统一 frame index 一致。

## 禁止事项

- 不伪造 essential、相机标定或 floor 参数；
- 不用 3D GT 关键点作为 RTM-pose 输入；
- 不把 Procrustes 指标当作完整世界坐标结果；
- 不修改中央 split/manifest；
- 不删除或覆盖原始序列。

## 回传格式

1. 修改文件列表（包括适配器、loader 修复、evaluator 和测试）；
2. 转换命令和示例 session；
3. essential/calibration/floor 状态；
4. smoke test 与 evaluator 测试输出；
5. 输出目录、DepthPro 版本/模型路径和失败帧；
6. 已知限制、阻塞项及下一步建议。
