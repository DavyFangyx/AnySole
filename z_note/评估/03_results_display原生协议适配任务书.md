# 任务 03：results_display 原生协议与能力感知展示适配

## 1. 任务目标

本任务只改造结果消费端 `results_display`。删除 common19 作为跨模型数值评测和可视化协议，改为按模型真实输出能力、原生表示和对应 GT 计算与展示结果。

统一的是同名指标的公式、时间轴、有效帧、单位和聚合规则，不统一各模型的结果文件格式，也不要求不同表示使用相同关节集合：

- SMPL 模型使用完整原生 SMPL-24 预测和 SMPL GT。
- Step2Motion 使用完整原生 BVH-23 预测和 BVH GT。
- V2T 使用已确认的 9 个压力指标。
- 指标所需的真实模型输出存在时，即使模型原生 evaluator 没有实现该指标，也要调用或补齐统一求解器。
- 指标所需输出不存在，或字段来自 GT 回填、启发式猜测、逆向求解、评测端注入模板时，该指标不适用，展示为 `—`。

本任务不执行旧结果删除。只生成精确的待清理清单，交由联合验收后执行。

## 2. 前置依赖与协作边界

### 2.1 依赖任务 01：指标协议与能力矩阵

开始实现前读取任务 01 的最终定义，以其为唯一指标口径，至少应已明确：

- 公共 motion 指标的公式、单位、有效帧和聚合规则；
- `root_orientation_deg`、`root_orientation_drift_deg` 复用 `metrics.py` 的 yaw 定义；
- contact 指标只接受模型显式预测，不能由关节高度、速度或压力阈值反推；
- Step2Motion 的可用指标和不适用指标；
- V2T 正式指标固定为 9 项；
- 指标求解器的必要输入声明。

若任务 01 尚未确定某项公式，本任务可以完成接口和缺失值通路，但不得自行创造第二套公式。

### 2.2 依赖任务 02：结果目录与生产端整理

任务 02 应提供正式模型结果根目录、预测入口和真实输出能力。目录迁移完成前可兼容读取旧路径，但最终注册表必须指向任务 02 规定的新路径，尤其是：

- 结果区统一使用 `results/baselines/pressure_toolkit`；
- `Baselines/pressure_tookit` 是历史源码仓库名，不得用全局替换改名；
- Step2Motion no-IMU 只是实验探针，不注册为正式模型，也不进入正式排名或 R_Test1/2/3 的自动扫描；
- GT fallback、仅为可视化生成的字段和模板注入必须在导出元数据中可识别。

本任务不得改训练代码、模型结构、推理输出语义或任务 02 管理的原生产物。

## 3. 已核验的现状问题

当前代码中 common19 不只用于 R_Test2 数值评测，也进入了 R_Test1 并排可视化：

- `results_display/script/utils/compare_core.py` 定义 `COMMON_JOINTS`、SMPL/BVH 映射、common19 骨架边以及 `select_common_*()`。
- `r_test2_compare.py` 将 `joint_set=common19` 写入 CSV、日志和各模型的 `test_comparison.json` 类结果。
- `r_test1_compare.py` 将所有预测裁剪到 common19，并用一份 GT 和一套骨架边计算、展示所有模型。
- `r_test3_traj.py --compare` 当前对所有模型使用 `load_session_gt(...)["joints"][:, 0]` 的同一 root GT，未按预测协议选择 SMPL GT 或 BVH GT。
- `r_test4_v2t.py` 仍输出 `contact_f1/contact_acc/contact_recall/air_recall`，与 V2T 9 项定义冲突。
- `models_modes.yaml` 仍引用结果路径 `pressure_tookit`，且没有明确记录协议和真实输出能力。

因此不能只删除一个常量或改展示文字，必须同时适配 R_Test1、R_Test2、R_Test3、R_Test4、注册表和文档。

## 4. 能力声明与指标分派

### 4.1 注册表接口

任务 02 扩展 `results_display/script/models_modes.yaml` 的正式模型条目。本任务按以下
schema 消费并校验，至少包括：

```yaml
protocol: smpl24 | bvh23 | pressure
capabilities:
  joint_positions: true | false
  root_translation: true | false
  root_rotation: true | false
  joint_rotations: true | false
  surface: true | false
  pressure: true | false
  contact: true | false
```

能力表示模型真实预测或可由其真实输出唯一确定的量，而不是“文件里碰巧存在这个数组”。以下内容不得将能力置为 `true`：

- 用 GT translation 或 GT pose 回填的字段；
- 从关节位置逆向求解的非唯一完整局部旋转；
- 从脚部高度、速度或压力阈值启发式生成的 contact；
- 仅为动画生成的占位数据；
- 评测端给模型套用的固定 body template 或 GT beta。

加载预测后还要检查实际字段是否满足声明。注册表声明为真但必要字段缺失时，结果状态是 `invalid`，并记录原因；能力声明为假时，指标状态是 `not_applicable`，展示为 `—`。不得把两者都静默转换为 0 或空模型结果。

### 4.2 求解器启用规则

每个指标求解器声明必要输入。模型原生 evaluator 未提供某个指标，不影响启用；只要真实输出满足必要输入，就必须使用任务 01 的公共求解器补算。

至少按下列关系门控：

| 指标族 | 必要真实输出 |
|---|---|
| MPJPE、PA-MPJPE | `joint_positions` |
| W-MPJPE、root ATE | `joint_positions` + `root_translation` |
| root orientation | `root_rotation` |
| MPJAE | 模型直接输出的 `joint_rotations` |
| world temporal / jitter | 有真实世界轨迹的 `joint_positions` 和时间轴 |
| PVE、shape | 模型真实输出且可公平比较的 `surface`/SMPL 参数 |
| contact | 显式训练并输出的 `contact` |
| V2T 9 项 | `pressure` |

### 4.3 pressure_toolkit 特殊约束

`pressure_toolkit` 使用模板/拟合过程得到表面。不得在公共评测端注入 GT beta 或固定模板来制造一个看似可比的 PVE。它的 `pve_mm`、`shape_vertex_std_mm` 以及其他依赖公平 surface/shape 输出的指标统一为 `not_applicable`，展示为 `—`。

其真实输出若满足 joint pose、root translation 或 root rotation 的必要条件，仍应补算对应关节、轨迹和朝向指标。不得因为取消 PVE 而连带取消这些可计算指标。

## 5. 实施内容

### 5.1 公共比较核心接口验收

`results_display/script/utils/compare_core.py` 由任务 01 负责。本任务不得重复修改其指标
公式或能力判断，只验收并消费以下接口：

1. common19 常量、选择器和专用逐帧 MPJPE 已删除。
2. SMPL/BVH 对应 GT loader、原生关节名和原生 skeleton edges 可用。
3. 指标能按 `protocol + capabilities + 实际字段` 分派并区分
   `not_applicable`、`invalid`、`missing`。
4. 正式 motion contact 不从 joints 推导；BVH 反解旋转不启用 MPJAE。
5. V2T 求解器只返回 9 项；聚合不把不适用值计入分母。

若接口不满足，本任务暂停对应消费端接入并把缺口交回 Agent A，不得在 R_Test 脚本里
另写一套指标公式。

### 5.2 R_Test2 数值评测

修改 `results_display/script/r_test2_compare.py`：

1. 删除 common19 import、裁剪和元数据。
2. 每个模型按其原生协议计算完整骨架指标：SMPL-24 对 SMPL GT，BVH-23 对 BVH GT。
3. 输出元数据至少包含 `evaluation_protocol`、capability 摘要、GT source、split/manifest hash 和指标缺失原因；不再写 `joint_set=common19`。
4. 跨模型 comparison schema 升版，禁止把旧 `mmvp_common_metrics_v1` 文件当成新结果读取。
5. CSV 内的不可用指标留空或使用原生缺失值；表格、PNG 和 mode overview 统一显示 `—`。
6. `missing`、`invalid` 和 `not_applicable` 必须区分：预测文件不存在是 `missing`，声明能力与文件矛盾是 `invalid`，模型没有该功能是 `not_applicable`。
7. Step2Motion 使用 BVH-23 原生评测；`pve_mm`、shape、MPJAE、contact 为 `—`。其主干若确有真实 translation/root rotation，则必须补算轨迹和 yaw 指标。
8. `pressure_toolkit` 的 PVE/shape 为 `—`；真实 joints/translation/rotation 可用的其他指标照常补算。
9. 各 mode 只展示任务 01 对该任务批准的字段；V2T 严格为 9 项。

### 5.3 R_Test1 动作可视化

修改 `results_display/script/r_test1_compare.py` 及必要的单模型可视化入口：

1. 删除 common19 的加载、骨架边和逐帧误差计算。
2. 每个模型使用自己的原生 joints、原生 skeleton edges 和同协议 GT。
3. 混合 SMPL/BVH 的一行中，不得再用一个 `GT Motion` 面板代表所有模型。每个模型面板应叠加或明确关联自己的 GT；若使用独立 GT 栏，则分别提供 SMPL GT 和 BVH GT 栏。
4. 每个面板标题或 footer 标明 `SMPL-24 native` 或 `BVH-23 native`。逐帧 MPJPE 只在模型及其配对 GT 之间计算。
5. 缺失预测保留 missing panel 和具体原因，不允许退回 common19 或其他模型的 GT。

R_Test1 是定性展示。它可以把不同原生骨架放在同一画布中观察，但不得把骨架点数量不同掩饰成完全相同的数值协议。

### 5.4 R_Test3 轨迹可视化

修改 `results_display/script/r_test3_traj.py`：

1. `--compare` 只纳入 `root_translation=true` 且预测文件确实含真实 root trajectory 的正式模型。
2. SMPL 模型配 SMPL root GT，BVH 模型配 BVH/Hips root GT；ATE 按模型与其配对 GT 分别计算。
3. 可以用所有预测和 GT 共同确定投影范围，但每个模型面板必须画自己的 paired GT，不能共用一条 SMPL GT。
4. 日志和静态图注明模型 protocol 与 GT source。
5. 检出 `root_translation_source=ground_truth_fallback` 或等价标记时拒绝进入正式轨迹比较，并记录原因。

### 5.5 R_Test4 V2T

修改 `results_display/script/r_test4_v2t.py`：

1. 正式 summary、report、aggregate 和图片只包含以下 9 项：
   - `T_corr`、`T_mae`、`T_rmse`
   - `pressure_force_mae`、`pressure_force_rmse`、`pressure_force_r2`
   - `pressure_cop_error_left`、`pressure_cop_error_right`、`pressure_cop_error_mean`
2. `contact_pred/contact_gt` 不再是新 archive 的强制字段。为读取旧 archive 可以兼容存在的字段，但不得输出 contact 指标。
3. R_Test2 和 R_Test4 必须调用同一个 pressure metric 实现；相同模型、session 和 valid mask 的 9 项数值必须一致。

### 5.6 注册表、路径与名称

`results_display/script/models_modes.yaml` 由任务 02 负责。本任务验收并消费以下结果：

- 每个正式条目已有 protocol 和 capabilities；
- pressure_toolkit 使用 `results://baselines/pressure_toolkit/...`；
- V2M motion 行名称反映 `VP-MoCap (FPP-Net + PoseTransOpt)` 真实生产链；
- 不包含 Step2Motion no-IMU 探针。

路径或声明不正确时交回 Agent B 修改，本任务不得自行改变生产端目录契约。

### 5.7 文档同步

更新以下现行说明中的消费端部分：

- `results_display/README.md`
- `z_note/EVALUATION_PROTOCOL_AUDIT.md`

`Baselines/command_manual.md` 由任务 02 负责；本任务只更新
`results_display/README.md` 和 `z_note/EVALUATION_PROTOCOL_AUDIT.md` 的消费端说明。

删除 common19 作为正式评测协议的描述，改为说明原生协议、对应 GT、能力门控、缺失值 `—`、V2T 9 项、路径规范和重新生成流程。历史记录如需保留，必须明确标为“旧协议，已废弃”，不能与现行命令混写。

## 6. 旧派生产物清单：本任务只盘点，不删除

实现完成后生成一份机器可核对的清单，至少覆盖：

```text
results_display/ResultTest/R1Test_visualize/compare/
results_display/ResultTest/R2Test_compare/
results_display/ResultTest/R3Test_traj/compare/
results_display/ResultTest/R4Test_v2t/
results/**/metrics/*_comparison.json
```

清单逐项记录：绝对或仓库相对路径、文件类型、大小、旧 schema、是否含 `joint_set=common19`、建议动作和可重建命令。

其中：

- R_Test1 compare、R_Test2、R_Test3 compare 属于 common19/混合 GT 的旧派生结果，待联合验收后重建。
- R_Test4 旧 CSV/JSON 若含 contact 字段，待联合验收后重建。
- `*_comparison.json` 只将 schema v1 或明确声明 common19 的文件列为待清理对象，不使用宽泛 glob 直接删除全部 metrics。
- 不将单模型 R_Test1、checkpoint、原始 prediction、原生 metrics 或模型训练日志列为删除对象。

本任务不得执行 `rm`、移动旧结果或覆盖原始结果。实际清理必须等任务 01、02、03 联合验收通过后单独授权。

## 7. 验证与验收

### 7.1 静态检查

- 执行代码和现行协议文档中不再出现 common19；仅允许明确标注为旧协议的历史审计记录出现。
- `models_modes.yaml` 的每个正式模型都有 protocol 和 capabilities。
- 结果路径使用 `pressure_toolkit`，源码仓库历史路径仍可正确解析。
- no-IMU 探针不在正式注册表和自动扫描结果中。

### 7.2 求解器与能力门控

- SMPL-24 预测等于 SMPL GT 时，可用的 native motion 指标为零或数值容差内为零。
- BVH-23 预测等于 BVH GT 时，可用的 native motion 指标为零或数值容差内为零。
- 删除某项必要真实输出后，对应指标变为 `invalid` 或 `not_applicable`，不能由 GT、零值或启发式输出替代。
- 模型具备必要输出但原生 evaluator 没有该指标时，公共端能正常补算。
- `pressure_toolkit` 的 PVE/shape 始终为 `—`，其他真实输出支持的指标不受影响。
- 当前没有显式 contact prediction 的 motion 模型 contact 均为 `—`。

### 7.3 各展示任务

- R_Test1 混合展示中，SMPL 模型使用 SMPL edges/GT，Step2Motion 使用 BVH edges/GT；不存在 common19 裁剪。
- R_Test2 的 Step2Motion 行标记 `bvh23-native`，SMPL 行标记 `smpl24-native`；缺失项显示 `—`。
- R_Test3 中 Step2Motion 的 ATE 对 BVH root GT 计算，SMPL 模型对 SMPL root GT 计算；GT fallback 模型不进入轨迹比较。
- R_Test4 与 R_Test2 的 V2T 结果严格只有 9 项；相同输入的数值逐项一致。
- 相同 test split 的 session 数、valid frame 数和帧权重在改造前后可追溯，协议改变不能静默改变 split。

### 7.4 清理清单

- 清单能精确识别旧 common19/schema v1/R4 contact 派生产物。
- 清单中没有 checkpoint、原始 predictions、native metrics 或 workspace 原生产物。
- 本任务完成时待清理文件仍然存在，没有发生删除。

## 8. 交付物

执行 Agent 完成后提交：

1. R_Test1/2/3/4 的代码改动，以及对任务 01 公共比较核心接口的验收记录；
2. 对任务 02 提供的 protocol/capabilities 正式模型注册表的消费验收记录；
3. 更新后的现行展示与运行文档；
4. 单元级能力门控测试和至少一个 SMPL、一个 BVH、一个 V2T 的端到端验收记录；
5. 旧派生产物待清理清单，明确注明“尚未删除”；
6. 按模型列出的“真实输出 → 启用指标 → `—` 指标”最终矩阵。

## 9. 不得触碰范围

- 不改任何模型训练结构、损失、输入模态或 checkpoint。
- 不把 Step2Motion no-IMU 提升为正式模型。
- 不修改 BVH、SMPL、压力等源 GT。
- 不统一或重写各模型的原生结果文件格式。
- 不用 GT 回填、反解旋转、启发式 contact 或模板注入扩展模型能力。
- 不删除、移动或覆盖旧派生结果及任何原始结果。
- 不处理任务 02 负责的 pressure_toolkit 大体量过程目录迁移。
