# Agent D Wave 2 交付报告（Step2Motion 适配）

任务：`z_note/重构执行/04_AgentD_Step2Motion适配.md`（S1 canonical split、S2 指标关节组、S3 压力增益）
状态：complete

## 修改文件

- `AnysoleWorkspace/tool/adapters/Step2Motion/__init__.py`（新增）
- `AnysoleWorkspace/tool/adapters/Step2Motion/build_gait.py`（新增，adapter 主入口）
- `AnysoleWorkspace/tool/adapters/Step2Motion/tests.py`（新增，§7 验收测试）
- `Baselines/Step2Motion/src/workspace.py`（重写：公共 URI 解析委托给
  `AnysoleWorkspace.tool.workspace.resolve_uri`；删除 legacy `workspace://derived`
  与 `sources/raw` BVH 回退；新增 `normalizer_path(config)`）
- `Baselines/Step2Motion/src/process_gait.py`（**删除**，用户裁定旧错误代码直接移除；
  转换内核原样迁入 build_gait.py，自划分/quality gate/旧目录读取全部移除）
- `Baselines/Step2Motion/src/normalizer.py`（输出路径改为 canonical model_inputs，
  新增 `--adapter-version`）
- `Baselines/Step2Motion/src/train.py` / `test.py`（normalizer 改为从 config 推导的
  canonical 路径；test.py 新增 `--no-visualize` 供 smoke）
- `Baselines/Step2Motion/src/bvh_export.py`（euler 写入 xyz→zyx，见「修复的本地适配问题」）
- `Baselines/Step2Motion/configs/config_gait.json` / `config_gait_noimu.json`
  （数据路径 workspace://derived → model-input://Step2Motion/adapter_v1/gait[_noimu]）

## 新增产物（`AnysoleWorkspace/model_inputs/Step2Motion/adapter_v1/`）

```
gait/{gait_train,gait_val,gait_test}.pt   (50 维输入，train 36MB / val 11.8MB / test 11.8MB)
gait/normalizer_gait.pth + split_sessions.json
gait_noimu/...                            (38 维输入 + normalizer_gait_noimu.pth + split_sessions.json)
artifact.json（触觉来源链、时间轴、clip 策略、normalizer provenance、池化声明）
pressure_ranges.csv（每 session value_min/max、日期、源文件）
```

## 保留的上游语义

- BVH-23 Skeleton3 关节序/层级（SRC_JOINTS/SRC_PARENTS）、Y-X-Z euler 解析、
  root translation、合成 IMU（synthesize_imu）、foot/force/CoP 通道契约 —
  全部原样保留（转换内核逐行迁入 adapter，只改了数据来源与时间轴）。
- MotionDatasetState clip 布局：每 clip 丢弃首帧、clips 索引、50/38 维 insole 布局、
  69 维输出（3 displacement + 66 pose）、40Hz、distances/offsets/quats 契约。
- 48→16 池化（pool_48_to_16，D_Test4 已验收 heel[0:8]+toe[8:16]）逐字未动；
  新数据 16ch 与 shared 48 格逐帧数值一致（max diff 0）。
- 扩散模型、translation head、loss、上游 metrics.py 实现未动。

## 修复的本地适配问题

1. **时间轴**：BVH 采样改用 shared `mocap_time_s`，压力直接消费 shared
   `pressure_48.npz`（不再从 CSV 重建 visual/pressure offset 插值）。
2. **split**：train/val/test 只读 `protocol/splits/default/splits.csv`
   （当前冻结文件为 **104/36/36**，见「建议主 Agent 重点复核」），
   自行划分与 quality gate 已删除；fake 帧不进入 clip（唯一空 session
   S12102：284 帧全 fake，记录为 `all_frames_fake`）。
3. **frame id 链**：每 clip 保留 shared frame id（`clip_frame_ids`/`clip_valid`，
   与 .pt 行一一对应，已按首帧丢弃同步裁剪）。
4. **bvh_export euler 口径**：`write_bvh` 原写 scipy 'xyz' euler，公共 evaluator 的
   BVH 解析（bvh_aligner，标准 BVH 通道序 X→Y→Z = Rx@Ry@Rz）会重建出不同旋转
   （实测 S13073 世界坐标 mean 38mm / 手部 max 231mm 偏差）。改为
   `as_euler("zyx")` 值倒序写入（Rx@Ry@Rz 参数化）后，evaluator 读取的预测与
   模型输出精确一致（实测 mean 0.011mm / max 0.22mm）；results_display r_test1
   渲染器用 scipy 'xyz' 读文件（其既有口径），改动后 raw 与 gen 两个面板的
   解析口径一致。
5. **normalizer 位置**：train.py/test.py 不再读 `dependencies/Step2Motion/normalizers`
   静态目录，改从 config `train_data` 推导 `model_inputs/Step2Motion/<ver>/<name>/normalizer_<name>.pth`。
6. **results 布局**：test.py 预测输出根由旧 `results/Step2Motion/predictions` 改为
   canonical `results/baselines/Step2Motion/predictions/<model>`；两份 config 的
   `models_dir` 改为 `results://baselines/Step2Motion/checkpoints`
   （07 书 §3 + command_manual：正式输出统一在 `results/baselines/<Model>/`）。

## 未修复的上游缺陷（按总控 §2.3 只记录，不静默修复）

1. `utils.skeleton_pos_to_rot`（上游方向对齐法恢复局部旋转）缺少 twist 信息：
   实测 S13073 pred=GT 时其重建位置与真值 mean 263mm / max 904mm 偏差。
   生产导出 `test.py::pred_to_bvh` 走该函数，因此正式预测 BVH 的公共指标
   会包含该恢复损失。修复需另立 `<Model>-corrected` 或由主 Agent 裁定导出契约。
2. 公共 evaluator（`Baselines/utils/gt_loading.py::resolve_repo_path`）不支持
   raw:// 等 canonical URI（实测产生 `/raw:/bvh/...` 不存在路径），且
   `Baselines/utils/evaluate.py` 默认 manifest/split 路径指向已删除的旧位置。
   属公共 resolver/evaluator 收口（Agent A Wave 3）；本 Agent 的 smoke 用
   绝对路径 manifest 行验证了 evaluator 的读取+求解链路，未改公共代码。
3. 公共 evaluator 对 BVH 预测的网格对齐缺陷：
   `evaluate_motion_row` 用 GT 的录音时间网格查询预测 BVH
   （`bvh_joints(pred_path, row)`），而 test.py 导出的预测文件已位于
   canonical 会话网格（文件第 k 帧 = 网格第 k 帧）→ 被整体平移
   mocap_start（S13073 ≈ 29 帧，实测 pred=GT 时 mpjpe 151mm；按预测自身
   轴读取为 0.011mm）。建议主 Agent 裁定：预测 BVH 按自身轴读取
   （`bvh_joints(pred_path)` 不带 row），并明确多 clip session 的
   `<sid>_cN_gen.bvh` 命名与会话网格对齐契约。

## 任务书 §8 专项

- **旧/新 gait.pt session 差异**：旧产物（derived/Step2Motion/gait）已于
  09-26/27 迁移中物理删除，旧 splits.csv 亦不在 git 历史，无法逐 session 对账。
  已知旧管线另有 quality gate（A/B）与旧 split（92/12/36 时期）；新数据 =
  canonical 104/36/36，唯一空 session S12102（全 fake）。
- **normalizer 重建源**：仅 `gait_train.pt`/`gait_noimu/gait_train.pt`
  （canonical train，103 个有数据 session；S12102 无帧贡献），
  provenance（源文件+sha256+输出 sha256）记于 adapter artifact.json。
- **保留的原生输入语义**：见上「保留的上游语义」。
- **native metrics 与 formal public metrics 分界**：`src/metrics.py`（MPJPE/MPEEPE/
  MRPE/legs 变体，硬编码索引）未修改，仅作上游调试产物；正式比较全部由
  `Baselines/utils` 公共 evaluator 按 BVH-23 关节名选腿/脚/趾
  （LeftFoot/RightFoot/LeftToeBase/RightToeBase = 17/21/18/22，测试已按名校验）。
- **压力增益异常报告位置**：`model_inputs/Step2Motion/adapter_v1/pressure_ranges.csv`
  （每 session 日期、value_min/max、单位、左/右源文件）；跨 session 标定未做
  （按任务书 §5 不属本任务）。上游审计已知跨日期峰值 857~1966（2.3× 增益差）。

## 建议主 Agent 重点复核

1. **canonical split 计数与任务书不符**：任务书/shared_schema/
   smoke_baseline_integration 均写 92/12/36，但冻结文件
   `protocol/splits/default/splits.csv`（09-27 19:58 重生成）为 **104/36/36**
   （val=test=S13+S14，与 07 书 §4「val/test 无重复」一致）。Agent C 亦已报告
   同一差异。本 Agent 按「只读 splits.csv」执行（104/36/36），未改公共常量。
2. 上述公共 evaluator 的 canonical URI/默认路径缺口（Agent A Wave 3 收口）。
3. `skeleton_pos_to_rot` 导出恢复损失的处理裁定（record vs `<Model>-corrected`）。
4. `results_display/README.md:273`、`AnysoleWorkspace/README.md:115-117` 仍引用
   process_gait.py 与旧 derived 路径（Agent A Wave 3 文档收口范围）。
5. 旧协议产物候选删除（Wave 4 清单）：`results/baselines/Step2Motion/`
   下旧 checkpoints（gait_model：copy_config 指向旧 workspace:// 数据）、
   `metrics/test_comparison.{json,csv}`（全 NaN、0 有效帧）、
   `assets/third_party/Step2Motion/normalizers/normalizer_gait[_noimu].pth`
   （旧数据生成，新 normalizer 已在 model_inputs）。
6. 本 Agent smoke 产物（可随时删除）：`results/baselines/Step2Motion/smoke/`、
   `predictions/gait_model_smoke/`、`checkpoints/gait_model_smoke/`。

## 开始前已存在改动 / 本轮未触碰

- 开始前已存在：用户对 AnysoleWorkspace/tool 各公共工具、anysole/**、configs/**
  等的改动（git status 快照见会话记录）；`Baselines_Backup/` 为既有归档未触碰。
- 本轮未触碰：公共 split/manifest/path 实现、AnySole/MotionPRO/VP-MoCap/
  pressure_toolkit 目录、公共 metric 公式、`src/metrics.py`、`src/utils.py`、
  上游 normalizer 静态目录（`assets/third_party/...` 未写入）。

## 执行的测试 / 测试结果

`AnysoleWorkspace/tool/adapters/Step2Motion/tests.py`（touch_gait 环境）— **2314 项全部通过**：

- split_sessions.json 与 canonical splits.csv 逐项一致（104/36/36）、split/manifest
  sha256 匹配、无重复、train 与 val/test subject-disjoint；S12102 记录
  `all_frames_fake`；
- normalizer provenance 只有 train .pt hash + train sessions（无 val/test 泄漏）；
- gait/gait_noimu 三 split 契约：insole 50/38 维、pose 69 维（3+66）、BVH-23
  关节序/层级、quats (N,23,4)、offsets (C,23,3)、distances (C,22)、40Hz、
  每 clip frame id 严格递增且与行数一致、valid 掩码对齐；
- 16ch parity：S10101/S11073/S13011 的 insole pressure16 ==
  pool_48_to_16(shared left48/right48)（max diff 0），D_Test4 冻结池化语义不变
  （旧数值产物已于 09-26/27 迁移物理删除，parity 由逐字保留的池化函数+逐帧比对保证）；
- pred=GT 公共指标归零（BVH-23 关节名）：mpjpe/pa/w_mpjpe/wa_mpjpe/root_ate/
  rte/root_orientation/drift/mpjae/accel 全部 ≤1e-3（度/弧度 1e-6）；腿/脚组按
  关节名选择（乱序名仍按名解析，未知名报错）；
- 导出 smoke：pred_to_bvh/cs_to_json 原生路径写出成功；GT-export 经公共
  evaluator CLI（smoke registry）status ok；evaluator 读取路径下 pred=GT
  mpjpe/pa/root_ate ≈ 0（0.011mm，未含公共 pred 网格查询缺陷）；
- one-batch train smoke：canonical config 下数据→模型前向+反传一步 loss 有限；
- one-session test.py smoke：全新随机模型（diffusion_T=4）跑通 test.py 全链路
  （exit 0），导出 `<sid>_gen.bvh/_cs.json/_meta.json/metrics_summary.json`，
  公共 evaluator CLI 对该真实预测 status ok。

## 未执行的重型测试及原因

- 正式全量训练/测试（epochs_pose=100/epochs_trans=200，GPU）：本 Wave 不重训；
  one-batch train smoke + one-session test.py smoke（T=4 新随机模型）已覆盖数据→模型链路。
