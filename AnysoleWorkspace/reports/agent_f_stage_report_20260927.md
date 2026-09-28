# Agent F Wave 2 交付报告：pressure_toolkit 语义保真适配

任务：`z_note/重构执行/06_AgentF_pressure_toolkit适配.md`
状态：**complete**（生产全量 fitting 待上游前端：SAM3.1 索引完整化、CLIFF 单人前端）

## 修改文件

`Baselines/pressure_tookit/**`（模型目录，按 .gitignore 约定不入 git）：

- `lib/utils/workspace.py` — 重写：委托冻结的 `AnysoleWorkspace.tool.workspace.resolve_uri`，删除 `workspace://`（含 derived/dependencies 旧路径）
- `lib/config/config.py` — 删除 `--depth-sample-stride`（默认 2）；新增 `--depth-approx-stride` 默认 **1**（显式近似/加速模式）；basdir/essential_root 默认改为 canonical URI
- `lib/core/fit_single_frame.py` — 恢复上游全采样口径；stride>1 仅在显式传入时生效
- `lib/dataextra/data_loader.py` — 全量重写为 shared frame id 契约（见下）
- `main_singleview.py` — frame-id 命名输出、`work://` 严格输出根、近似模式告警
- `run_full_mmvp.py` — 重写：protocol manifest/split 发现、fake 跳过、run artifact、导出调用
- `data_prep/rgb2depth.py` — 重写：shared facts rgb → adapter depth 树（只生成拟合帧）；删除旧 derived/positional 模式
- `configs/fit_smpl_rgbd.yaml`、`run.sh` — canonical 路径
- `tests/` — 新增 3 个测试文件 21 用例

`AnysoleWorkspace/tool/adapters/pressure_toolkit/`（新增）：

- `build_inputs.py`（主入口）、`common.py`、`floor.py`、`masks.py`、`insole.py`、`calibration.py`

`results_display/`（仅 D_Test6）：

- `script/d_test6_floor.py` + `DataTest/D6Test_floor/`（4 张审计图 + candidates CSV + README 确认清单）

`AnysoleWorkspace/tool/export_baseline_motion.py`（**越界最小修复**，请主 Agent 复核）：

- 该文件工作树版本存在 `resolve_uri` 局部重定义遮蔽冻结 resolver，导致 `session_parts` 传 `must_exist=True` 必现 TypeError，标准 prediction 导出无法运行。已改为委托冻结 resolver（保留 workspace:// 拒绝），19 行。未触碰任何指标公式。

## 新增产物

- `model_inputs/pressure_toolkit/v1/`：images（color 链接/depth/depth_mask/insole/calibration.npy/frame_ids.npy）、annotations/floor_info（floor_<subject>.npy + floor_artifact）、input keypoints、initialization（CLIFF 暂空）、session_reports、artifact.json。已为 S10101/S5011/S8011/S13011 构建（冒烟/审计）。
- `work/pressure_toolkit/<run_id>/`：fitting/results、meshes、yamls、temp、gt_depths、logs、artifact.json（launcher 生成；冒烟 run 已清）。
- D_Test6 四日期地面候选与审计图。

## 保留的上游语义

- 拟合数学/权重零改动（losses/contactTerm/depthTerm/camera/smpl_mmvp 拟合部分均未动，depthTerm 仅保留既有性能实现）
- 31×11→9 区域接触映射与 `Baselines_Backup` 上游逐值一致（有 parity 测试）
- HALPE-26 → 上游 loss 顺序由原有 `joint_mapper` 完成，未改 loss
- loader 3×3 mask 膨胀为上游原生行为，作用于交集后的人体 mask
- floor 算法为上游 `calculateFloorNormal`，棋盘格标定仅进入相机变换

## 修复的本地适配问题

- **P1 地面**：废弃棋盘格平面当行走地面的旧适配（`prepare_mmvp_observations.py` 的 `write_calibration_and_floor`）。四日期实测：行走地面拟合残差 1.5–2.4 mm；棋盘格平面偏移 **1.293–1.445 m**（20260810 = 1.445 m，即任务书所述约 1.45m）。D_Test6 待用户四日期确认。
- **P2 depth mask**：`human_mask_sam31 ∩ finite_depth ∩ [0.4m,5m]`，逐帧保存 frame_id、mask path/hash、时间误差、交集前后面积、空 mask 检查。实测 S10101 frame 4/5：human 70568/70408 px → 交集 69389/69293 px。
- **P3 采样**：默认恢复上游全采样；近似模式改名显式声明，正式 yaml 不含该开关。
- **P4 insole**：只读消费 `shared/representations/tactile/mmvp_31x11/v1`，逐帧包装 `{'insole':[L,R],'frame_id','source'}`，数值 parity 已验证（474/474）；fake/invalid 帧不生成拟合任务（frame_ids.npy 只含 valid 且非 fake），全 fake session 拒绝构建。
- **P5 CLIFF**：`load_init_pose` 按 shared frame id 精确取行，多人行 [0] 静默回退已删除，无 frame_id 的多行档案报契约错误。
- **P6 NaN**：回滚后 break 继续序列的控制流与上游一致（已单测）；旧 `reset_params` 崩溃调用此前已由 copy_ 内联替代。
- **P7 前帧接触**：进入 optimizer 前验证前一有效拟合帧 insole，缺失报 `ContractError`，负索引回绕已删除。
- **P8 HALPE**：keypoint 读取改为单受试者 dict 契约，多人数组 [0] 回退删除。

## 未修复的上游缺陷（记录，不静默修）

- `depth_utils.normalizeFloor`：拟合法向恰为 [0,1,0] 时叉积除零（实测地面不会触发）。
- `calculateFloorNormal` 无条件写 CWD `debug/depth_slice_rot.obj`（adapter 以临时目录隔离并清理）。

## 执行的测试

| 测试 | 结果 |
|---|---|
| `Baselines/pressure_tookit/tests/`（21 用例：P1 平面恢复/artifact/棋盘格审计、P2 交集与膨胀位置/空 mask、P3 stride 默认与正式配置、P4 insole parity/形状拒绝、P5 CLIFF 单人行选择/拒绝、P6 NaN 回滚、P7 前帧接触缺失×2+正常路径、P8 接触映射 parity/HALPE mapper/kp 契约、标定契约、导出+validator 链） | 21 passed |
| 真实数据 smoke：S10101 build_inputs（color 474、insole 474 parity、floor residual 6.2mm、棋盘格偏移 1.341m） | 通过 |
| one-frame fitting smoke（S10101 frame 4，init_shape，真实 depth+mask+kp+calib+floor，maxiters=3，GPU3） | 通过：smpl npz 全 finite，transl y≈1.0m（地面之上，无 1.45m 偏置） |
| one-session resume smoke（frame 5 tracking：前帧 insole/结果读取，契约通过） | 通过 |
| 标准 prediction export + Agent A validator（合成 2 帧 → `export_baseline_motion.py` → `validate_baseline_exports.validate_motion`） | 通过 |
| `run_full_mmvp.py --dry-run`（train split 发现 S10101、init_shape+init_pose 调度、artifact/日志/initialization 链接） | 通过 |
| D_Test6 四日期 36 候选 | 通过（图待用户确认） |

## 未执行的重型测试及原因

- 全量 140 session 生产 fitting：依赖 SAM3.1 索引完整化（当前索引仅 S10101 且 `time_error_s` 全为 None，按 20ms 容差全部判 invalid）与 Agent E 的 CLIFF 单人前端（未交付）。
- init_pose 真实 CLIFF smoke：同上（resume smoke 用契约形状的 dummy cliff，仅验证 tracking 控制流，已注明并清理）。

## 开始前已存在改动

- wave-1 未提交改动：`export_baseline_motion.py`（含遮蔽 bug）、`prepare_mmvp_observations.py`（棋盘格 floor 旧适配）、`run_rtmpose_halpe26.py`、pressure_washer 等。
- SAM3.1 索引在任务执行期间被并行重建（帧覆盖/字段在变化）。

## 本轮未触碰的用户改动

- `anysole/**`、`Baselines/MotionPRO|Step2Motion|VP-MoCap/**`、`Baselines/utils/**`、`configs/**`、`results_display/**`（除 D_Test6）、pressure_washer。

## 建议主 Agent 重点复核

1. `export_baseline_motion.py` 的 resolve_uri 越界最小修复（19 行）。
2. 收口清单（均在我写范围外，未动）：
   - `configs/tools/create_queue.py`/`configs/run.sh` 仍硬编码 `workspace://derived/pressure_toolkit`（旧路径已删，队列任务将失败）；
   - `AnysoleWorkspace/tool/prepare_mmvp_observations.py` 被本适配器取代（其 `write_calibration_and_floor` 即 P1 所述棋盘格地面 bug，且写入 VP-MoCap 树），建议删除；
   - `AnysoleWorkspace/tool/run_rtmpose_halpe26.py` 写入旧 `model_inputs/pressure_toolkit/input` 路径（非 v1），需指向 shared 前端或 v1 树；
   - `export_baseline_motion.py --pressure-root` 默认 `model-input://pressure_toolkit/fitting` 已过期（fitting 输出在 `work://`），run_full 已显式传绝对路径；
   - SAM3.1 索引需 Agent A 完成 140 session、20ms 容差字段；
   - 需 Agent E 的 `shared/frontends/cliff_hr48/v1`（adapter 按该契约消费：`<date>/<subject>/<session>/<session>_cliff_hr48.npz`，pose (T,72) + frame_id）。
3. 生产口径：D_Test6 四日期用户确认后，如确认 ROI 与默认（底部 55–90% × 30–70%）不同，用 `--floor-roi`/`--floor-frame` 固定（20260804 默认 ROI 残差 ~20mm，最佳候选来自更宽 ROI）。
