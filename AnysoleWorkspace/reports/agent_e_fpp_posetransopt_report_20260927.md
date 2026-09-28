# Agent E Wave 2 交付报告：FPP-Net 与 PoseTransOpt 语义保真适配

任务：`z_note/重构执行/05_AgentE_FPPNet与PoseTransOpt适配.md`
状态：complete（F0/V5、V1、V2、V3 全修；数据生产 140/140 完成；smoke 与契约测试 9/9 通过；正式全量训练待主 Agent 调度）

## 1. 修改文件

### Baselines/VP-MoCap/FPP-Net/**（FPP-Net 基线代码）

| 文件 | 变更 |
|---|---|
| `lib/Dataset/PressDataset/PED_tempKPCont.py` | F0/V5：删除 `contact_f6_soft.npy` 依赖与脚级标量广播链；恢复 `InsoleModule.press2Cont` 格级二值接触 + `getVertsPress(soft=False)` 192 维顶点级二值 GT；insole 改从公共 `shared://representations/tactile/mmvp_31x11/v1` 只读；keypoint 侧车校验 stored frame_id；batch 携带 temporal-5 中心 frame id |
| `lib/Dataset/InsoleModule.py` | 删除 `press2Cont` 内的 CWD `debug/sigmoid.png` 调试写盘；语义不变 |
| `lib/config/config.py` | URI 解析增加 `shared://`、`model-input://`、`work://`（走公共 `AnysoleWorkspace.tool.workspace.resolve_uri`）；新增 `dataset.tactile_root` / `dataset.prediction_root` |
| `configs/temporalKPSMPLCont_series5_mlp.yaml` | datadir/tv_fn 切 `model-input://FPP-Net/adapter_v1`；新增 `tactile_root`、`prediction_root`（`work://VP-MoCap/v1/fpp_predictions`）；checkpoint/log/result 移 `results://baselines/FPP-Net/`（总控 §2.7） |
| `app/infer_smplcont.py` | pred_contact_smpl 侧车带 `frame_id`、按 6 位帧号写入 `work://VP-MoCap/v1/fpp_predictions`（不再写进 datadir） |

### Baselines/VP-MoCap/PoseTransOpt/**（PoseTransOpt 基线代码）

| 文件 | 变更 |
|---|---|
| `lib/dataset/dataset_mmvp.py` | V1：改为按 frame id 装载三路输入（keypoints / CLIFF npz / pred_contact_smpl），每路校验 stored frame_id；savgol 与 quaternion 连续性修复按**连续片段**进行；`[2:-2]` 位置裁剪改为**逐片段端裁剪**（frame-id 感知），禁止跨模态按位置 zip |
| `app/optimize.py` | 输出默认根 `work://VP-MoCap/v1/pose_optimization/<date>/<subject>/<session>`；`opt_result.pth` 增加 `frame_ids`（优化后帧的 shared frame id）；写 `artifact.json`；URI 解析支持 canonical scheme |
| `run_full_mmvp.py` | 全部切换 canonical 路径（protocol/manifests、model-input://PoseTransOpt/adapter_v1）；移除旧 `derived/VP-MoCap` 与 `expected_fpp_windows` 位置守卫；改读 `join_manifest.json` 判定可运行性 |

### AnysoleWorkspace/tool/adapters/**（新适配器，Agent E 范围）

| 文件 | 职责 |
|---|---|
| `FPP-Net/build_metadata.py` | temporal-5 split 树（train 104 / val 36 == test 36，镜像 canonical val==test）+ V3 pixel_weight：站立帧 31×11 **mask 域**总和（`pixel_num=484` 同域），sub_info 记录 unit/domain/standing_frame_id/source_hash；sigmoid 输入/饱和率/动态范围报告（`fpp_metadata_report.json`） |
| `FPP-Net/build_inputs.py` | color 软链 + keypoints/公共 insole 校验 + adapter_manifest；**不复制**私有 31×11 树 |
| `FPP-Net/run_rtmpose.py` | RTMPose HALPE-26 keypoint 侧车（含 frame_id），一次推理可同时写 FPP/PoseTransOpt 两棵树；支持 shard/limit |
| `FPP-Net/export_v2t.py` | `results/baselines/FPP-Net/predictions/v2t/<session>.npz`（frame-id 放置；`contact_gt_source=press2Cont_binary_vertex`） |
| `PoseTransOpt/run_cliff.py` | V2 单人 CLIFF NPZ：按 `visual_time_s` 最近邻（≤20ms）匹配 mask 行、**从 mask PNG 重算 bbox**（不信任 CSV bbox）、无多人回退；NPZ 一行/帧 + frame_id/valid/bbox/time_error/来源 hash；既有文件行数违约自动重建 |
| `PoseTransOpt/build_inputs.py` | frame-id inner join（keypoints/CLIFF/FPP 三路）+ `join_manifest.json`（每路帧数、join 数、缺帧列表与原因、fake/invalid 剔除、片段边界、<21 帧短片段剔除）；color 仅链最终帧；FPP 侧车链入 PoseTransOpt 树 |
| `PoseTransOpt/make_template_scene.py` | 首个 valid 帧 DepthPro RGB-D 模板（1624×1240 口径） |
| `PoseTransOpt/export_v2m.py` | `results/baselines/VP-MoCap/predictions/eval_motion/<session>.npz`（frame-id 放置，废弃 `start=2` 位置重建） |

### 测试

`tests/test_fpp_posetransopt_adapter.py`（T1–T9，见 §5）。

### 删除（用户裁定：错误代码/文件路径直接移除，不保留历史版本）

- `Baselines/VP-MoCap/FPP-Net/{checkpoints,log,results}`（f6_soft 时代产物）；
- `Baselines/VP-MoCap/PoseTransOpt/outputs/2026-09-{24,26,27}`（错位数据时代优化产物）；
- `AnysoleWorkspace/tool/build_fpp_metadata.py`（被 `adapters/FPP-Net/build_metadata.py` 取代，含 V3 口径错误）。

## 2. 新增产物

- `AnysoleWorkspace/model_inputs/FPP-Net/adapter_v1/{dataset_split_temporal5.npy,<date>/sub_info.npy,fpp_metadata_report.json,<date>/<subject>/<session>/{color,keypoints,adapter_manifest.json}}`
- `AnysoleWorkspace/model_inputs/PoseTransOpt/adapter_v1/<date>/<subject>/<session>/{CLIFF_results.npz,color,keypoints,pred_contact_smpl,template_scene_rgbd.npy,join_manifest.json}`
- `AnysoleWorkspace/work/VP-MoCap/v1/{fpp_predictions,pose_optimization}/...`
- `results/baselines/FPP-Net/predictions/v2t/<session>.npz`、`results/baselines/VP-MoCap/predictions/eval_motion/<session>.npz`

## 3. 保留的上游语义（V4/V6/V7 裁定）

| 项 | 裁定 | 证据 |
|---|---|---|
| V4 flip 增强不对称（kp x 取负不换关节索引、insole/contact 换脚） | **上游保留**（不关闭、不改写） | `PED_tempKPCont.py` flip 块与上游逐行一致（基线语义审计总报告 §V4：「与上游逐行一致，属继承缺陷被激活」）；本 Agent 重写时逐行保留，仅把 contact GT 计算移到 flip 之后以保持「GT 随输入镜像」语义。`aug.is_aug` 保持上游默认 True，未为提高结果关闭 |
| V6 主点用图像中心（cx=image_width/2, cy=image_height/2）弃真实标定 | **上游保留** | `depth_to_pointcloud` 的 cx/cy 调用位于 `dataset_mmvp.py`/`initial_trans.py`，与上游 MMVP 代码一致（1624×1240 ZED、focal 1394 图像中心口径）；`calibration.npy` 为项目本地资产，上游从不消费。记录不修 |
| V7 loss_2d 置信度索引顺序错配（位置经 HALPE→OpenPose 映射、置信度未重排） | **上游保留** | `lib/loss/losses.py` 的 OPENPOSE/HALPE 分组与上游逐行一致（审计 §V7「继承上游」）；PoseTransOpt 优化目标未改动 |

以上均为上游原生行为：正式 baseline 保留并记录，不在 loss/augmentation 层修。

## 4. 修复的本地适配问题

- F0/V5：f6_soft 脚级广播 GT 回退为 press2Cont 原生顶点接触（192 维二值）；训练 target shape 与接触头结构不变。
- V1：三路输入按 frame id join；禁止 `[2:-2]` 位置 zip；savgol/裁剪逐片段。
- V2：CLIFF 恢复单人契约——mask PNG 重算 bbox + 时间容差 20ms；invalid 零行不回退多人检测；133 份旧异常 NPZ 已随旧树删除，新树 140 份全量重建且生成器按行数/frame_id 校验。
- V3：pixel_weight 改为 31×11 mask 域总和（与 `pixel_num=484` 同域），废弃 96 格和口径；sub_info 记录 unit/domain/standing_frame_id/source_hash，报告 sigmoid 输入/饱和率/动态范围。
- 旧 `workspace://derived/VP-MoCap`、`splits/default`（workspace 根）等死路径全部清除。

## 5. 执行的测试与结果

| # | 测试 | 结果 |
|---|---|---|
| T1 | FPP dataset one-batch：GT (B,192) 二值原生接触、无 f6 依赖、frame id 在 batch 内 | **通过**（S14011 frame 2，unique={0,1}） |
| T2 | 原生接触脚内差异统计（120 帧）：四区二值变化 L toe 120/120、L heel 120/120、R toe 0/120、R heel 22/120；脚内不同值帧 142；无广播链 | **通过** |
| T3 | S14011 5 帧公共 31×11 insole 与已审计 4×12→31×11 映射逐 bit 一致 | **通过** |
| T4 | model_inputs/FPP-Net 无私有 31×11 副本 | **通过** |
| T5 | FPP contact loss one-step 有限（loss 0.584，max\|grad\| 2.2e-2） | **通过** |
| T6 | 20 个 valid/fake 缺口 session 的 frame-id join 零静默错位 | **通过（20/20）** |
| T7 | 140 份 CLIFF NPZ 行数==帧数、frame id 唯一、valid==mask 重算判据、不回退多人 | **通过（140/140）** |
| T8 | FPP V2T one-session smoke + 标准 archive validator | **通过**（S14011，393/456 帧） |
| T9 | PoseTransOpt/V2M one-session smoke（含 fake 缺口 session，不因 CLIFF 行数崩溃） | **通过**（S14011，join 393→final 381，30 帧 smoke 限额） |
| T10 | RTMPose keypoints 全量（52,612/52,612 双树） | **通过** |
| T11 | CLIFF one-session smoke（S11013：360 行、346 mask-valid、时间误差 ≤12.5ms、mask 重算 bbox） | **通过** |
| T12 | 全量 join manifests（140/140，无失败） | **通过** |
| T13 | FPP 3-epoch 原生 GT 训练 smoke（loss 0.34→0.24）+ train/val/test 全 split 推理（51,145 侧车） | **通过**（smoke 产物已从 formal results 清除） |

未执行的重型测试：FPP-Net 全量 1410-epoch 正式训练与正式推理（smoke 3-epoch 训练仅用于打通管线；正式训练在最终验收前由主 Agent 调度）。

## 6. 开始前已存在改动 / 本轮未触碰

- 开始前：`AnysoleWorkspace/tool/{build_fpp_metadata,run_cliff_mmvp,run_rtmpose_halpe26,prepare_mmvp_observations,export_baseline_motion}.py` 等旧工具已修改（M，评估整改时代）；本轮除 `build_fpp_metadata.py`（FPP 专属、已删除）外均未触碰。
- 未触碰：`Baselines/pressure_tookit/**`、其他模型 adapter、`results_display/**`、公共指标公式、Agent A 的 shared/protocol 文件。

## 7. 建议主 Agent 重点复核

1. **human-mask frontend 与总控 §2.2 契约不一致（接口需求）**：当前 `shared/frontends/human_masks/sam31/v1` 为 `direct_csv_bbox` 模式——按 `visual_frame` 行号匹配（非 visual_time_s）、bbox 直取 CSV（24,150 个"valid"帧 bbox 全零，如 S5011 491/491 行 `skipped_existing`）。总控要求「bbox 从实际 mask 重算、CSV bbox 只作审计、20ms 容差」。Agent E 未改该文件，在 `run_cliff.py` 适配层自行实现了时间匹配 + mask PNG 重算 bbox（实测全部帧 ≤13ms）。请 Agent A 裁定是否把 frontend 重建为 mask-recompute 模式。
2. **split 计数**：`protocol/splits/default/splits.csv` 实际为 train 104 / val 36 == test 36（Agent C 已报同一发现，待 Agent A 裁定）。FPP split 树已按「val==test 镜像」生成；若裁定改为 92/12/36 需重跑 `build_metadata.py --force`。
3. `export_baseline_motion.py`（共享工具）的 `export_fpp_v2t`（`contact_gt_source=f6_soft`）与 `export_vp_mocap`（`start=2` 位置重建）为旧口径，Agent E 已在自己的 adapter 中提供 frame-id 版本；建议 Agent A 收口时切换共享工具到新适配器实现。
4. RTMPose 环境修复（机器级，非仓库）：`bbox_scan` 环境的 `mmpose.egg-link` 与 `easy-install.pth` 仍指向已删除的 `AnysoleWorkspace/dependencies/mmpose`，本轮已改为 `AnysoleWorkspace/assets/third_party/mmpose`（已验证 `import mmpose` 通过）。
5. `FPP-Net/configs` 的 `trainer.module` 字段仍写 `trainer_tempkpSMPLCont_mse`（不存在的模块名；trainer 实际经 `trainer.path` 装载，未改动，仅记录）。
