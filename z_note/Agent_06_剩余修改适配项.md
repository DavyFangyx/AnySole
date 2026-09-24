# Agent 06 · 剩余修改适配项（2026-09-21，触觉落盘完成后）

> 本文档是 `Agent_06_三基线统一复现任务书.md` 的**现行工作清单**：以任务书为基准，
> 吸收 2026-09-21 的进展（下载就位、触觉三树落盘、nearest-cell 口径）与用户口径
> 修正，逐项列出剩余需要修改与适配的地方。每完成一项打勾并回填数值。

## 0. 口径修正（相对 Agent_06 原文）

> **关于"修 bug"的分类**：本文中带"修"字的项分三类——
> ① 上游真实缺陷（已逐处核验行号，如 `data_loader.py:244` 尾随空格、FPP-Net BCE
> 维度 192vs484、`optimize.py:146` Windows 反斜杠等，共 6 处）；
> ② 硬编码作者实验室路径（`/data1/shenghao/...`、`D:/Dataset/...`、`../../bodyModels/...`，
> 属路径重指适配，不动模型逻辑）；
> ③ 为 MMVP 数据集/30fps 写死的假设（31×11 insole、dt=0.033、sub_info），
> 属确定性映射/参数化适配。只有 ① 是真 bug，且 VP-MoCap README 自认
> "TODO: further polish the code"。模型结构/损失语义在任何一类里都不动。

| 编号 | 修正 | 说明 |
|---|---|---|
| C1 | **三基线 = MMVP（pressure_tookit + VP-MoCap 两模型）+ MotionPRO + Step2Motion** | 用户口径；原任务书把 pressure_tookit / VP-MoCap / MotionPRO 并列、未含 Step2Motion，书滞后 |
| C2 | **D2 作废**：不采用 4×12→31×11 分块膨胀，实际采用 **nearest-cell 模板足底系映射** | `AnysoleWorkspace/tool/generate_baseline_tactile.py` 已实装（口径逐函数一致于 `d_test4_baseline_tactile.py`，落盘代替内存） |
| C3 | **D3 被 nearest-cell 方案消解**：落盘 insole 就在上游 31×11 网格 + 模板足底系坐标系上 → FPP-Net 的 contact GT（`getVertsPress` 用上游 `insole2smplL/R.npy`）**直接复用上游文件，无需重建 insole2smpl**；剩一个数值验证项（E2） | 上游 `essentials/insole2cont/` 全套在仓库内 |
| C4 | test split = **36**（train 92 / val 12 / test 36） | 原书"31 session"是旧数 |
| C5 | 映射统计 meta 实际落盘为 `derived/baseline_tactile/<sid>.json` | 原生成器 docstring 写 `<sid>/meta.json`，以实际为准 |
| C6 | 触觉生成覆盖 **140 个 split session + S12072（示例）**，无缺口；manifest 中 S12072/S12081/S14101/S6101 未入 split（非 eligible），不需要生成 | 已核验 |

## 1. 已完成（2026-09-21 前/当日）

- [x] 下载 5 项解包落位：essential（含 SMPL_MALE/smpl_uv）、CLIFF、rtmpose-m halpe26、SMPL_MALE、V02_05
- [x] 触觉三树落盘（`generate_baseline_tactile.py`），契约核验通过：
  - `derived/MotionPRO/pressure_96/<sid>.npz`：`(T,96,96)` float32、0-255，与 `image_pressure.py` 现算 bilinear（align_corners=False）**最大偏差 0.0**
  - `derived/Step2Motion/pressure_16ch/<sid>.npz`：`left16/right16 (T,16)`、0-255
  - `derived/pressure_tookit/images/<date>/<sub>/<sid>/insole/%03d.npy` 与 `derived/VP-MoCap/<date>/<sub>/<sid>/insole/%03d.npy`：`{'insole': [L(31,11), R(31,11)]}`、0-255、帧号 0..T-1、**两树逐帧内容一致（抽 3 session 0/中帧核验 True）**
- [x] Step2Motion 磁盘代码已部分恢复（src/train.py、normalizer.py、dataset.py 等在）
- [x] 上游 FPP-Net essentials 确认仓库内自带（insole2cont 全套 + smpl_uv.obj + dataset_split 参考 + sub_info 参考）

## 2. 剩余项

### A. 通用/共享（MMVP 两条线 + 队列）

- [ ] **A1 mmvp 环境**：`conda create -n mmvp python=3.8` + torch2.4.0+cu121 + legacy `human_body_prior==0.8.5.0`（当前 PyPI 索引没有 `2.2.2.0`，使用 `--no-deps` 防止其拉低 Torch）+ open3d==0.18.x + torchgeometry 0.1.2 + pyrender + hydra-core 1.3 + trimesh + smplx + tensorboardX + numpy<1.24 + configargparse/icecream/xrprimer（toolkit 依赖）。实测 touch_gait 已含 torchgeometry/pyrender/hydra/trimesh/smplx/chumpy，只缺前两者；报错再退 py3.7/torch1.12
- [ ] **A2 RTM-pose keypoints**：新 env（mmpose，`openmim install mmpose`）+ `dependencies/rtmpose/rtmpose-m_..._body7-halpe26_700e-256x192-4d3e73dd_20230605.pth`（config 用 mmpose 包内 `rtmpose-m_8xb256-700e_body7-halpe26-256x192.py`）；对真实 RGB 逐帧出 HALPE-26 → 双路装配：toolkit `input/<sub>/<seq>/keypoints/%03d.npy` + FPP `datadir/<date>/<sub>/<seq>/keypoints/%03d.npy`（同内容两树，写入映射表）；**禁止用 3D GT 投影**
- [ ] **A3 CLIFF 双格式**：`dependencies/CLIFF/demo.py` + 现有 `dependencies/MotionPRO/cliff_ckpt/hr48-...3dpw.pt`，一次前向出：toolkit 的 `<seq>_cliff_hr48.npz`（`pose` 72 维，进 init_data_dir）与 PoseTransOpt 的 `CLIFF_results.npz`（`shape/pose/global_t`）
- [ ] **A4 深度**：把 backup 的 `rgb2depth.py` 合入 `Baselines/pressure_tookit/data_prep/`（depthpro env），生成逐帧 `depth/*.png`（uint16 毫米 + meta.json）+ `depth_mask/*.png`（有效性规则写进适配器测试）；PoseTransOpt 的 `template_scene_rgbd.npy` 单帧地面深度从同一源取（替代 ZoeDepth，记录偏离）
- [ ] **A5 calibration / floor**：`calibration.npy` 从 `AnysoleWorkspace/calibration/*.json`（cam3 K/D/R/t）生成；`floor_<sub>.npy` 从标定生成，与 PoseTransOpt 地面变换同源。**先排查 doctor 对 `insole_templates.json` 的校验错误**（缺 cameras K/D/R/t）——`calibration/foot_sensor_layout` 已被生成器使用，说明标定目录部分就位，剩余项需逐文件核验；无可信标定/地面的 subject 阻塞标红
- [ ] **A6 build_sub_info.py**（D7）：`AnysoleWorkspace/tool/build_sub_info.py`——每受试者第一个动作中「双脚着地 + BVH 速度最小」帧，weight=该帧左右脚压力总和（/255 口径）、max_value=255、height=-1，产出上游同构 `{'<sub>': {...}}`
- [ ] **A7 队列编排**：`offline_train/run.sh` 新增 `fpp_train` / `fpp_infer` / `posetransopt` case；pressure_toolkit 任务 `CONDA_ENV` 由 depthpro 改 **mmvp**（fitting 用；rgb2depth 仍 depthpro）；`create_queue.py` 同步

### B. MotionPRO（SMPL 派）

- [ ] **B1 恢复集成版**：`git checkout 56f5187 -- Baselines/MotionPRO`（磁盘现只剩 `lib/` + `data/`，app/config/data_prep 全缺）；确认 workspace.py、image_pressure.py、FRAPPE.py、test_frappe.py、data_prep 齐全
- [ ] **B2 D1 GT 重导**：新脚本（`AnysoleWorkspace/tool/`，旧 data_prep 已出 index）从 manifest `smpl_path`（`motion_neutral_smpl.npz`）重导出 `smpl.npy / keypoints.npy`：40Hz 统一网格重采样（实测 T 不变）、**63 维 pose_body → 69 维映射**（不得沿用 `BVH_TO_SMPL_BODY`）、betas 真实化、**numpy2 pickle 坑**（训练端走 `load_smpl_npy()` 或 numpy1 写入）
- [ ] **B3 触觉切换**：`image_pressure.py` 改读 `derived/MotionPRO/pressure_96/<sid>.npz` 落盘（去掉每 epoch 内存 bilinear；0 偏差已核验，切换无风险；保留内存 resize 作 fallback 或直接删并加存在性检测）
- [ ] **B4 标签与沿用**：contact 标签用 `AnysoleWorkspace/tool/contact_labels.py`（bvh_h/bvh_soft）与 D1 同批重生成；`bbox.npy / feature_hrnet.pth / fake_mask.npy` 沿用现有
- [ ] **B5 训练与出口**：队列 `MODEL=motionpro`、`RUN_DIR=results/baselines/MotionPRO`、EPOCHS=1000/BATCH=16 → `test_frappe.py` 评估 → 补导出步骤写 `results/baselines/MotionPRO/predictions/eval_motion/<sid>.npz`（SMPL-24 joint_xyz_world + joint_names + valid_mask）→ `r_test2_compare.py` → `r_test1_visualize_motionpro.py`

### C. Step2Motion（BVH 派）

- [ ] **C1 补回 process_gait.py**：磁盘缺（backup 有）；从 `Baselines_backup/Step2Motion/src/process_gait.py` 或 git 历史恢复
- [ ] **C2 压力源切换 + 交叉验证**：process_gait 的 `pool_48_to_16`（Moticon 16ch 冻结口径 heel[0-7]+toe[8-15]）与落盘 `pressure_16ch` **数值一致断言**（逐帧 max 偏差=0 期望）；COP/force/IMU 仍由 process_gait 从 48 点 CSV 算（16ch 落盘不含 cop/force），改读落盘 16ch + 保留 48 点读 cop/force，或保留原路径加交叉验证
- [ ] **C3 gait pt 与现行 split/manifest 对齐核验**：现有 `derived/Step2Motion/gait/gait_{train,val,test}.pt` 为 BVH 时代产物——核验 T/fake mask/session 集与现行 splits.csv 一致；GT 本来就是 BVH-23（BVH 派，政策不变），一致则沿用、不一致则重生成
- [ ] **C4 训练与出口**：队列 case `step2motion` 已有（config_gait.json、normalizer_gait.pth 已就位）→ 训练 → 推理 → 导出 BVH-23 `joint_xyz_world + joint_names` → 统一评估（`r_test2_compare` 已支持 bvh23 协议）

### D. MMVP 方法线 A：pressure_tookit

- [ ] **D1 合并备份集成**：`Baselines_backup/pressure_tookit` 的 `lib/utils/workspace.py`、`data_prep/rgb2depth.py`、`main_singleview.py` resolve_path、config workspace:// 化合入当前干净副本；修 3 处 bug：`data_loader.py:220` keypoint 相对路径（改相对 basdir）、`:244` `'init_shape '` 尾随空格、根目录 `run.sh` 指向不存在的 smpl_fitting.py
- [ ] **D2 三阶段拟合**：`init_shape → init_pose → tracking`（create_queue 已生成任务；CONDA_ENV=mmvp；test split 36 session，先单序列 smoke 测单帧耗时再全量）；insole 输入直接指向已落盘的 `images/<date>/<sub>/<sid>/insole/`
- [ ] **D3 导出与评估**：逐帧 `smpl_<idx>.npz`（body_pose 69 + global_rot + transl 地面系米 + betas + scale）→ FK（含 model_scale_opt/betas[:,0] 尺度）→ 地面系→世界系（标定变换）→ `results/baselines/pressure_tookit/predictions/eval_motion/<sid>.npz` → Test2 + 可视化

### E. MMVP 方法线 B：FPP-Net

- [ ] **E1 上游 bug 修复**：`trainer_tempkpSMPLCont.py:50` BCE 维度 192 vs 484（改用 `contact_smpl` 或与推理一致）；`record.py` 硬编码 `../../bodyModels/.../smpl_uv.obj` 改指 `essentials/smpl_uv.obj`；`datadir` / `InsoleModule('/data/PressureDataset')` 配置化
- [ ] **E2 contact GT 验证（D3 消解后的残余验证项）**：用落盘 insole + 上游 `insole2smplL/R.npy` 跑 `getVertsPress`，核对 (2,96) 接触 GT 的物理合理性（脚跟帧→后跟顶点点亮、脚尖帧→前足顶点点亮；与 D_Test5 面板已核验的 corr 口径一致）
- [ ] **E3 tv_fn 生成**：splits.csv → 上游 `{'train': {date: {sub: {seq: [frame,...]}}}, 'test': ...}` 格式（写生成器 + 单测）
- [ ] **E4 训练与推理**：`fpp_train`（mmvp env；datadir=derived/VP-MoCap、sub_info=A6、tv_fn=E3）→ test 推理 → `derived/VP-MoCap/<date>/<sub>/<sid>/pred_contact_smpl/%03d.npy`（契约键 `contact_smpl.pred`，(2,96)）；报告 pressure MSE / cont MSE / BCE

### F. MMVP 方法线 C：PoseTransOpt

- [ ] **F1 上游可移植性 bug**：`app/optimize.py:146` `split('\\')`；`task/MMVP.yaml` Windows 占位路径；`transolver.py:13` `models\SMPL_NEUTRAL.pkl`
- [ ] **F2 帧率参数化**：`lib/initial_trans/initial_trans.py` 飞行段抛物线 `dt=0.033`（30fps 硬编码，已核验）→ 参数化 1/40（配置级）；`lambda_timing` 留 smoke 后按需 ×40/30，记录偏离
- [ ] **F3 models 装配**：`models/SMPL_NEUTRAL.pkl` ← `dependencies/smpl`；`SMPL_MALE.pkl` ← `dependencies/VP-MoCap/smpl`（已就位）；`V02_05/` 已解到 `models/`（已就位）
- [ ] **F4 运行与出口**：per-session `input_path_base=derived/VP-MoCap/<date>/<sub>/<sid>/`（color/keypoints/pred_contact_smpl/CLIFF_results.npz/template_scene_rgbd 全部来自共享产物）→ `app.optimize`（mmvp env）→ `opt_result.pth` → FK SMPL-24 → 世界系 → 统一导出 npz（范围同 D6：先 test 36）

## 3. 执行顺序建议

1. **Phase 1 MotionPRO**（B1–B5）：最快出 Test2 数值行，且与 MMVP 前置并行
2. **Phase 2 MMVP 共享前置**（A1–A6）：env 最耗时先建；A5 标定排查是唯一潜在阻塞点
3. **Phase 3 MMVP 方法线**（D→E→F）：A 完成后 D 与 E/F 可并行（E 需 A2/A6/E3）
4. **Phase 4 Step2Motion**（C1–C4）：独立，随时可插
5. **Phase 5 收口**：三基线 predictions 齐全 → `r_test2_compare.py` → Test2 五行（AnySole/MotionPRO/Step2Motion/pressure_tookit/VP-MoCap）→ Test1 可视化 → 更新 `总体验收矩阵.md`

## 4. 验收要点（沿用 Agent_06 §8 禁止事项）

- 模型代码改动逐条说明语义不变性；无 3D GT 冒充 2D 观测；不伪造标定/floor；不重写时间轴/split；val 不当 test；单位/帧率/掩码回传显式标注（40Hz、mm、fake 排除）
- 触觉层已完成且 0 偏差核验——后续任何改动不得重新引入内存转换口径差异
