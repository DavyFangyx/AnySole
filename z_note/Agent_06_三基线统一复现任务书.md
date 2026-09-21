# Agent 06：三基线（MotionPRO / pressure_tookit / VP-MoCap）统一复现任务书

> 版本：2026-09-20，SMPL-24 时代。本任务书取代 Agent_02（MotionPRO）与
> Agent_04（pressure toolkit）中已过时的部分（Skeleton3/23 关节、`data/` 相对路径、
> BVH 派生 GT 口径等），并新增 VP-MoCap 工作包。原则不变：**模型代码沿用上游，
> 数据、训练、测试全部接入 AnysoleWorkspace 统一体系**。

## 1. 目标

在统一数据/划分/评估体系上复现三个 baseline：

| baseline | 方法性质 | 输入观测 | 输出 |
|---|---|---|---|
| MotionPRO | 训练式回归（FRAPPE） | HRNet 视觉特征 + 压力图 | SMPL 参数逐帧回归 |
| pressure_tookit | 逐帧优化拟合（MMVP 官方，SMPLify 风格） | RGB + 深度 + RTM-pose 2D + 压力接触 | SMPL 参数逐帧拟合 |
| VP-MoCap | FPP-Net（训练式）+ PoseTransOpt（逐帧优化） | RTMPose 2D → 压力/接触预测 → 姿态+平移优化 | SMPL 参数逐帧优化 |

产出：各 baseline 在统一 test split 上的 checkpoint/拟合结果、统一格式预测
（`results/<Model>/predictions/eval_motion/`）、Test2 统一对比行、可视化。

## 2. 当前实际情况盘点（2026-09-20，与旧文档的差异）

### 2.1 统一体系现状（已冻结，直接使用）

- **manifest**：`AnysoleWorkspace/manifests/session_manifest.jsonl`。逐 session 含
  `smpl_path`（原生 SMPL-24 NPZ，动捕 GT 的规范来源）、`bvh_path`（baseline-only，
  仅供 Step2Motion）、`pressure_path`、`video_path`、`visual_start_s / offset_s /
  n_frames / target_fps=40`、`fake/valid_frame_indices`、`split_iid / split_ood`。
- **split**：`AnysoleWorkspace/splits/default/splits.csv`（列 `index,train,val,test`，
  subject-disjoint IID，val==test，train≈70 / test=31 session）。下游只读不生成。
- **统一序列**：`derived/MotionPRO/sequences/cam3/<date>/<subject>/<session>/` 九件套
  （`align_meta.json, color/*.jpg, pressure.npz (T,160,120 @0-255), smpl.npy,
  keypoints.npy (T,24,3), contact.npy (T,10), contact_<method>.npy ×8, bbox.npy,
  feature_hrnet.pth (T,2048), fake_mask.npy`）。日期 20260804/07/08/10。
- **依赖**：`dependencies/smpl/{SMPL_NEUTRAL.pkl, smpl_mean_params.npz}` ✓；
  `dependencies/MotionPRO/{cliff_ckpt/hr48-...pt, mmdetection/yolox...pth}` ✓；
  `dependencies/pressure_tookit/depthpro/{config.json, model.safetensors,
  preprocessor_config.json}` ✓；`dependencies/pressure_tookit/essential/` **缺**。
- **运行编排**：`offline_train/` 队列（`create_queue.py → queues/GPU<N>/pending →
  scheduler.sh → run.sh`）。`run.sh` 已有 case：`anysole / motionpro / step2motion /
  pressure_toolkit`；`create_queue.py` 已会为 pressure_toolkit 按
  `init_shape → init_pose → tracking` 三阶段、逐 session 生成任务（同 GPU 串行）。
- **统一评估契约**：`results/<Model>/predictions/eval_motion/<session>.npz`
  （`joint_xyz_world (T,J,3) + joint_names + valid_mask`，SMPL-24 或 BVH-23 协议自识别），
  由 `results_display/script/evaluate_compare.py` 做 19 关节语义交集对比，产出
  `results_display/Test2_comparison/comparison_{per_session,summary}.csv`。
  当前 MotionPRO 行为全 NaN（`results/MotionPRO` 无预测）。
- **环境**：`touch_gait`（py3.8/torch2.4）、`depthpro`（py3.10/torch2.4）、
  `bbox_scan`、`vibe`、`wham` 均存在。

### 2.2 三个 baseline 代码状态

- **MotionPRO**：`Baselines/MotionPRO` 是嵌套 git（上游 `wjrzm/MotionPRO`）。
  **磁盘工作区当前被还原成上游原版**（未提交）；完整集成版在外层 HEAD
  （提交 `2789889 框架`、`56f5187 contact-method-test`）与
  `Baselines_backup/MotionPRO/`（含 `lib/eval/`、`data_prep/`、`workspace.py`、
  已修的 FRAPPE MHA permute / footloss 除零、`workspace://` 配置等）。
  `results/MotionPRO` 为空：**从未完成正式训练/评估**。
- **pressure_tookit**：当前副本 = 干净上游（`haolyuan/pressure_tookit@frame_refine`，
  MMVP 官方逐帧优化）。部分集成在 `Baselines_backup/pressure_tookit/`（`lib/utils/
  workspace.py`、`data_prep/rgb2depth.py`、`configs/fit_smpl_rgbd.yaml` 的
  `workspace://` 化、`main_singleview.py` 的 `resolve_path`、`z_temp/` 三篇笔记），
  **尚未合入当前副本**。`derived/pressure_tookit/` 与 `essential/` 均空。
- **VP-MoCap**：`Baselines/VP-MoCap` 未跟踪、全新 clone（上游 `wjrzm/VP-MoCap`，
  MMVP 论文两段式：FPP-Net + PoseTransOpt）。**零集成，无任务书，无官方 ckpt**
  （FPP-Net README 的 checkpoint 链接是占位符 `url_here`，必须自训）。

## 3. 统一接入契约（三基线共同遵守）

1. **数据**：所有观测与 GT 从 `AnysoleWorkspace`（manifest + splits.csv +
   derived 九件套 / 原生 SMPL NPZ）读取；适配产物写入
   `derived/<baseline>/`，不落回原始数据目录。
2. **两条执行通道，终点是同一批 py 入口**（对齐 README_usage.md 现有模式）：
   - **通道 A（交互式，README_usage 文档化）**：SH 命令 → `python 脚本 + 参数`
     （`workspace://` 前缀路径），用于数据准备、smoke、调试；
   - **通道 B（后台队列）**：`offline_train/` 的 `.conf` → `run.sh`（SH）→
     同一批 py 入口，用于正式训练/全量拟合。两通道共用参数名与语义，禁止
     第三类入口（裸改代码路径跑一遍不计入结果）。
3. **改动分层（复现基线原则）**：
   - 接口/配置层可改：路径、帧率/窗口等配置、batch、GPU、导出格式；
   - 协议层**谨慎再谨慎**：关节顺序、单位（米/毫米）、时间轴、fake/valid 掩码、
     观测来源（不得用 3D GT 冒充 2D 观测）一律不改——只允许在适配层做确定性
     映射并写单测。
4. **输出**：统一导出 `results/<Model>/predictions/eval_motion/<session>.npz`
   （SMPL-24 协议 `joint_xyz_world` + `joint_names` + `valid_mask`），fake 帧不参与。
5. **评估**：`evaluate_compare.py` 统一口径（MPJPE/PA-MPJPE/WMPJPE/WAMPJPE/RTE/
   Accel/Jitter @40Hz，19 关节语义交集），Test2 对比表必须有数值行。

## 4. 分基线方案

### 4.1 MotionPRO（工作量最小，先跑通）

1. **恢复集成版代码**：`git checkout HEAD -- Baselines/MotionPRO`（或从
   `Baselines_backup/MotionPRO` 覆盖），确认 `workspace.py`、`image_pressure.py`
   （split_csv/fake_mask/contact_method/96×96 压力缩放/valid 掩码）、`FRAPPE.py`
   （MHA permute、footloss 除零、smpl 路径）、`app/test_frappe.py`（
   `evaluate_checkpoint` 表 3/4 指标）、`data_prep/` 均在。
2. **GT 口径（D1，已定：方案 b）**：按 SMPL-24 协议从 manifest `smpl_path`
   重导出 `smpl.npy / keypoints.npy`（复用 `anysole.data.smpl_io.load_smpl` +
   40Hz 重采样），contact 标签用 `results_display/script/contact_methods.py`
   的 bvh_h/bvh_soft（Test5 结论优于触觉系）。理由：GT 属于协议层
   （§3.3 红线），复现基线只改接口/配置，不引入与 AnySole 不同的 GT 基础。
3. **数据侧补齐**：`bbox.npy / feature_hrnet.pth / pressure.npz / fake_mask.npy`
   沿用现有；`keypoints.npy / smpl.npy / contact*.npy` 按步骤 2 重生成
   （`gen_bbox` 走 bbox_scan 环境，其余走 touch_gait）。
4. **训练**：队列任务 `MODEL=motionpro`、`RUN_DIR=results/MotionPRO`、
   `EPOCHS=1000, BATCH_SIZE=16`（集成版默认）。GPU 按现有调度。
5. **导出与评估**：`test_frappe.py` 已能产 `test_metrics.csv`；补一个导出步骤
   （SessionAccumulator 拼回整段 → FK SMPL-24 → `joint_xyz_world` 写
   `results/MotionPRO/predictions/eval_motion/<session>.npz`）→
   `evaluate_compare.py` → Test2 行非 NaN。
6. **可视化**：`results_display/script/visualize_motionpro.py` 已存在，补跑。

### 4.2 pressure_tookit（中等）

1. **合并备份集成**：把 `Baselines_backup/pressure_tookit` 的 `lib/utils/workspace.py`、
   `data_prep/rgb2depth.py`、`main_singleview.py` 的 `resolve_path`、config 的
   `workspace://` 化合入当前干净副本；顺手修两处 loader bug：
   - `lib/dataextra/data_loader.py:220` keypoint 路径改相对 `basdir`（Agent_04 已点名）；
   - 同文件 `:244` `stage == 'init_shape '` 尾随空格笔误；
   - 根目录 `run.sh` 指向不存在的 `lib/utils/smpl_fitting.py` → 改为
     `main_singleview.py`（或删除）。
2. **essential 下载**（用户手动，见 §5）：放入
   `dependencies/pressure_tookit/essential/`（bodyModels/smpl、smplify_essential、
   pressure/RegionInsole2SMPL*_enhanced.npy、foot_related、hand_ids.txt）。
3. **数据适配器**（新脚本，写 `derived/pressure_tookit/`）：
   - `images/<ds>/<sub>/<seq>/color|depth|depth_mask|insole/ + calibration.npy`：
     color 从统一 `color/*.jpg` 按帧号生成；depth 用既有 `rgb2depth.py`
     （DepthPro，uint16 毫米 + meta.json）；depth_mask 由深度有效性定义
     （如 0.4–5 m，规则写入适配器测试）；calibration.npy 从
     `AnysoleWorkspace/calibration/*.json` 的 cam3 K/D/R/t 生成。
   - `annotations/<ds>/floor_info/floor_<sub>.npy`：从标定生成；无可信标定/地面
     参数的 subject **阻塞并标红**（不伪造）。
   - `input/<sub>/<seq>/keypoints/*.npy`：RTM-pose 对真实 RGB 的 2D 观测
     （§5 下载，新 env），字段 `keypoints + keypoint_scores`，
     `read_rtm_kpts()` 可直接读；**禁止**用 `keypoints.npy` 3D GT 投影。
   - insole：统一 `pressure.npz` → 逐帧 `insole/*.npy`（`{'insole': [left, right]}`）。
     **D2（已定）两套网格的确定性映射，举例说明：**

     我们的布局（`anysole/data/pressure.py` `cop_from_grid` 冻结定义）：
     **4 行 × 12 列/脚；列 = 脚长方向，col 0 = 脚尖，col 11 = 脚跟**
     （`heel_to_toe = 11 - mean_col`）；行 0–3 = 脚宽方向（内外侧方向由标定确认）。

     MMVP 的布局（代码 + 模板实测，corr(顶点z, 行号) = **-0.996**）：
     **31 行 × 11 列/脚**（mask 242 有效像素/脚）；`load_contact` 把行分四段
     `[0,8)→label 0/1`、`[8,15)→2/3/4`、`[15,23)→5/6`、`[23,31)→7/8`
     （range 上界 31 恰好覆盖 31 行数组的 23–30），列为 3 段且左右脚镜像。
     用 FPP-Net 自带 `essentials/insole2cont/smpl_template.obj` +
     `insole2smplL.npy` 实测各段顶点 z：**row 0 = 脚尖（z 均值 +0.124）、
     row 30 = 脚跟（z −0.054）**。

     适配 = 确定性分块膨胀 4×12 → 31×11（toolkit 代码零改动）：
     - 长度方向 `our_col = round(r × 11/30)`：
       our col 0–2（脚尖）→ rows 0–7（label 0/1）；col 3–5（前掌）→ rows 8–14
       （label 2/3/4）；col 6–8（中足）→ rows 15–22（label 5/6）；col 9–11
       （脚跟）→ rows 23–30（label 7/8）。四段比例 3:3:3:3 ≈ MMVP 行段
       26%:23%:26%:26%，一致。
     - 宽度方向 4 行 → 11 列按 3 组分配（左/右镜像分开处理）；**内外侧方向
       以标定为准**，适配器单测断言：标定指定为内侧的行 → 必须点亮 MMVP
       内侧列组。
     - 写入值 = 适配层统一 /255 后的 0–255（与 `pressure.npz` 同口径）；
       `load_contact` 只判 `!= 0`，阈值语义不变。
     单测：脚尖单点只点亮 rows 0–7；脚跟单点只点亮 rows 23–30；左右脚不串；
     全零帧不产生接触。
   - CLIFF 初始化：clone CLIFF 代码（§5，ckpt `hr48-...3dpw.pt` 已有）对每序列
     生成 `{seq}_cliff_hr48.npz` 到 `init_data_dir`（`load_init_pose(form='cliff')`
     的格式：`pose` 72 维）。
4. **三阶段拟合**：沿用 `create_queue.py` 已生成的逐 session
   `init_shape → init_pose → tracking` 任务（同 GPU 串行）。**范围先只做 test split**
   （决策点 D6）：先单序列 smoke 测单帧耗时，再排全量 31 session。
   注意 config 的 `maxiters/权重` CLI 参数实际不生效（`main_singleview.py` 未 pop），
   如需调参改 `fitting.py` 默认或修 pop——记录偏离。
5. **导出**：逐帧 `smpl_{idx}.npz`（body_pose 69 轴角 + global_rot + transl 地面系米
   + betas + scale）→ FK SMPL（含 `model_scale_opt`/betas[:,0] 尺度）→ 地面系→
   世界系（标定变换）→ `results/pressure_tookit/predictions/eval_motion/<session>.npz`。
6. **评估与可视化**：`evaluate_compare.py` → Test2；trimesh 渲染 →
   `results_display/Test1_visualization/pressure_tookit/`。

### 4.3 VP-MoCap（最大，全新工作包）

分两段，共享前置：**RTM-pose HALPE-26 keypoints + CLIFF_results.npz + 场景深度**，
与 §4.2 的 keypoints/CLIFF 前置复用（写入 `derived/VP-MoCap/`）。

**A. FPP-Net（训练式）**
0. **上游 essentials 已在仓库内**（`FPP-Net/essentials/`：`insole2cont/` 全套、
   `smpl_uv.obj`、`dataset_split_temporal5.npy`、`sub_info/sub_info_*.npy` 参考）
   ——§5 的 smpl_uv 下载项取消；`record.py` 硬编码的
   `../../bodyModels/smpl/smpl_uv/smpl_uv.obj` 改指 `essentials/smpl_uv.obj`。
1. 数据适配到其 `datadir/{data_id}/{sub}/{seq}/insole|keypoints/ + sub_info.npy`：
   - insole：48 点按 §4.2 的同一 31×11 确定性膨胀（FPP-Net 与 toolkit 共用一份
     适配产物）；sigmoid 归一化由 FPP-Net 内部 `sigmoidNorm(p, weight/484)` 完成，
     适配层只负责写入与 weight 同单位的原始压力。
   - `sub_info.npy`（D7 已定）：新脚本 `AnysoleWorkspace/tool/build_sub_info.py`。
     判据：每受试者取第一个动作中「双脚着地（左右脚 48 点总和均 > 阈值）且 BVH
     运动速度最小」的一帧，`weight = 该帧左右脚压力总和`（与 insole 写入同一
     /255 口径），`max_value = 255`、`height = -1`。产出与上游同构的
     `{'<sub>': {'weight', 'max_value', 'height'}}`（实测上游 S01 weight≈453、
     max_value≈72 —— 上游 weight 本就是压力单位不是 kg，本方案与上游口径一致）。
   - keypoints：与 §4.2 共用 HALPE-26 npy。
   - **难点（D3 已定路线，先 probe）**：上游
     `essentials/insole2cont/insole2smplL.npy` 是 `{vertex_id(str): (rows, cols)}`，
     96 个足底顶点（`footL_ids.txt`，SMPL 官方 6890 顶点序 → **可直接用于我们的
     SMPL_NEUTRAL.pkl**）每个顶点映射到其覆盖的 insole 像素（`getVertsPress`
     方向：顶点压力 = 其像素之和）。这条映射是 MMVP 硬件几何（31×11 掩码 242
     像素/脚），**必须用我们的标定重建**：在 §4.2 同一膨胀网格上，按标定给出的
     cell 物理位置把每个 cell 分配给其覆盖的足底顶点，产出同格式
     `insole2smpl{L,R}.npy`（242 像素掩码、96 顶点、dict 结构不变 →
     PoseTransOpt 的 `(2,96)` 契约不受影响）。QA 复用模板验证法（顶点 z 与
     insole 行的 corr 应 ≈ ±1）；标定不足以重建即阻塞标红，降级为像素级对比报告。
2. 修上游 bug：`trainer_tempkpSMPLCont.py:50` BCE 维度 192 vs 484（改用
   `contact_smpl` 或与推理脚本一致的 MSE）；硬编码 `datadir` /
   `InsoleModule('/data/PressureDataset')` 改配置化。
3. 训练：新队列 case `fpp_train`（config `temporalKPSMPLCont_series5_mlp.yaml`，
   datadir/tv_fn 指向统一 split 生成物；`tv_fn` 由 splits.csv 生成其
   `{'train': {date: {sub: {seq: [frame,...]}}}, 'test': ...}` 格式）。
   环境用新建的 `mmvp`（见 B，含 tensorboardX/chumpy/yacs，numpy<1.24）。
   输出 `results/VP-MoCap/checkpoints/fpp/`。
4. 推理：test split → `derived/VP-MoCap/<session>/pred_contact_smpl/*.npy`
   （契约键 `contact_smpl.pred`，(2,96)）；报告 pressure MSE / cont MSE / BCE。

**B. PoseTransOpt（逐帧优化）**
1. 环境（D5 已定：新建 `mmvp`，实测依据）：`vibe` 环境已坏（torch import 失败）；
   `wham` 是 torch1.11 + numpy 2.0.1（numpy2 与 human_body_prior/torchgeometry
   老代码不兼容）；`touch_gait`（py3.8/torch2.4）**已有** torchgeometry 0.1.2、
   pyrender 0.1.45、hydra 1.3.2、trimesh、smplx、chumpy、yacs，仅缺
   human_body_prior、open3d、tensorboardX。因此：
   `conda create -n mmvp python=3.8` + torch2.4.0+cu121（与 touch_gait 同代）+
   `human_body_prior==2.2.2.0` + `open3d==0.18.x`（py3.8 有 wheel）+ torchgeometry
   0.1.2 + pyrender + hydra-core 1.3 + trimesh + smplx + tensorboardX + numpy<1.24。
   torchgeometry 在 torch2.4 已被 touch_gait 证明可装；human_body_prior 是纯 MLP，
   风险低，报错则退回 README 原栈（py3.7/torch1.12）。**不动 touch_gait/wham；
   FPP-Net 训练也用它。**
2. 依赖：`models/SMPL_NEUTRAL.pkl` 复用 `dependencies/smpl`；`SMPL_MALE.pkl`、
   VPoser `V02_05/` 下载（§5）。
3. 修上游可移植性 bug：`app/optimize.py:146` `split('\\')`；`task/MMVP.yaml`
   的 Windows 占位路径 `D:/Dataset/...`、`scene_rgbd`；`transolver.py:13`
   `models\SMPL_NEUTRAL.pkl`。
4. 每 session 输入装配：`input_path_base=derived/VP-MoCap/<session>/`：
   `color/ + keypoints/ + pred_contact_smpl/ + CLIFF_results.npz +
   template_scene_rgbd.npy`（单帧地面深度：DepthPro 已有，替代 README 的
   ZoeDepth——口径与其他基线一致，记为偏离）。
5. 运行 `python -m app.optimize` → `opt_result.pth`（pose (B,24,3,3) + beta + trans）
   → FK SMPL-24 → 世界系 → 统一导出 npz。范围同 D6：先 test split。
6. 评估：`evaluate_compare.py` → Test2；渲染 → Test1_visualization/VP-MoCap/。

## 5. 需要下载的依赖与 ckpt（用户手动下载清单）

| # | 用途 | 下载地址 | 放置位置 | 状态 |
|---|---|---|---|---|
| 1 | pressure_tookit essential（bodyModels/smpl、smplify_essential/gmm_08.pkl、pressure/RegionInsole2SMPL*、foot_related、hand_ids） | https://drive.google.com/file/d/15D2I9W4oYXZ2rbN94aFFevacua2JjOj9/view?usp=drive_link （readme.md 官方链接） | `AnysoleWorkspace/dependencies/pressure_tookit/essential/` | **缺，待用户下载** |
| 2 | SMPL_MALE.pkl（PoseTransOpt） | https://smpl.is.tue.mpg.de/ （注册后下载；SMPL_NEUTRAL 已有，不用再下） | `AnysoleWorkspace/dependencies/VP-MoCap/smpl/`（或 PoseTransOpt/models/） | **缺，待用户下载** |
| 3 | VPoser V02_05（PoseTransOpt 姿态先验） | https://smpl-x.is.tue.mpg.de/ （注册；VPoser 下载页） | `PoseTransOpt/models/V02_05/` | **缺，待用户下载** |
| 4 | RTM-pose 2D 关键点模型（**已定：halpe26 rtmpose-m**，pressure_tookit + VP-MoCap 共用） | https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose （model zoo 的 halpe26 配置+ckpt；具体文件名按 zoo 表钉死后回填） | `AnysoleWorkspace/dependencies/rtmpose/`；mmpose 代码装进新 env | **缺，待用户下载** |
| 5 | CLIFF 代码（pressure_tookit 姿态初始化 + VP-MoCap CLIFF_results.npz；ckpt hr48 已有） | https://github.com/huawei-noah/noah-research/tree/master/CLIFF | `AnysoleWorkspace/dependencies/CLIFF/` | **缺，待用户下载** |
| 6 | ZoeDepth（可选：PoseTransOpt README 的原始选择；默认改用已有 DepthPro，不下载） | https://github.com/isl-org/ZoeDepth | — | 可选 |
| 7 | FPP-Net 官方 checkpoint | README 为占位符 `url_here`，**官方未发布** | — | 必须自训（§4.3-A） |

无需下载：SMPL_NEUTRAL.pkl、smpl_mean_params.npz、CLIFF hr48 ckpt、YOLOX ckpt、
DepthPro 权重、MotionPRO/Step2Motion 全部依赖——均已在
`AnysoleWorkspace/dependencies/` 就位（doctor 已核验）。**smpl_uv.obj 与 FPP-Net
essentials 全套在 `Baselines/VP-MoCap/FPP-Net/essentials/` 仓库内自带，无需下载。**
human_body_prior / open3d / tensorboardX 为 pip 安装，不需要手动下载文件。

## 6. 决策点（已全部拍板，2026-09-20）

| 编号 | 决定 | 依据/备注 |
|---|---|---|
| D1 MotionPRO GT 口径 | **已定 (b)**：从原生 SMPL NPZ 重导出 smpl.npy/keypoints.npy | GT 属协议层（§3.3）：复现基线只改接口/配置，与 AnySole 同 GT 基础 |
| D2 toolkit insole 适配 | **已定**：4×12 → 31×11 确定性分块膨胀（toolkit 代码零改动） | 展开与举例见 §4.2 |
| D3 VP-MoCap insole2smpl | **已定路线**：从标定重建同格式映射，先 probe 可行性 | 展开与举例见 §4.3-A；不可行则阻塞标红、降级像素级报告 |
| D4 RTM-pose 选型 | **已定**：halpe26 rtmpose-m（用户下载） | 当初列决策点的原因**不是时间成本**：2D 观测是两个模型共享的观测协议——toolkit 的 `joint_mapper` 是 25 点映射表（openposemap/halpemap），VP-MoCap 用 26 点 HALPE（滤波/裁剪在 26 点上做）；模型选定即钉死点数、字段与置信度口径，三个读取点（`read_rtm_kpts`、`joint_mapper`、PoseTransOpt）必须一次对齐，属协议层决定。实现时对齐 toolkit 对 26 点的截断/映射行为并写单测 |
| D5 PoseTransOpt 环境 | **已定**：新建 `mmvp`（py3.8 + torch2.4，与 touch_gait 同代） | 实测：vibe 坏、wham numpy2 不兼容、touch_gait 只缺 2 个包；详见 §4.3-B。移植到 touch_gait 不采纳（不污染主训练环境） |
| D6 拟合范围 | **已定**：先只跑 test split（31 session） | train split 另行排队 |
| D7 FPP-Net 体重 | **已定**：`build_sub_info.py` 静立帧方案（§4.3-A） | 与上游口径同构：实测上游 sub_info 的 weight≈453 本就是压力单位（站立帧总压），非 kg |

## 7. 执行顺序与验收

**Phase 0（并行前置）**：essential 下载解包（#1）；RTM-pose env+ckpt 就位并对
真实 RGB 出 2D 关键点（#4）；CLIFF 代码 clone 并出初始化 pose（#5）；VPoser/SMPL_MALE
下载（#2/3）；新建 `mmvp` 环境（§4.3-B）并装 human_body_prior/open3d/tensorboardX；
`build_sub_info.py`（D7）。验收：三份前置产物对 1 个 smoke session 通过
`read_rtm_kpts`/`load_init_pose` 形状检查，无 GT 泄漏；sub_info 的 weight 与
静立帧压力总和一致（数值复核）。

**Phase 1（MotionPRO，最快出数）**：恢复集成代码 → D1 口径重导 → smoke
（`task.epochs=1`）→ 正式训练 → 统一导出 → Test2 行非 NaN。验收（沿用 Agent_02）：
doctor/py_compile/smoke 通过；CSV session 与 test 列一致无 NaN/Inf；随机抽查 2 个
session 帧数与预测可追溯；可视化存在。

**Phase 2（pressure_tookit）**：合并备份集成 + loader 修复 → 适配器（color/depth/
depth_mask/insole/calibration/floor/keypoints/CLIFF）→ 单序列三阶段 smoke →
test split 全量 → 导出/评估。验收（沿用 Agent_04）：适配器单测（压力布局、深度
毫米单位、帧号对齐）；无 3D GT 当观测；floor/标定缺失即阻塞；GT→GT 自检误差为 0。

**Phase 3（VP-MoCap FPP-Net）**：数据适配 + 上游 bug 修复 → smoke → 正式训练 →
test 推理 pred_contact_smpl。验收：训练 loss 收敛曲线存档；推理输出契约键形状
(2,96)；contact 预测可被 PoseTransOpt 读取。

**Phase 4（VP-MoCap PoseTransOpt）**：mmvp 环境 → 依赖就位 → 单 session 优化
smoke → test split 全量 → 导出。验收：opt_result.pth 形状/平滑检查；世界系导出
与 GT 在同一坐标系（RTE 量级合理）；路径修复后两目录启动均可运行。

**Phase 5（统一评估收口）**：三模型 `predictions/eval_motion/` 齐全 →
`evaluate_compare.py` → Test2 四行（AnySole/MotionPRO/Step2Motion/pressure_tookit/
VP-MoCap）全有数值 → `results_display/Test1_visualization/` 各基线可视化 →
更新 `z_note/总体验收矩阵.md` 状态列与协议表。

## 8. 禁止事项（继承 Agent_01/02/04，仍有效）

- 不修改三个 baseline 的模型结构/损失语义（允许修路径、修 bug、加适配层，逐项记录）。
- 不用 3D GT 关键点投影冒充 RTM-pose 观测；不伪造 essential/标定/floor/contact。
- 不重写统一时间轴/split/fake-mask；下游只读 manifest 与 splits.csv。
- 不把 val 当 test 回报；单位/帧率/掩码在回传中显式说明（40Hz、mm、fake 帧排除）。
- 不覆盖原始数据与既有 derived 产物；需要 `--overwrite` 先确认。
- pressure 48 点→其他表示的适配必须确定性、可单测；不得把 16 通道池化当原始协议。

## 9. 回传格式（各工作包完成后）

```text
[Agent 06 <工作包> 回传]
状态：PASS / PARTIAL / BLOCKED
代码版本：<git commit>；改动文件列表及用途（模型代码改动必须逐条说明语义不变性）
环境：<conda env；torch/cuda 版本；新增 env 与安装记录>
数据：split=<path+sha256>；train/val/test=<n>/<n>/<n>；适配产物路径与映射表
执行命令：<逐条列出实际命令，含 GPU、覆盖参数>
输出：<checkpoint/拟合结果/统一导出 npz/metrics/可视化 绝对路径>
指标：<Test2 行原样摘要及单位；baseline 自身指标（FPP-Net MSE/BCE 等）>
测试：smoke=<结果>；适配器单测=<结果>；GT→GT 自检=<结果>
下载物状态：<#1-#7 各项已就位/缺失，及 essential 包内容清单>
问题/偏离：<无则写"无"；阻塞项按 D1-D7 编号标红>
```
