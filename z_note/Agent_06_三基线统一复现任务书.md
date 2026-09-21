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

pressure_tookit 与 VP-MoCap **同属 MMVP 工作**（arXiv 2403.17610）的两条方法线：
观测协议与依赖完全相同，§4.2 按一个工作包处理（共享前端 + 两条方法线）。

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
  由 `results_display/script/r_test2_compare.py` 做 19 关节语义交集对比，产出
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
  MMVP 论文两段式：FPP-Net + PoseTransOpt）。**零集成，无官方 ckpt**
  （FPP-Net README 的 checkpoint 链接是占位符 `url_here`，必须自训）。
  **与 pressure_tookit 同属 MMVP 工作**，按 §4.2 统一工作包处理。

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
5. **评估**：`r_test2_compare.py` 统一口径（MPJPE/PA-MPJPE/WMPJPE/WAMPJPE/RTE/
   Accel/Jitter @40Hz，19 关节语义交集），Test2 对比表必须有数值行。

## 4. 分基线方案

### 4.1 MotionPRO（工作量最小，先跑通）

1. **恢复集成版代码**：`git checkout 56f5187 -- Baselines/MotionPRO`
   （`contact-method-test` 是最新集成版提交，已核验在历史中仍可达；`Baselines/`
   现已出 git index 并加入 .gitignore，checkout 直接在工作区写回文件不受影响）
   ——或从 `Baselines_backup/MotionPRO` 覆盖。确认 `workspace.py`、
   `image_pressure.py`（split_csv/fake_mask/contact_method/96×96 压力缩放/valid
   掩码）、`FRAPPE.py`（MHA permute、footloss 除零、smpl 路径）、
   `app/test_frappe.py`（`evaluate_checkpoint` 表 3/4 指标）、`data_prep/` 均在。
2. **GT 口径（D1，已定：方案 b）——清除 BVH 分支历史遗留通路**：
   **实测盘面（S7011）**：当前 derived `smpl.npy` 的 `betas` 恒为 0（BVH→SMPL
   转换的特征，`56f5187` 的 `motion_to_smpl()` 写死 `betas=zeros(10)`），
   `align_meta.json` 带 `bvh_path`，且文件是 numpy2 pickle（touch_gait 的
   numpy1.23 读报 `numpy._core`）——即当前 derived GT 仍是 BVH_Motion 分支时代
   的产物（后被重组工作用 numpy2 重存，内容未变）。`prepare_sequences.py`
   已在 `a8ff4d7 整理文件` 随 Baselines 出 index。
   **分支政策**：SMPL_Motion 下「用 BVH 的用 BVH、用 SMPL 的用 SMPL」——
   MotionPRO 的 GT 表示是 SMPL 参数（FRAPPE 回归 85 维、smplx FK），属于 SMPL
   派，必须从 manifest `smpl_path`（`motion_neutral_smpl.npz`：真实 `betas`、
   `poses (T_src,72)`、120Hz）重导出 `smpl.npy / keypoints.npy`（复用
   `anysole.data.smpl_io.load_smpl` + 40Hz 统一网格重采样；实测 derived T=389
   与统一 n_frames=389 一致，**换源不换时间轴**）；Step2Motion 继续走
   `bvh_path`（baseline-only 字段）；MMVP 两条线的评估 GT 同样用原生 SMPL。
   重导脚本放 `AnysoleWorkspace/tool/`（旧 `Baselines/MotionPRO/data_prep/`
   已无跟踪位置）；contact 标签用 `AnysoleWorkspace/tool/contact_labels.py`
   的 bvh_h/bvh_soft（Test5 结论优于触觉系）。
   **实现坑**：① 新写的 `smpl.npy` 若由 numpy2 环境产出，touch_gait 训练端
   必须走集成版 `load_smpl_npy()`（已兼容）或改用 numpy1 写入；② `motion_
   neutral_smpl.npz` 的 `pose_body` 是 63 维（21 关节×3），重导时按
   `anysole` 的 SMPL-24 协议补齐/映射到 69 维 body_pose，不得沿用
   `BVH_TO_SMPL_BODY`。
3. **数据侧补齐**：`bbox.npy / feature_hrnet.pth / pressure.npz / fake_mask.npy`
   沿用现有；`keypoints.npy / smpl.npy / contact*.npy` 按步骤 2 重生成
   （`gen_bbox` 走 bbox_scan 环境，其余走 touch_gait）。
4. **训练**：队列任务 `MODEL=motionpro`、`RUN_DIR=results/MotionPRO`、
   `EPOCHS=1000, BATCH_SIZE=16`（集成版默认）。GPU 按现有调度。
5. **导出与评估**：`test_frappe.py` 已能产 `test_metrics.csv`；补一个导出步骤
   （SessionAccumulator 拼回整段 → FK SMPL-24 → `joint_xyz_world` 写
   `results/MotionPRO/predictions/eval_motion/<session>.npz`）→
   `r_test2_compare.py` → Test2 行非 NaN。
6. **可视化**：`results_display/script/r_test1_visualize_motionpro.py` 已存在，补跑。

### 4.2 MMVP 工作包：pressure_tookit + VP-MoCap（同一工作，共享前端与依赖）

pressure_tookit 与 VP-MoCap 同属 MMVP 工作的两条方法线，观测协议完全相同
（insole 31×11、RTM-pose HALPE-26、CLIFF 初始化、DepthPro 深度、floor/标定），
差异只在"怎么从观测到 SMPL"：

| 方法线 | 性质 | 观测 → 输出 |
|---|---|---|
| A. pressure_tookit | MMVP 官方逐帧 SMPLify 拟合 | RGB-D + 2D 关键点 + 压力接触 → 逐帧 SMPL 参数 |
| B+C. VP-MoCap | FPP-Net（训练式，自训）+ PoseTransOpt（逐帧优化） | 2D 关键点 → 压力/足底接触预测 → 姿态+平移优化 |

因此**数据适配、下载依赖、环境设计只做一次，两条方法线共用**（§4.2.1）；
只有各自的方法代码与运行入口分开（§4.2.2–4.2.4）。

#### 4.2.1 共享前置（做一次，两条方法线消费）

0. **环境（D5 已定：新建 `mmvp`，实测依据）**：`vibe` 环境已坏（torch import
   失败）；`wham` 是 torch1.11 + numpy 2.0.1（numpy2 与 human_body_prior/
   torchgeometry 老代码不兼容）；`touch_gait`（py3.8/torch2.4）**已有**
   torchgeometry 0.1.2、pyrender 0.1.45、hydra 1.3.2、trimesh、smplx、chumpy、
   yacs，仅缺 human_body_prior、open3d、tensorboardX。因此：
   `conda create -n mmvp python=3.8` + torch2.4.0+cu121（与 touch_gait 同代）+
   `human_body_prior==2.2.2.0` + `open3d==0.18.x`（py3.8 有 wheel）+ torchgeometry
   0.1.2 + pyrender + hydra-core 1.3 + trimesh + smplx + tensorboardX + numpy<1.24，
   另加 toolkit 的 configargparse/icecream/xrprimer。toolkit 拟合、FPP-Net 训练、
   PoseTransOpt **统一用 mmvp**；rgb2depth 保持 depthpro 环境。torchgeometry 在
   torch2.4 已被 touch_gait 证明可装；human_body_prior 是纯 MLP，风险低，报错
   再退回 README 原栈（py3.7/torch1.12）。**不动 touch_gait/wham。**
1. **essential 下载**（用户手动，见 §5）：放入
   `dependencies/pressure_tookit/essential/`（bodyModels/smpl、smplify_essential、
   pressure/RegionInsole2SMPL*_enhanced.npy、foot_related、hand_ids.txt）。
   FPP-Net 的 essentials（`insole2cont/` 全套、`smpl_uv.obj`、dataset_split、
   sub_info 参考）**在 `Baselines/VP-MoCap/FPP-Net/essentials/` 仓库内自带**，
   无需下载。
2. **共享数据适配器**（新脚本，写 `derived/pressure_tookit/` 与
   `derived/VP-MoCap/`，同一生成器；一份产物两份 repo 共用）：
   - `images/<ds>/<sub>/<seq>/color|depth|depth_mask/ + calibration.npy`：
     color 从统一 `color/*.jpg` 按帧号生成；depth 用既有 `rgb2depth.py`
     （DepthPro，depthpro env，uint16 毫米 + meta.json）；depth_mask 由深度
     有效性定义（如 0.4–5 m，规则写入适配器测试）；calibration.npy 从
     `AnysoleWorkspace/calibration/*.json` 的 cam3 K/D/R/t 生成。
     PoseTransOpt 的 `template_scene_rgbd.npy`（单帧地面深度，README 用
     ZoeDepth → **改用同一 DepthPro 源**，口径与其他基线一致，记为偏离）
     也从同一深度产物取。
   - `annotations/<ds>/floor_info/floor_<sub>.npy`：从标定生成，与 PoseTransOpt
     的地面变换**同源**；无可信标定/地面的 subject **阻塞标红**（不伪造）。
   - keypoints：RTM-pose 对真实 RGB 的 2D 观测（§5 #4，新 env），字段
     `keypoints + keypoint_scores`，`read_rtm_kpts()` 可直接读；**禁止**用
     `keypoints.npy` 3D GT 投影。一份产物两处消费：toolkit 走
     `input/<sub>/<seq>/keypoints/`，FPP-Net 走 `datadir/<date>/<sub>/<seq>/
     keypoints/`（同文件软链/复制，写入映射表）。
   - CLIFF 初始化：clone CLIFF 代码（§5，ckpt `hr48-...3dpw.pt` 已有），
     **一次前向转两种格式**：toolkit 的 `{seq}_cliff_hr48.npz`
     （`load_init_pose(form='cliff')` 的 `pose` 72 维，到 `init_data_dir`）与
     PoseTransOpt 的 `CLIFF_results.npz`（`shape/pose/global_t`）。
   - `build_sub_info.py`（D7 已定，FPP-Net 用）：`AnysoleWorkspace/tool/
     build_sub_info.py`。判据：每受试者取第一个动作中「双脚着地（左右脚 48 点
     总和均 > 阈值）且 BVH 运动速度最小」的一帧，`weight = 该帧左右脚压力总和`
     （与 insole 写入同一 /255 口径），`max_value = 255`、`height = -1`。产出与
     上游同构的 `{'<sub>': {'weight','max_value','height'}}`（实测上游 S01
     weight≈453 —— 上游 weight 本就是压力单位不是 kg，本方案与上游口径一致）。
   - insole：统一 `pressure.npz` → 逐帧 `insole/*.npy`（`{'insole': [left, right]}`）。
     toolkit 的 `load_contact` 与 FPP-Net 的 `PED_tempKPCont` 读同一文件。
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
3. **时间轴与帧率（协议层，统一策略）**：统一 40Hz 网格是唯一时间轴，所有适配
   产物按 `frame_index 0..T-1` 编号（`%03d` 文件名）；各 baseline 内部不重新
   定义时间轴、不各自重采样——原始源速率差异（压力 CSV 高频 `t_us`、SMPL
   120Hz、BVH 120Hz）只在中央管线重采样一次，帧率以「帧间隔 = 1/40 s」单一
   常数进入各方法。逐点核对结论（已核验代码）：
   - MotionPRO / FPP-Net：无 fps 假设（窗口 20 帧 / seqlen 5 帧，逐帧消费），
     无处理。
   - pressure_tookit tracking：防滑时间损失 `calcTempLoss`
     （`lib/fitSMPL/contactTerm.py:92`）为**逐帧位移口径**，`tfoot_weights=10.0`
     （`loadweights.py`）是上游在 30fps 上调的；40Hz 下自然幅度 ≈×0.75 →
     先 smoke 实测脚滑行为，不足则在 `loadweights.py` 乘 40/30≈13.3（config 的
     权重 CLI 本就不生效，集中在此改，记录偏离）。
   - PoseTransOpt：`lib/initial_trans/initial_trans.py` 飞行段抛物线
     **硬编码 `dt=0.033`（30fps）+ 重力项**（已核验）→ 必须参数化为 1/40，
     否则飞行段全局轨迹量级错；`lambda_timing=200` 的逐帧时序一致性同为
     逐帧位移口径 → 与 toolkit 同策略（先 smoke，必要时 ×40/30，记录偏离）。
   - 评估侧：`r_test2_compare.py` 全 @40Hz（Accel/Jitter 的 fps² 缩放已统一）。
#### 4.2.2 方法线 A：pressure_tookit 逐帧拟合

1. **合并备份集成**：把 `Baselines_backup/pressure_tookit` 的 `lib/utils/workspace.py`、
   `data_prep/rgb2depth.py`、`main_singleview.py` 的 `resolve_path`、config 的
   `workspace://` 化合入当前干净副本；顺手修两处 loader bug：
   - `lib/dataextra/data_loader.py:220` keypoint 路径改相对 `basdir`（Agent_04 已点名）；
   - 同文件 `:244` `stage == 'init_shape '` 尾随空格笔误；
   - 根目录 `run.sh` 指向不存在的 `lib/utils/smpl_fitting.py` → 改为
     `main_singleview.py`（或删除）。
2. **三阶段拟合**：沿用 `create_queue.py` 已生成的逐 session
   `init_shape → init_pose → tracking` 任务（同 GPU 串行，队列任务 CONDA_ENV
   统一改 mmvp）。**范围先只做 test split**（决策点 D6）：先单序列 smoke 测单帧
   耗时，再排全量 31 session。
   注意 config 的 `maxiters/权重` CLI 参数实际不生效（`main_singleview.py` 未 pop），
   如需调参改 `fitting.py` 默认或修 pop——记录偏离。
3. **导出**：逐帧 `smpl_{idx}.npz`（body_pose 69 轴角 + global_rot + transl 地面系米
   + betas + scale）→ FK SMPL（含 `model_scale_opt`/betas[:,0] 尺度）→ 地面系→
   世界系（标定变换）→ `results/pressure_tookit/predictions/eval_motion/<session>.npz`。
4. **评估与可视化**：`r_test2_compare.py` → Test2；trimesh 渲染 →
   `results_display/Test1_visualization/pressure_tookit/`。

#### 4.2.3 方法线 B：VP-MoCap · FPP-Net（训练式）

0. **上游 essentials 已在仓库内**（`FPP-Net/essentials/`：`insole2cont/` 全套、
   `smpl_uv.obj`、`dataset_split_temporal5.npy`、`sub_info/sub_info_*.npy` 参考）
   ——§5 的 smpl_uv 下载项取消；`record.py` 硬编码的
   `../../bodyModels/smpl/smpl_uv/smpl_uv.obj` 改指 `essentials/smpl_uv.obj`。
1. 数据装配到其 `datadir/{data_id}/{sub}/{seq}/insole|keypoints/ + sub_info.npy`：
   **全部来自 §4.2.1 共享产物**（31×11 insole 膨胀、HALPE-26 keypoints、
   sub_info）。sigmoid 归一化由 FPP-Net 内部 `sigmoidNorm(p, weight/484)` 完成，
   适配层只负责写入与 weight 同单位的原始压力。
2. **难点（D3 已定路线，先 probe）**：上游
   `essentials/insole2cont/insole2smplL.npy` 是 `{vertex_id(str): (rows, cols)}`，
   96 个足底顶点（`footL_ids.txt`，SMPL 官方 6890 顶点序 → **可直接用于我们的
   SMPL_NEUTRAL.pkl**）每个顶点映射到其覆盖的 insole 像素（`getVertsPress`
   方向：顶点压力 = 其像素之和）。这条映射是 MMVP 硬件几何（31×11 掩码 242
   像素/脚），**必须用我们的标定重建**：在 §4.2.1 共享膨胀网格上，按标定给出的
   cell 物理位置把每个 cell 分配给其覆盖的足底顶点，产出同格式
   `insole2smpl{L,R}.npy`（242 像素掩码、96 顶点、dict 结构不变 →
   PoseTransOpt 的 `(2,96)` 契约不受影响）。QA 复用模板验证法（顶点 z 与
   insole 行的 corr 应 ≈ ±1）；标定不足以重建即阻塞标红，降级为像素级对比报告。
3. 修上游 bug：`trainer_tempkpSMPLCont.py:50` BCE 维度 192 vs 484（改用
   `contact_smpl` 或与推理脚本一致的 MSE）；硬编码 `datadir` /
   `InsoleModule('/data/PressureDataset')` 改配置化。
4. 训练：新队列 case `fpp_train`（config `temporalKPSMPLCont_series5_mlp.yaml`，
   datadir/tv_fn 指向统一 split 生成物；`tv_fn` 由 splits.csv 生成其
   `{'train': {date: {sub: {seq: [frame,...]}}}, 'test': ...}` 格式）。
   环境 mmvp（§4.2.1）。输出 `results/VP-MoCap/checkpoints/fpp/`。
5. 推理：test split → `derived/VP-MoCap/<session>/pred_contact_smpl/*.npy`
   （契约键 `contact_smpl.pred`，(2,96)）；报告 pressure MSE / cont MSE / BCE。

#### 4.2.4 方法线 C：VP-MoCap · PoseTransOpt（逐帧优化）

1. 依赖：`models/SMPL_NEUTRAL.pkl` 复用 `dependencies/smpl`；`SMPL_MALE.pkl`、
   VPoser `V02_05/` 下载（§5）。
2. 修上游可移植性 bug：`app/optimize.py:146` `split('\\')`；`task/MMVP.yaml`
   的 Windows 占位路径 `D:/Dataset/...`、`scene_rgbd`；`transolver.py:13`
   `models\SMPL_NEUTRAL.pkl`。
3. 每 session 输入装配：`input_path_base=derived/VP-MoCap/<session>/`：
   `color/ + keypoints/ + pred_contact_smpl/ + CLIFF_results.npz +
   template_scene_rgbd.npy`（**全部来自 §4.2.1 共享产物**；scene 深度用 DepthPro
   替代 README 的 ZoeDepth——口径与其他基线一致，记为偏离）。
4. 运行 `python -m app.optimize`（mmvp 环境）→ `opt_result.pth`
   （pose (B,24,3,3) + beta + trans）→ FK SMPL-24 → 世界系 → 统一导出 npz。
   范围同 D6：先 test split。
5. 评估：`r_test2_compare.py` → Test2；渲染 → Test1_visualization/VP-MoCap/。

## 5. 需要下载的依赖与 ckpt（用户手动下载清单）

| # | 用途 | 下载地址 | 放置位置 | 状态 |
|---|---|---|---|---|
| 1 | pressure_tookit essential（bodyModels/smpl、smplify_essential/gmm_08.pkl、pressure/RegionInsole2SMPL*、foot_related、hand_ids；**另含 SMPL_MALE/FEMALE 与 smpl_uv**） | https://drive.google.com/file/d/15D2I9W4oYXZ2rbN94aFFevacua2JjOj9/view?usp=drive_link （readme.md 官方链接） | `AnysoleWorkspace/dependencies/pressure_tookit/essential/` | **已就位**（2026-09-21 解包核验） |
| 2 | SMPL_MALE.pkl（PoseTransOpt） | https://smpl.is.tue.mpg.de/ （注册后下载；SMPL_NEUTRAL 已有，不用再下） | `AnysoleWorkspace/dependencies/VP-MoCap/smpl/SMPL_MALE.pkl` | **已就位**（官方 SMPL v1.1.0 包；essential 内另有副本作双保险） |
| 3 | VPoser V02_05（PoseTransOpt 姿态先验） | https://smpl-x.is.tue.mpg.de/ （注册；VPoser 下载页） | `Baselines/VP-MoCap/PoseTransOpt/models/V02_05/` | **已就位**（yaml + snapshots ×2） |
| 4 | RTM-pose 2D 关键点模型（**已定：halpe26 rtmpose-m**，pressure_tookit + VP-MoCap 共用） | https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose （model zoo 的 halpe26 配置+ckpt） | `AnysoleWorkspace/dependencies/rtmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth`；mmpose 代码装进新 env | **已就位**（body7 变体；config 由 mmpose 包自带，装包时取 rtmpose-m_8xb256-700e_body7-halpe26-256x192.py） |
| 5 | CLIFF 代码（pressure_tookit 姿态初始化 + VP-MoCap CLIFF_results.npz；ckpt hr48 已有） | https://github.com/huawei-noah/noah-research/tree/master/CLIFF | `AnysoleWorkspace/dependencies/CLIFF/` | **已就位**（从 noah-research-master 解出 CLIFF/；requirements torch≥1.11+torchgeometry+pyrender+smplx+yacs 全部由 mmvp 环境覆盖；ckpt 用现有 `dependencies/MotionPRO/cliff_ckpt/hr48-...3dpw.pt`） |
| 6 | ZoeDepth（可选：PoseTransOpt README 的原始选择；默认改用已有 DepthPro，不下载） | https://github.com/isl-org/ZoeDepth | — | 可选 |
| 7 | FPP-Net 官方 checkpoint | README 为占位符 `url_here`，**官方未发布** | — | 必须自训（§4.2.3） |

无需下载：SMPL_NEUTRAL.pkl、smpl_mean_params.npz、CLIFF hr48 ckpt、YOLOX ckpt、
DepthPro 权重、MotionPRO/Step2Motion 全部依赖——均已在
`AnysoleWorkspace/dependencies/` 就位（doctor 已核验）。**smpl_uv.obj 与 FPP-Net
essentials 全套在 `Baselines/VP-MoCap/FPP-Net/essentials/` 仓库内自带，无需下载。**
human_body_prior / open3d / tensorboardX 为 pip 安装，不需要手动下载文件。

## 6. 决策点（已全部拍板，2026-09-20）

| 编号 | 决定 | 依据/备注 |
|---|---|---|
| D1 MotionPRO GT 口径 | **已定 (b)**：从原生 SMPL NPZ 重导出 smpl.npy/keypoints.npy | GT 属协议层（§3.3）：复现基线只改接口/配置，与 AnySole 同 GT 基础 |
| D2 toolkit insole 适配 | **已定**：4×12 → 31×11 确定性分块膨胀（toolkit 代码零改动） | 展开与举例见 §4.2.1 |
| D3 VP-MoCap insole2smpl | **已定路线**：从标定重建同格式映射，先 probe 可行性 | 展开与举例见 §4.2.3；不可行则阻塞标红、降级像素级报告 |
| D4 RTM-pose 选型 | **已定**：halpe26 rtmpose-m（用户下载） | 当初列决策点的原因**不是时间成本**：2D 观测是两个模型共享的观测协议——toolkit 的 `joint_mapper` 是 25 点映射表（openposemap/halpemap），VP-MoCap 用 26 点 HALPE（滤波/裁剪在 26 点上做）；模型选定即钉死点数、字段与置信度口径，三个读取点（`read_rtm_kpts`、`joint_mapper`、PoseTransOpt）必须一次对齐，属协议层决定。实现时对齐 toolkit 对 26 点的截断/映射行为并写单测 |
| D5 PoseTransOpt 环境 | **已定**：新建 `mmvp`（py3.8 + torch2.4，与 touch_gait 同代），MMVP 工作包三条线共用 | 实测：vibe 坏、wham numpy2 不兼容、touch_gait 只缺 2 个包；详见 §4.2.1。移植到 touch_gait 不采纳（不污染主训练环境） |
| D6 拟合范围 | **已定**：先只跑 test split（31 session） | train split 另行排队 |
| D7 FPP-Net 体重 | **已定**：`build_sub_info.py` 静立帧方案（§4.2.1） | 与上游口径同构：实测上游 sub_info 的 weight≈453 本就是压力单位（站立帧总压），非 kg |

## 7. 执行顺序与验收

**Phase 0（并行前置：下载与环境）**：essential 下载解包（#1）；RTM-pose env+ckpt
就位并对真实 RGB 出 2D 关键点（#4）；CLIFF 代码 clone 并出初始化 pose（#5）；
VPoser/SMPL_MALE 下载（#2/3）；新建 `mmvp` 环境（§4.2.1）并装
human_body_prior/open3d/tensorboardX；`build_sub_info.py`（D7）。验收：三份前置
产物对 1 个 smoke session 通过 `read_rtm_kpts`/`load_init_pose` 形状检查，无 GT
泄漏；sub_info 的 weight 与静立帧压力总和一致（数值复核）。

**Phase 1（MotionPRO，最快出数）**：恢复集成代码 → D1 口径重导 → smoke
（`task.epochs=1`）→ 正式训练 → 统一导出 → Test2 行非 NaN。验收（沿用 Agent_02）：
doctor/py_compile/smoke 通过；CSV session 与 test 列一致无 NaN/Inf；随机抽查 2 个
session 帧数与预测可追溯；可视化存在。

**Phase 2（MMVP 工作包 · 共享数据适配 + D3 probe）**：共享适配器（31×11 insole
膨胀、keypoints 双路装配、DepthPro depth+depth_mask+scene_rgbd、calibration/
floor、CLIFF 双格式输出）→ 帧率项处理（PoseTransOpt dt 参数化 1/40；toolkit
tfoot 权重留 smoke 实测）→ D3 insole2smpl 标定重建 probe → 适配器单测。验收
（沿用 Agent_04）：压力布局/深度毫米单位/帧号对齐单测通过；无 3D GT 当观测；
floor/标定缺失即阻塞；D3 probe 结论（可行 → 产出同格式映射 + corr QA；
不可行 → 标红降级像素级报告）。

**Phase 3（方法线 A：pressure_tookit）**：合并备份集成 + loader 修复 → 单序列
三阶段 smoke → test split 全量 → 导出/评估。验收：单测通过；GT→GT 自检误差为 0。

**Phase 4（方法线 B+C：VP-MoCap）**：FPP-Net 训练（loss 收敛存档）→ test 推理
pred_contact_smpl（(2,96) 契约）→ PoseTransOpt 单 session smoke → test split
全量 → 导出。验收：推理输出可被 PoseTransOpt 读取；opt_result.pth 形状/平滑
检查；世界系导出与 GT 同坐标系（RTE 量级合理）。

**Phase 5（统一评估收口）**：三模型 `predictions/eval_motion/` 齐全 →
`r_test2_compare.py` → Test2 四行（AnySole/MotionPRO/Step2Motion/pressure_tookit/
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
