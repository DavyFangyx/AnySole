# EasyMocap 集成登记（E3 线 1「外部标定多视角基线」）

2026-10-30 裁定与执行记录。

## 位置

- 上游：`z_note/HMR2.0/EasyMocap`（git submodule @ e6006fd，zju3dv/EasyMocap master；与
  4D-Humans 同方案，只读 0 改动）。**不进 Baselines/**——Baselines 库契约为七个单视角
  基线与统一单视角评估，EasyMocap 只服务 E3 表 4 的 per-(会话, 视角数, 组合) 口径。
- 驱动：`anysole_tta/easymocap_baseline.py`（touch_gait env；与 calib_baseline 并列）。
- 资产：`AnysoleWorkspace/assets/third_party/easymocap/`（SMPL_NEUTRAL.pkl 软链 + 
  J_regressor_body25.npy）。
- 产物：`results/baselines/HMR2.0/predictions/easymocap/{sid}_n{n}_{camstr}.npz`
  （joints (T,24,3) 世界系 m + frame_id，与 calib_baseline 同契约，存在即跳过）。

## 角色与口径

E3 表 4 线 1「外部标定多视角基线（真值外参，参考上限）」= **真实外部基线**：
RTMPose 检测 + EasyMocap 原生管线。calib_baseline（GT 投影 2D + 自写拟合）保留为
同线 GT-kpt 变体（纯上限参照，不另占表 4 行）。

- 会话：val 12 + test 36（表 4 正式数字 = test）；帧轴 = facts 40Hz 正典网格。
- 视角组合：n=2/3/4 共 11 种（n2×6、n3×4、n4×1）。**n=1 已删除
  （2026-10-31 用户裁定）**：原生管线要求 ≥2 视角，单视角 GMM 兜底不代表
  标定方法能力，不列入 E3 表 4（表中该行填 —）；n=1 拟合代码保留于
  fit_1v/_init_pnp 备查。
- 指标：PA-MPJPE（`pa_mpjpe`）/ MPJPE（`pelvis_align` + `mean_point_error`）/
  GlobalMPJPE（原始世界系 `mean_point_error`）——官方评估层同款函数
  （README_metrics 指标族表；2026-10-31 补 PA）；均值 ± 组合间标准差
  （n=4 单一组合无 std，表注）；GT = 协议 FK 关节（load_gt_joints）。
- 导出器：`results_display/script/e_results_export.py` `_line1_stats` 读缓存填
  表 4 线 1（PA-MPJPE/MPJPE/GlobalMPJPE 三列）。

## 管线口径（与上游 mv1p 的对应）

| 步骤 | 实现 |
|---|---|
| 2D 关键点 | RTMPose-Halpe26 检测缓存注入（EasyMocap 自带检测器不启用）→ OpenPose body25 序映射（25 点全为实检测点，无近似） |
| 去畸变 | kp2d 经 cv2.undistortPoints（P=K）无畸变化；Pall = K@[R\|T]，世界系 m |
| 标定 | protocol/calibration 全链：R = Rc@Rm，T = (Rc@tm + tc)/100 |
| 三角化 | `simple_recon_person` 逐帧 + mv1p 同款重投影清洗（残差 > 50px 观测置零重三角化） |
| 阈值 | check_keypoints conf < 0.3 置零（上游 --thres2d 默认） |
| 拟合（n≥2） | `smpl_from_keypoints3d2d`：形状（体15 骨长比）→ 全局 RT → 3D 姿态 → 2D 精修；上游默认权重（k3d 1 / smooth_body 5 / smooth_poses 1 / smooth_Rh 1 / reg_poses 1e-3 / reg_poses_zero 1e-2 / k2d 1e-4） |
| 拟合（n=1） | **已删除（2026-10-31 用户裁定，不列入表 4）**。实现备查：上游生产路径 = SPIN 学习初始化（1v1p，权重不可下载）、纯优化无姿态先验实测塌缩 → GMM 姿态先验 CHUNK=16 批 LBFGS（calib_baseline.fit_2d 同款口径）+ PnP 旋转/平面深度混合初始化 |
| 输出关节 | 协议 FK 关节（官方导出契约，决策/11 W1：fk_pose6d_np + GT offsets/parents，与 protocol_gt 关节定义一致）——2026-10-31 用户审查修正：此前导出模型回归关节与契约不符。拟合系 → 协议系：该 pkl 模板相对协议骨架为纯平移 s（均值姿态 native 骨盆 ≈ (0,−0.223,0.028)，逐关节验证），trans 校正 = Th + R_total·s（R_total = R(Rh)·R(poses 根槽)） |

est/gt 对齐：body25 回归器输出与 body25 检测 GT 同定义逐位比较，无近似映射、
无补零。

## 冒烟（2026-10-30，S12011 全 15 组合）

n=4 四视角（协议 FK 合同数字）：PA 27.0 / MPJPE 35.5 / GMPJPE 61.7——
旧回归关节口径 33.1/38.4 作废（关节定义与官方契约不符）。逐关节结构无
异常：EasyMocap 足部优于 calib_baseline（43–52 vs 76–82mm），膝部相反
（53 vs 35–40mm）。单组合拟合 n≥2 ≈12s（325 帧，GPU 0）。

## 全量数字（528 单元 = 48 会话 × 11 组合，n=2/3/4）

E3 表 4 线 1（test 36，组合间口径——每组合取会话均值、跨组合 mean±std，
n=4 单一组合无 std；MPJPE / GlobalMPJPE，mm）：

| n | PA-MPJPE | MPJPE | GlobalMPJPE |
|---|---|---|---|
| 2 | 45.7 ± 7.5 | 68.9 ± 12.8 | 104.3 ± 42.7 |
| 3 | 35.4 ± 1.4 | 52.0 ± 4.0 | 60.8 ± 8.7 |
| 4 | 33.9 | 48.5 | 52.6 |

聚合入口 = `easymocap_baseline.py --split test --aggregate`（touch_gait；
读 test 会话缓存重算并写入 `metrics/e3_multiview.json` 的 easymocap 键，与 calib_baseline
并列）；表 4 CSV = `e_results_export.py --experiment E3`（`_line1_stats`
首选读该键——纯 json、导出器无 touch_gait 依赖也可跑；键缺失才从 npz
兜底重算）。

E7 多视角参考线（2026-10-30 用户裁定接入 Vi/ViT 阶段，n=4 四相机，
踝足 MPJPE mm，协议 FK 口径）：walk 59.2（Vi）/ 45.8（ViT）、
run 53.6 / 30.1、squat 70.5 / 48.0——`e7_driver.py` 读同源缓存，表/图
已刷新（Etest/E7_DynamicModalityInputs/）。
