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
- 视角组合：15 种（n1×4、n2×6、n3×4、n4×1），与 calib_baseline 相同枚举。
- 指标：MPJPE（pelvis 对齐）+ GlobalMPJPE（世界系），均值 ± 组合间标准差
  （n=4 单一组合无 std，表注）；评估函数 = calib_baseline 同款
  （`anysole.utils.metrics.pelvis_align/mean_point_error`，GT = 协议 FK 关节）。
- 导出器：`results_display/script/e_results_export.py` `_line1_stats` 读缓存填
  表 4 线 1（MPJPE/GlobalMPJPE 列；PA-MPJPE 列留 —）。

## 管线口径（与上游 mv1p 的对应）

| 步骤 | 实现 |
|---|---|
| 2D 关键点 | RTMPose-Halpe26 检测缓存注入（EasyMocap 自带检测器不启用）→ OpenPose body25 序映射（25 点全为实检测点，无近似） |
| 去畸变 | kp2d 经 cv2.undistortPoints（P=K）无畸变化；Pall = K@[R\|T]，世界系 m |
| 标定 | protocol/calibration 全链：R = Rc@Rm，T = (Rc@tm + tc)/100 |
| 三角化 | `simple_recon_person` 逐帧 + mv1p 同款重投影清洗（残差 > 50px 观测置零重三角化） |
| 阈值 | check_keypoints conf < 0.3 置零（上游 --thres2d 默认） |
| 拟合（n≥2） | `smpl_from_keypoints3d2d`：形状（体15 骨长比）→ 全局 RT → 3D 姿态 → 2D 精修；上游默认权重（k3d 1 / smooth_body 5 / smooth_poses 1 / smooth_Rh 1 / reg_poses 1e-3 / reg_poses_zero 1e-2 / k2d 1e-4） |
| 拟合（n=1） | 无三角化，`multi_stage_optimize2d` 纯 2D（上游 1v1p 同路径；k2d 1.0 主项 + 平滑/正则）；trans 初始化 = 骨盆射线（body25 MidHip）× 站立骨盆高度平面（y=0.9m 先验，非 GT）求交——与 calib_baseline.fit_2d 同款，零初始化实测落入深度歧义坏极小（MPJPE 207 → 110 量级） |
| 输出关节 | 拟合参数经模型 forward（return_smpl_joints）取回归 SMPL 原生 24 关节——与 calib_baseline 的 smplx 层回归关节同族；GT 端为协议 FK 关节（两基线同规则，差异一致计入） |

est/gt 对齐：body25 回归器输出与 body25 检测 GT 同定义逐位比较，无近似映射、
无补零。

## 冒烟（2026-10-30，S13011 n=4 四视角 325 帧）

MPJPE 27.1 / GMPJPE 34.1（同单元 calib_baseline：25.0 / 19.4——检测 2D 的
姿态代价 +2.1mm，世界系全局定位代价 +14.7mm）。逐关节结构无异常：EasyMocap
足部优于 calib_baseline（43–52 vs 76–82mm），膝部相反（53 vs 35–40mm），
手部 7mm（= 腕部实检测点）。单组合拟合约 12s（325 帧，GPU 0）。

## 全量数字

（test 36 + val 12 全组合跑完后回填，并入 e3_multiview.json / 表 4 CSV）
