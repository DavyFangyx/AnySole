# RGB SMPL 三基线两套指标对齐评估

metrics分支保留AnySole公共公式，并按当前协议对齐2_Code。两套评估从保存的
WHAM、SAM3DB和GVHMR预测独立重算，不读取旧评估值或重新运行基线推理。

## 指标协议

- MPJPE：24个原生FK关节，分别减去关节1、2的髋均值。
- PA-MPJPE：未减骨盆的关节逐帧相似变换对齐，包含尺度、旋转和平移。
- MPJAE：21个局部body旋转测地角，排除root和hand末端。
- PVE：预测自身beta与GT beta的6890顶点，分别减髋均值。
- PVE-T：各自beta的零姿态网格，髋均值对齐，无尺度校正。
- ACC-ERR和Jerk：骨盆相对轨迹，全部配对帧，实际时间差分；不额外除10。
- W-MPJPE100：100输入帧窗口，前两帧拟合相似变换。
- Root ATE：世界关节0的未对齐平均距离，不是ATE-RMSE。
- Foot sliding：关节7、8、10、11；GT速度小于0.3m/s时，预测三维位移均值。

GT旋转使用SO(3)插值，translation线性插值；重复时间取均值，时序长缺口分段。
先取单位均值，再在同一物理会话内等权，最后按257个会话宏平均。
AnySole原主模型不预测beta，同GT beta的PVE仍为已知体型诊断；RGB三基线
都有预测beta，其PVE和PVE-T评价预测体型。

## 代码位置

公共公式：anysole/utils/metrics.py。指标装配：anysole/utils/vision_smpl_metrics.py。
数据和SMPL辅助：anysole/data/rgb_smpl，选择性移植原Tactile四模块，运行时
不导入2_Code。并行入口：AnysoleWorkspace/tool/eval_rgb_smpl.py。
结果消费：results_display/script/compare_rgb_smpl_metrics.py，仅读取CSV/JSON。

## 运行与输出

所有Python命令使用Tactile Conda prefix，并配置工作区内TMPDIR。
在AnySole目录运行python AnysoleWorkspace/tool/eval_rgb_smpl.py --workers 7
--output /private/Tactile/3_Result/SMPL_evaluation/.aligned_latest。
2_Code目录运行python -m smpl_evaluation.cli --workers 7
--output /private/Tactile/3_Result/SMPL_evaluation/.aligned_latest/.2_code_run。
两套入口拒绝覆盖已有运行目标，先完成全量验证再发布，避免暴露部分结果。

在AnySole目录运行python results_display/script/compare_rgb_smpl_metrics.py
--root /private/Tactile/3_Result/SMPL_evaluation/.aligned_latest
--two-code-run /private/Tactile/3_Result/SMPL_evaluation/.aligned_latest/.2_code_run。
消费端从本次2_Code重跑拆分版本，并检查两套逐单元差值和覆盖。

正式结果每个baseline仅保留2_code_aligned和anysole_aligned；版本内含
unit/session/aggregate_metrics.csv、run.json、report.md。根目录保存总体及
分层差值、基线配对bootstrap、数值审计、来源与比较分析_两套指标对齐.md。
发布验收后清除旧历史/旧版本，只保留最新评估结果。

数值容差：毫米类1e-4mm，MPJAE 1e-6度，ACC 1e-4m/s²，Jerk 0.005m/s³。
比较失败返回1并保留报告，不用显示精度或放宽容差掩盖差异。
针对回归：python -m pytest tests/test_rgb_smpl_metrics.py -q -s -p no:cacheprovider。
依赖未提供Baselines包的旧消费者测试仍不能直接运行；新增协议有独立测试。
