# 本地 RGB→SMPL 三基线指标复核

本入口在 `metrics` 分支，基于 `e9accee0`。读取已有 WHAM、SAM3DB、GVHMR
预测及 marker 拟合 GT′，不运行基线推理。公共指标公式仍以
`anysole/utils/metrics.py` 为唯一来源，未修改该文件。

从 `/private/Tactile/AnySole` 执行，使用 Tactile 环境：

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p /private/Tactile/.ai/tmp/anysole-e9accee0-eval
export TMPDIR=/private/Tactile/.ai/tmp/anysole-e9accee0-eval
/opt/conda/bin/conda run --no-capture-output --prefix /opt/conda/envs/Tactile \
  python AnysoleWorkspace/tool/eval_rgb_smpl.py --workers 7
/opt/conda/bin/conda run --no-capture-output --prefix /opt/conda/envs/Tactile \
  python results_display/script/compare_rgb_smpl_metrics.py
```

入口拒绝覆盖已有版本目录。再次完整复核应通过 `--output` 指向
`/private/Tactile/3_Result/SMPL_evaluation/` 下新的子目录，并将同一目录
传给比较脚本的 `--root`。`--limit 8 --workers 2 --output <pilot>` 可做跨日期、
方法、相机的有限试跑；有限试跑不满足完整比较脚本的 2056-unit 验收条件。

结果按 `3_Result/SMPL_evaluation/<baseline>/<metric_version>/` 保存：

| 版本 | 含义 |
|---|---|
| `2_code_reference_20261006` | 原有发布表格的字节相同副本；非重跑 |
| `anysole_e9accee0` | 原生 SMPL-24 关节、全部24局部旋转、世界关节时序、公共关节足滑 |
| `anysole_e9accee0_formula_control` | 旧输入/掩码的公式控制；body21、局部时序、顶点足滑是诊断控制 |

新增指标装配代码在 `anysole/utils/vision_smpl_metrics.py`；数据读取、坐标变换、
SMPL 前向与并行控制入口在 `AnysoleWorkspace/tool/eval_rgb_smpl.py`；
`results_display/script/compare_rgb_smpl_metrics.py` 只消费评估结果。
仅只读复用 `2_Code/smpl_evaluation/{io,alignment,coordinates,model}.py`，
通过隔离 namespace 避免其 package 初始化时引入旧 metrics；没有导入旧具体
指标、runner 或 GVHMR evaluation。底层工具没有再复制一份。

每个新版本有 unit/session/aggregate CSV、run.json 和比较报告；比较端另写
逐 unit 差值、根版本比较 CSV、基线配对 bootstrap CSV、公式控制审计 JSON
与 `比较分析_AnySole_e9accee0.md`。run.json 保存基底提交、分支、源码、模型、
GT、标定与 manifest SHA-256，以及采样与聚合策略。完整运行的 worker
进度、失败表与运行汇总位于结果根的 `.rgb_smpl_run_<pid>/`。

比较入口严格控制检查失败时退出1，但仍完整输出报告与差值。本次30项控制
中28项通过；WHAM/SAM3DB的W-MPJPE100各17个unit超过0.0001mm预设容差，
最大差0.000256/0.000181mm。没有放宽容差；源码AST与ULP敏感性证据另保留
在结果根。它们不改变完整原生协议“结果不同”的结论。

PVE-T 在 e9accee0 已明确废除，因此为 null/CSV空值/N/A；不能移植旧 T-pose
指标补造。三基线生成 beta 和表面，所以 PVE 使用各自预测形状。

采样继续使用相同 RGB 帧与 CSV 锚定的真实 mocap 时间；不将不规则帧假定
为40Hz，也不切换主模型的split/cam3。AnySole主模型自己的40Hz网格、
训练/推理与触觉流程不在此次复核范围。时序函数沿用原实现的重复时间合并、
长间隔分段；原生协议保留全部配对帧，公式控制按历史规则剔除修复参考帧。

针对验证：

```bash
/opt/conda/bin/conda run --no-capture-output --prefix /opt/conda/envs/Tactile \
  python -m pytest tests/test_rgb_smpl_metrics.py -q -s -p no:cacheprovider
```

已有依赖 `Baselines` 包的消费者测试需要完整上游包；该提交本地不包含它。
本次使用 canonical 公式、真实数据试跑和针对协议测试验收，不伪造原测试通过。
