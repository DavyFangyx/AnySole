# Agent 02：AnySole × MotionPRO baseline 复现任务

## 目标

在当前仓库中，以 AnySoleWorkspace 的统一数据/依赖布局复现 MotionPRO baseline（PressureWasher 输入，cam3），产出可追溯的训练 checkpoint、测试指标和可视化结果；不改变 AnySole 主模型或 MotionPRO 算法语义。所有命令均从仓库根目录或明确指定的 MotionPRO 目录执行，并记录环境、配置、commit 和实际数据范围。

## 固定路径与约定

仓库根目录：`/data/fangyuxuan/projects/gait`；MotionPRO 工程目录：`Baselines/MotionPRO`。统一工作区为 `AnysoleWorkspace/`：

```text
AnysoleWorkspace/derived/MotionPRO/sequences/cam3/<date>/<subject>/<session_id>/
  align_meta.json color/ pressure.npz smpl.npy contact.npy fake_mask.npy
  bbox.npy feature_hrnet.pth keypoints.npy
AnysoleWorkspace/splits/default/splits.csv
AnysoleWorkspace/dependencies/smpl/SMPL_NEUTRAL.pkl
AnysoleWorkspace/dependencies/MotionPRO/cliff_ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
AnysoleWorkspace/dependencies/MotionPRO/mmdetection/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
```

结果统一写入 `results/MotionPRO/{checkpoints,metrics,logs,tensorboard}/`，动画写入 `results_display/Test1_visualization/MotionPRO/`。不要重新下载或复制已存在的权重；路径可由 `ANYSOLE_WORKSPACE`、`ANYSOLE_RESULTS`、`ANYSOLE_RESULTSDISPLAY` 覆盖，但回传时必须说明覆盖值。

## 执行步骤

1. 环境与工作区检查：

   ```bash
   cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
   conda activate touch_gait
   python AnysoleWorkspace/tool/workspace.py doctor
   test -f ../../AnysoleWorkspace/splits/default/splits.csv
   test -d ../../AnysoleWorkspace/derived/MotionPRO/sequences/cam3
   ```

   若 `doctor` 失败，先报告缺失项；不得擅自改软链接、删除数据或移动依赖。

2. 如统一序列尚未生成，按 README 的实际入口转换（会覆盖已有目录的命令必须先确认）：

   ```bash
   python data_prep/prepare_sequences.py --cam-id 3 --overwrite
   ```

   检查每个纳入 session 至少有上述 9 个文件；质量表标记为 C/D 的 session 不得纳入。

3. 补齐训练输入，按依赖环境分步运行：

   ```bash
   conda activate bbox_scan
   python -m lib.util.gen_bbox --cam-id 3
   conda activate touch_gait
   python -m lib.util.gen_image_feature --cam-id 3
   python -m lib.util.gen_kps --cam-id 3
   ```

   `python -m lib.util.*` 必须在 `Baselines/MotionPRO` 内执行。核验同一 session 的 `feature_hrnet.pth`、`pressure.npz`、`smpl.npy`、`keypoints.npy`、`contact.npy` 帧数一致，且三模态按同帧索引使用。

4. 生成或核对 canonical split（默认 IID；如任务另有 OOD/exclude 要求，必须把参数写入记录）：

   ```bash
   python data_prep/make_splits.py --cam-id 3
   ```

   只允许使用 `../../AnysoleWorkspace/splits/default/splits.csv`，确认列为 `train,val,test`，并保存 session 数量及 subject/action 分布。

5. 先做最小 smoke run，确认数据、模型和输出链路可用：

   ```bash
   CUDA_VISIBLE_DEVICES=6 python -m app.train_frappe task.epochs=1
   ```

   smoke run 失败时保留完整 traceback 和配置，不以空指标或手工生成文件充数。

6. 正式训练与独立测试。正式 epoch、学习率和 GPU 以复现实验配置为准，示例：

   ```bash
   CUDA_VISIBLE_DEVICES=6 python -m app.train_frappe
   python -m app.test_frappe
   ```

   若换 checkpoint，显式传 `task.checkpoint_path=...`。训练结束自动测试也必须检查是否确实使用 best checkpoint 和 split 的 `test` 列。

7. 生成 test 可视化（可选但作为完整验收推荐执行）：

   ```bash
   python ../../results_display/script/visualize_motionpro.py
   ```

## 统一输出清单

至少提交以下相对根目录的文件/目录：

- `results/MotionPRO/checkpoints/<task>/<loss>/<lr>/imagepressure2smpl_best.pth`
- `results/MotionPRO/metrics/<task>/<loss>/<lr>/test_metrics.csv` 与 `test_metrics.log`
- `results/MotionPRO/logs/` 下本次训练日志及配置快照
- `results_display/Test1_visualization/MotionPRO/<task>/<loss>/<lr>/<ckpt_stem>/gif/`、`mp4/`（若执行可视化）
- 使用的 `AnysoleWorkspace/splits/default/splits.csv` 的 hash、session 数量和 train/val/test 数量

指标须保留 OVERALL 及逐 session 行：MPJPE、P-MPJPE、PVE、W-MPJPE、WA-MPJPE、RTE、WBCE、Accel、Jitter，并注明单位（毫米、m/s²、10^-3 m/s²）和 `eval_fps=40`。不得只回传截图或四舍五入后的单个数字。

## 测试与验收标准

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
conda activate touch_gait
python -m pytest -q app/test_eval_metrics.py
python -m py_compile app/train_frappe.py app/test_frappe.py data_prep/*.py
python -m app.test_frappe
```

验收必须同时满足：

1. `doctor`、指标单测和 `py_compile` 通过；
2. smoke run 成功产生 checkpoint/日志，正式 run 成功产生非空 `test_metrics.csv`；
3. CSV 的 session 与 split 的 test 列一致，无缺失输入、NaN/Inf 或空 OVERALL；
4. 日志显示使用统一 workspace、cam3、明确 checkpoint 和 split；
5. 至少随机抽查 2 个 session，确认输入帧数、预测帧数和指标行可追溯；可视化文件可读且路径符合约定。

## 禁止事项

- 不修改 `anysole/`、`Baselines/MotionPRO/` 代码、配置默认值或第三方源码；本任务只允许新增/更新本任务报告文件和结果输出。
- 不删除、覆盖或移动 `AnysoleWorkspace`、`results` 中既有数据/结果；需要 `--overwrite` 时先取得明确批准。
- 不从论文公开数据集替换 PressureWasher 数据，不混用旧 `Baselines/MotionPRO/data/` 路径与 canonical workspace。
- 不跳过 bbox、image feature、keypoints 任一输入，不伪造 contact、指标、checkpoint 或可视化。
- 不把 `val` 当 `test` 回报，不跨 session 随意切分或改变帧率/单位；任何偏离都必须标红说明。

## 回传格式

```text
[Agent 02 MotionPRO 复现回传]
状态：PASS / PARTIAL / BLOCKED
代码版本：<git commit>
环境：<conda env；关键 torch/cuda 版本>
数据：cam=<id>；split=<path+sha256>；train/val/test=<n>/<n>/<n>
执行命令：<逐条列出实际命令，含 GPU、覆盖参数>
输出：<checkpoint、metrics.csv/log、日志、可视化绝对路径>
指标：<OVERALL 行原样摘要及单位>
测试：doctor=<结果>；pytest=<结果>；py_compile=<结果>；test_frappe=<结果>
问题/偏离：<无则写“无”；失败附 traceback 路径和阻塞原因>
```
