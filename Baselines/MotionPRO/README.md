# MotionPRO

本目录是 MotionPRO 在本机上的 baseline 工作区，数据来自 PressureWasher，不走原论文公开数据集下载流程。

统一目录：输入与依赖位于 `../../AnysoleWorkspace/`，checkpoint、指标和日志位于
`../../results/MotionPRO/`，GIF/MP4 位于 `../../results_display/Test1_visualization/MotionPRO/`。
可分别通过 `ANYSOLE_WORKSPACE`、`ANYSOLE_RESULTS` 和
`ANYSOLE_RESULTSDISPLAY` 覆盖根路径。

原始说明见 [Readme_origin.md](Readme_origin.md)。

## 使用方式

所有命令都在这个目录执行：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
conda activate touch_gait
```

环境是 `touch_gait`。权重已经放好：

- `../../AnysoleWorkspace/dependencies/smpl/SMPL_NEUTRAL.pkl`
- `../../AnysoleWorkspace/dependencies/MotionPRO/cliff_ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt`
- `../../AnysoleWorkspace/dependencies/MotionPRO/mmdetection/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth`

### 1. 转序列

把 PressureWasher 的对齐结果转成 MotionPRO 序列目录：

```bash
# 会先删掉已有目录，再整段重转。默认相机 3，输出到统一 workspace
python data_prep/prepare_sequences.py --cam-id 3 --overwrite
# 只刷新接触标签，不重做完整转换
python data_prep/prepare_sequences.py --cam-id 3 --refresh-contact
```

输出在 `../../AnysoleWorkspace/derived/MotionPRO/sequences/cam<id>/<date>/<subject>/<session_id>/`，至少包含：

```text
align_meta.json
color/
pressure.npz
smpl.npy
contact.npy
fake_mask.npy
```

质量表里标成 `C` 或 `D` 的 session 会跳过。`contact.npy` 是 `(T, 10)`，只有 6/7 两列（左右脚）可能为 1。

### 2. 生成训练文件

序列转完后，再补 bbox、图像特征和 3D 关键点：

`gen_image_feature` 需要先有 `bbox.npy`。bbox 用 `bbox_scan`，特征和关键点用 `touch_gait`。`python -m lib.util.*` 必须在 MotionPRO 根目录执行；如果当前目录是 `~/projects/gait`，会报 `No module named 'lib'`。

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
conda activate bbox_scan
python -m lib.util.gen_bbox --cam-id 3

# bbox 跑完后再：
conda activate touch_gait
python -m lib.util.gen_image_feature --cam-id 3
python -m lib.util.gen_kps --cam-id 3
```

完成后一条完整序列是：

```text
../../AnysoleWorkspace/derived/MotionPRO/sequences/cam3/20260808/S12/S12021/
├── align_meta.json
├── color/
├── pressure.npz
├── smpl.npy
├── contact.npy
├── fake_mask.npy
├── bbox.npy
├── feature_hrnet.pth
└── keypoints.npy
```

训练读的是 `feature_hrnet.pth`、`pressure.npz`、`smpl.npy`、`keypoints.npy`、`contact.npy`。视频、压力、SMPL 按同一帧索引切片，不再做时间戳对齐。

### 3. 划分 train / val / test

训练只读 `task.split_csv`，默认 `../../AnysoleWorkspace/splits/default/splits.csv`。完整序列默认都入组，`val` 与 `test` 相同。默认按 IID 切：同一 subject + action 只有 1 条则进 train；多于 1 条则最大 trial 进 val/test。例如 `S12101`、`S12102` 训练，`S12103` 测试。`--ood` 把指定受试者全体放到 val/test，其余人仍走 IID。`--exclude` 把指定受试者整组踢出这份 CSV，不进 train/val/test。

```bash
python data_prep/make_splits.py --cam-id 3
python data_prep/make_splits.py --cam-id 3 --ood S11,S10
python data_prep/make_splits.py --cam-id 3 --exclude S5,S6
```

只写一份 `../../AnysoleWorkspace/splits/default/splits.csv`，列是 `train,val,test`。

### 4. 训练

默认读统一 split，数据来自 `../../AnysoleWorkspace/derived/MotionPRO/sequences/cam3`。`task.cam_id=3` 只是选这个目录，不是相机内参。配置里已经默认 cam3，下面这行等价于不写 `task.cam_id`：

```bash
CUDA_VISIBLE_DEVICES=6 python -m app.train_frappe
```

本机 wandb 默认在 `config/config.yaml` 里是关闭的。要记日志时：

```bash
CUDA_VISIBLE_DEVICES=6 python -m app.train_frappe --wandb_mode offline --wandb_project MotionPRO --wandb_entity davyfangyuxuan-nanjing-university-of-aeronautics-and-ast
```

`--wandb_*` 会在 Hydra 之前被摘掉，所以后面仍可写 `task.epochs=1`、`task.gpu=0`。日志目录是 `../../results/MotionPRO/logs/`，checkpoint 在 `../../results/MotionPRO/checkpoints/<task>/<loss>/<lr>/`，默认 best 文件是 `imagepressure2smpl_best.pth`。

训练过程只记窗口级验证指标；训练结束后会自动用 best checkpoint 对同一份 `splits.csv` 的 `test` 列做完整测试，写出表 3/4：

- 训练 step：`loss/total`、`loss/foot`、`loss/joint`、`grad_norm/total`
- 每个 eval epoch：`val/mpjpe`、`val/contact_iou`
- 按 session：`val_by_session/<session_id>/mpjpe`、`val_by_session/<session_id>/contact_iou`

Contact IoU 用窗口内地平面上的脚高，SMPL 关节 10/11，阈值 5 cm。验证集读 `split_csv` 的 `val` 列。完整测试按非重叠窗口拼回整段 session，用 GT `beta` 和预测 `theta/trans` 过 SMPL，再按 session 算：

- MPJPE / P-MPJPE / PVE / W-MPJPE / WA-MPJPE / RTE / WBCE：毫米
- Accel：m/s^2
- Jitter：10^-3 m/s^2

`eval_fps` 默认 40。OVERALL 里 RTE 按 session 等权平均，其余按有效帧数加权。结果写到：

- `../../results/MotionPRO/metrics/<task>/<loss>/<lr>/test_metrics.csv`
- `../../results/MotionPRO/metrics/<task>/<loss>/<lr>/test_metrics.log`

### 5. 单独复跑测试（可选————当前训练结束之后自动完整流程测试）

训练结束后已经出过完整指标。只有换 checkpoint、换 GPU，或者想再跑一遍时才需要这一步。默认仍读 `task.split_csv` 的 `test` 列，未指定 `task.checkpoint_path` 时加载 `imagepressure2smpl_best.pth`。

```bash
python -m app.test_frappe
python -m app.test_frappe task.checkpoint_path=results://MotionPRO/checkpoints/imagepressure2smpl/init/5e-05/imagepressure2smpl_best.pth
CUDA_VISIBLE_DEVICES=1 python -m app.test_frappe
python -m app.test_frappe task.gpu=1
```

GPU 规则和训练一样：已有 `CUDA_VISIBLE_DEVICES` 时不覆盖，否则用 `task.gpu`。

### 6. 可视化对比

默认扫 `../../results/MotionPRO/checkpoints` 下每个 `.pth`，对 `splits.csv` 的 `test` 列逐条渲对比动画。每个 ckpt 单独一个目录，只写 gif 和 mp4，不写 png。已经有完整输出的 session 会跳过；`--force` 才覆盖。

```bash
python ../../results_display/script/visualize_motionpro.py
python ../../results_display/script/visualize_motionpro.py --session '{S14103,S14023}' --force
python ../../results_display/script/visualize_motionpro.py --checkpoint results://MotionPRO/checkpoints/imagepressure2smpl/init/5e-05/imagepressure2smpl_best.pth
```

输出在：

```text
../../results_display/Test1_visualization/MotionPRO/<task>/<loss>/<lr>/<ckpt_stem>/gif/<session_id>_compare.gif
../../results_display/Test1_visualization/MotionPRO/<task>/<loss>/<lr>/<ckpt_stem>/mp4/<session_id>_compare.mp4
```

找不到的 session（例如测试集里没有对应序列）会告警并跳过。

## 数据约定

Session id 形如 `S14103`：subject=`S14`，action=`10`，trial=`3`。

三种模态按同帧索引使用：

```text
feature[t] <-> pressure[t] <-> SMPL[t] <-> keypoints[t] <-> contact[t]
```

`ImagePressureDataset` 用 `task.split_csv` 里的 session id 去统一 sequences 目录下找 `feature_hrnet.pth`。某个 session 缺文件时直接报错退出。
