# Fast SAM 3D Body 全自动 RGB→SMPL Pipeline

`2_Code/sam3db_baseline/` 包含当前 pipeline 的全部自有实现：数据准备、
RGB→MHR→SMPL 推理、overlay、7 GPU × 2 worker 调度、断点续跑和审计。

它依赖以下仓库内容，不能只复制本目录运行：

- `2_Code/wham_baseline/`：数据 inventory 和 annotation 索引；
- `third_party/fast-sam-3d-body/`、`third_party/dinov3/`、`third_party/mhr/`；
- `models/` 中的 MoGe 和 neutral SMPL；
- `/opt/conda/envs/Tactile`。

## 1. 输入和输出

输入格式：

```text
1_Dataset/<date>/
├── RGB/<method>/<session>/camera_<N>/*.{jpg,jpeg,png,bmp}
└── annotations/records.parquet
```

- `method` 当前只支持 `method_1_offset` 和 `method_2_projection`；
- 图片文件名必须能解析出 frame ID；
- Parquet 必须包含 `date`、`session`、`camera`、`frame`、
  `bbox_xyxy_px`；每张 RGB 必须有唯一有效 bbox。

固定输出根为 `3_Result/SAM3DB_baseline`。每个成功 unit 只有三个公开文件：

```text
3_Result/SAM3DB_baseline/<date>_<m1|m2>_<session>_c<camera>/
├── result.pkl
├── overlay.mp4
└── state.json
```

manifest、日志、失败记录和运行状态统一位于
`3_Result/SAM3DB_baseline/.run/`。

## 2. 环境和代码入口

从工作区根执行：

```bash
cd /private/Tactile
source /opt/conda/bin/activate /opt/conda/envs/Tactile
export PYTHONPATH=/private/Tactile/2_Code
```

主要文件：

| 文件 | 用途 |
| --- | --- |
| `prepare.py` | 扫描数据、生成 manifest/annotation index/config hash |
| `runtime.py` | Fast SAM3DB、MHR→SMPL 和投影 |
| `worker.py` | 处理 unit 并原子发布三个输出文件 |
| `pipeline.py` | 多 GPU 调度、续跑、状态和审计 |
| `cli.py` | `prepare`、`run`、`status`、`audit` |
| `launch.sh` | 脱离终端运行 coordinator |

首次运行或修改代码后检查：

```bash
pytest -q 2_Code/tests/test_sam3db_baseline.py
bash -n 2_Code/sam3db_baseline/launch.sh
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ps -eo pid,ppid,pgid,sid,stat,args | rg 'sam3db_baseline.(cli run|worker)'
```

启动前 GPU 应空闲，且不能存在使用同一结果根的 coordinator/worker。

## 3. 先跑小批量

选择一个短 session，例如：

```bash
python -m sam3db_baseline.cli prepare \
  --dates 20260808 \
  --methods method_1_offset \
  --sessions S11073

bash 2_Code/sam3db_baseline/launch.sh bounded \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --retries 1
```

`launch.sh` 返回 PID 后任务已脱离当前终端。等待它自然退出，再执行：

```bash
python -m sam3db_baseline.cli status
python -m sam3db_baseline.cli audit
```

确认 `audit.clean == true`，并人工查看每路 `overlay.mp4` 的首、中、末帧，
确认 bbox 选中目标人物、skeleton 对齐且视频帧数正确。小批量未通过或未经用户
确认时，不启动全量任务。

可用的额外筛选参数：

```text
--cameras camera_1,camera_2
--max-frames-per-unit 120
```

## 4. 启动全量自动任务

全量运行前再次确认没有活动的 SAM3DB 进程。然后生成无筛选完整 manifest：

```bash
python -m sam3db_baseline.cli prepare
sed -n '1,220p' 3_Result/SAM3DB_baseline/.run/ready.json
```

必须确认：

- `status` 为 `ready`；
- `full_manifest` 为 `true`；
- dates/methods/sessions/cameras 筛选均为 `null`；
- unit/frame 数量符合当前数据集。

获得用户明确全量授权后启动：

```bash
bash 2_Code/sam3db_baseline/launch.sh full user-approved \
  --gpus 0,1,2,3,4,5,6 \
  --workers-per-gpu 2 \
  --retries 1 \
  --force
```

生产配置固定为 7 GPU × 2 worker，共 14 workers。不要改成单 worker，也不要
使用未经验证的 3 workers/GPU。`--force` 用于处理 pilot 与 full config hash
不同的同名 unit；旧无效 unit 会移入 `.run/attempts/`，不会直接删除。

## 5. 监控

任务无需 Agent 持续在线。按需执行只读检查：

```bash
python -m sam3db_baseline.cli status
sed -n '1,240p' 3_Result/SAM3DB_baseline/.run/run.json
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ps -eo pid,ppid,pgid,sid,stat,args | rg 'sam3db_baseline.(cli run|worker)'
find 3_Result/SAM3DB_baseline/.run/failures -maxdepth 1 -type f -print
```

正常状态：14 workers、每张 GPU 两个进程、成功 unit/frame 持续增加、失败记录
不增加。日志位于 `.run/logs/gpu-<gpu>-slot-<slot>-attempt-<N>.log`。

`status` 会校验现有结果 hash，结果很多时不要高频轮询。

## 6. 中断和恢复

如果任务中断：

1. 确认旧 coordinator 和所有 worker 已退出；
2. 检查 `.run/run.json`、失败日志和 GPU；
3. **不要重新运行 `prepare`**；
4. 使用与原任务完全相同的启动命令重新启动。

```bash
bash 2_Code/sam3db_baseline/launch.sh full user-approved \
  --gpus 0,1,2,3,4,5,6 \
  --workers-per-gpu 2 \
  --retries 1 \
  --force
```

有效 unit 会被校验并跳过，只重算 pending/invalid unit。不要 broad `pkill`，
不要手改 `state.json`，也不要在进程运行时清理 `.run/stage`。

如果修改会影响模型、输出或处理语义，应开始新的 manifest snapshot，而不是用
旧 config hash 续跑。

## 7. 最终验收

coordinator 自然退出后执行：

```bash
python -m sam3db_baseline.cli status
python -m sam3db_baseline.cli audit
```

只有同时满足以下条件才算完成：

- `run.json.status == "completed"`；
- succeeded units/frames 等于 manifest totals；
- `audit.clean == true` 且 `audit.issues` 为空；
- coordinator/worker 均已退出，GPU 已释放；
- 抽样 overlay 正常。

## 8. 新增数据后复用

新增数据仍使用第 3～7 节的流程，但必须注意：当前 `config_hash` 绑定整个
manifest、annotation、代码和模型资产。新增日期/session/frame 后重新
`prepare`，旧输出不会自动成为新完整 baseline 的一部分。

推荐流程：

1. 按第 1 节放置新增 RGB 和 `records.parquet`；
2. 确认旧任务已结束；
3. 让用户决定旧 `3_Result/SAM3DB_baseline` 的保留/归档位置；
4. 获得明确授权后整体移动旧结果根，不要直接删除；
5. 在干净结果根先跑新增数据的 bounded pilot；
6. 用户确认后执行无筛选 `prepare`，生成包含全部新旧数据的完整 manifest；
7. 用 7×2 命令启动全量任务；
8. 执行最终 `status`、`audit` 和视频抽检。

不要在包含旧 public unit 的同一结果根只执行 `prepare --dates <new-date>` 后将其
视为完整 baseline；audit 会把不属于当前 manifest 的旧目录报告为 unexpected。

## 9. 不能回退的运行约束

- 所有 Python 命令使用 `/opt/conda/envs/Tactile`；
- 运行中禁止 `prepare`，它会替换 `.run` 控制文件；
- 保留本地 DINOv3/MoGe 和 offline 配置，不允许网络下载 fallback；
- `FOV_MODEL_PATH` 必须指向 MoGe 的 `model.pt` 文件；
- 保持 TensorRT/compile/CUDA graph 关闭；
- 使用 annotation bbox 选择目标人物，不使用 detector 自动猜测；
- public unit 只保留 `result.pkl`、`overlay.mp4`、`state.json`；
- 全量并发为 2×7；完成状态只认 `status` 和 clean `audit`。
