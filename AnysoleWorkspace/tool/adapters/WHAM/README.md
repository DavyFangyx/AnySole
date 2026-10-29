# WHAM 全自动推理操作手册

本文只保留启动和维护全量 pipeline 所需内容。工作区固定为
`/private/Tactile`；数据集只读，结果只写入
`/private/Tactile/3_Result/WHAM_baseline` 及其子目录。

## 固定运行契约

- 调度、测试、制表和可视化使用 `/opt/conda/envs/Tactile`；WHAM GPU
  推理使用 `/opt/conda/envs/WHAM/bin/python`。
- 输入单元是
  `1_Dataset/<date>/RGB/<method>/<session>/camera_<n>/frame*.jpg`；目标人物框来自
  `<date>/annotations/records.parquet`。
- 推理只使用逐帧目标框和 WHAM 默认相机启发式；禁止传入数据集
  `K/D/R/t`、`--calib` 或启用全局 DPVO。输出为 CFR 30 FPS。
- 当前主机配置：GPU `0..6`、每卡 2 个独立推理 worker、8 个 CPU
  可视化 worker、可视化队列 64。
- 每个公开单元目录只保留 `result.pkl`、`overlay.mp4`、`state.json`；控制、
  日志和临时缓存位于结果根的 `.run/`。
- 清空旧结果和启动全量推理是两个独立的破坏性/高资源动作，均须得到当次
  用户明确授权。`launch_full.sh` 本身不会删除任何文件。

缺少环境时才按本目录 `environment-wham.yml`、`constraints-wham.txt` 重建；
正式启动前必须确认 `third_party/WHAM/demo.py` 与
`lib/models/preproc/detector.py` 中的目标框兼容改动仍在，并由证书哈希校验。

## 当前语料：直接启动或恢复

当前已验证语料为 3,560 个单元、1,399,904 帧、445 个 session；两个方法各
1,780 个单元、699,952 帧。2026-09-18 的完整运行已 clean 完成。若代码、
模型、标注或语料未变化，使用现有证书和启动器即可校验、恢复或重跑缺失单元。

先做只读检查，不得与另一个 coordinator、GPU owner 或孤儿 worker 重复启动：

```bash
cd /private/Tactile
nvidia-smi
ps -ef | grep -E '[w]ham_baseline|[d]emo.py'
jq . 3_Result/WHAM_baseline/.run/summary.json
```

验证代码和启动器：

```bash
cd /private/Tactile/2_Code
source /opt/conda/bin/activate /opt/conda/envs/Tactile
export PYTHONPATH=/private/Tactile/2_Code
pytest -q tests/test_wham_baseline*.py
bash -n wham_baseline/launch_full.sh
```

只有得到本次明确的全量启动授权后才执行：

```bash
cd /private/Tactile
bash 2_Code/wham_baseline/launch_full.sh user-approved
```

启动器会运行 `prepare.py`，校验语料、ROI、代码、模型、运行时、磁盘容量和
compact 名称，然后以新 session/PGID 后台启动 coordinator。命令返回后任务
不依赖 agent 会话，可长期自主推理、重试、落盘、生成视频并最终自动审计；
它不是重启服务，主机重启后需按“恢复”处理。

## 语料扩增：生成新一代任务

新增或修改日期、方法、session、相机、帧或标注后，旧证书和旧配置全部失效。
必须生成新的完整 manifest/ROI/证书并使用新的结果子目录；不要伪造旧
`state.json` 或复用旧哈希。

### 1. 构建并核对候选控制文件

```bash
cd /private/Tactile
source /opt/conda/bin/activate /opt/conda/envs/Tactile
export PYTHONPATH=/private/Tactile/2_Code
export WHAM_CANDIDATE_DIR="$(mktemp -d /private/Tactile/.ai/tmp/wham-expanded.XXXXXX)"

python -m wham_baseline.runner inventory \
  --dataset-root /private/Tactile/1_Dataset \
  --manifest "$WHAM_CANDIDATE_DIR/full.jsonl" --strict
python -c 'from wham_baseline.manifest_index import build_index; import os,json; print(json.dumps(build_index(os.environ["WHAM_CANDIDATE_DIR"]+"/full.jsonl"),indent=2))'
python -m wham_baseline.constraints build-index \
  --dataset-root /private/Tactile/1_Dataset \
  --output "$WHAM_CANDIDATE_DIR/roi.sqlite"
python -c 'from wham_baseline.prepare import coverage; from wham_baseline.manifest_index import load_index; import os,json; d=os.environ["WHAM_CANDIDATE_DIR"]; print(json.dumps(coverage(d+"/full.jsonl",d+"/roi.sqlite",load_index(d+"/full.jsonl",verify_hash=True)),indent=2))'
python -m wham_baseline.constraints inspect-index \
  --index "$WHAM_CANDIDATE_DIR/roi.sqlite"
```

只接受 `coverage.status=passed`，并人工确认 index 中的 unit/frame/session、
逐方法计数和 manifest SHA256。数据仍在写入、存在空单元/重复 ID、异常方法、
缺失标注或命名不兼容时停止。

### 2. 更新门槛并做有界实测

根据已确认的新计数修改并测试以下安全门槛：

- `pipeline.plan()`：完整语料的 unit/frame/session 和逐方法精确计数；
- `compact.short_name()`：仅在新增合法方法或命名格式时修改，并证明无碰撞；
- `launch_full.sh`：新证书、新结果子目录；若预算超过 1 TB，同时向 pipeline
  传入相同的 `--min-free-bytes`。

代码冻结前运行 focused tests，并用有界 manifest 做真实 7×2 推理和视频检查：

```bash
python -m wham_baseline.runner inventory \
  --dataset-root /private/Tactile/1_Dataset \
  --manifest "$WHAM_CANDIDATE_DIR/sample.jsonl" --strict \
  --max-units 14 --max-frames-per-unit 60
python -c 'from wham_baseline.manifest_index import build_index; import os; build_index(os.environ["WHAM_CANDIDATE_DIR"]+"/sample.jsonl")'
python -m wham_baseline.pipeline bounded-run \
  --manifest "$WHAM_CANDIDATE_DIR/sample.jsonl" \
  --result-root /private/Tactile/3_Result/WHAM_baseline/prelaunch-expanded \
  --annotation-index "$WHAM_CANDIDATE_DIR/roi.sqlite" \
  --gpus 0,1,2,3,4,5,6 --workers-per-gpu 2 \
  --viz-workers 8 --viz-queue-limit 64 \
  --max-test-units 14 --max-test-frames 840 \
  --use-target-bbox --wait-for-gpu --compact-output
```

检查 14 个 `result.pkl`、`overlay.mp4`、`state.json`，人工查看代表性视频。
若主机/GPU、分辨率、最长序列或 WHAM 模式变化，先重新校准并据实修改并发；
不要把有界测试当成全量授权。

### 3. 生成新证书

用有界样本测量三类公开文件的空间，按完整帧数外推，并增加至少 30% 与
运行中缓存余量；正式 reserve 不得低于 1 TB。代码冻结后运行以下脚本，
其输出先保存为结果目录外的 task artifact，状态保持 `candidate`：

```bash
python - <<'PY'
import json, os
from pathlib import Path
from wham_baseline.constraints import read_index_metadata
from wham_baseline.manifest_index import file_sha256, load_index
from wham_baseline.pipeline import fingerprint, parser

d = Path(os.environ["WHAM_CANDIDATE_DIR"])
m, roi = d / "full.jsonl", d / "roi.sqlite"
idx = load_index(m, verify_hash=True)
a = parser().parse_args([
    "full-run", "--manifest", str(m),
    "--result-root", "/private/Tactile/3_Result/WHAM_baseline/generation-expanded",
    "--annotation-index", str(roi), "--compact-output", "--viz-workers", "8",
])
a.gpus = list(range(7))
_, config, _ = fingerprint(a, idx)
meta = read_index_metadata(roi)
print(json.dumps({
    "status": "candidate",
    "output_layout": "compact-v1",
    "dataset_root": "/private/Tactile/1_Dataset",
    "manifest_sha256": idx["manifest_sha256"],
    "roi_sha256": file_sha256(roi),
    "annotation_sources": {x["path"]: x["sha256"] for x in meta["sources"]},
    "asset_hashes": {p: h for p, h in config["files"].items()
                     if p != str(roi.resolve())},
    "required_free_bytes": 1_000_000_000_000,
    "viz_workers": 8,
}, indent=2, sort_keys=True))
PY
```

从空临时目录再次生成 manifest/ROI，要求哈希一致。只有 tests、有界推理、
人工视频检查、容量预算和冷重建全部通过后，才把新证书改为 `status=passed`。
证书通过不等于用户授权启动。

### 4. 准备并启动新一代

优先使用 `WHAM_baseline` 下的新结果子目录，避免删除旧结果。若必须复用基准
根目录，先盘点进程、symlink、内容和大小，并另行取得精确删除授权；旧 `.run`
会阻止新语料发现，旧公开状态会产生 config-hash 冲突。

将审核后的 launcher 指向新证书和新结果根，保留 `prepare`、磁盘、GPU 等待、
目标框、resume、compact 和 approval 门槛。先执行 `prepare.py` 和
`pipeline dry-run`，核对计划与 `ready.json`；向用户提交计数、哈希、测试视频、
容量和精确启动命令。获得明确全量授权后才调用 launcher 的
`user-approved` 入口。启动后确认 `launch.json` 的 PID/PGID、manifest hash、
14 个独立推理 slot、空 `pause_reason` 和第一批终态。

## 监控、恢复和验收

被动监控不会干预运行：

```bash
cd /private/Tactile
source /opt/conda/bin/activate /opt/conda/envs/Tactile
export PYTHONPATH=/private/Tactile/2_Code
python -m wham_baseline.pipeline status \
  --manifest 3_Result/WHAM_baseline/.run/full.jsonl \
  --result-root 3_Result/WHAM_baseline --compact-output
jq . 3_Result/WHAM_baseline/.run/summary.json
tail -n 50 3_Result/WHAM_baseline/.run/events.jsonl
nvidia-smi
```

coordinator 持有 `pipeline.lock` 时不要另跑 audit 或重复启动。普通推理与视频
失败最多各尝试 3 次；首次 OOM 会排空活动任务并以每卡单 worker 隔离重试；
磁盘不足、资产/配置变化或连续系统错误会暂停。

异常退出或暂停时，先查看 `pause_reason`、`coordinator.log`、events、记录的
PID/PGID、worker 终态与 `nvidia-smi`。确认没有活动 coordinator/孤儿进程并
修复真实原因后，在原授权仍适用的前提下重跑同一 launcher；resume 会校验并
跳过有效单元，只补缺失/损坏阶段。不得手工篡改状态或哈希。

coordinator 会在结束时自动审计。仅在所有进程结束后才可手动复核：

```bash
python -m wham_baseline.pipeline audit \
  --manifest /private/Tactile/3_Result/WHAM_baseline/.run/full.jsonl \
  --result-root /private/Tactile/3_Result/WHAM_baseline \
  --annotation-index /private/Tactile/3_Result/WHAM_baseline/.run/roi.sqlite \
  --compact-output
```

完成条件：`summary.status=completed`、`audit.clean=true`；推理/视频单元数和帧数
均等于 manifest；issues、failed、pending、unexpected outputs 均为空；每个
公开单元恰有三个有效文件；没有 coordinator、worker、孤儿状态或 GPU 进程。
