#!/usr/bin/env python3
"""Generate baseline task confs into the shared queue (configs/queue/).

GPU 归属不再在这里指定：每个 worker（CUDA_VISIBLE_DEVICES=N bash
configs/bg.sh）从共享队列抢任务。

pressure_toolkit：一个 session 一个 conf，conf 里只有整段拟合
（`FITTING_STAGE=init_pose`，首个缺失帧由 launcher 决定，tracking 自动接续），
不再有 init_shape -> init_pose -> tracking 三阶段拓扑；per-subject 的
init_shape 由 launcher 在 per-subject flock 下按需生成一次后复用。续跑
（`START_IDX=auto`）同样在 launcher 内按"第一个缺失帧"决定，因此同一 session
可被任意 worker 取走，也可以安全重复入队。

run-root 不变式（`Baselines/pressure_tookit/run_full_mmvp.py` 启动即校验）：

    run_root      = work://pressure_toolkit/<run_id>
    OUTPUT_DIR    = <run_root>/fitting
    INIT_DATA_DIR = <run_root>

因为 `main_singleview.py` 把逐帧结果写到 `<OUTPUT_DIR>/results/...`，而 tracking
读前帧 `<INIT_DATA_DIR>/fitting/results/<date>/<subject>/<seq>/smpl_<prev>.npz`
（`lib/dataextra/data_loader.py:398`）；两者指向无关的根会让 tracking 静默失效。

任务级 staging 与过程产物同在 run 根（`work://pressure_toolkit/<run_id>/logs`
与 `artifact.json`），见 configs/run.sh 的 pressure 分支。

出 conf 的判据是 adapter 目录存在（splits.csv 是评估划分，不是拟合范围），
depth/depth_mask/keypoints/CLIFF/地面这些细则由 launcher 的 pre-flight 判定：
判不过的 session 会在 run artifact 里记下理由后跳过，不崩、不空跑 fitting。
注意 done/ 里已有同名 conf 的 session 不会被重新生成（含"只记了跳过理由、
没真正拟合"的情况）；要把这类 session 重新入队，先删掉
`configs/done/*_pressure_<session>.conf` 再跑一次生成器。
"""
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT.parent
REPO = CONFIG_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from AnysoleWorkspace.tool.workspace import resolve_uri

# 受试者性别口径：与 AnysoleWorkspace/tool/export_baseline_motion.py 的
# --female 默认值一致（S14 为女性受试者，其余为男性）。
FEMALE_SUBJECTS = {'S14'}

def put(name, text):
    path = CONFIG_DIR / 'queue' / f'{name}.conf'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): path.write_text(text)

def pressure_queued(session):
    """该 session 是否已在队列/运行中/已完成；failed 允许重新入队。"""
    for stage_dir in ('queue', 'running', 'done'):
        if list((CONFIG_DIR / stage_dir).glob(f'*_pressure_{session}.conf')):
            return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True)
    parser.add_argument('--manifest', default='AnysoleWorkspace/protocol/manifests/session_manifest.jsonl')
    parser.add_argument('--split', choices=('train', 'val', 'test', 'all'), default='all')
    parser.add_argument(
        '--pressure-run-id', default='v1',
        help='pressure_toolkit run 根：work://pressure_toolkit/<run_id>；同一 run 根'
             '内口径参数必须一致（launcher 的 artifact.json 会拒绝口径漂移）')
    args = parser.parse_args()
    run_id = str(args.pressure_run_id or '').strip().strip('/')
    if not run_id or run_id in {'.', '..'} or '/' in run_id:
        raise SystemExit(f'--pressure-run-id must be one path segment: {args.pressure_run_id!r}')
    args.pressure_run_id = run_id
    models = [x.strip() for x in args.models.split(',') if x.strip()]
    index = 0
    for model in models:
        if model in {'pressure_toolkit', 'fpp_train', 'fpp_infer', 'posetransopt'}:
            continue
        index += 1
        text = f'MODEL={model}\nRUN_NAME={model}\nCONDA_ENV=touch_gait\n'
        if model.startswith('anysole'): text += 'CONFIG_FILE=anysole/configs/v1.yaml\n'
        elif model == 'step2motion': text += 'CONFIG_FILE=configs/config_gait.json\n'
        put(f'{index:03d}_{model}', text)
    if not any(model in models for model in ('pressure_toolkit', 'fpp_train', 'fpp_infer', 'posetransopt')):
        return
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    all_rows = list(rows)
    if args.split != 'all':
        split_path = REPO / 'AnysoleWorkspace/protocol/splits/default/splits.csv'
        with split_path.open(encoding='utf-8-sig', newline='') as handle:
            selected = {value.strip() for item in __import__('csv').DictReader(handle)
                        for value in [item.get(args.split, '')] if value.strip()}
        rows = [row for row in rows if row.get('session_id') in selected]
    rows.sort(key=lambda row: row.get('session_id', ''))

    def date_from_row(row):
        return resolve_uri(row['video_path'], must_exist=True).parts[-3]

    if 'fpp_train' in models:
        index += 1
        put(f'{index:03d}_fpp_train',
            'MODEL=fpp_train\nRUN_NAME=fpp_train\nCONDA_ENV=mmvp\n'
            'CONFIG_FILE=Baselines/VP-MoCap/FPP-Net/configs/temporalKPSMPLCont_series5_mlp.yaml\n')
    if 'fpp_infer' in models:
        infer_phases = ('train', 'val', 'test') if args.split == 'all' else (args.split,)
        for phase in infer_phases:
            index += 1
            put(f'{index:03d}_fpp_infer_{phase}',
                f'MODEL=fpp_infer\nRUN_NAME=fpp_infer_{phase}\nCONDA_ENV=mmvp\n'
                f'FPP_PHASE={phase}\n'
                'CONFIG_FILE=Baselines/VP-MoCap/FPP-Net/configs/temporalKPSMPLCont_series5_mlp.yaml\n')

    if 'posetransopt' in models:
        for row in rows:
            index += 1
            date = date_from_row(row)
            session = row['session_id']
            base = f'model-input://PoseTransOpt/adapter_v1/{date}/{row["subject_id"]}/{session}'
            # 评估整改任务 02：staging（task.conf/日志）落在 results/logs，
            # results/baselines/VP-MoCap/ 只保留 predictions/ 与 metrics/。
            text = (f'MODEL=posetransopt\nRUN_NAME=posetransopt_{session}\nCONDA_ENV=mmvp\n'
                    f'INPUT_PATH_BASE={base}\nSCENE_RGBD={base}/template_scene_rgbd.npy\n')
            put(f'{index:05d}_posetransopt_{session}', text)

    if 'pressure_toolkit' not in models:
        return
    run_root = f'work://pressure_toolkit/{args.pressure_run_id}'
    # pressure 的取件范围是 adapter 树，不按 split 过滤：adapter 里有什么就拟合
    # 什么（splits.csv 是评估划分，不是拟合范围）。若按 --split test 过滤，train
    # 列的 adapter session（如 S10101）会被整体漏掉。adapter 目录不存在的
    # session 不出 conf：pre-flight 会再按 depth/depth_mask/CLIFF/地面判据细化
    # 跳过，但未接入的 session 连目录都没有，出 conf 只会得到一堆空跑任务。
    for row in sorted(all_rows, key=lambda item: item.get('session_id', '')):
        subject, session = row.get('subject_id'), row.get('session_id')
        if not subject or not session: continue
        date = date_from_row(row)
        adapter_session = (f'model-input://pressure_toolkit/v1/images/'
                           f'{date}/{subject}/{session}')
        if not resolve_uri(adapter_session).is_dir():
            continue
        if pressure_queued(session):
            continue
        index += 1
        # 口径（M1/M2/M3）：CPU 对应点、maxiters 101、画布 640x576；三者在
        # launcher 写入 run 根的 artifact.json 里作为参数记录，同一 run 根内
        # 口径漂移会被拒绝。
        text = (
            f'MODEL=pressure_toolkit\nRUN_NAME=pressure_{session}\nCONDA_ENV=mmvp\n'
            'CONFIG_FILE=Baselines/pressure_tookit/configs/fit_smpl_rgbd.yaml\n'
            f'DATASET={date}\nSUB_IDS={subject}\nSEQ_NAME={session}\n'
            f'MODEL_GENDER={"female" if subject in FEMALE_SUBJECTS else "male"}\n'
            'SPLIT=all\n'
            # 整段拟合：入口 init_pose，tracking 自动接续；START_IDX=auto =
            # 续跑起点取"第一个缺失帧"（launcher 运行时决定）。
            'FITTING_STAGE=init_pose\nSTART_IDX=auto\nEND_IDX=-1\n'
            f'RUN_ID={args.pressure_run_id}\n'
            f'OUTPUT_DIR={run_root}/fitting\n'
            f'INIT_DATA_DIR={run_root}\n'
            'PRESSURE_ICP_DEVICE=cpu\n'
            'PRESSURE_MAXITERS=101\n'
            'PRESSURE_CANVAS=640,576\n'
            # M4 裁定：每卡并发 4 个 fitting 进程。队列里真正的并发度由 worker
            # 数量决定，这个值同时用于推导 PRESSURE_KDTREE_WORKERS。
            'PRESSURE_PER_GPU=4\n'
            # A5：可视化默认关闭（每帧 OBJ 约 3MB，正式跑只留 npz 结果）；
            # 排查时把它改成 0 即可保留 meshes/gt_depths。
            'PRESSURE_NO_EXPORT_OBJ=1\n'
            'PRESSURE_EXPORT=1\n'
        )
        put(f'{index:05d}_pressure_{session}', text)

if __name__ == '__main__': main()
