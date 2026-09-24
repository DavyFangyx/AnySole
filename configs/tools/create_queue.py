#!/usr/bin/env python3
"""Generate baseline task confs into the shared queue (configs/queue/).

GPU 归属不再在这里指定：每个 worker（CUDA_VISIBLE_DEVICES=N bash
configs/bg.sh）从共享队列抢任务。注意 pressure_toolkit 每个 session 的
init_shape -> init_pose -> tracking 三阶段依赖顺序，共享队列下请只开一个
worker（或分批入队），否则阶段可能被不同 worker 并行取走。
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT.parent
REPO = CONFIG_DIR.parent

def put(name, text):
    path = CONFIG_DIR / 'queue' / f'{name}.conf'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): path.write_text(text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True)
    parser.add_argument('--manifest', default='AnysoleWorkspace/manifests/session_manifest.jsonl')
    parser.add_argument('--split', choices=('train', 'val', 'test', 'all'), default='all')
    args = parser.parse_args()
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
    if args.split != 'all':
        rows = [row for row in rows if args.split in str(row.get('split_iid', '')).split(',')]
    rows.sort(key=lambda row: row.get('session_id', ''))

    def date_from_row(row):
        parts = Path(row['pressure_path']).parts
        return parts[parts.index('cam3') + 1]

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
            base = f'workspace://derived/VP-MoCap/{date}/{row["subject_id"]}/{session}'
            text = (f'MODEL=posetransopt\nRUN_NAME=posetransopt_{session}\nCONDA_ENV=mmvp\n'
                    f'RUN_DIR=results/baselines/VP-MoCap/posetransopt/{session}\n'
                    f'INPUT_PATH_BASE={base}\nSCENE_RGBD={base}/template_scene_rgbd.npy\n')
            put(f'{index:05d}_posetransopt_{session}', text)

    if 'pressure_toolkit' not in models:
        return
    for row in rows:
        subject, session = row.get('subject_id'), row.get('session_id')
        if not subject or not session: continue
        for stage in ('init_shape', 'init_pose', 'tracking'):
            index += 1
            text = (f'MODEL=pressure_toolkit\nRUN_NAME=pressure_{session}_{stage}\nRUN_DIR=results/baselines/pressure_toolkit/{session}\n'
                    f'INIT_DATA_DIR={REPO}/results/baselines/pressure_toolkit/{session}\nCONDA_ENV=mmvp\n'
                    'CONFIG_FILE=Baselines/pressure_tookit/configs/fit_smpl_rgbd.yaml\n'
                    f'DATASET={date_from_row(row)}\nSUB_IDS={subject}\nSEQ_NAME={session}\n'
                    f'FITTING_STAGE={stage}\nSTART_IDX=0\nEND_IDX=-1\n'
                    'BASDIR=workspace://derived/pressure_tookit\n'
                    'ESSENTIAL_ROOT=workspace://dependencies/pressure_tookit/essential\n')
            put(f'{index:05d}_pressure_{session}_{stage}', text)

if __name__ == '__main__': main()
