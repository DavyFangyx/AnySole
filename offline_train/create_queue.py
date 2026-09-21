import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

def put(gpu, name, text):
    path = ROOT / 'queues' / f'GPU{gpu}' / 'pending' / f'{name}.conf'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): path.write_text(text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True)
    parser.add_argument('--gpu-list', required=True)
    parser.add_argument('--manifest', default='AnysoleWorkspace/manifests/session_manifest.jsonl')
    args = parser.parse_args()
    gpus = [x.strip() for x in args.gpu_list.split(',') if x.strip()]
    if not gpus: raise SystemExit('gpu-list is empty')
    models = [x.strip() for x in args.models.split(',') if x.strip()]
    index = 0
    for model in models:
        if model == 'pressure_toolkit': continue
        index += 1
        text = f'MODEL={model}\nRUN_NAME={model}\nCONDA_ENV=touch_gait\n'
        if model.startswith('anysole'): text += 'CONFIG_FILE=anysole/configs/v1.yaml\n'
        put(gpus[(index - 1) % len(gpus)], f'{index:03d}_{model}', text)
    if 'pressure_toolkit' not in models: return
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    for row in rows:
        subject, session = row.get('subject_id'), row.get('session_id')
        if not subject or not session: continue
        session_gpu = gpus[(index) % len(gpus)]
        for stage in ('init_shape', 'init_pose', 'tracking'):
            index += 1
            text = (f'MODEL=pressure_toolkit\nRUN_NAME=pressure_{session}_{stage}\nRUN_DIR=results/offline/pressure_toolkit/{session}\n'
                    f'INIT_DATA_DIR={REPO}/results/offline/pressure_toolkit/{session}\nCONDA_ENV=depthpro\n'
                    'CONFIG_FILE=Baselines/pressure_tookit/configs/fit_smpl_rgbd.yaml\n'
                    f'DATASET={row.get("date", "20230422")}\nSUB_IDS={subject}\nSEQ_NAME={session}\n'
                    f'FITTING_STAGE={stage}\nSTART_IDX=0\nEND_IDX=-1\n'
                    'BASDIR=workspace://derived/pressure_tookit\n'
                    'ESSENTIAL_ROOT=workspace://dependencies/pressure_tookit/essential\n')
            put(session_gpu, f'{index:05d}_pressure_{session}_{stage}', text)

if __name__ == '__main__': main()
