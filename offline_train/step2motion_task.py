import json, os, subprocess, sys
from pathlib import Path

def conf(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1); out[key] = value.strip().strip("'\"")
    return out

task = conf(sys.argv[1]); output = Path(sys.argv[2]); output.mkdir(parents=True, exist_ok=True)
base = Path(task.get('CONFIG_FILE', 'Baselines/Step2Motion/configs/config_gait.json'))
cfg = json.loads(base.read_text()); cfg['name'] = task.get('RUN_NAME', 'gait_model')
cfg['models_dir'] = str((output / 'checkpoints').resolve())
for key in ('epochs_pose', 'epochs_trans', 'batch_size'):
    if key.upper() in task: cfg[key] = int(task[key.upper()])
effective = output / 'config.json'; effective.write_text(json.dumps(cfg, indent=2))
os.chdir(Path(__file__).resolve().parents[1] / 'Baselines/Step2Motion')
subprocess.run([task.get('PYTHON_BIN', sys.executable), 'src/train.py', '--config', str(effective)], check=True)
