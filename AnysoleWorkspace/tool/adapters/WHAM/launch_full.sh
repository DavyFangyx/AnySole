#!/usr/bin/env bash
# Conda's activation hooks probe optional variables and are not nounset-safe.
set -eo pipefail
# Prepared only; execute after explicit user approval of results and videos.
if [[ "${1:-}" != "user-approved" ]]; then
  echo 'Full inference awaits explicit user approval. Then use: bash launch_full.sh user-approved' >&2
  exit 2
fi
source /opt/conda/bin/activate /opt/conda/envs/Tactile
set -u
cd /private/Tactile
export PYTHONPATH=/private/Tactile/2_Code
WHAM_RESULT_ROOT=/private/Tactile/3_Result/WHAM_baseline
WHAM_CERTIFICATE=/private/Tactile/.ai/runtime/tasks/active/TASK-20260916-1811-wham-baseline-reproduction/artifacts/compact.json
# Rebuild only control inputs after an explicitly authorized clean start.
# prepare refuses changed assets/corpus or mixed legacy output directories.
python -m wham_baseline.prepare --result-root "$WHAM_RESULT_ROOT" --certificate "$WHAM_CERTIFICATE" > /dev/null
WHAM_VIZ_WORKERS=$(python - <<'PY'
import json
from pathlib import Path
import shutil
v=json.loads(Path('/private/Tactile/3_Result/WHAM_baseline/.run/ready.json').read_text())
if v['status']!='passed':
    raise SystemExit('Clean-start preparation has not passed')
if shutil.disk_usage('/private/Tactile/3_Result/WHAM_baseline').free<v['required_free_bytes']:
    raise SystemExit('Insufficient free disk capacity for the validated budget')
print(v['viz_workers'])
PY
)
python -m wham_baseline.pipeline launch \
  --dataset-root /private/Tactile/1_Dataset \
  --manifest /private/Tactile/3_Result/WHAM_baseline/.run/full.jsonl \
  --result-root /private/Tactile/3_Result/WHAM_baseline \
  --wham-root /private/Tactile/third_party/WHAM \
  --wham-python /opt/conda/envs/WHAM/bin/python \
  --ffmpeg-bin /opt/conda/envs/Tactile/bin/ffmpeg \
  --annotation-index /private/Tactile/3_Result/WHAM_baseline/.run/roi.sqlite \
  --gpus 0,1,2,3,4,5,6 --workers-per-gpu 2 \
  --viz-workers "$WHAM_VIZ_WORKERS" --viz-queue-limit 64 \
  --use-target-bbox --resume --wait-for-gpu --compact-output --approved-full-run
