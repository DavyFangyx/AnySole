#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && -f "$1" ]] || exit 2
cfg="$1"; root="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$root/.." && pwd)"
source "$cfg"; : "${MODEL:?}"; : "${RUN_NAME:?}"
cd "$repo"
run_dir="${RUN_DIR:-$repo/results/offline/$RUN_NAME}"
[[ "$run_dir" = /* ]] || run_dir="$repo/$run_dir"
mkdir -p "$run_dir"; cp "$cfg" "$run_dir/task.conf"; date -Is > "$run_dir/started_at"
export PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-disabled}" ANYSOLE_WORKSPACE="${ANYSOLE_WORKSPACE:-$repo/AnysoleWorkspace}" ANYSOLE_RESULTS="${ANYSOLE_RESULTS:-$repo/results}"
py="${PYTHON_BIN:-python}"
if [[ "${CONDA_ENV:-touch_gait}" != none && "${CONDA_ENV:-touch_gait}" != 无 ]]; then
  command -v conda >/dev/null || { echo 'conda is required' >&2; exit 2; }
  eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV:-touch_gait}"
fi
case "$MODEL" in
  anysole|anysole_insole_drift)
    modal="${MODAL:-anysolev1}"; [[ "$MODEL" == anysole_insole_drift ]] && modal=anysolev1_insole_drift
    "$py" -m anysole.train --config "${CONFIG_FILE:-anysole/configs/v1.yaml}" --modal "$modal" --device cuda --out-dir "$run_dir/checkpoints" --epochs "${EPOCHS:-200}" --batch-size "${BATCH_SIZE:-256}" ;;
  motionpro)
    cd Baselines/MotionPRO
    "$py" -m app.train_frappe task.gpu="$CUDA_VISIBLE_DEVICES" task.checkpoint_dir="$run_dir/checkpoints" task.result_dir="$run_dir/metrics" task.output_dir="$run_dir/debug" task.epochs="${EPOCHS:-1000}" task.batch_size="${BATCH_SIZE:-16}" wandb_mode=disabled ;;
  step2motion) "$py" "$root/step2motion_task.py" "$cfg" "$run_dir" ;;
  pressure_toolkit)
    cd Baselines/pressure_tookit
    "$py" main_singleview.py -c "$repo/$CONFIG_FILE" --dataset "$DATASET" --sub_ids "$SUB_IDS" --seq_name "$SEQ_NAME" --fitting_stage "$FITTING_STAGE" --start_idx "$START_IDX" --end_idx "$END_IDX" --output_dir "$run_dir" --basdir "$BASDIR" --essential_root "$ESSENTIAL_ROOT" --model_gender "${MODEL_GENDER:-male}" ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 2;;
esac
date -Is > "$run_dir/finished_at"
touch "$run_dir/.done"
