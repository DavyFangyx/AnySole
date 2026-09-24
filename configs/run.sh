#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && -f "$1" ]] || exit 2
cfg="$1"; root="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$root/.." && pwd)"
# 任务 conf（带 TASK/EXPERIMENT_ID）继承 defaults.conf；baseline 型
# conf（只带 MODEL=）不受影响。
if grep -qE '^(TASK|EXPERIMENT_ID)=' "$cfg"; then
  source "$root/defaults.conf"
fi
source "$cfg"
# 任务 conf 只要求 TASK；baseline conf 必须带 MODEL。
[[ -n "${TASK:-}" ]] || : "${MODEL:?}"
: "${RUN_NAME:=task_${EXPERIMENT_ID:-unknown}}"
cd "$repo"
run_dir="${RUN_DIR:-$repo/results/logs/task_staging/$RUN_NAME}"
[[ "$run_dir" = /* ]] || run_dir="$repo/$run_dir"
mkdir -p "$run_dir"; cp "$cfg" "$run_dir/task.conf"; date -Is > "$run_dir/started_at"
export PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-disabled}" WANDB_DIR="${WANDB_DIR:-$repo/results/logs/wandb}" ANYSOLE_WORKSPACE="${ANYSOLE_WORKSPACE:-$repo/AnysoleWorkspace}" ANYSOLE_RESULTS="${ANYSOLE_RESULTS:-$repo/results}"
py="${PYTHON_BIN:-python}"
if [[ "${CONDA_ENV:-touch_gait}" != none && "${CONDA_ENV:-touch_gait}" != 无 ]]; then
  command -v conda >/dev/null || { echo 'conda is required' >&2; exit 2; }
  eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV:-touch_gait}"
fi
# 任务 conf（一个 conf = 一个模型任务 / 一次 display）→ runner task；
# DRY_RUN=1 只打印命令，供验证与排查。
if [[ -n "${TASK:-}" ]]; then
  dry_flag=""; [[ "${DRY_RUN:-0}" = 1 ]] && dry_flag="--dry-run"
  if "$py" configs/tools/runner.py task "$cfg" $dry_flag; then
    date -Is > "$run_dir/finished_at"; touch "$run_dir/.done"
    exit 0
  fi
  exit 1
fi
case "$MODEL" in
  anysole|anysole_insole_drift)
    "$py" -m anysole.train --config "${CONFIG_FILE:-anysole/configs/v1.yaml}" --device cuda --out-dir "$run_dir/checkpoints" --epochs "${EPOCHS:-200}" --batch-size "${BATCH_SIZE:-256}" ;;
  motionpro)
    cd Baselines/MotionPRO
    "$py" -m app.train_frappe task.gpu="$CUDA_VISIBLE_DEVICES" task.checkpoint_dir="$run_dir/checkpoints" task.result_dir="$run_dir/metrics" task.output_dir="$run_dir/debug" task.epochs="${EPOCHS:-1000}" task.batch_size="${BATCH_SIZE:-16}" wandb_mode=disabled ;;
  step2motion)
    cd Baselines/Step2Motion
    "$py" src/train.py --config "${CONFIG_FILE:-configs/config_gait.json}" ;;
  pressure_toolkit)
    cd Baselines/pressure_tookit
    "$py" main_singleview.py -c "$repo/$CONFIG_FILE" --dataset "$DATASET" --sub_ids "$SUB_IDS" --seq_name "$SEQ_NAME" --fitting_stage "$FITTING_STAGE" --start_idx "$START_IDX" --end_idx "$END_IDX" --output_dir "$run_dir" --basdir "$BASDIR" --essential_root "$ESSENTIAL_ROOT" --model_gender "${MODEL_GENDER:-male}" ;;
  fpp_train)
    cd Baselines/VP-MoCap/FPP-Net
    fpp_cfg="${CONFIG_FILE:-configs/temporalKPSMPLCont_series5_mlp.yaml}"; [[ "$fpp_cfg" = /* ]] || fpp_cfg="$repo/$fpp_cfg"
    "$py" app/train_temporal.py --config "$fpp_cfg" --batch_size "${BATCH_SIZE:-32}" --num_threads "${NUM_THREADS:-0}" --gpus "${CUDA_VISIBLE_DEVICES:-cpu}" ;;
  fpp_infer)
    cd Baselines/VP-MoCap/FPP-Net
    fpp_cfg="${CONFIG_FILE:-configs/temporalKPSMPLCont_series5_mlp.yaml}"; [[ "$fpp_cfg" = /* ]] || fpp_cfg="$repo/$fpp_cfg"
    "$py" app/infer_smplcont.py --config "$fpp_cfg" --phase "${FPP_PHASE:-test}" --batch_size "${BATCH_SIZE:-1}" --num_threads "${NUM_THREADS:-0}" --gpus "${CUDA_VISIBLE_DEVICES:-cpu}" ;;
  posetransopt)
    cd Baselines/VP-MoCap/PoseTransOpt
    "$py" -m app.optimize task.input_path_base="${INPUT_PATH_BASE:?}" task.scene_rgbd="${SCENE_RGBD:?}" gpu="${CUDA_VISIBLE_DEVICES:-0}" ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 2;;
esac
date -Is > "$run_dir/finished_at"
touch "$run_dir/.done"
