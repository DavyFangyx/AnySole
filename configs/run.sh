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
# pressure_toolkit：权威落点是 run 根（work://pressure_toolkit/<run_id>），
# 逐帧结果、run artifact（artifact.json）与逐 session 日志都在那里；任务级
# staging（task.conf/started_at/.done）也收敛到 run 根的 logs/，避免同一任务
# 的日志分处两地。其他模型维持统一的 results/logs/task_staging。
if [[ "${MODEL:-}" = pressure_toolkit && "${INIT_DATA_DIR:-}" = work://* ]]; then
  # INIT_DATA_DIR 即 run 根（work://pressure_toolkit/<run_id>）；$repo/AnysoleWorkspace
  # 与解析器里的 WORKSPACE 常量一致（resolver 不读环境变量）。
  run_dir="${RUN_DIR:-$repo/AnysoleWorkspace/work/${INIT_DATA_DIR#work://}/logs/task_staging/$RUN_NAME}"
fi
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
    # M 系列整改：一个 conf = 一个 session 的整段拟合（入口 init_pose，之后自动
    # 接续 tracking），入口统一走 Baselines/pressure_tookit/run_full_mmvp.py。
    # 它负责 depth pre-flight（缺 depth/depth_mask/CLIFF/地面的 session 记录后
    # 跳过，不崩）、per-subject init_shape（flock 下算一次复用）、skip-existing
    # 续跑（START_IDX=auto → 第一个缺失帧）、run artifact 与预测导出；
    # fitting 本身仍然只由 main_singleview.py 执行，口径不受调度影响。
    # run-root 不变式（launcher 启动即校验，conf 里显式声明）：
    #   run_root      = work://pressure_toolkit/$RUN_ID   (INIT_DATA_DIR)
    #   output_dir    = <run_root>/fitting                (OUTPUT_DIR)
    # 正式口径固定为 M1/M2/M3：CPU 对应点、画布 640x576、maxiters 101。
    # launcher 把 toolkit 配置固定为常量 configs/fit_smpl_rgbd.yaml，conf 若
    # 指向别的配置会被忽略（口径只允许一条）。
    if [[ -n "${CONFIG_FILE:-}" && "${CONFIG_FILE##*/}" != fit_smpl_rgbd.yaml ]]; then
      echo "WARNING: pressure CONFIG_FILE=$CONFIG_FILE ignored; run_full_mmvp.py uses configs/fit_smpl_rgbd.yaml" >&2
    fi
    per_gpu="${PRESSURE_PER_GPU:-4}"   # M4 裁定：每卡并发 4 个 fitting 进程
    export PRESSURE_ICP_DEVICE="${PRESSURE_ICP_DEVICE:-cpu}"
    # PRESSURE_KDTREE_WORKERS 与并发度挂钩：每个 fitting 进程默认会按
    # os.cpu_count() 起 cKDTree 线程（外加 BLAS 线程），per_gpu 个进程会把
    # 96 核打满。未显式给定时按 cores/per_gpu 分，下限 2。
    if [[ -z "${PRESSURE_KDTREE_WORKERS:-}" ]]; then
      cores="$(nproc 2>/dev/null || echo 4)"
      PRESSURE_KDTREE_WORKERS=$(( cores / per_gpu ))
      if (( PRESSURE_KDTREE_WORKERS < 2 )); then PRESSURE_KDTREE_WORKERS=2; fi
      export PRESSURE_KDTREE_WORKERS
    fi
    pressure_args=( --run-root "${INIT_DATA_DIR:?pressure conf must set INIT_DATA_DIR=<run_root>}"
                    --output-dir "${OUTPUT_DIR:?pressure conf must set OUTPUT_DIR=<run_root>/fitting}"
                    --subject "${SUB_IDS:?}" --gender "${MODEL_GENDER:-male}"
                    --dataset "${DATASET:-}" --split "${SPLIT:-all}"
                    --sessions "${SEQ_NAME:?}"
                    --start-idx "${START_IDX:-auto}" --end-idx "${END_IDX:--1}"
                    --canvas "${PRESSURE_CANVAS:-640,576}"
                    --maxiters "${PRESSURE_MAXITERS:-101}"
                    --per-gpu "$per_gpu"
                    --icp-device "${PRESSURE_ICP_DEVICE}" )
    # GPU 由 worker 的 CUDA_VISIBLE_DEVICES 继承（scheduler.sh → run.sh → 子进程）
    # FITTING_STAGE=init_shape 表示只补 per-subject 形状文件后退出；其余值都表示
    # 整段拟合，由 launcher 自行决定 init_pose/tracking 与续跑起点。
    if [[ "${FITTING_STAGE:-init_pose}" = init_shape ]]; then
      pressure_args+=( --stage init_shape )
    fi
    # A5：可视化默认关闭（每帧 OBJ ~3MB）；PRESSURE_NO_EXPORT_OBJ=0 时保留。
    if [[ "${PRESSURE_NO_EXPORT_OBJ:-1}" = 1 ]]; then
      pressure_args+=( --no-export-obj )
    fi
    if [[ "${PRESSURE_EXPORT:-0}" = 1 ]]; then
      pressure_args+=( --export )
    fi
    # 与任务 conf 一致：DRY_RUN=1 交给 launcher 打印计划与将执行的命令，不拟合。
    if [[ "${DRY_RUN:-0}" = 1 ]]; then
      pressure_args+=( --dry-run )
    fi
    "$py" "$repo/Baselines/pressure_tookit/run_full_mmvp.py" "${pressure_args[@]}" ;;
  fpp_train)
    cd Baselines/VP-MoCap/FPP-Net
    fpp_cfg="${CONFIG_FILE:-configs/temporalKPSMPLCont_series5_mlp.yaml}"; [[ "$fpp_cfg" = /* ]] || fpp_cfg="$repo/$fpp_cfg"
    "$py" app/train_temporal.py --config "$fpp_cfg" --batch_size "${BATCH_SIZE:-32}" --num_threads "${NUM_THREADS:-0}" --gpus "${CUDA_VISIBLE_DEVICES:-cpu}" ;;
  fpp_infer)
    cd Baselines/VP-MoCap/FPP-Net
    fpp_cfg="${CONFIG_FILE:-configs/temporalKPSMPLCont_series5_mlp.yaml}"; [[ "$fpp_cfg" = /* ]] || fpp_cfg="$repo/$fpp_cfg"
    "$py" app/infer_smplcont.py --config "$fpp_cfg" --phase "${FPP_PHASE:-test}" --batch_size "${BATCH_SIZE:-32}" --num_threads "${NUM_THREADS:-4}" --gpus "${CUDA_VISIBLE_DEVICES:-cpu}" ;;
  posetransopt)
    cd Baselines/VP-MoCap/PoseTransOpt
    "$py" -m app.optimize task.input_path_base="${INPUT_PATH_BASE:?}" task.scene_rgbd="${SCENE_RGBD:?}" gpu="${CUDA_VISIBLE_DEVICES:-0}" ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 2;;
esac
date -Is > "$run_dir/finished_at"
touch "$run_dir/.done"
