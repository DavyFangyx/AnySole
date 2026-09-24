#!/usr/bin/env bash
# bg.sh — 后台启动 scheduler worker；同一 GPU 可重复执行以追加 worker
# 用法: CUDA_VISIBLE_DEVICES=N bash bg.sh
# 所有 worker 共享同一个 configs/queue/，通过原子 mv 抢任务；GPU 归属由
# 本进程的 CUDA_VISIBLE_DEVICES 决定。重复执行会给同一张卡追加一个 worker，
# 各自持有独立日志 GPU{N}_worker{M}.log。
set -euo pipefail

# 每张卡可能有多个训练进程，绑定原生数学线程池，避免 CPU 超订。
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

[[ $# -eq 0 ]] || exit 2
gpu="${CUDA_VISIBLE_DEVICES:-}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'set CUDA_VISIBLE_DEVICES=N' >&2; exit 2; }
root="$(cd "$(dirname "$0")" && pwd)"

# 扫描既有 worker 日志，取下一个空闲编号
shopt -s nullglob
next_worker=1
for f in "$root"/GPU"${gpu}"_worker*.log; do
  name="$(basename "$f")"
  if [[ "$name" =~ ^GPU${gpu}_worker_?([0-9]+)\.log$ ]]; then
    id="${BASH_REMATCH[1]}"
    if [ "$id" -ge "$next_worker" ]; then
      next_worker=$((id + 1))
    fi
  fi
done

log="$root/GPU${gpu}_worker${next_worker}.log"
nohup bash "$root/scheduler.sh" "$root/queue" "$log" >>"$log" 2>&1 < /dev/null &
echo "PID: $!"; echo "log: $log"; echo "停止: kill $!"
