#!/usr/bin/env bash
# bg.sh — 每卡后台启动一个 scheduler worker
# 用法: CUDA_VISIBLE_DEVICES=N bash bg.sh
# 所有 worker 共享同一个 configs/queue/，通过原子 mv 抢任务；GPU 归属由
# 本进程的 CUDA_VISIBLE_DEVICES 决定。flock 防止同卡重复启动。
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
gpu="${CUDA_VISIBLE_DEVICES:-}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'set CUDA_VISIBLE_DEVICES=N' >&2; exit 2; }
root="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$root/locks"
exec 9>"$root/locks/GPU${gpu}.lock"; flock -n 9 || exit 1
log="$root/scheduler_GPU${gpu}.log"; nohup bash "$root/scheduler.sh" >"$log" 2>&1 < /dev/null &
echo "PID: $!"; echo "log: $log"
