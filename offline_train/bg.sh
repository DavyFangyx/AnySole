#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
gpu="${CUDA_VISIBLE_DEVICES:-}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'set CUDA_VISIBLE_DEVICES=N' >&2; exit 2; }
root="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$root/logs" "$root/locks"
exec 9>"$root/locks/GPU${gpu}.lock"; flock -n 9 || exit 1
log="$root/logs/GPU${gpu}_scheduler.log"; nohup bash "$root/scheduler.sh" >"$log" 2>&1 < /dev/null &
echo "PID: $!"; echo "log: $log"
