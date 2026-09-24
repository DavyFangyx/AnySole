#!/usr/bin/env bash
# scheduler.sh — 单一共享队列的串行 worker（对齐 SurvPGC configs/scheduler.sh）
#
# 每张卡可由 bg.sh 启动一个或多个 worker（CUDA_VISIBLE_DEVICES=N），所有 worker
# 从同一个 configs/queue/ 原子 mv 抢任务，互不重复；bg.sh 会把独立日志路径作为
# $2 传入（缺省仍为 scheduler_GPU{N}.log）。取件规则：
#   * 优先取非 display 任务（字典序最早者，seq 前缀 = 拓扑序）；
#   * display 任务只在 running 清空后才可取件，保证依赖产物齐全；
#   * 队列空且 running 非空时等待（可能有其他 worker 还在跑）。
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
queue="${1:-$root/queue}"
running_dir="$root/running"
done_dir="$root/done"
failed="$root/failed"
mkdir -p "$queue" "$running_dir" "$done_dir" "$failed"
shopt -s nullglob

while true; do
  cfg=""
  for f in "$queue"/*.conf; do
    name="$(basename "$f")"
    [[ "$name" == *__display.conf ]] && continue
    cfg="$f"
    break
  done
  if [[ -z "$cfg" ]]; then
    running_files=("$running_dir"/*.conf)
    displays=("$queue"/*__display.conf)
    if ((${#running_files[@]} == 0)) && ((${#displays[@]})); then
      cfg="${displays[0]}"
    fi
  fi
  if [[ -z "$cfg" ]]; then
    all_files=("$queue"/*.conf)
    running_files=("$running_dir"/*.conf)
    if ((${#all_files[@]} == 0)) && ((${#running_files[@]} == 0)); then
      break
    fi
    sleep 5
    continue
  fi

  name="$(basename "$cfg")"
  # 原子 claim：多 worker 竞争时只有一个 mv 成功，失败者重选。
  if ! mv "$cfg" "$running_dir/$name" 2>/dev/null; then
    continue
  fi
  gpu="${CUDA_VISIBLE_DEVICES:-shared}"
  log="${2:-$root/scheduler_GPU${gpu}.log}"
  echo "START $(date -Is) $name" >> "$log"
  if bash "$root/run.sh" "$running_dir/$name" >> "$log" 2>&1; then
    mv "$running_dir/$name" "$done_dir/$name"
    echo "[$(date -Is)] OK     $name"
  else
    mv "$running_dir/$name" "$failed/$name"
    echo "[$(date -Is)] FAIL   $name"
  fi
done
echo "FINISHED $(date -Is)"
