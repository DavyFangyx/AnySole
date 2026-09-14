#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"; gpu="${CUDA_VISIBLE_DEVICES:-}"
queue="$root/queues/GPU${gpu}"; mkdir -p "$queue"/{pending,running,done,failed,logs}; shopt -s nullglob
while true; do
  files=("$queue/pending"/*.conf); ((${#files[@]})) || break
  cfg="${files[0]}"; name="$(basename "$cfg")"; mv "$cfg" "$queue/running/$name" || continue
  log="$queue/logs/${name%.conf}.log"; echo "START $(date -Is) $name" >> "$log"
  if bash "$root/run.sh" "$queue/running/$name" >> "$log" 2>&1; then mv "$queue/running/$name" "$queue/done/$name"; else mv "$queue/running/$name" "$queue/failed/$name"; fi
done
echo "FINISHED $(date -Is)" 
