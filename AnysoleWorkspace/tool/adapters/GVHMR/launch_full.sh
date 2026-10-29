#!/usr/bin/env bash
set -eo pipefail

workspace=/private/Tactile
result_root="$workspace/3_Result/GVHMR_baseline"
if [ "${1:-}" != "user-approved" ] || [ "$#" -ne 1 ]; then
  echo "usage: launch_full.sh user-approved" >&2
  exit 2
fi

source /opt/conda/bin/activate /opt/conda/envs/Tactile
set -u
export PYTHONPATH="$workspace/2_Code${PYTHONPATH:+:$PYTHONPATH}"
if [ ! -f "$result_root/.run/manifest.json" ]; then
  echo "full GVHMR manifest is missing" >&2
  exit 2
fi
if [ -f "$result_root/.run/supervisor.json" ]; then
  old_pid=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid", 0))' "$result_root/.run/supervisor.json")
  if [ "$old_pid" -gt 0 ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "GVHMR supervisor is already alive: $old_pid" >&2
    exit 2
  fi
fi
mkdir -p "$result_root/.run/logs"
setsid nohup python -m gvhmr_baseline.supervisor --output-root "$result_root" >> "$result_root/.run/supervisor.log" 2>&1 < /dev/null &
pid=$!
echo "launched GVHMR supervisor pid/pgid $pid"
