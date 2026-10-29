#!/usr/bin/env bash
set -eo pipefail

workspace=/private/Tactile
result_root="$workspace/3_Result/SAM3DB_baseline"

if [ "$#" -lt 1 ]; then
  echo "usage: launch.sh bounded [run args...] | full user-approved [run args...]" >&2
  exit 2
fi

scope=$1
shift
full_args=()
if [ "$scope" = "full" ]; then
  if [ "${1:-}" != "user-approved" ]; then
    echo "full launch requires literal user-approved token" >&2
    exit 2
  fi
  shift
  full_args=(--full --authorization user-approved)
elif [ "$scope" != "bounded" ]; then
  echo "scope must be bounded or full" >&2
  exit 2
fi

# Conda activation is intentionally before nounset; this avoids the known
# activation failure that was fixed in the WHAM launcher.
source /opt/conda/bin/activate /opt/conda/envs/Tactile
set -u
export PYTHONPATH="$workspace/2_Code${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$result_root/.run/logs"

setsid nohup python -m sam3db_baseline.cli run \
  --result-root "$result_root" "${full_args[@]}" "$@" \
  >> "$result_root/.run/coordinator.log" 2>&1 < /dev/null &
pid=$!
echo "launched pid/pgid $pid"
