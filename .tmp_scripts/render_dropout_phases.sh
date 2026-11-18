#!/usr/bin/env bash
set -u
cd /data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
S=.tmp_scripts/e5_demo_teaser.py
JOBS=(
 "S6041 fS6S7"
 "S12042 fS5S11S12"
 "S5033 fS5S11S12"
 "S5083 fS5S11S12"
)
run_job(){ $PY $S --cond gap060 --sid "$1" --fold "$2" --no-text --gap-phases > "/tmp/drop_phase_$1.log" 2>&1; echo "DONE $1 exit $?"; }
pids=()
for j in "${JOBS[@]}"; do run_job $j & pids+=($!); if [ ${#pids[@]} -ge 4 ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi; done
wait
echo ALL DONE
