#!/usr/bin/env bash
set -u
cd /data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
S=.tmp_scripts/e5_demo_teaser.py
JOBS=(
 "occ020 S11053 fS5S11S12"
 "occ090 S11051 fS5S11S12"
 "occ090 S11053 fS5S11S12"
)
for j in "${JOBS[@]}"; do
  set -- $j
  $PY $S --cond "$1" --sid "$2" --fold "$3" --no-text --world-phys --tag pending > "/tmp/pending_$1_$2.log" 2>&1
  echo "DONE $1 $2 exit $?"
done
echo ALL DONE
