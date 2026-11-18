#!/usr/bin/env bash
# Locomotion-only E5 teasers, no text captions/legend.
set -u
cd /data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
S=.tmp_scripts/e5_demo_teaser.py
JOBS=(
  "occ020 occ020 S7031 fS6S7"
  "occ020 occ020 S12042 fS5S11S12"
  "occ020 occ020 S5031 fS5S11S12"
  "occ020 occ020 S5083 fS5S11S12"
  "occ020 occ020 S7053 fS6S7"
  "occ090 occ090 S5031 fS5S11S12"
  "occ090 occ090 S5033 fS5S11S12"
  "occ090 occ090 S7031 fS6S7"
  "occ090 occ090 S7053 fS6S7"
  "occ090 occ090 S6041 fS5S6S11"
  "dropout gap060 S12042 fS5S11S12"
  "dropout gap060 S11051 fS5S11S12"
  "dropout gap060 S7053 fS6S7"
  "dropout gap060 S6041 fS6S7"
  "dropout gap060 S7031 fS6S7"
)
run_job () {
  local scene=$1 cond=$2 sid=$3 fold=$4
  local npz="results/diagnostics/20261116_e5_teaser_motion/V8_IMU_Residual/$cond/$fold/eval_motion/${sid}_VT2M.npz"
  if [ ! -s "$npz" ]; then echo "MISS $scene $sid ($npz)"; return 0; fi
  $PY $S --cond "$cond" --sid "$sid" --fold "$fold" --no-text > "/tmp/render_loco_${scene}_${sid}.log" 2>&1
  echo "DONE $scene $sid exit $?"
}
pids=()
for job in "${JOBS[@]}"; do
  run_job $job &
  pids+=($!)
  if [ ${#pids[@]} -ge 4 ]; then
    wait "${pids[0]}"; pids=("${pids[@]:1}")
  fi
done
wait
echo "ALL DONE"
