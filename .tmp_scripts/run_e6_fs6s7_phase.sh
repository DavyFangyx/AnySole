#!/bin/bash
# E6 fS6S7 phase analyses with the corrected V8_Cv5 s10 predictions.
set -u
REPO=/data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
OUT="$REPO/results_display/Etest_crossfit_v8_s10/E6_ContactPhaseAnalysis"
run_one() {
  metric="$1"
  E6_SPLIT_CSV="../crossfit_woS13/fS6S7.csv" \
  E6_SPLIT_COLUMN="fS6S7" \
  E6_PRED_SUBDIR="overlap_s10/eval_motion" \
  "$PY" "$REPO/anysole/e6_phase_analysis.py" \
    --models "V8_Cv5_fS6S7_s1,V8_Cv7_fS6S7_s1" \
    --metric "$metric" \
    --contact-method f6_soft_time_accel_state_smpl24_v1 \
    --out-dir "$OUT/$metric/fold_fS6S7" \
    --no-figure --skip-missing-sessions \
    > "$OUT/${metric}_fS6S7.log" 2>&1
  echo "[done] $metric rc=$? $(date '+%F %T')"
}
run_one pa_mpjpe &
run_one global_mpjpe &
run_one ankle_foot_mpjpe &
wait
echo "[e6 fs6s7 phase done] $(date '+%F %T')"
