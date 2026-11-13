#!/bin/bash
# E6 supplementary: global_mpjpe + ankle_foot_mpjpe phase analyses (s10).
set -u
REPO=/data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
OUT="$REPO/results_display/Etest_crossfit_v8_s10/E6_ContactPhaseAnalysis"
mkdir -p "$OUT"
run_one() {
  metric="$1"; fold="$2"
  echo "[start] $metric $fold $(date '+%F %T')"
  E6_SPLIT_CSV="../crossfit_woS13/${fold}.csv" \
  E6_SPLIT_COLUMN="$fold" \
  E6_PRED_SUBDIR="overlap_s10/eval_motion" \
  "$PY" "$REPO/anysole_tta/e6_phase_analysis.py" \
    --models "V8_Cv5_${fold}_s1,V8_Cv7_${fold}_s1" \
    --metric "$metric" \
    --contact-method f6_soft_time_accel_state_smpl24_v1 \
    --out-dir "$OUT/$metric/fold_${fold}" \
    --no-figure --skip-missing-sessions \
    > "$OUT/${metric}_${fold}.log" 2>&1
  echo "[done] $metric $fold rc=$? $(date '+%F %T')"
}
(
  run_one global_mpjpe fS5S6S11
  run_one global_mpjpe fS5S11S12
) &
(
  run_one global_mpjpe fS6S7
  run_one global_mpjpe fS5S6S8
) &
wait
(
  run_one ankle_foot_mpjpe fS5S6S11
  run_one ankle_foot_mpjpe fS5S11S12
) &
(
  run_one ankle_foot_mpjpe fS6S7
  run_one ankle_foot_mpjpe fS5S6S8
) &
wait
echo "[e6 supp done] $(date '+%F %T')"
