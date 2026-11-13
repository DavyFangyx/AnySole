#!/bin/bash
# E6 phase analysis for s10: Cv5 + Cv7, per fold, pa_mpjpe (+ pen metrics).
set -u
REPO=/data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
OUT="$REPO/results_display/Etest_crossfit_v8_s10/E6_ContactPhaseAnalysis"
mkdir -p "$OUT"
run_fold() {
  fold="$1"
  echo "[start] $fold $(date '+%F %T')"
  E6_SPLIT_CSV="../crossfit_woS13/${fold}.csv" \
  E6_SPLIT_COLUMN="$fold" \
  E6_PRED_SUBDIR="overlap_s10/eval_motion" \
  "$PY" "$REPO/anysole_tta/e6_phase_analysis.py" \
    --models "V8_Cv5_${fold}_s1,V8_Cv7_${fold}_s1" \
    --metric pa_mpjpe \
    --contact-method f6_soft_time_accel_state_smpl24_v1 \
    --out-dir "$OUT/pa_mpjpe/fold_${fold}" \
    --no-figure > "$OUT/phase_${fold}.log" 2>&1
  echo "[done] $fold rc=$? $(date '+%F %T')"
}
run_fold fS5S6S11 &
run_fold fS5S11S12 &
run_fold fS6S7 &
run_fold fS5S6S8 &
wait
echo "[e6 phase done] $(date '+%F %T')"
