#!/bin/bash
# V8_Cv7 s10 predictions for E6: 4 folds, motion + session detail, all 3 modes.
set -u
REPO=/data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
run_one() {
  fold="$1"; gpu="$2"
  cell="V8_Cv7_${fold}_s1"
  ckpt="$(cat "$REPO/results/cells/$cell/checkpoint_used.txt")"
  cfg="$REPO/anysole/configs/v1_fold_${fold#f}.yaml"
  outdir="$REPO/results/cells/$cell/metrics/overlap_s10"
  mkdir -p "$outdir"
  echo "[start] $cell gpu=$gpu $(date '+%F %T')"
  "$PY" -m anysole.eval --config "$cfg" --ckpt "$ckpt" --split test \
    --overlap-stride 10 --dump-detail --no-robustness \
    --write-motion "$REPO/results/cells/$cell/predictions/overlap_s10/eval_motion" \
    --metrics-out "$outdir/test_brief.json" --protocol-out "$outdir/test.json" \
    --device "cuda:$gpu" > "$outdir/run.log" 2>&1
  echo "[done] $cell rc=$? $(date '+%F %T')"
}
(
  run_one fS5S6S11 2
  run_one fS5S11S12 2
) &
(
  run_one fS6S7 7
  run_one fS5S6S8 7
) &
wait
echo "[cv7 done] $(date '+%F %T')"
