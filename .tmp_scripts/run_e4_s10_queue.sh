#!/bin/bash
# E4 s10 eval queue: 4 models x 4 folds. E4 tables only need clean metrics,
# so skip motion export and robustness. Resumable: skips cells with test.json.
set -u
REPO=/data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python

run_one() {
  model="$1"; fold="$2"; gpu="$3"
  cell="${model}_${fold}_s1"
  outdir="$REPO/results/cells/$cell/metrics/overlap_s10"
  ckpt="$(cat "$REPO/results/cells/$cell/checkpoint_used.txt")"
  cfg="$REPO/anysole/configs/v1_fold_${fold#f}.yaml"
  mkdir -p "$outdir"
  if [ -s "$outdir/test.json" ]; then
    echo "[skip] $cell"
    return 0
  fi
  echo "[start] $cell gpu=$gpu $(date '+%F %T')"
  : > "$outdir/run.log"
  "$PY" -m anysole.eval --config "$cfg" --ckpt "$ckpt" --split test \
    --overlap-stride 10 --no-write-motion --no-robustness \
    --metrics-out "$outdir/test_brief.json" --protocol-out "$outdir/test.json" \
    --device "cuda:$gpu" >> "$outdir/run.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[fail] $cell rc=$rc $(date '+%F %T')"
    rm -f "$outdir/test.json"
  else
    echo "[done] $cell $(date '+%F %T')"
  fi
}

(
  run_one V8_Cv7 fS5S6S11 3
  run_one V8_Cv7 fS5S11S12 3
  run_one V8_Cv7 fS6S7 3
  run_one V8_Cv7 fS5S6S8 3
) &
(
  run_one V8_Cv4 fS5S6S11 3
  run_one V8_Cv4 fS5S11S12 3
  run_one V8_Cv4 fS6S7 3
  run_one V8_Cv4 fS5S6S8 3
) &
(
  run_one V8_Cv2 fS5S6S11 6
  run_one V8_Cv2 fS5S11S12 6
  run_one V8_Cv2 fS6S7 6
  run_one V8_Cv2 fS5S6S8 6
) &
(
  run_one V8_Cv6 fS5S6S11 6
  run_one V8_Cv6 fS5S11S12 6
  run_one V8_Cv6 fS6S7 6
  run_one V8_Cv6 fS5S6S8 6
) &

wait
echo "[queue done] $(date '+%F %T')"
