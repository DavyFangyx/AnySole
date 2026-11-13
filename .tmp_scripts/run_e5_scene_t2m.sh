#!/bin/bash
# E5 scene T-only: rerun 8 scene cells with all 3 modes (adds T2M), after queue.
set -u
REPO=/data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
run_one() {
  fold="$1"; suffix="$2"; gpu="$3"
  cell="V8_Cv5_${fold}_s1${suffix}"
  ckpt="$(cat "$REPO/results/cells/$cell/checkpoint_used.txt")"
  cfg="$(/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -c "import json;print(json.load(open('$REPO/results/cells/$cell/effective_config.json'))['config'])")"
  outdir="$REPO/results/cells/$cell/metrics/overlap_s10"
  echo "[start] $cell gpu=$gpu $(date '+%F %T')"
  "$PY" -m anysole.eval --config "$cfg" --ckpt "$ckpt" --split test \
    --overlap-stride 10 --no-write-motion --no-robustness \
    --metrics-out "$outdir/test_brief.json" --protocol-out "$outdir/test.json" \
    --device "cuda:$gpu" > "$outdir/run_scene_t2m.log" 2>&1
  echo "[done] $cell rc=$? $(date '+%F %T')"
}
(
  run_one fS5S6S11 _evalsceneclean 3
  run_one fS5S6S11 _evalscene 3
) &
(
  run_one fS5S11S12 _evalsceneclean 6
  run_one fS5S11S12 _evalscene 6
) &
(
  run_one fS6S7 _evalsceneclean 3
  run_one fS6S7 _evalscene 3
) &
(
  run_one fS5S6S8 _evalsceneclean 6
  run_one fS5S6S8 _evalscene 6
) &
wait
echo "[scene t2m done] $(date '+%F %T')"
