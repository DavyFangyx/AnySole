#!/bin/bash
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
cd /data/fangyuxuan/projects/gait
ckpt=$(cat results/cells/V7_Cv5ARSS2_s1/checkpoint_used.txt)
outdir="results/cells/V7_Cv5ARSS2_s1_valfix"
mkdir -p "$outdir/metrics"
CUDA_VISIBLE_DEVICES=7 $PY -m anysole.eval \
  --config anysole/configs/v1.yaml --ckpt "$ckpt" \
  --split val --device cuda --no-write-motion \
  --metrics-out "$outdir/metrics/val_brief.json" \
  --protocol-out "$outdir/metrics/val.json" >> "$outdir/val.log" 2>&1
code=$?
echo "[arss2-val] exit $code"
[ $code -eq 0 ] && $PY -c "
import json
d=json.load(open('$outdir/metrics/val_brief.json'))
for mode in ('T2M','V2M','VT2M'):
    m=d['metrics'][mode]
    print(mode, 'PA', round(m['PA-MPJPE'],1), '| MPJPE', round(m['MPJPE'],1), '| Drift', round(m['RootOrientationDrift'],1), '| Jitter', round(m['Jitter_pred'],0))
"
