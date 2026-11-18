#!/usr/bin/env bash
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
D=results/diagnostics/20261112_v8_gate_frozen
OUT=results/diagnostics/20261116_e5_teaser_motion/V8_IMU_Residual
run_one () {
  local cond=$1 fold=$2 gpu=$3
  local stem=${fold#f}
  local out=$OUT/$cond/$fold
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY anysole/eval.py \
    --config "anysole/configs/v1_${cond}_fold_${stem}.yaml" \
    --ckpt "$D/router_ft_only_${fold}_v2rerun.pt" --split test \
    --metrics-out "$out/brief.json" --protocol-out "$out/protocol.json" \
    --write-motion "$out/eval_motion" --no-robustness --config-id V2M,VT2M \
    > "/tmp/e5_teaser_${cond}_${fold}.log" 2>&1
  echo "DONE $cond $fold exit $?"
}
run_one occ020 fS6S7 2 &
run_one occ090 fS5S6S11 3 &
wait
echo ALL DONE
