#!/usr/bin/env bash
# Export remaining folds needed for multi-session E5 teasers.
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
D=results/diagnostics/20261112_v8_gate_frozen
OUT=results/diagnostics/20261116_e5_teaser_motion/V8_IMU_Residual

run_one () {
  local cond=$1 fold=$2 gpu=$3
  local stem=${fold#f}
  local out=$OUT/$cond/$fold
  local cfg="anysole/configs/v1_${cond}_fold_${stem}.yaml"
  case $cond in gap*) cfg="anysole/configs/v1_gap_r${cond#gap}_fold_${stem}.yaml" ;; esac
  mkdir -p "$out"
  if [ -s "$out/eval_motion/.done" ]; then echo "SKIP $cond $fold"; return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY anysole/eval.py \
    --config "$cfg" --ckpt "$D/router_ft_only_${fold}_v2rerun.pt" \
    --split test --metrics-out "$out/brief.json" --protocol-out "$out/protocol.json" \
    --write-motion "$out/eval_motion" --no-robustness --config-id V2M,VT2M \
    > "/tmp/e5_teaser_${cond}_${fold}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && touch "$out/eval_motion/.done"
  echo "DONE $cond $fold exit $rc"
}

run_one occ020 fS5S6S8 2 &
run_one occ090 fS5S11S12 3 &
run_one gap060 fS5S11S12 6 &
wait
echo "ALL DONE"
