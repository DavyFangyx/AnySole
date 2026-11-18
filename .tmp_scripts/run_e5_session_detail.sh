#!/usr/bin/env bash
# E5 demo screening: per-session detail for V8_IMU_Residual @ occ020 / occ090.
# Two waves x 4 GPUs (2/3/6/7). Output kept separate from formal eval.
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
D=results/diagnostics/20261112_v8_gate_frozen
OUT=results/diagnostics/20261116_e5_session_detail
folds=(fS5S6S11 fS5S11S12 fS6S7 fS5S6S8)
gpus=(2 3 6 7)

run_one () {
  local cond=$1 fold=$2 gpu=$3
  local stem=${fold#f}
  local out=$OUT/V8_IMU_Residual/$cond/$fold
  mkdir -p "$out"
  if [ -s "$out/session_detail_test.json" ]; then
    echo "SKIP $cond $fold"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY anysole/eval.py \
    --config "anysole/configs/v1_${cond}_fold_${stem}.yaml" \
    --ckpt "$D/router_ft_only_${fold}_v2rerun.pt" \
    --split test \
    --metrics-out "$out/brief.json" \
    --protocol-out "$out/protocol.json" \
    --no-write-motion --no-robustness --config-id V2M,VT2M --dump-detail \
    > "/tmp/e5_session_${cond}_${fold}.log" 2>&1
  echo "DONE $cond $fold exit $?"
}

for cond in occ020 occ090; do
  for i in "${!folds[@]}"; do
    run_one "$cond" "${folds[$i]}" "${gpus[$i]}" &
  done
  wait
done
echo "ALL DONE"
