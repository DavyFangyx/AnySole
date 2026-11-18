#!/usr/bin/env bash
# Evaluate the three V8_IMU_* single-fold FT ablation pilots (fS5S6S11).
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
D=results/diagnostics/20261112_v8_gate_frozen
O=results/diagnostics/20261114_imu_ft_ablation
R=AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
FOLD=fS5S6S11
SPLIT=splits_test_S5S6S11_val_S12.csv

run_eval () {
  local name=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY $D/perf_table_router.py \
    --no-router --gate-ckpt "$O/${name}_${FOLD}.pt" \
    --split-csv AnysoleWorkspace/protocol/splits/default/$SPLIT \
    --clean-cache-root $R/cam3 --occ020-cache-root $R/cam3_occ_lower020 \
    --occ050-cache-root $R/cam3_occ_lower050 \
    --degraded-cache-root $R/cam3_occ_lower100 \
    --batch-size 64 > /tmp/v3_imu_ft_eval_${name}_${FOLD}.log 2>&1 &
  echo "$name pid $!"
}

run_eval V8_IMU_Residual_woEnum        2
run_eval V8_IMU_Residual_DecoderOnly   3
run_eval V8_IMU_Residual_CleanOnly     6
wait
echo IMU_FT_EVAL_DONE
