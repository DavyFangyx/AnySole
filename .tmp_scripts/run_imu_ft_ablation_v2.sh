#!/usr/bin/env bash
# V8_IMU_* FT training-strategy ablation, single-fold pilot (fS5S6S11).
# V2 protocol (2026-11-15): sequential per-condition update, batch-mean val.
# Three missing arms: no-enum FT, decoder-only FT, clean-only FT.
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
D=results/diagnostics/20261112_v8_gate_frozen
O=results/diagnostics/20261114_imu_ft_ablation_v2
R=AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet
B=results/AnySole
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
FOLD=fS5S6S11
SPLIT=splits_test_S5S6S11_val_S12.csv
ENUM=$B/V8_Cv7_PhysIMU_${FOLD}_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_last.pt
NOENUM=$B/V8_Cv7_PhysIMU_NoEnum_${FOLD}_s1_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_ep740_bs256_sd1/checkpoints/ckpt_last.pt
mkdir -p "$O"

run_one () {
  local name=$1 gpu=$2 base=$3
  shift 3
  local out=$O/${name}_${FOLD}.pt
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY $D/finetune_router_enum.py \
    --gate-ckpt "$base" --no-router "$@" \
    --split-csv AnysoleWorkspace/protocol/splits/default/$SPLIT \
    --clean-cache-root $R/cam3 --occ020-cache-root $R/cam3_occ_lower020 \
    --occ050-cache-root $R/cam3_occ_lower050 \
    --degraded-cache-root $R/cam3_occ_lower100 \
    --batch-size 64 --epochs 20 --lr 1e-4 --lambda-clean-rel 1.0 \
    --cond-mode sequential --val-mode batch --recipe-name "$name" --out "$out" \
    > /tmp/v2_imu_ft_${name}_${FOLD}.log 2>&1 &
  echo "$name pid $! -> $out"
}

run_one V8_IMU_Residual_woEnum        2 "$NOENUM"
run_one V8_IMU_Residual_DecoderOnly   3 "$ENUM" --decoder-only
run_one V8_IMU_Residual_CleanOnly     6 "$ENUM" --conditions clean
wait
echo IMU_FT_ABLATION_DONE
