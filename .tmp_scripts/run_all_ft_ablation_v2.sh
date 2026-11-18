#!/usr/bin/env bash
# V2 four-fold training for the remaining V8_IMU arms.
# 2 jobs per GPU on 2/3/6/7. Wave 1: woEnum + DecoderOnly; wave 2: CleanOnly + Router-only.
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
D=results/diagnostics/20261112_v8_gate_frozen
O=results/diagnostics/20261114_imu_ft_ablation_v2
RO=results/diagnostics/20261114_router_ablation_v2
R=AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet
B=results/AnySole
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
folds=(fS5S6S11 fS5S11S12 fS6S7 fS5S6S8)
gpus=(2 3 6 7)
splits=(splits_test_S5S6S11_val_S12.csv splits_test_S5S11S12_val_S13.csv \
        splits_test_S6S7_val_S14.csv splits_test_S5S6S8_val_S7.csv)
mkdir -p "$O" "$RO"

enum_base () {
  local fold=$1
  echo "$B/V8_Cv7_PhysIMU_${fold}_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_last.pt"
}
noenum_base () {
  local fold=$1
  echo "$B/V8_Cv7_PhysIMU_NoEnum_${fold}_s1_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_ep740_bs256_sd1/checkpoints/ckpt_last.pt"
}

run_one () {
  local arm=$1 fold=$2 gpu=$3 extra=$4 base=$5 out=$6
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY $D/finetune_router_enum.py \
    --gate-ckpt "$base" $extra \
    --split-csv AnysoleWorkspace/protocol/splits/default/${splits[$fold]} \
    --clean-cache-root $R/cam3 --occ020-cache-root $R/cam3_occ_lower020 \
    --occ050-cache-root $R/cam3_occ_lower050 \
    --degraded-cache-root $R/cam3_occ_lower100 \
    --batch-size 64 --epochs 20 --lr 1e-4 --lambda-clean-rel 1.0 \
    --cond-mode sequential --val-mode batch \
    --recipe-name "$arm" --out "$out" \
    > /tmp/v2_all_${arm}_${folds[$fold]}.log 2>&1 &
  echo "$arm ${folds[$fold]} gpu$gpu pid $!"
}

wave1 () {
  local -a pids=()
  for i in 0 1 2 3; do
    local gpu=${gpus[$i]} fold=${folds[$i]}
    run_one V8_IMU_Residual_woEnum $i $gpu "--no-router" "$(noenum_base $fold)" \
      "$O/V8_IMU_Residual_woEnum_${fold}.pt"
    run_one V8_IMU_Residual_DecoderOnly $i $gpu "--no-router --decoder-only" "$(enum_base $fold)" \
      "$O/V8_IMU_Residual_DecoderOnly_${fold}.pt"
  done
  wait
  echo WAVE1_DONE
}

wave2 () {
  for i in 0 1 2 3; do
    local gpu=${gpus[$i]} fold=${folds[$i]}
    run_one V8_IMU_Residual_CleanOnly $i $gpu "--no-router --conditions clean" "$(enum_base $fold)" \
      "$O/V8_IMU_Residual_CleanOnly_${fold}.pt"
    run_one V8_IMU_Router $i $gpu "--freeze-backbone" "$(enum_base $fold)" \
      "$RO/V8_IMU_Router_${fold}.pt"
  done
  wait
  echo WAVE2_DONE
}

wave1
wave2
echo ALL_V2_TRAINING_DONE
