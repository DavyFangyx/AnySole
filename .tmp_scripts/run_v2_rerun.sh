#!/usr/bin/env bash
# v2 rerun: sequential per-condition updates + batch-mean val (pre-v3 recipe).
# Runs the v2 main (full router) and the v2 FT (enum fine-tune, no router)
# concurrently, 4 folds each, 20 epochs, best-val.
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
D=results/diagnostics/20261112_v8_gate_frozen
R=AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet
B=results/AnySole
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
folds=(fS5S6S11 fS5S11S12 fS6S7 fS5S6S8)
gpus=(2 3 6 7)
splits=(splits_test_S5S6S11_val_S12.csv splits_test_S5S11S12_val_S13.csv \
        splits_test_S6S7_val_S14.csv splits_test_S5S6S8_val_S7.csv)
PIDS=()

launch_arm () {
  local tag=$1 base_kind=$2 name_fmt=$3
  shift 3
  for i in 0 1 2 3; do
    local gpu=${gpus[$i]} fold=${folds[$i]} split=${splits[$i]}
    local base
    case $base_kind in
      enum) base=$B/V8_Cv7_PhysIMU_${fold}_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_last.pt ;;
      gate) base=$D/gate_4fold_${fold}.pt ;;
      *) echo "bad base kind $base_kind"; return 1 ;;
    esac
    local out=$D/${name_fmt/\{fold\}/$fold}.pt
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY $D/finetune_router_enum.py \
      --gate-ckpt "$base" "$@" \
      --split-csv AnysoleWorkspace/protocol/splits/default/$split \
      --clean-cache-root $R/cam3 --occ020-cache-root $R/cam3_occ_lower020 \
      --occ050-cache-root $R/cam3_occ_lower050 \
      --degraded-cache-root $R/cam3_occ_lower100 \
      --batch-size 64 --epochs 20 --lr 1e-4 --lambda-clean-rel 1.0 \
      --cond-mode sequential --val-mode batch --out "$out" \
      > /tmp/v2rerun_${tag}_${fold}.log 2>&1 &
    PIDS+=($!)
  done
}

echo "=== v2 full router ==="
launch_arm full gate 'router_ft_enum_cr_{fold}_l1.0_v2rerun'
echo "=== v2 FT (no router) ==="
launch_arm ft enum 'router_ft_only_{fold}_v2rerun' --no-router
for p in "${PIDS[@]}"; do wait $p; echo "pid $p exit $?"; done
echo V2RERUN_DONE
