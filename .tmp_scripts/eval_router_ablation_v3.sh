#!/usr/bin/env bash
# Evaluate v3 ablation arms: base-only references + FT/Frozen/NoEnum/Full.
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
D=results/diagnostics/20261112_v8_gate_frozen
R=AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet
B=results/AnySole
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
SUF=${SUF:-v3}
folds=(fS5S6S11 fS5S11S12 fS6S7 fS5S6S8)
gpus=(2 3 6 7)
splits=(splits_test_S5S6S11_val_S12.csv splits_test_S5S11S12_val_S13.csv \
        splits_test_S6S7_val_S14.csv splits_test_S5S6S8_val_S7.csv)

run_eval () {
  local tag=$1 base_kind=$2 router_tag=$3
  local -a pids=()
  for i in 0 1 2 3; do
    local gpu=${gpus[$i]} fold=${folds[$i]} split=${splits[$i]}
    local base
    case $base_kind in
      enum)   base=$B/V8_Cv7_PhysIMU_${fold}_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_last.pt ;;
      noenum) base=$B/V8_Cv7_PhysIMU_NoEnum_${fold}_s1_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_ep740_bs256_sd1/checkpoints/ckpt_last.pt ;;
      gate)   base=$D/gate_4fold_${fold}.pt ;;
      *) echo "bad base kind $base_kind"; return 1 ;;
    esac
    local -a rargs=()
    case $router_tag in
      "")      rargs=(--base-only --gate-ckpt "$base") ;;
      ft)      rargs=(--no-router --gate-ckpt $D/router_ft_only_${fold}_${SUF}.pt) ;;
      frozen)  rargs=(--no-gate-base --gate-ckpt "$base" --router-ckpt $D/router_frozen_${fold}_${SUF}.pt) ;;
      noenum)  rargs=(--no-gate-base --gate-ckpt "$base" --router-ckpt $D/router_noenum_${fold}_${SUF}.pt) ;;
      full)    rargs=(--no-gate-base --gate-ckpt "$base" --router-ckpt $D/router_ft_enum_cr_${fold}_l1.0_${SUF}.pt) ;;
      *) echo "bad router tag $router_tag"; return 1 ;;
    esac
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY $D/perf_table_router.py \
      "${rargs[@]}" \
      --split-csv AnysoleWorkspace/protocol/splits/default/$split \
      --clean-cache-root $R/cam3 --occ020-cache-root $R/cam3_occ_lower020 \
      --occ050-cache-root $R/cam3_occ_lower050 \
      --degraded-cache-root $R/cam3_occ_lower100 \
      --batch-size 64 > /tmp/${SUF}_eval_${tag}_${fold}.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; echo "$tag pid $p exit $?"; done
}

echo "=== arm1 bare enum base (V/VT) ==="
run_eval arm1_enum enum ""
echo "=== arm2 V gate (V/VT) ==="
run_eval arm2_gate gate ""
echo "=== arm3 FT v3 (V/VT) ==="
run_eval ft_v3 enum ft
echo "=== arm4 Router_Frozen v3 ==="
run_eval frozen_v3 enum frozen
echo "=== arm6 Router_NoEnum v3 ==="
run_eval noenum_v3 noenum noenum
echo "=== arm5 Full Router v3 (lambda=1.0) ==="
run_eval full_v3 enum full
echo ALL_V3_EVAL_DONE
