#!/usr/bin/env bash
# Formal evaluation (anysole/eval.py) for the V8_IMU experiment groups.
# Frame-weighted protocol; E4 uses VT2M only; E2/E5 use all modes.
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
D=results/diagnostics/20261112_v8_gate_frozen
FT=results/diagnostics/20261114_imu_ft_ablation_v2
RO=results/diagnostics/20261114_router_ablation_v2/eval_ckpts
B=results/AnySole
OUT=results/diagnostics/20261114_imu_eval_v2
if [ "${GROUP:-e4}" != "e4" ]; then OUT=${OUT}_all; fi
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
folds=(fS5S6S11 fS5S11S12 fS6S7 fS5S6S8)
gpus=(2 3 6 7)

enum_base () { echo "$B/V8_Cv7_PhysIMU_${1}_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_last.pt"; }
cv5_base ()  { echo "$B/V8_Cv5_PhysIMU_${1}_f6_soft_time_accel_state_smpl24_v1/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_last.pt"; }

ckpt_for () {
  local model=$1 fold=$2
  case $model in
    V8_IMU_Residual) echo "$D/router_ft_only_${fold}_v2rerun.pt" ;;
    V8_IMU_Residual_woAdapt) enum_base "$fold" ;;
    V8_IMU_Residual_woEnum) echo "$FT/V8_IMU_Residual_woEnum_${fold}.pt" ;;
    V8_IMU_Residual_DecoderOnly) echo "$FT/V8_IMU_Residual_DecoderOnly_${fold}.pt" ;;
    V8_IMU_Residual_CleanOnly) echo "$FT/V8_IMU_Residual_CleanOnly_${fold}.pt" ;;
    V8_IMU_Router) echo "$RO/V8_IMU_Router_${fold}_eval.pt" ;;
    V8_IMU_Router_Full) echo "$RO/V8_IMU_Router_Full_${fold}_eval.pt" ;;
    V8_IMU_TGate) cv5_base "$fold" ;;
    V8_IMU_DualGate) echo "$D/gate_4fold_${fold}.pt" ;;
    *) echo "BAD_MODEL"; return 1 ;;
  esac
}

cfg_for () {
  local cond=$1 fold=$2
  local stem=${fold#f}  # v1_fold_S5S6S11.yaml style (no leading f)
  case $cond in
    clean) echo "anysole/configs/v1_fold_${stem}.yaml" ;;
    occ010) echo "anysole/configs/v1_occ010_fold_${stem}.yaml" ;;
    occ020) echo "anysole/configs/v1_occ020_fold_${stem}.yaml" ;;
    occ030) echo "anysole/configs/v1_occ030_fold_${stem}.yaml" ;;
    occ040) echo "anysole/configs/v1_occ040_fold_${stem}.yaml" ;;
    occ050) echo "anysole/configs/v1_occ050_fold_${stem}.yaml" ;;
    occ060) echo "anysole/configs/v1_occ060_fold_${stem}.yaml" ;;
    occ070) echo "anysole/configs/v1_occ070_fold_${stem}.yaml" ;;
    occ080) echo "anysole/configs/v1_occ080_fold_${stem}.yaml" ;;
    occ090) echo "anysole/configs/v1_occ090_fold_${stem}.yaml" ;;
    occ100) echo "anysole/configs/v1_occ100_fold_${stem}.yaml" ;;
    sceneocc) echo "anysole/configs/v1_sceneocc_${fold}.yaml" ;;
    sceneocc_clean) echo "anysole/configs/v1_sceneocc_clean_${fold}.yaml" ;;
    gap010) echo "anysole/configs/v1_gap_r010_fold_${stem}.yaml" ;;
    gap020) echo "anysole/configs/v1_gap_r020_fold_${stem}.yaml" ;;
    gap040) echo "anysole/configs/v1_gap_r040_fold_${stem}.yaml" ;;
    gap060) echo "anysole/configs/v1_gap_r060_fold_${stem}.yaml" ;;
    gap080) echo "anysole/configs/v1_gap_r080_fold_${stem}.yaml" ;;
    *) echo "BAD_COND"; return 1 ;;
  esac
}

run_eval () {
  local model=$1 cond=$2 fold=$3 gpu=$4 modes=$5
  local ck=$(ckpt_for "$model" "$fold")
  local cfg=$(cfg_for "$cond" "$fold")
  local out=$OUT/$model/$cond
  mkdir -p "$out"
  local brief=$out/${fold}_brief.json
  local proto=$out/${fold}.json
  if [ -s "$brief" ] && [ -s "$proto" ]; then
    echo "SKIP $model $cond $fold"
    return 0
  fi
  local -a mode_args=()
  if [ -n "$modes" ]; then mode_args=(--config-id "$modes"); fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY anysole/eval.py \
    --config "$cfg" --ckpt "$ck" --split test \
    --metrics-out "$brief" --protocol-out "$proto" --no-write-motion --no-robustness \
    "${mode_args[@]}" > /tmp/eval_v2_${model}_${cond}_${fold}.log 2>&1
  echo "DONE $model $cond $fold exit $?"
}

GROUP=${GROUP:-e4}
case $GROUP in
  e4)
    MODELS_A=(V8_IMU_Residual V8_IMU_Router V8_IMU_TGate V8_IMU_DualGate)
    MODELS_B=(V8_IMU_Residual V8_IMU_Residual_woAdapt V8_IMU_Residual_woEnum V8_IMU_Residual_DecoderOnly V8_IMU_Residual_CleanOnly)
    CONDS=(clean occ020 occ050 occ090 sceneocc sceneocc_clean)
    MODELS=("${MODELS_A[@]}" "${MODELS_B[@]}")
    ;;
  e2)
    MODELS=(V8_IMU_Residual)
    CONDS=(clean)
    ;;
  e5)
    MODELS=(V8_IMU_Residual)
    CONDS=(clean occ010 occ020 occ030 occ040 occ050 occ060 occ070 occ080 occ090 occ100 \
           sceneocc sceneocc_clean gap010 gap020 gap040 gap060 gap080)
    ;;
  e4curves)
    MODELS=(V8_IMU_Router V8_IMU_TGate V8_IMU_DualGate)
    CONDS=(occ100 gap010 gap020 gap040 gap060 gap080)
    ;;
  e4curves2)
    MODELS=(V8_IMU_Router V8_IMU_TGate V8_IMU_DualGate)
    CONDS=(occ010 occ030 occ040 occ060 occ070 occ080)
    ;;
  *) echo "bad GROUP"; exit 1 ;;
esac

# Deduplicate model list preserving order.
declare -A seen=()
uniq_models=()
for m in "${MODELS[@]}"; do
  if [ -z "${seen[$m]:-}" ]; then seen[$m]=1; uniq_models+=("$m"); fi
done
MODELS=("${uniq_models[@]}")

idx=0
running=0
for m in "${MODELS[@]}"; do
  for c in "${CONDS[@]}"; do
    for i in 0 1 2 3; do
      fold=${folds[$i]}; gpu=${gpus[$((idx % 4))]}
      modes="VT2M"
      if [ "$GROUP" != "e4" ]; then modes=""; fi
      run_eval "$m" "$c" "$fold" "$gpu" "$modes" &
      idx=$((idx+1))
      running=$((running+1))
      if [ "$running" -ge 16 ]; then wait -n; running=$((running-1)); fi
    done
  done
done
wait
echo FORMAL_EVAL_${GROUP}_DONE
