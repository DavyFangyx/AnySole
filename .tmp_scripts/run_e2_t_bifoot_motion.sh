#!/usr/bin/env bash
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
D=results/diagnostics/20261112_v8_gate_frozen
M=results/diagnostics/20261114_imu_eval_v2_motion
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
folds=(fS5S6S11 fS5S11S12 fS6S7 fS5S6S8)
cfgs=(v1_fold_S5S6S11 v1_fold_S5S11S12 v1_fold_S6S7 v1_fold_S5S6S8)
gpus=(2 3 6 7)
mkdir -p "$M"
for i in 0 1 2 3; do
  fold=${folds[$i]}; gpu=${gpus[$i]}
  mkdir -p "$M/$fold"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY anysole/eval.py \
    --config anysole/configs/${cfgs[$i]}.yaml \
    --ckpt $D/router_ft_only_${fold}_v2rerun.pt --split test \
    --config-id T2M --write-motion "$M/$fold" \
    --metrics-out "$M/$fold/_brief.json" --protocol-out "$M/$fold/_protocol.json" \
    --no-robustness > /tmp/e2_t_bifoot_motion_${fold}.log 2>&1 &
  echo "$fold pid $!"
done
wait
echo E2_T_BIFOOT_MOTION_DONE
