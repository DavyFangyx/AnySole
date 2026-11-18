#!/usr/bin/env bash
set -u
cd /data/fangyuxuan/projects/gait
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
D=results/diagnostics/20261112_v8_gate_frozen
M=results/diagnostics/20261114_imu_eval_v2_motion/VT_teaser
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
cfgs=(v1_fold_S5S6S11 v1_occ020_fold_S5S6S11 v1_occ050_fold_S5S6S11 v1_occ100_fold_S5S6S11)
names=(clean occ020 occ050 occ100)
gpus=(2 3 6 7)
for i in 0 1 2 3; do
  mkdir -p "$M/${names[$i]}"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} PYTHONPATH=. $PY anysole/eval.py \
    --config anysole/configs/${cfgs[$i]}.yaml \
    --ckpt $D/router_ft_only_fS5S6S11_v2rerun.pt --split test \
    --config-id V2M,VT2M --write-motion "$M/${names[$i]}" \
    --metrics-out /tmp/e6_vt_teaser_${names[$i]}_brief.json \
    --protocol-out /tmp/e6_vt_teaser_${names[$i]}.json \
    --no-robustness > /tmp/e6_vt_teaser_${names[$i]}.log 2>&1 &
  echo "${names[$i]} pid $!"
done
wait
echo E6_VT_TEASER_MOTION_DONE
