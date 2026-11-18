#!/bin/bash
# Retrain V8_Cv5 fS6S7 with the woS13 split (test S6/S7, val S14).
set -u
REPO=/data/fangyuxuan/projects/gait
PY=/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python
LOG="$REPO/.tmp_scripts/train_v8_cv5_fs6s7_woS13.log"
cd "$REPO"
"$PY" -m anysole.train \
  --config anysole/configs/v1_fold_S6S7.yaml \
  --contact-method f6_soft_time_accel_state_smpl24_v1 \
  --device cuda:3 \
  --epochs 740 --batch-size 256 --seed 1 \
  --model-name V8_Cv5_fS6S7 \
  --lr-warmup-frac 0.05 --grad-clip 5.0 \
  --config-probs 0.5 0.25 0.25 --condition-sampling enum \
  --final-eval-split none \
  --lambda-pose 3.0 --lambda-traj 1.0 --lambda-trec 0.0 --lambda-vrec 0.0 \
  --lambda-contact 1.0 --lambda-nll 0.0 --lambda-air 2.0 \
  --checkpoint-policy multi_metric --checkpoint-role auto \
  --smpl-support-policy exclude --wandb_mode disabled \
  --wandb_experiment_tag v7_variants \
  > "$LOG" 2>&1
echo "[train done] rc=$? $(date '+%F %T')" >> "$LOG"
