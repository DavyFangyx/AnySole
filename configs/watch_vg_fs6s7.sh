#!/usr/bin/env bash
# 临时收尾（2026-11-06）：V8_Cv5VG_fS6S7_s1 训完后，退回其 6 格 eval。
set -u
root=/data/fangyuxuan/projects/gait
cell="$root/results/cells/V8_Cv5VG_fS6S7_s1/.done"
fail="$root/configs/failed"
queue="$root/configs/queue"
log="$root/configs/logs/watch_vg_fs6s7.log"

IDS=(
  00482__V8_Cv5VG_fS6S7_s1_e6
  00498__V8_Cv5VG_fS6S7_s1_evalocc020
  00499__V8_Cv5VG_fS6S7_s1_evalocc040
  00500__V8_Cv5VG_fS6S7_s1_evalocc060
  00501__V8_Cv5VG_fS6S7_s1_evalocc080
  00502__V8_Cv5VG_fS6S7_s1_evalocc100
)
echo "[$(date -Is)] waiting for $cell" >> "$log"
while [ ! -f "$cell" ]; do sleep 60; done
echo "[$(date -Is)] base cell done — requeueing 6 evals" >> "$log"
for id in "${IDS[@]}"; do
  if [ -f "$fail/$id.conf" ]; then
    mv "$fail/$id.conf" "$queue/$id.conf" && echo "[$(date -Is)] requeued $id" >> "$log"
  fi
done
echo "[$(date -Is)] done" >> "$log"
