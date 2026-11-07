#!/usr/bin/env bash
# 临时守护（2026-11-06）：V8_Cv5VG/ST 全量训练（00472-476）不得落到 GPU4/7。
#
# 背景：GPU4 只剩 ~5 GB、GPU7 ~9 GB（yuqingchi 常驻），而全量训练需 ~13 GB。
# 策略：
#   phase 1  轮询队列，最早训练格距队首 <= THRESH 时进入处置；
#   phase 2  干净停掉 GPU4/7 的 worker（当前 eval 杀树 + conf 回 queue）；
#   phase 3  等 5 格训练全部被 GPU5/6 的 worker 抢走（running/done）或失败；
#   phase 4  重启 GPU4(1) + GPU7(4) 个 worker。
# 退出码：0=完成；3=训练格失败（不重试）；其他=崩溃（外层 wrapper 60s 后重试）。
set -u

root=/data/fangyuxuan/projects/gait
q="$root/configs/queue"
run="$root/configs/running"
done_dir="$root/configs/done"
fail_dir="$root/configs/failed"
log="$root/configs/logs/watch_vg_st.log"

TRAIN_IDS=(
  00472__V8_Cv5VG_fS5S6S11_s1
  00473__V8_Cv5VG_fS5S11S12_s1
  00474__V8_Cv5VG_fS6S7_s1
  00475__V8_Cv5VG_fS5S6S8_s1
  00476__V8_Cv5ST_fS5S6S11_s1
)
# 2026-11-06 15:34 启动的 GPU4/7 worker
STOP_PIDS=(3298376 3298605 3298645 3298971 3299010)
THRESH=8
POLL=30

log_msg() { echo "[$(date -Is)] $*" >> "$log"; }

is_train() {
  local b="$1" id
  for id in "${TRAIN_IDS[@]}"; do [[ "$b" == "$id" ]] && return 0; done
  return 1
}

killtree() {
  local p="$1" c
  for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done
  kill -TERM "$p" 2>/dev/null || true
}

# ---------- phase 1：等训练波接近队首 ----------
while :; do
  first="" idx=0 n=0
  for f in "$q"/*.conf; do
    [ -e "$f" ] || break
    n=$((n + 1))
    b="$(basename "$f" .conf)"
    if [ -z "$first" ] && is_train "$b"; then first="$b"; idx=$n; break; fi
  done
  remaining=0
  for id in "${TRAIN_IDS[@]}"; do
    { [ -f "$q/$id.conf" ] || [ -f "$run/$id.conf" ]; } && remaining=$((remaining + 1))
  done
  if [ "$remaining" -eq 0 ]; then
    log_msg "no pending training in queue/running — nothing to protect; exit"
    exit 0
  fi
  if [ -z "$first" ]; then
    log_msg "no training left in queue (all claimed) — exit without stopping"
    exit 0
  fi
  log_msg "earliest queued training $first at position $idx (pending=$remaining)"
  [ "$idx" -le "$THRESH" ] && break
  sleep "$POLL"
done

# ---------- phase 2：停 GPU4/7 worker，在跑 eval 回队 ----------
log_msg "threshold reached — stopping GPU4/7 workers"
for sid in "${STOP_PIDS[@]}"; do
  kill -0 "$sid" 2>/dev/null || { log_msg "worker $sid already gone"; continue; }
  runsh="$(pgrep -P "$sid" 2>/dev/null | head -1)"
  conf=""
  if [ -n "$runsh" ]; then
    conf="$(tr '\0' '\n' < "/proc/$runsh/cmdline" 2>/dev/null \
            | grep -o 'configs/running/[^ ]*\.conf' | head -1 | xargs -r basename)"
  fi
  log_msg "stopping worker $sid (job=${conf:-idle})"
  killtree "$sid"
  sleep 1
  if [ -n "$conf" ] && [ -f "$run/$conf" ]; then
    mv "$run/$conf" "$q/$conf" && log_msg "requeued $conf"
  fi
done
sleep 3
for sid in "${STOP_PIDS[@]}"; do kill -KILL "$sid" 2>/dev/null || true; done

# ---------- phase 3：等 5 格训练全部被 GPU5/6 抢走 ----------
while :; do
  pending=0 failed=0
  for id in "${TRAIN_IDS[@]}"; do
    [ -f "$fail_dir/$id.conf" ] && { failed=$((failed + 1)); continue; }
    [ -f "$run/$id.conf" ] && continue
    [ -f "$done_dir/$id.conf" ] && continue
    [ -f "$q/$id.conf" ] && pending=$((pending + 1))
  done
  log_msg "waiting trainings: pending=$pending failed=$failed"
  if [ "$failed" -gt 0 ]; then
    log_msg "ABORT: a training conf failed — inspect configs/failed/; GPU4/7 left stopped"
    exit 3
  fi
  [ "$pending" -eq 0 ] && break
  sleep "$POLL"
done

# ---------- phase 4：恢复 GPU4/7 ----------
log_msg "all 5 trainings claimed — restarting GPU4 (x1) + GPU7 (x4)"
CUDA_VISIBLE_DEVICES=4 bash "$root/configs/bg.sh" >> "$log" 2>&1
for _ in 1 2 3 4; do CUDA_VISIBLE_DEVICES=7 bash "$root/configs/bg.sh" >> "$log" 2>&1; done
log_msg "done"
exit 0
