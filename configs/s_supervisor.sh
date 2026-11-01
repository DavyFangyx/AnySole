#!/usr/bin/env bash
# s_supervisor.sh — S 显式收益监督实验：GPU 感知自动发射 + 10:00 硬停（2026-11-01）
#
# 背景：用户休息中，需要全自动布置 S 实验（queue 00206-00209）。但发射时
# 全部 8 张卡被占用（GPU0-3 = songpeixuan FSDP 训练；GPU4-7 = fangyuxuan
# hmr2_tta）。本脚本轮询空闲卡，一旦有卡释放（显存 <2GB 且无计算进程）就
# 在那张卡上起一个 scheduler worker（每个 worker 从共享 queue 原子抢一个
# conf），最多 4 个 worker；到 2026-11-01 10:00 硬停所有已起 worker 及其
# 训练子进程（SIGTERM → 宽限 → SIGKILL），不留任何训练跑过 10 点。
#
# 用法：nohup bash configs/s_supervisor.sh >>configs/logs/s_supervisor.log 2>&1 &
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
queue="$root/queue"
logs_dir="$root/logs"
mkdir -p "$logs_dir"
log="$logs_dir/s_supervisor.log"

DEADLINE="${S_DEADLINE:-2026-11-01 10:00:00}"
MAX_GPUS="${S_MAX_GPUS:-4}"       # 最多用几张卡
FREE_MEM_MB="${S_FREE_MEM_MB:-2000}"  # 空闲判据：显存 < 2GB 且无计算进程
POLL_SEC="${S_POLL_SEC:-30}"

# supervisor 由 nohup 非交互启动，PATH 里没有 conda，导致 run.sh 的
# `command -v conda` 失败（"conda is required"）。把 conda bin 提前到 PATH，
# 让 setsid 起的 scheduler worker 能继承到 conda 并 activate touch_gait。
CONDA_BIN="${S_CONDA_BIN:-/data/fangyuxuan/miniconda3/bin}"
if [[ -d "$CONDA_BIN" ]]; then
  export PATH="$CONDA_BIN:$PATH"
else
  log "WARN: conda bin 目录不存在：$CONDA_BIN（worker 将无法激活环境）"
fi

declare -A worker_pgid   # gpu_index -> setsid 进程组号（= 组长 PID）

log() { echo "[$(date -Is)] $*" >>"$log"; }

# 该 GPU 是否空闲：显存 < FREE_MEM_MB 且该 GPU uuid 无任何计算进程
is_free() {
  local idx="$1" uuid mem busy
  local line
  line=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v i="$idx" '$1==i{print $2","$3}')
  [[ -z "$line" ]] && return 1
  uuid="${line%%,*}"; mem="${line##*,}"
  mem=${mem:-999999}
  (( mem >= FREE_MEM_MB )) && return 1
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
         | tr -d ' ' | grep -c "$uuid" || true)
  [[ "$busy" = "0" ]] && return 0 || return 1
}

now_epoch() { date +%s; }
deadline_epoch() { date -d "$DEADLINE" +%s; }

stop_worker() {
  local idx="$1" pgid="${worker_pgid[$idx]:-}"
  [[ -z "$pgid" ]] && return 0
  if kill -0 -"$pgid" 2>/dev/null; then
    log "  STOP GPU $idx pgid=$pgid：SIGTERM"
    kill -TERM -- -"$pgid" 2>/dev/null || true
    # 宽限 90s 让训练保存 ckpt 并优雅退出
    for _ in $(seq 1 18); do
      kill -0 -"$pgid" 2>/dev/null || break
      sleep 5
    done
    if kill -0 -"$pgid" 2>/dev/null; then
      log "  STOP GPU $idx pgid=$pgid：SIGKILL"
      kill -KILL -- -"$pgid" 2>/dev/null || true
    fi
  fi
  unset worker_pgid["$idx"]
}

stop_all() {
  log "STOP_ALL：硬停所有已起 worker"
  for idx in "${!worker_pgid[@]}"; do
    stop_worker "$idx"
  done
}

started_count() {
  local n=0
  for _ in "${!worker_pgid[@]}"; do ((n+=1)); done
  echo "$n"
}

log "===== S supervisor 启动：deadline=$DEADLINE max_gpus=$MAX_GPUS poll=${POLL_SEC}s ====="

while true; do
  now=$(now_epoch); dl=$(deadline_epoch)
  if (( now >= dl )); then
    log "到达 $DEADLINE，停止发射并硬停"
    stop_all
    break
  fi

  # 还有 conf 且 worker 数未满 → 扫描空闲卡
  if compgen -G "$queue"/*.conf >/dev/null 2>&1; then
    n=$(started_count)
    if (( n < MAX_GPUS )); then
      for idx in 0 1 2 3 4 5 6 7; do
        [[ -n "${worker_pgid[$idx]:-}" ]] && continue
        if is_free "$idx"; then
          wlog="$logs_dir/GPU${idx}_s_worker.log"
          log "GPU $idx 空闲，发射 scheduler worker（log=$wlog）"
          # setsid：每个 worker 独立进程组，便于 10:00 整组硬停；
          # CUDA_VISIBLE_DEVICES 绑定到该空闲卡（scheduler 的 --device cuda
          # 依赖它定位物理卡）。
          CUDA_VISIBLE_DEVICES="$idx" \
            setsid bash "$root/scheduler.sh" "$queue" "$wlog" \
            >>"$wlog" 2>&1 < /dev/null &
          worker_pgid["$idx"]=$!
          log "  GPU $idx worker pgid=$!"
          break
        fi
      done
    fi
  fi
  sleep "$POLL_SEC"
done

log "===== S supervisor FINISHED ====="
