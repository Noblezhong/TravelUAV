#!/bin/bash
# pilot11 AHS-only 283 eval 自愈 supervisor(5090 tmux 常驻)。
# 监视 drl_scheduler_eval.py(与 0903 基线同 harness,无 TCM);UE Vulkan 崩/超时自愈重启。
# 逻辑复制自 pilot10_eval_supervisor.sh v2(已修 timeouts() 0 匹配双行 bug + 冷却防连杀)。
# 权重 pin pilot11-30000 步;SAVE 固定 → eval resume 按 episode 目录跳过已完成集。
ROOT=/code/TravelUAV-ppo
export SCHEDULER_MODEL_PATH="${SCHEDULER_MODEL_PATH:-$ROOT/drl_train_pilot11_0909-2109/scheduler_models/ppo_scheduler_20260909-210915-662188.zip}"
export EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-$ROOT/eval_drl_pilot11_ahsonly_283_fast_x10}"
export EVAL_LOG=/tmp/pilot11_ahsonly_eval.log
SAVE=$EVAL_SAVE_PATH
LOG=$EVAL_LOG
SUP=/tmp/pilot11_ahsonly_supervisor.log
LAUNCHER=$ROOT/scripts/eval_pilot11_ahsonly_283.sh
TOTAL=283
MAX_RESTARTS=20
seg=1
restarts=0
last_restart_ts=0

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$SUP"; }
records(){ grep -h -c episode_end "$SAVE"/profile_logs/*.jsonl 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
eval_alive(){ pgrep -f 'drl_scheduler_eval[.]py' >/dev/null && echo 1 || echo 0; }
now_ts(){ date +%s; }
timeouts(){ local n; n=$(grep -c 'Request timed out' "$LOG" 2>/dev/null); [ -n "$n" ] && echo "$n" || echo 0; }

rec=$(records); last_to=$(timeouts)
log "=== pilot11 AHS-only 283 supervisor start: model=$SCHEDULER_MODEL_PATH save=$SAVE records=$rec timeout=$last_to ==="

while true; do
  sleep 180
  rec=$(records); alive=$(eval_alive); to=$(timeouts)
  if [ "$rec" -ge "$TOTAL" ]; then
    log "ALL DONE: $rec/$TOTAL records"; touch "$SAVE/__SUPERVISOR_DONE__"
    log "=== supervisor exit (completed) ==="; exit 0
  fi

  if [ "$alive" = "1" ] && [ "$to" -le "$last_to" ]; then
    last_to=$to
    if [ $(( $(now_ts) / 180 % 3 )) -eq 0 ]; then log "healthy records=$rec timeouts=$to"; fi
    continue
  fi

  ts=$(now_ts); reason=""
  [ "$alive" = "0" ] && reason="eval_dead"
  [ "$to" -gt "$last_to" ] && reason="${reason:+$reason+}timeout_inc($last_to->$to)"
  if [ "$alive" = "1" ] && [ $((ts - last_restart_ts)) -lt 480 ]; then
    log "restart suppressed (grace, last_restart ${ts-last_restart_ts}s ago): $reason records=$rec"
    last_to=$to; continue
  fi
  restarts=$((restarts+1))
  log "RESTART #$restarts: $reason records=$rec"
  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    log "FATAL: exceeded $MAX_RESTARTS restarts without reaching $TOTAL. STOPPING (manual intervention needed)."
    log "=== supervisor exit (fatal) ==="; exit 1
  fi

  # 1) 杀 eval
  pkill -f 'drl_scheduler_eval[.]py' 2>/dev/null; sleep 4
  [ "$(eval_alive)" = "1" ] && { pkill -9 -f 'drl_scheduler_eval[.]py' 2>/dev/null; sleep 2; }
  # 2) 归档 seg 日志
  [ -f "$LOG" ] && mv "$LOG" "${LOG}_seg${seg}.log" && log "archived -> ${LOG}_seg${seg}.log"
  seg=$((seg+1))
  # 3) 重启 AirSim
  tmux kill-session -t airsim 2>/dev/null; sleep 1
  tmux new-session -d -s airsim "cd $ROOT && bash scripts/airsim_server_5090.sh > /tmp/airsim5090_sup_$(date +%H%M%S).log 2>&1"
  sleep 6
  # 4) 续跑 eval(inline env,防 tmux 不继承调用者 env)
  tmux kill-session -t evp11 2>/dev/null
  tmux new-session -d -s evp11 "cd $ROOT && SCHEDULER_MODEL_PATH=$SCHEDULER_MODEL_PATH EVAL_SAVE_PATH=$SAVE EVAL_LOG=$LOG bash $LAUNCHER > /tmp/evp11_launch.log 2>&1"
  # 5) 等 eval 起(模型加载 ~3min + 首个 episode header;最长 6min)
  ok=0
  for i in $(seq 1 24); do sleep 15
    if [ "$(eval_alive)" = "1" ] && grep -qE '\[episode 0000\]' "$LOG" 2>/dev/null; then ok=1; break; fi
    if [ "$(eval_alive)" = "0" ]; then log "eval died during startup (i=$i)"; break; fi
  done
  last_restart_ts=$(now_ts); last_to=$(timeouts)
  log "resume active seg$((seg+1)) eval_ok=$ok records=$rec restarts=$restarts"
done
