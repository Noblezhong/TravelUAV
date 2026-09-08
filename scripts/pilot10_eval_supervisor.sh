#!/bin/bash
# pilot10 1418 eval 自愈 supervisor v2 (5090 tmux 常驻)
# v2 修复：timeouts() 用 $(grep -c) 直接捕获 "0"，去掉 || echo（旧版 0 匹配时输出两行 → last_to="0\n0"
#          → 整数比较恒错 → 每次都被判不健康 → 连杀健康 seg）。另移除"重启后立刻要 ≥3 新记录"的脆断，
#          改为总重启次数上限（真实崩溃 ~7-12 次，上限 20 足够）+ 重启后 8min 冷却防连杀。
ROOT=/code/TravelUAV-ppo
SAVE=$ROOT/eval_drl_0904_pilot10_20k_1418_fast_x10
LOG=/tmp/pilot10_eval1418.log
SUP=/tmp/pilot10_supervisor.log
LAUNCHER=$ROOT/scripts/eval_pilot10_20k_1418.sh
TOTAL=1418
MAX_RESTARTS=20
seg=6        # 当前 seg6 在跑；下次归档从 seg6 起
restarts=0
last_restart_ts=0

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$SUP"; }
records(){ grep -h -c episode_end "$SAVE"/profile_logs/*.jsonl 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
eval_alive(){ pgrep -f 'drl_scheduler_eval[.]py' >/dev/null && echo 1 || echo 0; }
now_ts(){ date +%s; }
# 关键修复：grep -c 在 0 匹配时 stdout 已输出 "0"，直接捕获即可，切勿再 || echo（会产生第二行）
timeouts(){ local n; n=$(grep -c 'Request timed out' "$LOG" 2>/dev/null); [ -n "$n" ] && echo "$n" || echo 0; }

rec=$(records); last_to=$(timeouts)
log "=== supervisor v2 start: records=$rec timeout=$last_to (seg6 running, untouched) ==="

while true; do
  sleep 180
  rec=$(records); alive=$(eval_alive); to=$(timeouts)
  if [ "$rec" -ge "$TOTAL" ]; then
    log "ALL DONE: $rec/$TOTAL records"; touch "$SAVE/__SUPERVISOR_DONE__"
    log "=== supervisor exit (completed) ==="; exit 0
  fi

  if [ "$alive" = "1" ] && [ "$to" -le "$last_to" ]; then
    last_to=$to
    # 每 ~9min 一条健康日志（每 3 次循环打一次）
    if [ $(( $(now_ts) / 180 % 3 )) -eq 0 ]; then log "healthy records=$rec timeouts=$to"; fi
    continue
  fi

  # ==== 触发重启 ====
  ts=$(now_ts); reason=""
  [ "$alive" = "0" ] && reason="eval_dead"
  [ "$to" -gt "$last_to" ] && reason="${reason:+$reason+}timeout_inc($last_to->$to)"
  # 冷却：仅对 timeout 触发（非 eval_dead）生效，防止误杀刚重启的健康 eval
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
  # 4) 续跑 eval
  tmux kill-session -t ev1418r 2>/dev/null
  tmux new-session -d -s ev1418r "cd $ROOT && bash $LAUNCHER > /tmp/eval1418r_launch.log 2>&1"
  # 5) 等 eval 起（模型加载 ~3min + 首个 episode header；最长 6min）
  ok=0
  for i in $(seq 1 24); do sleep 15
    if [ "$(eval_alive)" = "1" ] && grep -qE '\[episode 0000\]' "$LOG" 2>/dev/null; then ok=1; break; fi
    if [ "$(eval_alive)" = "0" ]; then log "eval died during startup (i=$i)"; break; fi
  done
  last_restart_ts=$(now_ts); last_to=$(timeouts)
  log "resume active seg$((seg+1)) eval_ok=$ok records=$rec restarts=$restarts"
done
