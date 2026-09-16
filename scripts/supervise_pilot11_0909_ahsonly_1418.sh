#!/bin/bash
# 自愈 supervisor: AHS-only 1418 行。崩溃(如空图 cv2.imwrite)后自动续跑。
# 停止条件: 跑满 TARGET / 连续一次重试无进展(疑似确定性崩溃，避免空转)
LAUNCHER=/HDD1/code/TravelUAV/scripts/eval_pilot11_0909_ahsonly_1418.sh
SAVE=/HDD1/code/TravelUAV/eval_drl_pilot11_0909_ahsonly_1418_fast_x10
TARGET=1418
SLOG=/tmp/pilot11_ahsonly_1418_supervisor.log
cnt() { ls -d ${SAVE}/*/ 2>/dev/null | grep -vE "/profile_logs/" | wc -l; }
prev=0
for attempt in $(seq 1 30); do
  echo "[$(date "+%m-%d %H:%M:%S")] attempt $attempt 起跑 (已完成 $(cnt))" >> "$SLOG"
  bash "$LAUNCHER" >> "$SLOG" 2>&1
  rc=$?
  d=$(cnt)
  echo "[$(date "+%m-%d %H:%M:%S")] attempt $attempt 退出 rc=$rc, 已完成 $d/$TARGET" >> "$SLOG"
  [ "$d" -ge "$TARGET" ] && { echo "== 全部完成 ==" >> "$SLOG"; break; }
  if [ "$d" -le "$prev" ]; then echo "== 无进展($prev -> $d)，停止重试 ==" >> "$SLOG"; break; fi
  prev=$d
  sleep 20
done
