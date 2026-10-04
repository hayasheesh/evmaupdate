#!/usr/bin/env bash
# 20station pretrain の見張り。プロセスの生死は見ない（pgrep がない）。
# ログの更新時刻と完了マーカーだけで判定する。
# ep25/50/75/100 で hobo との桁比較を出し、以後 100 ごとに経過を出す。
#   usage: watch_s20.sh <tag> <run_dir>
set -u
TAG="$1"
RUN="$2"
cd "$(dirname "$0")/../.."
LOG="execute_results/ft_sweep/pretrain_${TAG}.log"
STALL=1800
last_report=0

age() { echo $(( $(date +%s) - $(stat -c %Y "$1") )); }
cur_ep() { grep -oE "^train[0-9]+" "$LOG" 2>/dev/null | tail -1 | tr -d 'train'; }

while true; do
  NOW=$(date '+%F %T')

  if grep -q "=== tag=${TAG} exit=" "$LOG" 2>/dev/null; then
    echo "[$NOW] ${TAG} 終了: $(grep "=== tag=${TAG} exit=" "$LOG" | tail -1)"
    break
  fi

  A=$(age "$LOG")
  if [ "$A" -gt "$STALL" ]; then
    echo "[$NOW] ${TAG} 停止の疑い: ログ無更新 ${A}s"
    tail -3 "$LOG"
    break
  fi

  EP=$(cur_ep)
  if [ -n "${EP:-}" ]; then
    MARK=0
    for m in 25 50 75 100; do
      if [ "$EP" -ge "$m" ] && [ "$last_report" -lt "$m" ]; then MARK=$m; fi
    done
    if [ "$MARK" -eq 0 ] && [ "$EP" -ge 200 ]; then
      B=$(( EP / 100 * 100 ))
      if [ "$B" -gt "$last_report" ]; then MARK=$B; fi
    fi
    if [ "$MARK" -gt 0 ]; then
      echo "[$NOW] ===== ${TAG} ep${EP} 桁比較 ====="
      python execute_results/ft_sweep/compare_to_hobo.py "$RUN" "$EP" 2>&1 | grep -v tensorflow | tail -24
      last_report=$MARK
    fi
  fi

  sleep 180
done
