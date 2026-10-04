#!/usr/bin/env bash
# 20station の能力分布測定の完了を待って、まとめだけ出す。
# 7station 側は 5日で打ち切り（cap_dist_7s_partial5days.log に保存）。
set -u
cd "$(dirname "$0")/../.."
B=execute_results/ft_sweep/cap_dist_20s.log
while true; do
  if grep -q "まとめ" "$B" 2>/dev/null; then
    echo "[$(date '+%F %T')] ===== 20station 完了 ====="
    grep -A4 "まとめ" "$B"
    exit 0
  fi
  AGE=$(( $(date +%s) - $(stat -c %Y "$B") ))
  if [ "$AGE" -gt 2400 ]; then
    echo "[$(date '+%F %T')] cap_dist_20s 無更新 ${AGE}s、止まった可能性"
    tail -2 "$B"
    exit 1
  fi
  sleep 180
done
