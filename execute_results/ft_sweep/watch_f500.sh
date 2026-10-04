#!/usr/bin/env bash
# dense8 run の見張り。100epごとにテスト追従率を検査し、崩壊なら学習を止めて抜ける。
# 抜けたことが通知になるので、その後の修正と再起動は上位でおこなう。
set -u
cd "$(dirname "$0")/../.."
LOG=execute_results/ft_sweep/pretrain_G.log
RUN=archive/prod_f500_dense8_ABG_7station_20260922_112241
STALL=1200
last=0

age() { echo $(( $(date +%s) - $(stat -c %Y "$1") )); }
cur_ep() { grep -oE "^train[0-9]+" "$LOG" 2>/dev/null | tail -1 | tr -d 'train'; }

while true; do
  NOW=$(date '+%F %T')

  # ログは再開のたびに追記される。過去の停止行を拾わないよう末尾だけ見る。
  if tail -5 "$LOG" 2>/dev/null | grep -qE "\[done\] work_dir|resume\] stopped safely"; then
    echo "[$NOW] RESULT=finished  ep=$(cur_ep)"; tail -2 "$LOG" | cut -c1-140; break
  fi

  A=$(age "$LOG")
  if [ "$A" -gt "$STALL" ]; then
    echo "[$NOW] RESULT=died  ep=$(cur_ep)  ログ無更新 ${A}s"; tail -3 "$LOG" | cut -c1-140; break
  fi

  EP=$(cur_ep)
  if [ -n "${EP:-}" ] && [ "$EP" -ge 200 ]; then
    B=$(( EP / 100 * 100 ))
    if [ "$B" -gt "$last" ]; then
      echo "[$NOW] ===== ep${EP} 学習の健全性 ====="
      # パイプ越しに $? を読むと grep の終了コードになる。python の側を取る。
      OUT=$(python execute_results/ft_sweep/health_report.py "$LOG" "$RUN" --check 2>&1); RC=$?
      OUT=$(echo "$OUT" | grep -viE "tensorflow|oneDNN")
      echo "$OUT" | tail -24
      if [ "$RC" -eq 2 ]; then
        echo "[$NOW] RESULT=collapsed  ep=${EP}  学習を停止します"
        python pre_train.py --request-stop "$RUN" 2>&1 | tail -1
        break
      fi
      last=$B
    fi
  fi
  sleep 240
done
