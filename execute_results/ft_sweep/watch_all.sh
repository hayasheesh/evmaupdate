#!/usr/bin/env bash
# 走っているものを見張る。
#
# プロセスの生死は見ない。Git Bash に pgrep はなく、PowerShell 経由の引用符も
# 通らなかった。どちらも一度は誤検知を出している。ログの更新時刻と完了マーカー
# だけで判定する。学習は 20 秒ごとに train 行を書くので、無更新が続けば死んでいる。
#
# 数字は合図に載せない。報告する数字は必ず出力ファイルから読み直す。
set -u
cd "$(dirname "$0")/../.."
STALL=900

age() { echo $(( $(date +%s) - $(stat -c %Y "$1") )); }

L=execute_results/ft_sweep/pretrain_hobomatch.log
L2=execute_results/ft_sweep/hm3_long.log
done_pt=0; done_ft=0

while true; do
  NOW=$(date '+%F %T')

  if [ "$done_pt" -eq 0 ]; then
    # 最後の再開マーカー以降だけを見る。ログには前回 ep2000 完了時の
    # マーカーが残っており、全体を grep すると即座に誤検知する。
    S=$(grep -n "=== resume tag=hobomatch started" "$L" | tail -1 | cut -d: -f1)
    if [ -n "$S" ] && tail -n +"$S" "$L" | grep -q "=== resume tag=hobomatch exit="; then
      echo "[$NOW] SIGNAL pretrain: 完了マーカーあり"; done_pt=1
    elif [ "$(age "$L")" -gt "$STALL" ]; then
      echo "[$NOW] SIGNAL pretrain: 無更新が続いている（完了マーカーなし）"; done_pt=1
    fi
  fi

  if [ "$done_ft" -eq 0 ]; then
    if grep -q "=== arm=hm3_long exit=" "$L2" 2>/dev/null; then
      echo "[$NOW] SIGNAL hm3_long: 完了マーカーあり"; done_ft=1
    elif [ "$(age "$L2")" -gt "$STALL" ]; then
      echo "[$NOW] SIGNAL hm3_long: 無更新が続いている（完了マーカーなし）"; done_ft=1
    fi
  fi

  for f in "$L" "$L2"; do
    tail -n 60 "$f" 2>/dev/null | grep -qE "Traceback|[A-Za-z]+Error:|CUDA error" \
      && echo "[$NOW] SIGNAL $(basename "$f"): 例外の痕跡"
  done

  # 3) BESS なしの MARL vs ルールベース比較。出力ディレクトリに日ごとの
  #    サマリが増えるので、それが止まったら合図を出す。
  for d in nb_marl nb_zero; do
    D="execute_results/ft_sweep/$d"
    [ -d "$D" ] || continue
    N=$(find "$D" -name "controller_precision_summary.json" 2>/dev/null | wc -l)
    LAST=$(find "$D" -type f -newermt "-20 minutes" 2>/dev/null | head -1)
    [ -z "$LAST" ] && [ "$N" -lt 5 ] && echo "[$NOW] SIGNAL $d: 20分 出力が増えていない"
  done

  FREE=$(df -P /c | awk 'NR==2{print int($4/1048576)}')
  [ "$FREE" -lt 40 ] && echo "[$NOW] SIGNAL disk: 空きが少ない"

  [ "$done_pt" -eq 1 ] && [ "$done_ft" -eq 1 ] && { echo "[$NOW] SIGNAL 監視終了"; break; }
  sleep 120
done
