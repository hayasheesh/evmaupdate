#!/bin/bash
# finetune（count・control）の見張り。手を打つ必要があるときだけ1行出して終わる（Claude の Bash の run_in_background で動かす）。
#   bash ft_quiet.sh
# 見るもの: 学習の Traceback、ログが15分止まる、学習の終了、GPU の停止の印、評価の失敗と区切りの終了、
#           学習100回目以降の勾配の比（大域の割合が 0.40〜0.80 の外、本線の2000回目と比べて大きさが3倍超か1/3未満。
#           割合の上を 0.80 にしているのは、count・control とも 0.74〜0.77 で大きさは3倍以内のため。前に SoC が伸びなかった実行は 0.89〜0.92）。
#           値は30分ごとに ft_grad_log.txt に書く）。
H=/c/Users/admin/Desktop/EVMALOCALUPDATE/execute_results/finetune_day_20261019
ROOT=/c/Users/admin/Desktop/EVMALOCALUPDATE
REP=$H/ft_reported.txt; touch "$REP"
PIDF=$H/ft_quiet.pid
[ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null
echo $$ > "$PIDF"
last_grad=0
fire() { grep -qxF "$1" "$REP" && return 0; echo "$1" >> "$REP"; echo "$(date +%H:%M) $2"; exit 0; }
while true; do
  [ "$(cat "$PIDF")" = "$$" ] || exit 0
  now=$(date +%s)
  for m in count control; do
    res=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('result') or 'running')" "$H/${m}_run_status.json" 2>/dev/null)
    [ -n "$res" ] && [ "$res" != running ] && fire "ft_${m}_result_$res" "finetune $m の学習が $res"
    tb=$(grep -c Traceback "$H/logs/${m}_pretrain.stderr.log" 2>/dev/null)
    [ "${tb:-0}" -gt 0 ] && fire "ft_${m}_tb$tb" "finetune $m で Traceback $tb 件"
    mt=$(date -r "$H/logs/${m}_pretrain.stdout.log" +%s 2>/dev/null || echo "$now")
    [ $((now - mt)) -gt 900 ] && fire "ft_${m}_silent_$mt" "finetune $m のログが $(( (now - mt) / 60 )) 分止まっている"
  done
  [ -f "$H/FT_GPU_STOPPED.txt" ] && fire "ft_gpu_stop" "GPU の温度で finetune を止めた: $(cat "$H/FT_GPU_STOPPED.txt")"
  for f in "$H"/FAILED_*; do [ -e "$f" ] && fire "ft_${f##*/}" "finetune の評価が失敗: ${f##*/FAILED_}"; done
  for tag in $(grep -o '[^ ]* exit 0$' "$H/ft_eval_queue.log" 2>/dev/null | cut -d' ' -f1); do
    fire "ft_eval_$tag" "finetune の評価 $tag が済んだ（summary_day.json）"
  done
  if [ $((now - last_grad)) -gt 1800 ]; then
    last_grad=$now
    for m in count control; do
      d=$(ls -d "$ROOT"/archive/ftday_aemo6_20240201_${m}_2* 2>/dev/null | tail -1)
      [ -n "$d" ] || continue
      g=$(python "$ROOT/execute_results/wr_gs_6station_20261011/grad_check.py" "$(cygpath -w "$d")" --json 2>/dev/null | tail -1)
      [ -n "$g" ] || continue
      echo "$(date +%m-%dT%H:%M) $g" >> "$H/ft_grad_log.txt"
      ep=$(echo "$g" | python -c "import json,sys; print(json.load(sys.stdin).get('episode') or 0)" 2>/dev/null)
      # 基準は本線の2000回目（大域の割合 0.663、局所 11.46、大域 25.94。同じ grad_check.py で測った値）。hobo の時期ごとの表は使わない
      off=$(echo "$g" | python -c "
import json,sys
g=json.load(sys.stdin); o=[]
if not 0.40<=g['global_ratio']<=0.80: o.append('ratio%.2f'%g['global_ratio'])
for k,ref in (('local_raw',11.46),('global_raw',25.94)):
    q=g[k]/ref
    if q>3 or q<1/3: o.append('%sx%.1f'%(k,q))
print(' '.join(o))" 2>/dev/null)
      [ "${ep:-0}" -ge 100 ] && [ -n "$off" ] && fire "ft_${m}_grad_$ep" "finetune $m の勾配が範囲外（ep$ep: $off）"
    done
  fi
  sleep 300
done
