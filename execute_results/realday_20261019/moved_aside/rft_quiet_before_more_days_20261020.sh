#!/bin/bash
# realday の finetune（Windows の control、研究室の scratch_ml_*・multi・multift_ml_*）の見張り。手を打つ必要があるときだけ1行出して終わる
# （Claude の Bash の run_in_background で動かす）。
#   bash rft_quiet.sh
# 見るもの：学習の Traceback、ログが15分止まる、学習の終了、Windows の GPU の停止の印、評価の失敗（両方の機械）、評価の終了（500回と1000回だけ知らせる）、
#           研究室の評価の結果を Windows に写す（評価が済むたび）、勾配の比（Windows の2本、30分ごと、ft_grad_log.txt と同じ形で
#           rft_grad_log.txt に書く。大域の割合が 0.40〜0.80 の外、本線の2000回目と比べて大きさが3倍超か1/3未満で知らせる）。
H=/c/Users/admin/Desktop/EVMALOCALUPDATE/execute_results/realday_20261019
ROOT=/c/Users/admin/Desktop/EVMALOCALUPDATE
LAB=/home/hayashi/workspace/EVMALOCALUPDATE/execute_results/realday_20261019
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
REP=$H/rft_reported.txt; touch "$REP"
PIDF=$H/rft_quiet.pid
WMODES=""  # Windows で学習中の mode（C は 10/20 に1000回まで終わった）
[ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null
echo $$ > "$PIDF"
last_grad=0
unreach=0
fire() { grep -qxF "$1" "$REP" && return 0; echo "$1" >> "$REP"; echo "$(date +%H:%M) $2"; exit 0; }
while true; do
  [ "$(cat "$PIDF")" = "$$" ] || exit 0
  now=$(date +%s)
  for m in $WMODES; do
    res=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('result') or 'running')" "$H/${m}_run_status.json" 2>/dev/null)
    [ -n "$res" ] && [ "$res" != running ] && fire "w_${m}_result_$res" "Windows の $m の学習が $res"
    tb=$(grep -c Traceback "$H/logs/${m}_pretrain.stderr.log" 2>/dev/null)
    [ "${tb:-0}" -gt 0 ] && fire "w_${m}_tb$tb" "Windows の $m で Traceback $tb 件"
    mt=$(date -r "$H/logs/${m}_pretrain.stdout.log" +%s 2>/dev/null || echo "$now")
    [ $((now - mt)) -gt 900 ] && fire "w_${m}_silent_$mt" "Windows の $m のログが $(( (now - mt) / 60 )) 分止まっている"
  done
  [ -f "$H/RFT_GPU_STOPPED.txt" ] && fire "w_gpu_stop" "Windows の GPU の温度で止めた: $(cat "$H/RFT_GPU_STOPPED.txt")"
  for f in "$H"/FAILED_*; do [ -e "$f" ] && fire "w_${f##*/}" "Windows の評価が失敗: ${f##*/FAILED_}"; done
  # 評価が済んだ知らせは500回と1000回だけ（ほかは rft_eval_done.txt に書くだけ）
  for tag in $(grep -o '[^ ]* exit 0$' "$H/rft_eval_queue_windows.log" 2>/dev/null | cut -d' ' -f1); do
    grep -qxF "w_eval_$tag" "$H/rft_eval_done.txt" 2>/dev/null || echo "w_eval_$tag" >> "$H/rft_eval_done.txt"
    case $tag in *500|*500_*|*1000|*1000_*) fire "w_eval_$tag" "Windows の評価 $tag が済んだ";; esac
  done
  lab=$(timeout 90 $SSH smartgrid-gpu "cd $LAB 2>/dev/null && for m in scratch_ml_busy scratch_ml_quiet scratch_ml_average multi multift_ml_busy multift_ml_quiet multift_ml_average; do [ -e \${m}_run_status.json ] || continue; r=\$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1])).get('result') or 'running')\" \${m}_run_status.json 2>/dev/null); t=\$(grep -c Traceback logs/\${m}_pretrain.stderr.log 2>/dev/null); a=\$(( \$(date +%s) - \$(stat -c %Y logs/\${m}_pretrain.stdout.log 2>/dev/null || date +%s) )); echo \"st:\$m:\${r:-none}:\${t:-0}:\$a\"; done; for f in FAILED_*; do [ -e \"\$f\" ] && echo \"fail:\$f\"; done; [ -e CHAIN_FAILED.txt ] && echo chain:failed; grep -o '[^ ]* exit 0\$' rft_eval_queue_lab.log 2>/dev/null | cut -d' ' -f1 | sed 's/^/done:/'" 2>/dev/null)
  if [ -z "$lab" ]; then
    unreach=$((unreach + 1))
    [ $unreach -ge 3 ] && fire "lab_unreach_$(date +%Y%m%d%H)" "研究室に3回続けて繋がらない"
  else
    unreach=0
    for x in $lab; do
      case $x in
        st:*) IFS=: read -r _ m r t a <<< "$x"
              [ "$r" != running ] && [ "$r" != none ] && fire "l_${m}_result_$r" "研究室の $m の学習が $r"
              [ "${t:-0}" -gt 0 ] && fire "l_${m}_tb$t" "研究室の $m で Traceback $t 件"
              [ "${a:-0}" -gt 900 ] && [ "$r" = running ] && fire "l_${m}_silent_$((now / 3600))" "研究室の $m のログが $((a / 60)) 分止まっている";;
        fail:*) fire "l_${x#fail:}" "研究室の評価が失敗: ${x#fail:FAILED_}";;
        chain:*) fire "l_chain_failed" "研究室のいろいろな日の流れ（chain_multi）が失敗";;
        done:*) tag=${x#done:}
                if ! grep -qxF "l_eval_$tag" "$H/rft_eval_done.txt" 2>/dev/null; then
                  mkdir -p "$H/eval"; scp -o BatchMode=yes -q -r "smartgrid-gpu:$LAB/eval/*" "$H/eval/" 2>/dev/null
                  echo "l_eval_$tag" >> "$H/rft_eval_done.txt"
                fi
                case $tag in *500|*1000) fire "l_eval_$tag" "研究室の評価 $tag が済んだ（eval を Windows に写した）";; esac;;
      esac
    done
  fi
  if [ $((now - last_grad)) -gt 1800 ]; then
    last_grad=$now
    for m in $WMODES; do
      d=$(ls -d "$ROOT"/archive/realday_aemo6_${m}_2* 2>/dev/null | tail -1)
      [ -n "$d" ] || continue
      g=$(python "$ROOT/execute_results/wr_gs_6station_20261011/grad_check.py" "$(cygpath -w "$d")" --json 2>/dev/null | tail -1)
      [ -n "$g" ] || continue
      echo "$(date +%m-%dT%H:%M) $g" >> "$H/rft_grad_log.txt"
      ep=$(echo "$g" | python -c "import json,sys; print(json.load(sys.stdin).get('episode') or 0)" 2>/dev/null)
      off=$(echo "$g" | python -c "
import json,sys
g=json.load(sys.stdin); o=[]
if not 0.40<=g['global_ratio']<=0.80: o.append('ratio%.2f'%g['global_ratio'])
for k,ref in (('local_raw',11.46),('global_raw',25.94)):
    q=g[k]/ref
    if q>3 or q<1/3: o.append('%sx%.1f'%(k,q))
print(' '.join(o))" 2>/dev/null)
      [ "${ep:-0}" -ge 100 ] && [ -n "$off" ] && fire "w_${m}_grad_$ep" "Windows の $m の勾配が範囲外（ep$ep: $off）"
    done
  fi
  sleep 300
done
