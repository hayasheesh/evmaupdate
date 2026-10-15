#!/bin/bash
# 学習と評価の見張り。手を打つ必要が出たときだけ1行を出して終わる（Claude の Bash の run_in_background で動かす）。
# 何も起きない間は何も出さない。勾配の比は30分ごとに grad_log.txt へ書くだけで、範囲を外れたとき（OFF）だけ知らせる。
# 一度知らせた事柄は reported.txt に鍵を残し、起動し直しても同じ知らせは出さない。
#   bash watch.sh
# 見るもの:
#   Windows 6 station（wr_gs_6station_20261011）: Traceback の増加、ログが15分止まる、学習の終了
#   温度の見張り（gpu_guard.ps1）: 停止の印、記録の途絶、GPU 65 °C 以上
#   研究室 20 station（wr_gs_20station_*_20261011）: Traceback の増加、ログが15分止まる、段階の変化、空きメモリ 3 GB 未満、3回続けて繋がらない
#   蓄電池の大きさを振る評価（rule_bess_6station_20261011）: Traceback、終了
# 30分ごとに研究室の結果を archive へ写す（lab_mirror.sh）。6時間ごとに公開リポジトリへ送る（publish.sh、変更があるときだけ）。
ROOT=/c/Users/admin/Desktop/EVMALOCALUPDATE
HERE=$ROOT/execute_results/watch_quiet
W6=$ROOT/execute_results/wr_gs_6station_20261011
G=$ROOT/execute_results/hyb2_cap77wd_7station_20261009
BESS=$ROOT/execute_results/rule_bess_6station_20261011
REP=$HERE/reported.txt; touch "$REP"
# 同時に1つだけ動かす。前の見張りが残っていれば止めてから始める（Git Bash の ps は引数を出さないので PID を記録する）。
PIDF=$HERE/watch.pid
if [ -f "$PIDF" ]; then
  old=$(cat "$PIDF"); kill "$old" 2>/dev/null
  for c in $(ps -ef | awk -v p="$old" '$3==p {print $2}'); do kill "$c" 2>/dev/null; done
fi
echo $$ > "$PIDF"
GLOG=$HERE/grad_log.txt
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
unreach=0
last_grad=0
last_pub=$(date +%s)

fire() {  # $1 = 鍵, $2 = 知らせる文。まだ知らせていなければ出して終わる。
  grep -qxF "$1" "$REP" && return 0
  echo "$1" >> "$REP"
  echo "$(date +%H:%M) $2"
  exit 0
}

while true; do
  now=$(date +%s)
  # Windows 6 station の本線（ERCOT・PJM、10/13 20:55 に研究室から移した）。ログは logs/<task>_pretrain.win.*.log
  for RUN6 in "$ROOT/execute_results/resbid_6station_20261012"; do
  K=winrb
  for t in ercot pjm; do
    res=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('result') or 'running')" "$RUN6/${t}_run_status.json" 2>/dev/null)
    [ -n "$res" ] && [ "$res" != running ] && fire "${K}_${t}_result_$res" "$K $t 学習の状態が $res"
    tb=$(cat "$RUN6"/logs/${t}_pretrain.win.stderr.log 2>/dev/null | grep -c Traceback)
    [ "${tb:-0}" -gt 0 ] && fire "${K}_${t}_tb$tb" "$K $t Traceback $tb 件"
    log="$RUN6/logs/${t}_pretrain.win.stdout.log"
    # 学習が最後まで終わった実行（[done] の行がある）は、ログが止まっていても知らせない
    tail -c 2000 "$log" 2>/dev/null | grep -aq '^\[done\]' && continue
    mt=$(date -r "$log" +%s 2>/dev/null || echo "$now")
    [ $((now - mt)) -gt 900 ] && fire "${K}_${t}_silent_$mt" "$K $t ログが $(( (now - mt) / 60 )) 分止まっている"
  done
  done
  # PJM の規則の 6 station（pjmrule_6station_20261015、Windows）：入札2本のあと学習
  PR=$ROOT/execute_results/pjmrule_6station_20261015
  if [ -f "$PR/pjm_run_status.json" ]; then
    res=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('result') or ' '.join(k+':'+v['state'] for k,v in d['stages'].items()))" "$PR/pjm_run_status.json" 2>/dev/null)
    case "$res" in *failed*|complete) fire "pjmrule_$res" "PJM の規則の 6 station: $res";; esac
    case "$res" in *pretrain:running*) fire "pjmrule_pretrain_started" "PJM の規則の 6 station が学習に入った（勾配の比を確かめる）";; esac
    tb=$(cat "$PR"/logs/pjm_*.stderr.log 2>/dev/null | grep -c Traceback)
    [ "${tb:-0}" -gt 0 ] && fire "pjmrule_tb$tb" "PJM の規則の 6 station で Traceback $tb 件"
    # 最初の入札のLPが時間切れで打ち切られると、入札が計算機の混み具合で変わる
    grep -lq "keeping the Assessment-I seed\|LP stopped without a verdict" "$PR"/logs/pjm_*bank.stdout.log 2>/dev/null && fire "pjmrule_seed_fallback" "PJM の規則の入札で最初のLPが打ち切られた"
    case "$res" in *pretrain:running*)
      mt=$(date -r "$PR/logs/pjm_pretrain.stdout.log" +%s 2>/dev/null || echo "$now")
      [ $((now - mt)) -gt 900 ] && fire "pjmrule_silent_$mt" "PJM の規則の 6 station の学習のログが $(( (now - mt) / 60 )) 分止まっている";;
    esac
  fi
  # 順番待ちの仕事（queue_20261015）：失敗したとき、一覧が尽きたとき（次の仕事を足す）
  Q=$ROOT/execute_results/queue_20261015
  for f in "$Q"/FAILED_*; do [ -e "$f" ] && fire "q_win_${f##*/}" "Windows の順番待ちの仕事が失敗: ${f##*/FAILED_}"; done
  # 空になった知らせは、最後に終えた仕事（RUNNING_ の中身）ごとに一度だけ出す
  for f in "$Q"/IDLE_*; do [ -e "$f" ] || continue; qn=${f##*/IDLE_}; last=$(cut -d" " -f2 "$Q/RUNNING_$qn" 2>/dev/null); fire "q_idle_win_${qn}_${last:-none}" "Windows の順番待ち $qn が空になった"; done
  ql=$(timeout 60 $SSH smartgrid-gpu 'cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/queue_20261015 2>/dev/null && for f in FAILED_* IDLE_* DONE_sel_aemo_cnt6_build; do [ -e "$f" ] || continue; q=${f#IDLE_}; echo "$f:$(cut -d" " -f2 RUNNING_$q 2>/dev/null)"; done' 2>/dev/null)
  for x in $ql; do case $x in FAILED_*) fire "q_lab_${x%%:*}" "研究室の順番待ちの仕事が失敗: ${x%%:*}";; IDLE_*) fire "q_idle_lab_${x%%:*}_${x#*:}" "研究室の順番待ち ${x%%:*} が空になった";; DONE_sel_aemo_cnt6_build:*) fire "sel_aemo_cnt6_lab_done" "研究室で AEMO cnt6 の入札ができた（Windows に写して、Windows の cnt6 を止める。run.py の起動し直しは windows2 の未見の測定が尽きてから）";; esac; done
  # 研究室の GPU メモリ（10/15 23時に学習4本で 15.0/16.3 GB）：16,000 MiB を超えたら知らせる
  gm=$(timeout 60 $SSH smartgrid-gpu 'docker exec hayashi-gpu nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' 2>/dev/null | head -1 | tr -d ' ')
  [ "${gm:-0}" -gt 16000 ] && fire "lab_gpu_mem_$(date +%Y%m%d%H)" "研究室の GPU メモリが ${gm} MiB（16,303 MiB 中）"
  # PJM の入札・制御の損失分析（Windows の winpjmloss、研究室の lab4 の pjm20_info_*）：両方終わったら知らせる
  if [ -f "$Q/DONE_pjm6_info_ev" ] || [ -f "$Q/FAILED_pjm6_info_ev" ]; then
    l5=$(timeout 60 $SSH smartgrid-gpu 'cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/queue_20261015 && ls DONE_pjm20_info_ev FAILED_pjm20_info_ev 2>/dev/null' 2>/dev/null)
    [ -n "$l5" ] && fire "pjm_loss_done" "PJM の損失分析（6・20 station）が終わった（集計して結果と実験計画と下書きへ）"
  fi
  # 熱いあいだ Windows の順番待ちを一時停止する見張り（heat_guard.py）：止まった、30分以上一時停止が続く
  HG="$Q/heat_guard_state.json"
  hgm=$(date -r "$HG" +%s 2>/dev/null || echo 0)
  [ $((now - hgm)) -gt 180 ] && fire "heat_guard_silent_$hgm" "温度の見張り（heat_guard.py）の記録が $(( (now - hgm) / 60 )) 分止まっている"
  hp=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d['paused_pids']))" "$HG" 2>/dev/null)
  if [ "${hp:-0}" -gt 0 ]; then
    [ -f "$HERE/heat_paused_since" ] || echo "$now" > "$HERE/heat_paused_since"
    hs=$(cat "$HERE/heat_paused_since")
    [ $((now - hs)) -gt 1800 ] && fire "heat_paused_long_$hs" "熱で Windows の仕事が $(( (now - hs) / 60 )) 分一時停止したまま"
  else
    rm -f "$HERE/heat_paused_since"
  fi
  # 研究室の GPU の順番待ちで動いている学習（B0、PJM の規則の 20 station）：ログが30分止まる
  qs=$(timeout 60 $SSH smartgrid-gpu 'cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/queue_20261015 && for f in RUNNING_labgpu_*; do q=${f#RUNNING_}; [ -e IDLE_$q ] || [ -e WAITING_$q ] && continue; id=$(cut -d" " -f2 $f); [ -f DONE_$id ] || [ -f FAILED_$id ] && continue; echo "$id $(( $(date +%s) - $(stat -c %Y logs/$id.log 2>/dev/null || date +%s) ))"; done' 2>/dev/null)
  while read -r id age; do [ -n "$id" ] && [ "${age:-0}" -gt 1800 ] && fire "q_silent_${id}_$(date +%Y%m%d%H)" "研究室の $id のログが $((age / 60)) 分止まっている"; done <<< "$qs"
  # 研究室の lab4（20 station のルールベース＋蓄電池）は、B0 ERCOT の学習が始まってから起動する（同時に空きメモリを見て両方が始まらないように）
  if ! grep -qxF "lab4_started" "$REP"; then
    l4=$(timeout 60 $SSH smartgrid-gpu 'cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/queue_20261015 && [ -e DONE_b0_ercot_rehome ] && [ ! -e IDLE_labgpu_b0_ercot ] && [ ! -e WAITING_labgpu_b0_ercot ] && grep -q b0_ercot_pretrain RUNNING_labgpu_b0_ercot && [ RUNNING_labgpu_b0_ercot -nt DONE_b0_ercot_rehome ] && echo go' 2>/dev/null)
    if [ "$l4" = go ]; then
      timeout 60 $SSH smartgrid-gpu "docker exec -d hayashi-gpu bash -lc 'cd /workspace/EVMALOCALUPDATE/execute_results/queue_20261015 && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup /workspace/.venvs/evma20/bin/python -X utf8 -u queue.py lab4 >> logs/queue_lab4.log 2>&1'" && echo "lab4_started" >> "$REP"
    fi
  fi
  # 温度の見張り
  [ -f "$G/GPU_GUARD_STOPPED.txt" ] && fire "guard_stopped_$(date -r "$G/GPU_GUARD_STOPPED.txt" +%s)" "温度の見張りが学習を止めた"
  gm=$(date -r "$G/guard_temperature.csv" +%s 2>/dev/null || echo 0)
  [ $((now - gm)) -gt 600 ] && fire "guard_silent_$gm" "温度の記録が $(( (now - gm) / 60 )) 分止まっている"
  gt=$(tail -1 "$G/guard_temperature.csv" 2>/dev/null | cut -d, -f2 | cut -d. -f1)
  [ "${gt:-0}" -ge 65 ] && fire "gpu_hot_$(date +%Y%m%d%H)" "GPU ${gt} °C"
  # 研究室 20 station
  lab=$(timeout 90 $SSH smartgrid-gpu 'cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results; for t in aemo ercot; do d=wr_gs_20station_${t}_20261011; st=$(python3 -c "import json; print(json.load(open(\"$d/status.json\"))[\"stage\"])" 2>/dev/null); tb=$(cat $d/*.stderr.log 2>/dev/null | grep -c Traceback); n=$(ls -t $d/*.stdout.log 2>/dev/null | head -1); m=$(stat -c %Y $n 2>/dev/null || date +%s); echo "$t $st $tb $m"; done; echo "mem $(free -m | sed -n 2p | tr -s " " | cut -d" " -f7) $(date +%s)"' 2>/dev/null)
  if [ -z "$lab" ]; then
    unreach=$((unreach + 1))
    [ $unreach -ge 3 ] && fire "lab_unreachable_$(date +%Y%m%d%H)" "研究室に3回続けて繋がらない"
  else
    unreach=0
    lnow=$(echo "$lab" | awk '$1=="mem"{print $3}')
    mem=$(echo "$lab" | awk '$1=="mem"{print $2}')
    [ "${mem:-99999}" -lt 3000 ] && fire "lab_mem_$(date +%Y%m%d%H)" "研究室の空きメモリ ${mem} MB"
    while read -r t st tb m; do
      [ "$t" = mem ] && continue
      [ "$st" != pretrain ] && fire "lab_${t}_stage_$st" "研究室 20 station $t の段階が $st"
      [ "${tb:-0}" -gt 0 ] && fire "lab_${t}_tb$tb" "研究室 20 station $t Traceback $tb 件"
      [ "$st" = pretrain ] && [ $((lnow - m)) -gt 900 ] && fire "lab_${t}_silent_$m" "研究室 20 station $t ログが $(( (lnow - m) / 60 )) 分止まっている"
    done <<< "$lab"
  fi
  # 蓄電池の評価
  # EVの選び方・指令の本数・予測精度の表（selection_table_6station_20261015）：AEMO は Windows、ERCOT は研究室
  SEL=$ROOT/execute_results/selection_table_6station_20261015
  if [ -f "$SEL/status_aemo.json" ]; then
    grep -lq Traceback "$SEL"/logs/*aemo*.log 2>/dev/null && fire "sel_aemo_tb_$(ls "$SEL"/logs/*aemo*.log | xargs grep -l Traceback | head -1 | xargs basename)" "選び方の表 AEMO で Traceback"
    sst=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['stage'])" "$SEL/status_aemo.json" 2>/dev/null)
    case "$sst" in *failed*) fire "sel_aemo_$sst" "選び方の表 AEMO が止まった（$sst）";; done) fire "sel_aemo_done" "選び方の表 AEMO が終わった";; esac
    fb=$(grep -l "keeping the Assessment-I seed\|LP stopped without a verdict" "$SEL"/logs/build_aemo_*.log 2>/dev/null | head -1)
    [ -n "$fb" ] && fire "sel_aemo_seed_fallback_${fb##*/}" "選び方の表 AEMO で最初のLPが打ち切られた: ${fb##*/}"
    sl=$(timeout 60 $SSH smartgrid-gpu 'd=/home/hayashi/workspace/EVMALOCALUPDATE/execute_results/selection_table_6station_20261015; tb=$(grep -l Traceback $d/logs/*ercot*.log 2>/dev/null | head -1); st=$(python3 -c "import json; print(json.load(open(\"$d/status_ercot.json\"))[\"stage\"])" 2>/dev/null); echo "$st ${tb##*/}"' 2>/dev/null)
    set -- $sl
    [ -n "$2" ] && fire "sel_ercot_tb_$2" "選び方の表 ERCOT（研究室）で Traceback: $2"
    case "$1" in *failed*) fire "sel_ercot_$1" "選び方の表 ERCOT（研究室）が止まった（$1）";; done) fire "sel_ercot_done" "選び方の表 ERCOT（研究室）が終わった";; esac
  fi
  if [ -f "$BESS/aemo.err" ]; then
    grep -q Traceback "$BESS"/aemo.err "$BESS"/ercot.err 2>/dev/null && fire "bess_tb_$(date -r "$BESS/aemo.err" +%s)" "蓄電池の評価で Traceback"
    n=$(/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_rule_bess' }).Count" 2>/dev/null | tr -d '\r')
    [ "${n:-1}" = 0 ] && fire "bess_done_$(date -r "$BESS/aemo.log" +%s)" "蓄電池の評価が終わった"
  fi
  # 表3・図4（Windows 6 station）：Traceback、全部終わった、途中で親が消えた
  T3=$ROOT/execute_results/table3_6station_20261012
  tb=$(cat "$T3"/logs/*.err 2>/dev/null | grep -c Traceback)
  [ "${tb:-0}" -gt 0 ] && fire "t3w_tb$tb" "Windows の表3・図4で Traceback $tb 件"
  if [ -f "$T3/DONE_table3" ] && [ -f "$T3/DONE_aemo" ] && [ -f "$T3/DONE_ercot" ]; then
    fire "t3w_done" "Windows の表3・図4が終わった"
  else
    n=$(/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_windows.py' }).Count" 2>/dev/null | tr -d '\r')
    [ "${n:-1}" = 0 ] && fire "t3w_dead" "Windows の表3・図4の親（run_windows.py）が終わったのに DONE がそろっていない"
  fi
  # 表3・図4（研究室 20 station）
  t3l=$(timeout 60 $SSH smartgrid-gpu 'cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/table3_20station_20261012 && echo "$(cat *.err 2>/dev/null | grep -c Traceback) $(ls DONE_* 2>/dev/null | wc -l) $(ps -eo args | grep -c "[r]un_lab.sh")"' 2>/dev/null)
  if [ -n "$t3l" ]; then
    read -r tb nd alive <<< "$t3l"
    [ "${tb:-0}" -gt 0 ] && fire "t3l_tb$tb" "研究室の表3・図4で Traceback $tb 件"
    [ "${nd:-0}" -ge 3 ] && fire "t3l_done" "研究室の表3・図4が終わった"
    [ "${alive:-1}" = 0 ] && [ "${nd:-0}" -lt 3 ] && fire "t3l_dead" "研究室の表3・図4の run_lab.sh が終わったのに DONE がそろっていない"
  fi
  # PJM の入札（Windows 6 station と研究室 20 station）
  P6=$ROOT/execute_results/pjm_bank_6station_20261012
  [ -f "$P6/FAILED" ] && fire "pjm6_failed" "Windows の PJM 入札が失敗"
  [ -f "$P6/DONE" ] && fire "pjm6_done" "Windows の PJM 入札が終わった"
  tb=$(cat "$P6"/logs/*.stderr.log 2>/dev/null | grep -c Traceback)
  [ "${tb:-0}" -gt 0 ] && fire "pjm6_tb$tb" "Windows の PJM 入札で Traceback $tb 件"
  p20=$(timeout 60 $SSH smartgrid-gpu 'cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/pjm_bank_20station_20261012 && echo "$(ls DONE FAILED 2>/dev/null | tr "
" " ")|$(cat *.err 2>/dev/null | grep -c Traceback)"' 2>/dev/null)
  case "$p20" in *FAILED*) fire "pjm20_failed" "研究室の PJM 入札が失敗";; esac
  case "$p20" in DONE*) fire "pjm20_done" "研究室の PJM 入札が終わった";; esac
  [ "${p20##*|}" != "" ] && [ "${p20##*|}" != 0 ] && fire "pjm20_tb${p20##*|}" "研究室の PJM 入札で Traceback ${p20##*|} 件"
  # 研究室 50 station AEMO（入札→学習）
  w50=$(timeout 60 $SSH smartgrid-gpu 'cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/wr_gs_50station_aemo_20261012 && echo "$(cat status.txt 2>/dev/null)|$(cat *.stderr.log 2>/dev/null | grep -c Traceback)"' 2>/dev/null)
  case "$w50" in failed*) fire "w50_failed" "研究室の 50 station が失敗";; pretrain*) fire "w50_pretrain" "研究室の 50 station が学習に入った（勾配の比を確かめる）";; esac
  [ "${w50##*|}" != "" ] && [ "${w50##*|}" != 0 ] && fire "w50_tb${w50##*|}" "研究室の 50 station で Traceback ${w50##*|} 件"
  # 研究室 20 station の ABG（本線の設定、Q_MIX 1.0）：AEMO・ERCOT
  for t in aemo ercot; do
    ab=$(timeout 60 $SSH smartgrid-gpu "cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/abg_resbid_20station_20261013/logs && [ -f ${t}_pretrain.stdout.log ] && echo \"\$(cat ${t}_pretrain.exit 2>/dev/null)|\$(grep -c Traceback ${t}_pretrain.stderr.log 2>/dev/null)|\$(( \$(date +%s) - \$(stat -c %Y ${t}_pretrain.stdout.log 2>/dev/null || date +%s) ))\"" 2>/dev/null)
    [ -z "$ab" ] && continue
    IFS='|' read -r abx abt aba <<< "$ab"
    [ -n "$abx" ] && fire "ab20_${t}_exit_$abx" "研究室の 20 station ABG $t が終わった（終了コード $abx）"
    [ "${abt:-0}" -gt 0 ] && fire "ab20_${t}_tb$abt" "研究室の 20 station ABG $t で Traceback $abt 件"
    [ -z "$abx" ] && [ "${aba:-0}" -gt 1800 ] && fire "ab20_${t}_silent_$(date +%Y%m%d%H)" "研究室の 20 station ABG $t のログが $((aba / 60)) 分止まっている"
  done
  # 研究室 20 station の本線（resbid の設定）：AEMO・ERCOT・PJM（PJM は 10/15 開始、報酬の倍率は大域0.4・局所1.5倍）
  for t in aemo ercot pjm; do
    ab=$(timeout 60 $SSH smartgrid-gpu "cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/resbid_20station_20261014/logs && [ -f ${t}_pretrain.stdout.log ] && echo \"\$(cat ${t}_pretrain.exit 2>/dev/null)|\$(grep -c Traceback ${t}_pretrain.stderr.log 2>/dev/null)|\$(( \$(date +%s) - \$(stat -c %Y ${t}_pretrain.stdout.log 2>/dev/null || date +%s) ))\"" 2>/dev/null)
    [ -z "$ab" ] && continue
    IFS='|' read -r abx abt aba <<< "$ab"
    [ -n "$abx" ] && fire "rb20_${t}_exit_$abx" "研究室の 20 station 本線 $t が終わった（終了コード $abx）"
    [ "${abt:-0}" -gt 0 ] && fire "rb20_${t}_tb$abt" "研究室の 20 station 本線 $t で Traceback $abt 件"
    [ -z "$abx" ] && [ "${aba:-0}" -gt 1800 ] && fire "rb20_${t}_silent_$(date +%Y%m%d%H)" "研究室の 20 station 本線 $t のログが $((aba / 60)) 分止まっている"
  done
  # 6時間ごと: 公開リポジトリへ送る（変更があるときだけ）
  if [ $((now - last_pub)) -ge 21600 ]; then
    last_pub=$now
    msg=$(bash $HERE/publish.sh)
    [ -n "$msg" ] && fire "publish_fail_$(date +%Y%m%d%H)" "GitHub への送信に失敗: $msg"
  fi
  # 30分ごと: 研究室の写しと勾配の比
  if [ $((now - last_grad)) -ge 1800 ]; then
    last_grad=$now
    bash $HERE/lab_full_mirror.sh 'wr_cap77wd_*plan_20station_resbid_tanh_ls717_2*' >/dev/null 2>&1
    bash $HERE/lab_full_mirror.sh 'wr_cap77wd_pjmplan_20station_resbid_tanh_g04l15_2*' >/dev/null 2>&1
    bash $HERE/lab_full_mirror.sh 'wr_cap77wd_*plan_6station_b0_tanh_ls717_2*' 'wr_cap77wd_pjmplan_20station_pjmrule75_g04l15_2*' >/dev/null 2>&1
    # 研究室：B0（6 station AEMO・ERCOT）と PJM の規則の 20 station
    for pat in wr_cap77wd_aemoplan_6station_b0_tanh_ls717_2 wr_cap77wd_ercotplan_6station_b0_tanh_ls717_2 wr_cap77wd_pjmplan_20station_pjmrule75_g04l15_2; do
      cat $W6/grad_check.py | timeout 120 $SSH smartgrid-gpu "docker exec -i hayashi-gpu bash -lc 'd=\$(ls -dt /workspace/EVMALOCALUPDATE/archive/${pat}* 2>/dev/null | head -1); [ -n \"\$d\" ] && cat > /tmp/grad_check.py && /workspace/.venvs/evma20/bin/python /tmp/grad_check.py \$d'" 2>/dev/null | grep " ep" | sed "s/^/$(date +%m%d_%H%M) lab /" >> "$GLOG"
    done
    # 学習の初め（200エピソード未満）は勾配がまだ小さく、大きさの比べようがないので知らせない。
    for t in aemo ercot pjm; do
      name=resbid_tanh_ls717; [ $t = pjm ] && name=resbid_tanh_g04l15
      cat $W6/grad_check.py | timeout 120 $SSH smartgrid-gpu "docker exec -i hayashi-gpu bash -lc 'd=\$(ls -dt /workspace/EVMALOCALUPDATE/archive/wr_cap77wd_${t}plan_20station_${name}_2* 2>/dev/null | head -1); [ -n \"\$d\" ] && cat > /tmp/grad_check.py && /workspace/.venvs/evma20/bin/python /tmp/grad_check.py \$d'" 2>/dev/null | grep " ep" | sed "s/^/$(date +%m%d_%H%M) lab /" >> "$GLOG"
    done
    # Windows：PJM の規則の 6 station
    d=$(ls -dt "$ROOT"/archive/wr_cap77wd_pjmplan_6station_pjmrule75_g04l15_2* 2>/dev/null | head -1)
    [ -n "$d" ] && timeout 120 python "$W6/grad_check.py" "$d" 2>/dev/null | grep " ep" | sed "s/^/$(date +%m%d_%H%M) win /" >> "$GLOG"
    off=$(tail -10 "$GLOG" | grep OFF | awk '{for(i=1;i<=NF;i++) if($i ~ /^ep[0-9]+:$/){e=substr($i,3)+0; if(e>=200) print; break}}' | head -1)
    # 同じ実行で同じ種類の範囲外（ratio、globalx など）は一度だけ知らせる。種類が増えたらまた知らせる。
    [ -n "$off" ] && fire "grad_off_$(echo "$off" | awk '{k=$NF; sub(/^OFF:/, "", k); gsub(/[0-9.]/, "", k); print $3 "_" k}')" "勾配が範囲外: $off"
  fi
  [ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] || exit 0
  sleep 300
done
