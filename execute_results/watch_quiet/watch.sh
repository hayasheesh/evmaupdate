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
  # Windows 6 station（本線と通常の MADDPG）
  for RUN6 in "$W6" "$ROOT/execute_results/maddpg_6station_20261012" "$ROOT/execute_results/abg_6station_20261012" "$ROOT/execute_results/resbid_6station_20261012"; do
  K=$(basename "$RUN6" | cut -c1-6)
  for t in aemo ercot; do
    res=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('result') or 'running')" "$RUN6/${t}_run_status.json" 2>/dev/null)
    [ -n "$res" ] && [ "$res" != running ] && fire "${K}_${t}_result_$res" "$K $t 学習の状態が $res"
    tb=$(cat "$RUN6"/logs/${t}_*.stderr.log 2>/dev/null | grep -c Traceback)
    [ "${tb:-0}" -gt 0 ] && fire "${K}_${t}_tb$tb" "$K $t Traceback $tb 件"
    log=$(ls -t "$RUN6"/logs/${t}_*.stdout.log 2>/dev/null | head -1)
    mt=$(date -r "$log" +%s 2>/dev/null || echo "$now")
    [ $((now - mt)) -gt 900 ] && [ "$res" = running ] && fire "${K}_${t}_silent_$mt" "$K $t ログが $(( (now - mt) / 60 )) 分止まっている"
  done
  done
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
  # 研究室 6 station AEMO（観測と報酬を戻した版、局所3.26倍）
  rb=$(timeout 60 $SSH smartgrid-gpu 'cd /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/resbid_6station_20261012/logs && echo "$(cat aemo_pretrain.exit 2>/dev/null)|$(grep -c Traceback aemo_pretrain.stderr.log 2>/dev/null)|$(( $(date +%s) - $(stat -c %Y aemo_pretrain.stdout.log 2>/dev/null || date +%s) ))"' 2>/dev/null)
  if [ -n "$rb" ]; then
    IFS='|' read -r rbx rbt rba <<< "$rb"
    [ -n "$rbx" ] && fire "rb_exit_$rbx" "研究室の 6 station AEMO（resbid）が終わった（終了コード $rbx）"
    [ "${rbt:-0}" -gt 0 ] && fire "rb_tb$rbt" "研究室の 6 station AEMO（resbid）で Traceback $rbt 件"
    [ -z "$rbx" ] && [ "${rba:-0}" -gt 900 ] && fire "rb_silent_$(date +%Y%m%d%H)" "研究室の 6 station AEMO（resbid）のログが $((rba / 60)) 分止まっている"
  fi
  # 6時間ごと: 公開リポジトリへ送る（変更があるときだけ）
  if [ $((now - last_pub)) -ge 21600 ]; then
    last_pub=$now
    msg=$(bash $HERE/publish.sh)
    [ -n "$msg" ] && fire "publish_fail_$(date +%Y%m%d%H)" "GitHub への送信に失敗: $msg"
  fi
  # 30分ごと: 研究室の写しと勾配の比
  if [ $((now - last_grad)) -ge 1800 ]; then
    last_grad=$now
    bash $W6/lab_mirror.sh 'wr_obsdrop_cap77wd_*_20station_gs025_*' >/dev/null 2>&1; bash $W6/lab_mirror.sh 'wr_cap77wd_*_6station_resbid_tanh_ls717_*' >/dev/null 2>&1
    for t in aemo ercot; do
      :
      cat $W6/grad_check.py | timeout 120 $SSH smartgrid-gpu "docker exec -i hayashi-gpu bash -lc 'd=\$(ls -dt /workspace/EVMALOCALUPDATE/archive/wr_obsdrop_cap77wd_${t}plan_20station_gs025_2* 2>/dev/null | head -1); [ -n \"\$d\" ] && cat > /tmp/grad_check.py && /workspace/.venvs/evma20/bin/python /tmp/grad_check.py \$d'" 2>/dev/null | grep " ep" | sed "s/^/$(date +%m%d_%H%M) lab /" >> "$GLOG"
    done
    # 学習の初め（200エピソード未満）は勾配がまだ小さく、大きさの比べようがないので知らせない。
    cat $W6/grad_check.py | timeout 120 $SSH smartgrid-gpu "docker exec -i hayashi-gpu bash -lc 'd=\$(ls -dt /workspace/EVMALOCALUPDATE/archive/wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_2* 2>/dev/null | head -1); [ -n \"\$d\" ] && cat > /tmp/grad_check.py && /workspace/.venvs/evma20/bin/python /tmp/grad_check.py \$d'" 2>/dev/null | grep " ep" | sed "s/^/$(date +%m%d_%H%M) lab /" >> "$GLOG"
    off=$(tail -5 "$GLOG" | grep OFF | awk '{for(i=1;i<=NF;i++) if($i ~ /^ep[0-9]+:$/){e=substr($i,3)+0; if(e>=200) print; break}}' | head -1)
    [ -n "$off" ] && fire "grad_off_$(echo "$off" | awk '{print $1 $2 $3}')" "勾配が範囲外: $off"
  fi
  [ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] || exit 0
  sleep 300
done
