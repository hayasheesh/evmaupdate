#!/bin/bash
# Windows と研究室PCの実行状態をまとめて読む。読むだけで、何も起動・停止しない。
ROOT=/c/Users/admin/Desktop/EVMALOCALUPDATE
E=$ROOT/execute_results
PY=/c/Users/admin/AppData/Local/Programs/Python/Python310/python.exe
W=$E/watch_20261002
age() { echo $(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || echo 0) ))s; }
alive() { tasklist //FI "PID eq $1" 2>/dev/null | grep -q " $1 " && echo alive || echo gone; }
pid_of() { sed -n 's/.*"pid":  *\([0-9]*\).*/\1/p' "$1" 2>/dev/null | head -1; }

echo "== windows $(date '+%m-%d %H:%M')"
ERCOT=$ROOT/archive/prod_ercotplan_AB_7station_20260930_221551
ER=$E/ercot_ab_resume_noboost_2_20261002
echo "ERCOT learner $(alive $(pid_of $ER/status.json)), launcher $(alive 43124), $(grep -o '"stage": *"[a-z]*"' $ER/status.json), stdout age $(age $ER/pretrain.stdout.log), stop file $( [ -f $ERCOT/resume/STOP_REQUESTED ] && echo present || echo none)"
echo "boost AC/DC: $(powercfg //query SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 2>/dev/null | grep -o '0x[0-9a-f]*$' | tail -2 | tr '
' ' ')"
grep -E '^train[0-9]+' $ER/pretrain.stdout.log | tail -1 | cut -c1-120
PYTHONUTF8=1 $PY $W/health.py $ERCOT --last-windows 2 2>/dev/null | tail -n +2
grep -m2 Traceback $ER/pretrain.stderr.log

U=$E/marl_unseen_aemo_ab2200_20261002
echo "AEMO unseen MARL eval: $(grep -o '"stage": *"[a-z]*"' $U/status.json), day dirs $(ls -d $U/results/day_* 2>/dev/null | wc -l)/5"
[ -f $U/summary_checked.json ] && PYTHONUTF8=1 $PY -c "
import json; s=json.load(open(r'$(cygpath -w $U/summary_checked.json)',encoding='utf-8'))
print({k: s[k] for k in ('excluded_training_commands','rollouts','tracking_pct_weighted','soc_pct_weighted','tracking_plus_soc','rollouts_without_miss')})"
grep -m2 Traceback $U/evaluate.stderr.log

GPULOG=$W/gpu_log.csv
if [ -f $GPULOG ]; then
  echo "GPU last: $(tail -1 $GPULOG)"
  echo "GPU last 30 min: max temp $(tail -60 $GPULOG | cut -d, -f2 | grep -E '^[0-9]+$' | sort -n | tail -1)C, max power $(tail -60 $GPULOG | cut -d, -f5 | grep -E '^[0-9.]+$' | sort -n | tail -1)W, logger rows $(($(wc -l < $GPULOG) - 1))"
fi
G=$E/gb_128cmd_marl_20261002
echo "GB resume: $(grep -o '"stage": *"[a-z_]*"' $G/resume_status.json 2>/dev/null) error=$(grep -o '"error": *[^,]*' $G/resume_status.json 2>/dev/null | cut -c1-120), learner $(alive $(pid_of $G/resume_status.json)), launcher $(alive $(sed -n 's/.*"launcherPid": *\([0-9]*\).*/\1/p' $G/resume_status.json 2>/dev/null | head -1)), stdout age $(age $G/resume.stdout.log)"
[ -f $G/wait_status.json ] && echo "GB waiter (利用者のGPUの処理の終わりを待って再開): $(grep -o '"stage": *"[a-z_]*"' $G/wait_status.json), resumeFrom $(grep -o '"resumeFromEpisode": *[0-9]*' $G/wait_status.json | grep -o '[0-9]*$'), waiter $(alive $(sed -n 's/.*"launcherPid": *\([0-9]*\).*/\1/p' $G/wait_status.json | head -1)), user job 39436 $(alive 39436), error=$(grep -o '"error": *[^,]*' $G/wait_status.json | cut -c1-120)"
echo "GPU log age $(age $GPULOG) (30秒ごとに書く。数分を超えたら記録用プロセスが止まっている)"
for s in bid_train bid_test; do
  [ -f $G/$s.stdout.log ] && echo "  $s: completed $(grep -c 'completed index' $G/$s.stdout.log), failed $(grep -ci 'failed' $G/$s.stdout.log), log age $(age $G/$s.stdout.log)"
done
GBRUN=$(ls -d $ROOT/archive/prod_elexonplan_AB_7station_* 2>/dev/null | tail -1)
if [ -n "$GBRUN" ]; then
  cat $G/pretrain.stdout.log $G/resume.stdout.log 2>/dev/null | grep -E '^train[0-9]+' | tail -1 | cut -c1-120
  PYTHONUTF8=1 $PY $W/health.py $GBRUN --last-windows 6 2>/dev/null | tail -n +2
fi
for f in $G/*.stderr.log; do [ -s "$f" ] && grep -q Traceback "$f" && echo "[traceback] $f"; done

"/c/Windows/System32/OpenSSH/ssh.exe" -o BatchMode=yes -o ConnectTimeout=20 smartgrid-gpu \
  'bash /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/remote_20station_bid_unseen_20261002/lab_status.sh' 2>&1 || echo "[lab] ssh failed"

# 研究室 20 station AB の結果の写しを archive/<同じ実行名>/ に置く（グラフ・CSV・テスト記録・TensorBoard の記録）。
# 学習状態やチェックポイントは写さない。研究室側は読むだけ。
# ABORTED.txt のある実行（本線でないもの）は写さない。
for LABRUN in $("/c/Windows/System32/OpenSSH/ssh.exe" -o BatchMode=yes -o ConnectTimeout=20 smartgrid-gpu 'for d in /home/hayashi/workspace/EVMALOCALUPDATE/archive/prod_aemoplan128_AB_20station_5080_scaledreward_* /home/hayashi/workspace/EVMALOCALUPDATE/archive/prod_ercotplan128_AB_20station_5080_scaledreward_*; do [ -d "$d" ] && [ ! -f "$d/ABORTED.txt" ] && echo "$d"; done'); do
  MIRROR=$ROOT/archive/$(basename "$LABRUN")
  mkdir -p $MIRROR/results $MIRROR/performance $MIRROR/runs
  for f in test_performance_metrics.png test_performance_metrics_legend.png test_performance_metrics.csv train_performance_metrics.png train_performance_metrics_legend.png train_performance_metrics.csv test_history.json episode_rewards_all.png episode_rewards_all.csv test_episode_rewards_all.png test_episode_rewards_all.csv; do
    "/c/Windows/System32/OpenSSH/scp.exe" -q -o BatchMode=yes -o ConnectTimeout=20 "smartgrid-gpu:$LABRUN/results/$f" $MIRROR/results/ 2>/dev/null
  done
  for d in performance runs; do
    "/c/Windows/System32/OpenSSH/scp.exe" -q -o BatchMode=yes -o ConnectTimeout=20 "smartgrid-gpu:$LABRUN/$d/events.out.tfevents.*" $MIRROR/$d/ 2>/dev/null
  done
  printf '%s
' "研究室PCの実行の一部の写し（グラフ・CSV・テスト記録・TensorBoardの記録）。学習状態とチェックポイントは入っていない。" "元: smartgrid-gpu:$LABRUN" "更新: $(date '+%Y-%m-%d %H:%M')" > $MIRROR/LAB_MIRROR.txt
  echo "lab mirror updated $(date '+%m-%d %H:%M') -> $MIRROR"
done
