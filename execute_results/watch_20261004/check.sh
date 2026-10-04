#!/bin/bash
# 作り直し（EV を各局の実セッションから作る版）の状態を、Windows と研究室PCでまとめて読む。
# 読むだけで、何も起動・停止しない。研究室の学習結果の一部（グラフ・CSV・テスト記録・TensorBoard）は archive/ に写す。
ROOT=/c/Users/admin/Desktop/EVMALOCALUPDATE
E=$ROOT/execute_results
PY=/c/Users/admin/AppData/Local/Programs/Python/Python310/python.exe
HEALTH=$E/watch_20261002/health.py
O=$E/sessions_7station_20261004
SSH="/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=20"
SCP="/c/Windows/System32/OpenSSH/scp.exe -q -o BatchMode=yes -o ConnectTimeout=20"
age() { echo $(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || echo 0) ))s; }
alive() { tasklist //FI "PID eq $1" 2>/dev/null | grep -q " $1 " && echo alive || echo gone; }

echo "== windows $(date '+%m-%d %H:%M')"
echo "boost AC/DC: $(powercfg //query SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 2>/dev/null | grep -o '0x[0-9a-f]*$' | tail -2 | tr '\n' ' ')"
echo "GPU temp,util,mem,power: $(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null)"
ORCH=$(sed -n 's/.*"orchestrator_pid": *\([0-9]*\).*/\1/p' $O/status.json | head -1)
echo "orchestrator $(alive $ORCH), status age $(age $O/status.json)"
PYTHONUTF8=1 $PY - "$(cygpath -w $O/status.json)" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding='utf-8'))
print('stage', s.get('stage'), s.get('error') or '')
print('banks    ', {k: v['state'] + (' ' + str(v.get('exit_codes')) if v['state'] == 'failed' else '') for k, v in s['banks'].items()})
print('trainings', {k: v['state'] + (' ' + str(v.get('exit_codes')) if v['state'] == 'failed' else '') for k, v in s['trainings'].items()})
EOF
for set in aemo_plan_deviation ercot_plan_deviation elexon_plan_deviation pjm_regd_phase_shift; do
  t=$(ls $E/bid_banks/sessions_train_25_7station_128cmd_3ev_$set/days/*/fixed_bid.pkl 2>/dev/null | wc -l)
  v=$(ls $E/bid_banks/sessions_validation_5_7station_128cmd_3ev_$set/days/*/fixed_bid.pkl 2>/dev/null | wc -l)
  echo "  bank $set: train $t/25, validation $v/5"
done
for f in $O/logs/*.stderr.log; do [ -s "$f" ] && grep -q Traceback "$f" && echo "[traceback] $f"; done
for task in aemo ercot gb pjm ercot_maddpg; do
  LOG=$O/logs/${task}_pretrain.stdout.log
  [ -f $LOG ] || continue
  case $task in
    aemo) M=sessions_aemoplan_AB_7station;; ercot) M=sessions_ercotplan_AB_7station;;
    gb) M=sessions_elexonplan_AB_7station;; pjm) M=sessions_pjmregd_AB_7station;;
    ercot_maddpg) M=sessions_ercotplan_MADDPGstd_7station;;
  esac
  RUN=$(ls -d $ROOT/archive/${M}_* 2>/dev/null | tail -1)
  echo "-- $task: $(basename "$RUN"), stdout age $(age $LOG)"
  grep -E '^train[0-9]+' $LOG | tail -1 | cut -c1-150
  grep -o 'TEST[0-9]* summary.*MAE=[0-9.]*' $LOG | tail -1 | cut -c1-200
  [ -n "$RUN" ] && PYTHONUTF8=1 $PY $HEALTH $RUN --last-windows 3 2>/dev/null | tail -n +2
done

$SSH smartgrid-gpu 'bash /home/hayashi/workspace/EVMALOCALUPDATE/execute_results/sessions_watch_20261004/lab_status.sh' 2>&1 || echo "[lab] ssh failed"

# 研究室 20 station の学習結果の写しを archive/<同じ実行名>/ に置く。学習状態やチェックポイントは写さない。
for LABRUN in $($SSH smartgrid-gpu 'for d in /home/hayashi/workspace/EVMALOCALUPDATE/archive/sessions_*_AB_20station_*; do [ -d "$d" ] && echo "$d"; done'); do
  MIRROR=$ROOT/archive/$(basename "$LABRUN")
  mkdir -p $MIRROR/results $MIRROR/performance $MIRROR/runs
  for f in test_performance_metrics.png test_performance_metrics.csv train_performance_metrics.png train_performance_metrics.csv test_history.json episode_rewards_all.png episode_rewards_all.csv test_episode_rewards_all.png test_episode_rewards_all.csv; do
    $SCP "smartgrid-gpu:$LABRUN/results/$f" $MIRROR/results/ 2>/dev/null
  done
  for d in performance runs; do
    $SCP "smartgrid-gpu:$LABRUN/$d/events.out.tfevents.*" $MIRROR/$d/ 2>/dev/null
  done
  printf '%s\n' "研究室PCの実行の一部の写し（グラフ・CSV・テスト記録・TensorBoardの記録）。学習状態とチェックポイントは入っていない。" "元: smartgrid-gpu:$LABRUN" "更新: $(date '+%Y-%m-%d %H:%M')" > $MIRROR/LAB_MIRROR.txt
  echo "lab mirror updated -> $(basename $MIRROR)"
done
