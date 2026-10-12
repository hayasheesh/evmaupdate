#!/bin/bash
# 研究室 20 station の学習結果の写しを archive/<同じ実行名>/ に置く。学習状態やチェックポイントは写さない。
# 研究室の実行は読むだけで、止めたり消したりしない。
ROOT=/c/Users/admin/Desktop/EVMALOCALUPDATE
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
for LABRUN in $(timeout 60 $SSH smartgrid-gpu 'for d in /home/hayashi/workspace/EVMALOCALUPDATE/archive/low2max_*_AB_20station_* /home/hayashi/workspace/EVMALOCALUPDATE/archive/obs2_*_AB_7station_* /home/hayashi/workspace/EVMALOCALUPDATE/archive/allin_*_AB_20station_*; do [ -d "$d" ] && echo "$d"; done'); do
  MIRROR=$ROOT/archive/$(basename "$LABRUN")
  mkdir -p $MIRROR/results $MIRROR/performance $MIRROR/runs
  for f in test_performance_metrics.png test_performance_metrics.csv train_performance_metrics.png train_performance_metrics.csv test_history.json episode_rewards_all.png episode_rewards_all.csv test_episode_rewards_all.png test_episode_rewards_all.csv; do
    timeout 300 $SCP "smartgrid-gpu:$LABRUN/results/$f" $MIRROR/results/ 2>/dev/null
  done
  for d in performance runs; do
    timeout 300 $SCP "smartgrid-gpu:$LABRUN/$d/events.out.tfevents.*" $MIRROR/$d/ 2>/dev/null
  done
  printf '%s\n' "研究室PCの実行の一部の写し（グラフ・CSV・テスト記録・TensorBoardの記録）。学習状態とチェックポイントは入っていない。" "元: smartgrid-gpu:$LABRUN" "更新: $(date '+%Y-%m-%d %H:%M')" > $MIRROR/LAB_MIRROR.txt
done
