#!/bin/bash
# 研究室の学習結果の写しを Windows の archive に置く。学習状態やチェックポイントは写さない。研究室の実行は読むだけで、止めたり消したりしない。
#   bash lab_mirror.sh '<研究室の archive の下の名前のパターン>' [写し先。既定は archive]
# 見張り（watch_gs6.sh）が30分ごとに今の20 stationの実行を写す。前の世代は写し先を archive/過去 にして一度だけ写す。
ROOT=/c/Users/admin/Desktop/EVMALOCALUPDATE
PATTERN=${1:-'wr_obsdrop_cap77wd_*_20station_gs025_*'}
DEST=${2:-$ROOT/archive}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
for LABRUN in $(timeout 60 $SSH smartgrid-gpu "for d in /home/hayashi/workspace/EVMALOCALUPDATE/archive/$PATTERN; do [ -d \"\$d\" ] && echo \"\$d\"; done"); do
  MIRROR=$DEST/$(basename "$LABRUN")
  mkdir -p $MIRROR/results $MIRROR/performance $MIRROR/runs
  for f in test_performance_metrics.png test_performance_metrics.csv train_performance_metrics.png train_performance_metrics.csv test_history.json episode_rewards_all.png episode_rewards_all.csv test_episode_rewards_all.png test_episode_rewards_all.csv; do
    timeout 300 $SCP "smartgrid-gpu:$LABRUN/results/$f" $MIRROR/results/ 2>/dev/null
  done
  for d in performance runs; do
    timeout 300 $SCP "smartgrid-gpu:$LABRUN/$d/events.out.tfevents.*" $MIRROR/$d/ 2>/dev/null
  done
  printf '%s\n' "研究室PCの実行の一部の写し（グラフ・CSV・テスト記録・TensorBoardの記録）。学習状態とチェックポイントは入っていない。" "元: smartgrid-gpu:$LABRUN" "更新: $(date '+%Y-%m-%d %H:%M')" > $MIRROR/LAB_MIRROR.txt
  echo "mirrored $(basename "$LABRUN") -> $DEST"
done
