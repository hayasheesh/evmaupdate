#!/bin/bash
# 研究室で 20 station の ABG（AEMO・ERCOT）を始め、学習のフォルダができてから止める見張り（abg_stop.py）を起動する。コンテナ内で bash start_lab.sh。
cd /workspace/EVMALOCALUPDATE/execute_results/abg_resbid_20station_20261013
PY=/workspace/.venvs/evma20/bin/python
for t in aemo ercot; do
  nohup bash -c "$PY -X utf8 -u stage_worker.py $t pretrain >> logs/${t}_pretrain.stdout.log 2>> logs/${t}_pretrain.stderr.log; echo \$? > logs/${t}_pretrain.exit" > /dev/null 2>&1 &
  echo "$t pid $!"
done
for i in $(seq 60); do
  [ -n "$(ls -d ../../archive/wr_cap77wd_aemoplan_20station_resbid_abg_2* 2>/dev/null)" ] && [ -n "$(ls -d ../../archive/wr_cap77wd_ercotplan_20station_resbid_abg_2* 2>/dev/null)" ] && break
  sleep 10
done
nohup $PY -X utf8 -u abg_stop.py > logs/abg_stop.stdout.log 2>&1 &
echo "abg_stop pid $!"
