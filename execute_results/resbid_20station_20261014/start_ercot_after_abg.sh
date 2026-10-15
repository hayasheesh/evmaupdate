#!/bin/bash
# 20 station 本線 ERCOT は、10/14 08:51 に GPU のメモリ不足（20 station ABG AEMO・本線 AEMO・入札と同時）で学習の最初に落ちた。
# 20 station ABG AEMO が終わる（logs/aemo_pretrain.exit ができる）のを待ってから、一から始め直す。コンテナ内で nohup bash start_ercot_after_abg.sh &。
cd /workspace/EVMALOCALUPDATE/execute_results/resbid_20station_20261014
PY=/workspace/.venvs/evma20/bin/python
until [ -f ../abg_resbid_20station_20261013/logs/aemo_pretrain.exit ]; do sleep 300; done
sleep 120
S=$(date +%Y%m%d_%H%M%S)
for f in ercot_pretrain.stdout.log ercot_pretrain.stderr.log ercot_pretrain.exit; do [ -f logs/$f ] && mv logs/$f logs/$f.oom_$S; done
$PY -X utf8 -u stage_worker.py ercot pretrain >> logs/ercot_pretrain.stdout.log 2>> logs/ercot_pretrain.stderr.log
echo $? > logs/ercot_pretrain.exit
