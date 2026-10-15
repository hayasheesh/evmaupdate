#!/bin/bash
# 2026-10-13 01:15 に止めた研究室の計算を起動し直す（コンテナ内で bash resume_lab.sh）。
#  1. resbid 6 station AEMO の学習を、保存した学習状態から再開する（STOP_REQUESTED と前の終了コードの記録は名前を変えて残す）
#  2. PJM 20 station の入札（run.sh）をもう一度起動する。済んだ日は飛ばして続きから作る
#  3. 50 station AEMO（run.sh）をもう一度起動する。PJM の入札が DONE になるのを待ってから入札と学習に進む
E=/workspace/EVMALOCALUPDATE/execute_results
PY=/workspace/.venvs/evma20/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
RUN=/workspace/EVMALOCALUPDATE/archive/wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_20261012_102306
[ -f $RUN/resume/STOP_REQUESTED ] && mv $RUN/resume/STOP_REQUESTED $RUN/resume/STOP_REQUESTED.done_$STAMP
cd $E/resbid_6station_20261012
[ -f logs/aemo_pretrain.exit ] && mv logs/aemo_pretrain.exit logs/aemo_pretrain.exit.done_$STAMP
nohup bash -c "$PY -X utf8 -u stage_worker.py aemo pretrain --resume-run $RUN >> logs/aemo_pretrain.stdout.log 2>> logs/aemo_pretrain.stderr.log; echo \$? > logs/aemo_pretrain.exit" > /dev/null 2>&1 &
echo "resbid pid $!"
cd $E/pjm_bank_20station_20261012 && nohup bash run.sh > /dev/null 2>&1 &
echo "pjm bank pid $!"
cd $E/wr_gs_50station_aemo_20261012 && nohup bash run.sh > /dev/null 2>&1 &
echo "50 station pid $!"
