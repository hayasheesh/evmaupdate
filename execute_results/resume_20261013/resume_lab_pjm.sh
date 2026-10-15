#!/bin/bash
# 10/13 14:45 に ep549 で止めた研究室の本線 PJM 6 station（resbid の設定）を、保存した学習状態から再開する（コンテナ内で bash resume_lab_pjm.sh）。
# STOP_REQUESTED と前の終了コードの記録は名前を変えて残す。
E=/workspace/EVMALOCALUPDATE/execute_results
PY=/workspace/.venvs/evma20/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
RUN=/workspace/EVMALOCALUPDATE/archive/wr_cap77wd_pjmplan_6station_resbid_tanh_ls717_20261013_021247
[ -f $RUN/resume/STOP_REQUESTED ] && mv $RUN/resume/STOP_REQUESTED $RUN/resume/STOP_REQUESTED.done_$STAMP
cd $E/resbid_6station_20261012
[ -f logs/pjm_pretrain.exit ] && mv logs/pjm_pretrain.exit logs/pjm_pretrain.exit.done_$STAMP
nohup bash -c "$PY -X utf8 -u stage_worker.py pjm pretrain --resume-run $RUN >> logs/pjm_pretrain.stdout.log 2>> logs/pjm_pretrain.stderr.log; echo \$? > logs/pjm_pretrain.exit" > /dev/null 2>&1 &
echo "pjm pid $!"
