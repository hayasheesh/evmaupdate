#!/bin/bash
# 研究室で、6 station の残り（ERCOT の cmd、AEMO と ERCOT の ev）を回す。GPU は使わない（CUDA_VISIBLE_DEVICES=-1）。
# コンテナ内で bash run_lab.sh。並列数は合わせて10。終わったら DONE_lab を置く。
# 最初に、Windows で済んだ AEMO cmd の1本（日0・指令0）を解き直し、check_aemo_cmd.json に書く（両機械で入札が同じになるかの確認）。
cd /workspace/EVMALOCALUPDATE/execute_results/info_bid_6station_20261012
export CUDA_VISIBLE_DEVICES=-1
PY=/workspace/.venvs/evma20/bin/python
$PY -c "
import json, info_bid as ib
rows = ib.run_job(('aemo', 'cmd', 0, 0, 0))
json.dump(rows, open('check_aemo_cmd.json', 'w'), ensure_ascii=False)
" > logs/check.log 2>&1 &
$PY -u info_bid.py ercot cmd 4 > logs/ercot_cmd.log 2>&1 &
$PY -u info_bid.py aemo ev 3 > logs/aemo_ev.log 2>&1 &
$PY -u info_bid.py ercot ev 2 > logs/ercot_ev.log 2>&1 &
wait
touch DONE_lab
