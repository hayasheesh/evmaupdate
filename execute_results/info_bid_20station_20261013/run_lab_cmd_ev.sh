#!/bin/bash
# 研究室で、20 station の「当日の指令だけを知る入札」（cmd、日と指令ごとに20本）と「当日のEVだけを知る入札」（ev、40本）を、
# AEMO と ERCOT で回す。組は oracle と同じ40組。GPU は使わない。並列は市場ごとに3（20 station ABG 2本と同じ機械なのでメモリを残す）。
# 終わったら DONE_cmd_ev を置く。記録は logs/<task>_<mode>.log。
cd /workspace/EVMALOCALUPDATE/execute_results/info_bid_20station_20261013
export CUDA_VISIBLE_DEVICES=-1
PY=/workspace/.venvs/evma20/bin/python
for t in aemo ercot; do
  ( for m in cmd ev; do $PY -u info_bid.py $t $m 3 >> logs/${t}_${m}.log 2>&1; done ) &
done
wait
touch DONE_cmd_ev
