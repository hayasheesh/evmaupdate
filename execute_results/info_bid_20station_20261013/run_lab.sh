#!/bin/bash
# 研究室で、20 station の「指令とEVを知る入札」（oracle、AEMO と ERCOT の各40組）を回す。GPU は使わない（CUDA_VISIBLE_DEVICES=-1）。
#   コンテナ内で bash run_lab.sh。並列数は市場ごとに4（研究室は20コア。学習2本と PJM 20 station の入札が動いている）。
#   終わったら DONE_oracle を置く。記録は logs/<task>_oracle.log。
cd /workspace/EVMALOCALUPDATE/execute_results/info_bid_20station_20261013
export CUDA_VISIBLE_DEVICES=-1
PY=/workspace/.venvs/evma20/bin/python
for t in aemo ercot; do
  $PY -u info_bid.py $t oracle 4 >> logs/${t}_oracle.log 2>&1 &
done
wait
touch DONE_oracle
