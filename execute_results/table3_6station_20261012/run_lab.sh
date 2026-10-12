#!/bin/bash
# 研究室の 20 station：表3（lp・rule・marl_force・marl、各40組）と、図4の入札の変種（検証用5日）と未見通過率を順に回す。
#   コンテナ内で bash run_lab.sh。終わったら DONE_table3 / DONE_<task> を置く。
cd /workspace/EVMALOCALUPDATE/execute_results/table3_20station_20261012
export TABLE3_SW_DIR=/workspace/EVMALOCALUPDATE/execute_results/wr_gs_20station_{task}_20261011
PY=/workspace/.venvs/evma20/bin/python
( for m in rule marl_force marl lp; do for t in aemo ercot; do $PY run_table3.py $t $m 4 2 4 >> table3.log 2>> table3.err; done; done; touch DONE_table3 ) &
for t in aemo ercot; do
  (
    $PY build_variant_bank.py $t session_count 128 1 3 >> bank_$t.log 2>> bank_$t.err
    $PY build_variant_bank.py $t low_connection_2_max_count 256 1 3 >> bank_$t.log 2>> bank_$t.err
    base=wr_cap77wd_min250_margin25_validation_5_20station_128cmd_3ev_${t}_plan_deviation
    $PY run_unseen_variants.py $t unseen_$t 4 $base ${base/_128cmd_3ev_/_128cmd_minmedmax_} ${base/_128cmd_3ev_/_256cmd_low2max_} >> unseen_$t.log 2>> unseen_$t.err
    touch DONE_$t
  ) &
done
wait
