#!/bin/bash
# 研究室で、入札時の損の表（6 station）を回す。GPU は使わない（CUDA_VISIBLE_DEVICES=-1）。コンテナ内で bash run_lab.sh。
#   1. prep：5日分の128通りと40組の当日のEVを data/ に保存する（128通りが入札データの作成時と一致しなければ止まる）
#   2. calibrate：精度95%の k と85%の σ を決める（calib.json）
#   3. run：AEMO と ERCOT を並べて、それぞれ並列2で18通りを回す（入札1本が1.3〜2.7 GB。20 station の ABG を足した10/13 20時に空きメモリが2.4 GB まで減ったので4＋4から下げた）
# 終わったら DONE を置く。記録は logs/。
cd /workspace/EVMALOCALUPDATE/execute_results/bid_loss_6station_20261013
export CUDA_VISIBLE_DEVICES=-1
PY=/workspace/.venvs/evma20/bin/python
$PY -u loss_bid.py prep 10 >> logs/prep.log 2>&1 || exit 1
[ -f calib.json ] || $PY -u loss_bid.py calibrate >> logs/calibrate.log 2>&1 || exit 1
for t in aemo ercot; do
  $PY -u loss_bid.py run $t 2 >> logs/run_$t.log 2>&1 &
done
wait
touch DONE
