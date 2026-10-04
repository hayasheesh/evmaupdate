#!/bin/bash
# 研究室PCの状態を読むだけのスクリプト。何も起動・停止しない。
W=/home/hayashi/workspace/EVMALOCALUPDATE
R=$W/execute_results
age() { echo $(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || echo 0) ))s; }
echo "== lab $(date '+%m-%d %H:%M') load $(cut -d' ' -f1-3 /proc/loadavg) memavail $(awk '/MemAvailable/{printf "%.1fG", $2/1048576}' /proc/meminfo) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
echo "container GPU visible: $(docker exec -u 1007:1007 hayashi-gpu /workspace/.venvs/evma20/bin/python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | tail -1)"
# 20 station、設計指令128本 × EV 3本（入札 → GPU の空きを待って AB 学習）
for M in aemo ercot; do
  P=$R/remote_5080_20station_${M}128_20261002
  [ -f $P/status.json ] || continue
  python3 -c "import json; s=json.load(open('$P/status.json')); print('${M}128 pipeline: stage=' + str(s.get('stage')) + ' error=' + str(s.get('error')))"
  for b in train_25_20station_128cmd_3ev_${M}_plan_deviation validation_5_20station_128cmd_3ev_${M}_plan_deviation; do
    echo "  $b: fixed_bid days $(ls $R/bid_banks/$b/days/*/fixed_bid.pkl 2>/dev/null | wc -l)"
  done
  for f in $P/[!f]*.stderr.log; do [ -s "$f" ] && grep -q Traceback "$f" && echo "[traceback] $f"; done
  RUN=$(ls -d $W/archive/prod_${M}plan128_AB_20station_5080_scaledreward_* 2>/dev/null | tail -1)
  if [ -n "$RUN" ]; then
    echo "  pretrain run $(basename $RUN), stdout age $(age $P/pretrain.stdout.log)"
    grep -E '^train[0-9]+' $P/pretrain.stdout.log | tail -1 | cut -c1-160
    docker exec -u 1007:1007 -e HOME=/home/hayashi -w /workspace/EVMALOCALUPDATE hayashi-gpu /workspace/.venvs/evma20/bin/python \
      execute_results/remote_20station_bid_unseen_20261002/health.py "/workspace/EVMALOCALUPDATE/archive/$(basename $RUN)" --last-windows 6 2>/dev/null
  fi
done
exit 0
