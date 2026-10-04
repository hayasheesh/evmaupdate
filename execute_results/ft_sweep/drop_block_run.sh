#!/usr/bin/env bash
# 落としたブロックが残りのブロックを楽にするか。実際に外して測り直す。
set -u
cd "$(dirname "$0")/../.."
MODEL=archive/prod_hobomatch_7station_20260918_040210
LOG=execute_results/ft_sweep/drop_block_run.log

export EVMA_NUM_STATIONS=7
export EVMA_EVAL_NO_BESS=1
export EVMA_EVAL_PIPELINE=marl_force
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_CONTEXT_OBS=1

echo "=== ブロック除外 実測 開始 $(date '+%F %T') ===" > "$LOG"
for TAG in thr5 thr4; do
  echo "--- ${TAG} 開始 $(date '+%F %T') ---" >> "$LOG"
  EVMA_EVAL_DROP_BLOCKS_JSON="execute_results/ft_sweep/drop_${TAG}.json" \
  python -u tools/evaluate_final_system_on_bid_bank.py \
    --model-dir "$MODEL" --episode 2000 \
    --command-scenarios 6 --ev-seeds 1 \
    --output-dir "execute_results/ft_sweep/drop_${TAG}" >> "$LOG" 2>&1
  echo "--- ${TAG} exit=$? $(date '+%F %T') ---" >> "$LOG"
done
echo "=== 完了 $(date '+%F %T') ===" >> "$LOG"
