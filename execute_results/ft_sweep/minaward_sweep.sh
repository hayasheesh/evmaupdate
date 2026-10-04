#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
MODEL=archive/prod_hobomatch_7station_20260918_040210
LOG=execute_results/ft_sweep/minaward_sweep.log
export EVMA_NUM_STATIONS=7 EVMA_EVAL_NO_BESS=1 EVMA_EVAL_PIPELINE=marl_force
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24 EVMA_ACTOR_EV_COUNT=1 EVMA_LOWER_BID_CONTEXT_OBS=1
echo "=== 最低約定スイープ 開始 $(date '+%F %T') ===" > "$LOG"
for A in 400 600 800; do
  echo "--- min_award=${A} 開始 $(date '+%F %T') ---" >> "$LOG"
  EVMA_EVAL_MIN_AWARD_KW="$A" python -u tools/evaluate_final_system_on_bid_bank.py \
    --model-dir "$MODEL" --episode 2000 --command-scenarios 6 --ev-seeds 1 \
    --output-dir "execute_results/ft_sweep/minaw${A}" >> "$LOG" 2>&1
  echo "--- min_award=${A} exit=$? $(date '+%F %T') ---" >> "$LOG"
done
echo "=== 完了 $(date '+%F %T') ===" >> "$LOG"
