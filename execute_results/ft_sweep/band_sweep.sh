#!/usr/bin/env bash
# 約定はそのまま、許容帯だけ広げる。完全分散で 99.8% に必要な帯幅を探る。
set -u
cd "$(dirname "$0")/../.."
MODEL=archive/prod_hobomatch_7station_20260918_040210
LOG=execute_results/ft_sweep/band_sweep.log
export EVMA_NUM_STATIONS=7 EVMA_EVAL_NO_BESS=1 EVMA_EVAL_PIPELINE=marl_force
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24 EVMA_ACTOR_EV_COUNT=1 EVMA_LOWER_BID_CONTEXT_OBS=1
echo "=== 帯幅スイープ 開始 $(date '+%F %T') ===" > "$LOG"
for F in 0.10 0.15 0.20 0.30 0.45; do
  TAG="band$(echo $F | tr -d '.')"
  echo "--- frac=${F} 開始 $(date '+%F %T') ---" >> "$LOG"
  EVMA_EVAL_BAND_FRACTION="$F" python -u tools/evaluate_final_system_on_bid_bank.py \
    --model-dir "$MODEL" --episode 2000 --command-scenarios 6 --ev-seeds 1 \
    --output-dir "execute_results/ft_sweep/${TAG}" >> "$LOG" 2>&1
  echo "--- frac=${F} exit=$? $(date '+%F %T') ---" >> "$LOG"
done
echo "=== 完了 $(date '+%F %T') ===" >> "$LOG"
