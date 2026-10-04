#!/usr/bin/env bash
# いまの run の ep1560 を force のみ / force+BESS で測る。
# 学習時と同じ観測レイアウト・同じ指令ライブラリ・同じ検証バンクに揃える。
set -u
cd "$(dirname "$0")/../.."
MODEL=archive/prod_f500_dense8_noalloc_7station_20260921_102504
BANK=execute_results/bid_banks/validation_5_7s_f500_dense8
EP=1560
LOG=execute_results/ft_sweep/eval_force_bess.log

export EVMA_NUM_STATIONS=7
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24
export EVMA_LOWER_BID_CONTEXT_OBS=1
export EVMA_ACTIVATION_SCENARIO_DIR=data/aemo/nem/processed_5min/dense8
export EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW=500

echo "=== 開始 $(date '+%F %T')  model=${MODEL} ep=${EP} ===" > "$LOG"
for P in marl_force marl_force_bess; do
  echo "--- pipeline=${P} 開始 $(date '+%F %T') ---" >> "$LOG"
  EVMA_EVAL_PIPELINE="$P" python -u tools/evaluate_final_system_on_bid_bank.py \
    --model-dir "$MODEL" --episode "$EP" --bid-bank-dir "$BANK" \
    --command-scenarios 6 --ev-seeds 1 \
    --output-dir "execute_results/ft_sweep/ep${EP}_${P}" >> "$LOG" 2>&1
  echo "--- pipeline=${P} exit=$? $(date '+%F %T') ---" >> "$LOG"
done
echo "=== 完了 $(date '+%F %T') ===" >> "$LOG"
