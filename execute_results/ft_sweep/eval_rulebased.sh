#!/usr/bin/env bash
# ルールベース層（force + 中央残差分配 = 固定反復の重み付き water-filling）を
# 上限の比較対象として測り直す。ゼロアクタ版は MARL を一切使わない下限で、
# 学習済み actor との差が「方策そのものの値打ち」になる。
set -u
cd "$(dirname "$0")/../.."
MODEL=archive/prod_f500_dense8_noalloc_7station_20260921_102504
BANK=execute_results/bid_banks/validation_5_7s_f500_dense8
EP=1560
LOG=execute_results/ft_sweep/eval_rulebased.log

export EVMA_NUM_STATIONS=7 EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24 EVMA_LOWER_BID_CONTEXT_OBS=1
export EVMA_ACTIVATION_SCENARIO_DIR=data/aemo/nem/processed_5min/dense8
export EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW=500

echo "=== 開始 $(date '+%F %T') ep=${EP} ===" > "$LOG"
run () {  # $1=tag $2=pipeline $3=zero
  echo "--- ${1}  pipeline=${2} zero_actor=${3}  $(date '+%F %T') ---" >> "$LOG"
  EVMA_EVAL_PIPELINE="$2" EVMA_EVAL_ZERO_ACTOR="$3" \
  python -u tools/evaluate_final_system_on_bid_bank.py \
    --model-dir "$MODEL" --episode "$EP" --bid-bank-dir "$BANK" \
    --command-scenarios 6 --ev-seeds 1 \
    --output-dir "execute_results/ft_sweep/ep${EP}_${1}" >> "$LOG" 2>&1
  echo "--- ${1} exit=$? $(date '+%F %T') ---" >> "$LOG"
}
run marl_force_central marl_force_central 0
run rule_only         marl_force_central 1
run rule_bess         system              1
echo "=== 完了 $(date '+%F %T') ===" >> "$LOG"
