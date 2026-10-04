#!/usr/bin/env bash
# 入札を縮めると完全分散（force のみ）でアセスメントIIを通るか。
# up_plan/down_plan を倍率で縮める。目標も帯も再計算されるので整合する。
set -u
cd "$(dirname "$0")/../.."
MODEL=archive/prod_hobomatch_7station_20260918_040210
LOG=execute_results/ft_sweep/bid_scale_sweep.log

export EVMA_NUM_STATIONS=7
export EVMA_EVAL_NO_BESS=1
export EVMA_EVAL_PIPELINE=marl_force
# 学習時の観測構成に合わせる。合わせないと actor の入力次元が 333 対 260 で落ちる。
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_CONTEXT_OBS=1

{
  echo "=== 入札倍率スイープ 開始 $(date '+%F %T') ==="
  echo "model=${MODEL} ep=2000  pipeline=marl_force (force あり/中央なし/BESSなし)"
} > "$LOG"

for S in 1.00 0.80 0.65 0.50; do
  TAG="bs${S/./}"
  echo "--- scale=${S} 開始 $(date '+%F %T') ---" >> "$LOG"
  EVMA_EVAL_BID_SCALE="$S" python -u tools/evaluate_final_system_on_bid_bank.py \
    --model-dir "$MODEL" --episode 2000 \
    --command-scenarios 6 --ev-seeds 1 \
    --output-dir "execute_results/ft_sweep/${TAG}" >> "$LOG" 2>&1
  echo "--- scale=${S} exit=$? $(date '+%F %T') ---" >> "$LOG"
done
echo "=== スイープ完了 $(date '+%F %T') ===" >> "$LOG"
