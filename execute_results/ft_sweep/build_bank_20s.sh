#!/usr/bin/env bash
# 20局・床500kW・直接LP閾値400 で train25 + validation5 の入札bankを作る。
# 1日あたり447秒(32ワーカー・日単位1並列)の実測に条件を揃える。
set -u
cd "$(dirname "$0")/../.."

export EVMA_NUM_STATIONS=20
export EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW=500
export EVMA_LOWER_TRAIN_UPPER_BID_DIRECT_ORACLE_MAX_EVS=400
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_CONTEXT_OBS=1
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24

TRAIN=execute_results/bid_banks/train_25_20s_f500
TEST=execute_results/bid_banks/validation_5_20s_f500

echo "=== bank構築 開始 $(date '+%F %T') ==="
echo "局数20 / 床500kW / 直接LP閾値400 / scenario-workers 16 / 日単位1並列"

for split in train test; do
  if [ "$split" = "train" ]; then D=25; OUT=$TRAIN; else D=5; OUT=$TEST; fi
  echo "--- split=$split days=$D -> $OUT  $(date '+%T') ---"
  python tools/build_training_bid_bank.py \
    --split "$split" --days "$D" \
    --paired-train-days 25 --paired-test-days 5 --train-split-count 25 \
    --workers 1 --scenario-workers 16 \
    --output-dir "$OUT"
  rc=$?
  echo "--- split=$split exit=$rc $(date '+%T') ---"
  [ $rc -ne 0 ] && { echo "失敗のため中止"; exit $rc; }
done
echo "=== bank構築 完了 $(date '+%F %T') ==="
