#!/usr/bin/env bash
# 落ちた hobomatch を最新の resume チェックポイントから続ける。
# 環境変数は run_pretrain_production.sh と、その呼び出し側が渡した2つ
# (slope と global critic のクリップ) を合わせたもの。1つでも欠けると
# pre_train.py の resume ガードが弾くので、ここを勝手に変えない。
# 照合は pre_train.py 自身が行う。自作の照合を置くと本物とずれる。
set -u
cd "$(dirname "$0")/../.."

RUN_DIR="archive/prod_hobomatch_7station_20260918_040210"
LOG="execute_results/ft_sweep/pretrain_hobomatch.log"

export EVMA_NUM_STATIONS=7
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_CONTEXT_OBS=1
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24
export EVMA_LOWER_TRAIN_SCENARIO_WORKERS=16
export EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR=0
export EVMA_GLOBAL_BALANCE_REWARD_MODE=legacy_tolerance_band
export EVMA_BALANCE_REWARD_SLOPE=0.0272
export EVMA_GRAD_CLIP_MAX_GLOBAL=20.0

echo "=== resume tag=hobomatch started $(date '+%F %T') ===" >> "$LOG"
python pre_train.py --episodes 2000 --resume-run "$RUN_DIR" >> "$LOG" 2>&1
echo "=== resume tag=hobomatch exit=$? $(date '+%F %T') ===" >> "$LOG"
