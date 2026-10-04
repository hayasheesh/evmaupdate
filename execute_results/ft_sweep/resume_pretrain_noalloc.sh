#!/usr/bin/env bash
# 停止した noalloc_7station の pretrain を ep212 の状態から続きから走らせる。
# 環境変数は run_pretrain_gamma.sh と同じ。1つでも欠けると resume の
# ランタイム照合が落ちるので、ここを勝手に変えない。
set -u
cd "$(dirname "$0")/../.."

RUN_DIR="archive/gamma_noalloc_7station_7station_20260917_181704"
LOG="execute_results/ft_sweep/pretrain_noalloc_7station.log"

export EVMA_NUM_STATIONS=7
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_CONTEXT_OBS=1
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24
export EVMA_LOWER_TRAIN_SCENARIO_WORKERS=16
export EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR=0

echo "=== resume tag=noalloc_7station started $(date '+%F %T') ===" >> "$LOG"
python pre_train.py --episodes 2000 --resume-run "$RUN_DIR" >> "$LOG" 2>&1
echo "=== resume tag=noalloc_7station exit=$? $(date '+%F %T') ===" >> "$LOG"
