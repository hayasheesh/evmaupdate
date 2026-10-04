#!/usr/bin/env bash
# A: 報酬の裾を直線にする（切替 60 kW）
# B: 前ステップの艦隊合計残差を観測に入れる（スカラー1個、全局共通）
#
# A の根拠: ep1560 で、外したステップの平均誤差は 113.8 kW。そこでの tanh の
# 傾きは成功時（16.7 kW）の 60%、300 kW では 14分の1。残る失敗は全部この裾に
# ある。60 kW から直線にすると 114 kW での傾きが 1.45 倍、300 kW で 12 倍に
# なる。原点の傾き 0.01333 は不変、ゼロ交差は 82.4 -> 81.0 kW。
# 代償として下に非有界になるので、大域批評家のクリップ率は上がる見込み。
#
# B の根拠: ep1800 で、126 kW を超える指令の送出率が 0.79 で一定。外した
# ステップの 56〜63% ではどの局も自分の限界に達していない。全局が少しずつ
# 控えて合計が届いていないが、それを告げる入力が観測に無い。渡すのは連系点
# 蓄電池が既に読んでいるのと同じスカラー1個で、個別EVの状態は含まない。
set -eu
cd "$(dirname "$0")/../.."

python - <<'PY'
import EnvConfig as E, sys
if E.USE_CENTRAL_EV_RESIDUAL_ALLOCATOR: sys.exit("中央残差分配が有効。中止。")
if E.TRAIN_USE_RESIDUAL_BESS:           sys.exit("学習時BESSが有効。中止。")
print("[preflight] 分配器 off / 学習時BESS off を確認")
PY

EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR=0 \
EVMA_ACTOR_EV_COUNT=1 \
EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24 \
EVMA_GLOBAL_BALANCE_REWARD_MODE=bounded_absolute_error \
EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW=150 \
EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW=60 \
EVMA_LOCAL_FLEET_RESIDUAL_OBS=1 \
EVMA_GRAD_CLIP_MAX=5.0 \
EVMA_GRAD_CLIP_MAX_GLOBAL=5.0 \
EVMA_ACTIVATION_SCENARIO_DIR=data/aemo/nem/processed_5min/dense8 \
EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW=500 \
python -u pre_train.py \
  --episodes 2000 \
  --resume-checkpoint-interval 20 \
  --model-name prod_f500_dense8_AB_7station \
  --bank-dir execute_results/bid_banks/train_25_7s_f500_dense8 \
  --test-bank-dir execute_results/bid_banks/validation_5_7s_f500_dense8
