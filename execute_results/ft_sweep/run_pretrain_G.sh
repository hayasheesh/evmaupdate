#!/usr/bin/env bash
# A+B はそのまま、行動器の勾配を大域批評家だけにする（Q_MIX_GLOBAL_WEIGHT=1.0）。
#
# 根拠: A+B 走行は ep440-500 で 91.78% / MAE 41.9 に達したあと、ep1000-1260 の
# 平均で 89.68% まで下がった。同じ区間で大域報酬は 0.2769 -> 0.2751 とほぼ動かず、
# 局所報酬だけが -0.0006 -> +0.0176 と単調に上がっている。混合 g_mix =
# 0.5*g_l + 0.5*g_g のうち大域側が平らになった後、更新の向きを局所側が決めている。
# 重み 1.0 では skip_local が立ち、局所批評家の学習と局所勾配の両方が外れる。
#
# バンクの5日はすべて全情報LPで実行可能と証明済み（joint_certification:
# all_scenarios_ok=True, soc_all_ok=True, 256指令シナリオ, 許容失敗0）。
# したがって 100% との差は入札側ではなく制御側にある。
# 入札先読みは入れない（EnvConfig 既定 0）。24 ブロック分を見せていた頃に
# 学習後の重みを測ると actor は7局すべてで初期値より下げており、この層の
# 入力分散の取り分も 1.4% から 1.2% へ減っていた。A+B 走行の起動ファイルで
# 24 を指定してしまっていたので戻す。局所観測は 137 -> 65 次元。
set -eu
cd "$(dirname "$0")/../.."

python - <<'PY'
import EnvConfig as E, sys
if E.USE_CENTRAL_EV_RESIDUAL_ALLOCATOR: sys.exit("中央残差分配が有効。中止。")
if E.TRAIN_USE_RESIDUAL_BESS:           sys.exit("学習時BESSが有効。中止。")
print("[preflight] 分配器 off / 学習時BESS off を確認")
PY

EVMA_Q_MIX_GLOBAL_WEIGHT=1.0 \
EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR=0 \
EVMA_ACTOR_EV_COUNT=1 \
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
  --model-name prod_f500_dense8_ABG_7station \
  --bank-dir execute_results/bid_banks/train_25_7s_f500_dense8 \
  --test-bank-dir execute_results/bid_banks/validation_5_7s_f500_dense8
