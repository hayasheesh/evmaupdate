#!/usr/bin/env bash
# 7局・dense8・最低入札500kW の本番 pretrain。中央残差分配は入れない。
# 提案するシステムは完全分散なので、actor の行動を書き換える中央層は構成に無い。
# EnvConfig の既定も 0 にしてあるが、ここでも明示する。既定に頼った起動で
# 分配器アリのまま 717ep 回したことがあるため、二度と黙って入らないようにする。
# 入札先読みは入れない。EnvConfig の既定 0 に任せる。24ブロック分の
# baseline/up/down を見せていたが、学習後の重みを測ると actor は7局すべてで
# 初期値より下げており（-6.8%、7/7）、この層の入力分散に占める取り分も
# 1.4% から 1.2% へ減っていた。残る instruction_scale_kw だけは大域批評家が
# 拾っている（+14.8%）ので LOWER_BID_CONTEXT_USE_OBS は既定の 1 のまま。
# 局所観測は 136 次元から 64 次元になり、replay の 1 遷移が約半分になる。
#
# 2026-09-21 に起動した prod_f500_dense8_noalloc_7station は先読み 24 で
# 学習している。あの run を --resume-run で再開するときは
# EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24 を明示すること。観測次元が合わないと
# 保存済みの observation_config 署名に弾かれる。
set -eu
cd "$(dirname "$0")/../.."

python - <<'PY'
import EnvConfig, sys
if EnvConfig.USE_CENTRAL_EV_RESIDUAL_ALLOCATOR:
    sys.exit("中央残差分配が有効になっている。起動を中止する。")
if EnvConfig.TRAIN_USE_RESIDUAL_BESS:
    sys.exit("学習時 BESS が有効になっている。起動を中止する。")
print("[preflight] 中央残差分配 off / 学習時BESS off を確認")
PY

EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR=0 \
EVMA_ACTOR_EV_COUNT=1 \
EVMA_GLOBAL_BALANCE_REWARD_MODE=bounded_absolute_error \
EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW=150 \
EVMA_GRAD_CLIP_MAX=5.0 \
EVMA_GRAD_CLIP_MAX_GLOBAL=5.0 \
EVMA_ACTIVATION_SCENARIO_DIR=data/aemo/nem/processed_5min/dense8 \
EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW=500 \
python -u pre_train.py \
  --episodes 2000 \
  --model-name prod_f500_dense8_noalloc_7station \
  --bank-dir execute_results/bid_banks/train_25_7s_f500_dense8 \
  --test-bank-dir execute_results/bid_banks/validation_5_7s_f500_dense8
