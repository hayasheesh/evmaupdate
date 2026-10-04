#!/usr/bin/env bash
# 本番の長時間 pretrain。hobo100100 と同じ帯のステップ関数、傾きは短期試行で
# 選んだ値。中央残差配分は学習中 OFF、実行時のみ ON（評価器が自前で入れる）。
#   usage: EVMA_BALANCE_REWARD_SLOPE=<値> run_pretrain_production.sh <tag> [episodes]
set -u
TAG="$1"
EPISODES="${2:-2000}"
cd "$(dirname "$0")/../.."

export EVMA_NUM_STATIONS=7
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_CONTEXT_OBS=1
# 入札先読みは入れない。EnvConfig の既定 0 に任せる。24ブロック分の
# baseline/up/down を見せていたが、学習後の重みを測ると actor は7局すべてで
# 初期値より下げており（-6.8%、7/7）、この層の入力分散に占める取り分も
# 1.4% から 1.2% へ減っていた。残る instruction_scale_kw だけは大域批評家が
# 拾っている（+14.8%）ので LOWER_BID_CONTEXT_USE_OBS は既定の 1 のまま。
# 局所観測は 136 次元から 64 次元になり、replay の 1 遷移が約半分になる。
export EVMA_LOWER_TRAIN_SCENARIO_WORKERS="${EVMA_LOWER_TRAIN_SCENARIO_WORKERS:-16}"
export EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR=0
export EVMA_GLOBAL_BALANCE_REWARD_MODE=legacy_tolerance_band

LOG="execute_results/ft_sweep/pretrain_${TAG}.log"
{
  echo "=== production pretrain tag=${TAG} started $(date '+%F %T') ==="
  echo "balance reward mode = ${EVMA_GLOBAL_BALANCE_REWARD_MODE}"
  echo "slope               = ${EVMA_BALANCE_REWARD_SLOPE:-default(0.01333)}"
  echo "central allocator   = ${EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR}"
  echo "episodes            = ${EPISODES}"
  echo "scenario workers    = ${EVMA_LOWER_TRAIN_SCENARIO_WORKERS}"
} > "$LOG"

python - >> "$LOG" 2>&1 <<'PY'
import os, sys; sys.path.insert(0, ".")
from EnvConfig import (GLOBAL_BALANCE_REWARD, GLOBAL_BALANCE_REWARD_MODE,
                       GLOBAL_BALANCE_REWARD_SLOPE, USE_CENTRAL_EV_RESIDUAL_ALLOCATOR,
                       LOWER_TRAIN_UPPER_BID_BANK_DIR as TR,
                       LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR as TE)
from Config import GAMMA, GAMMA_GLOBAL, NUM_STATIONS, GRAD_CLIP_MAX, GRAD_CLIP_MAX_GLOBAL
print(f"[preflight] mode={GLOBAL_BALANCE_REWARD_MODE} R={GLOBAL_BALANCE_REWARD} "
      f"slope={GLOBAL_BALANCE_REWARD_SLOPE:.5f}/kW", flush=True)
print(f"[preflight] gamma={GAMMA}/{GAMMA_GLOBAL} clip={GRAD_CLIP_MAX}/{GRAD_CLIP_MAX_GLOBAL} "
      f"stations={NUM_STATIONS} 中央残差配分={'ON' if USE_CENTRAL_EV_RESIDUAL_ALLOCATOR else 'OFF'}", flush=True)
if GLOBAL_BALANCE_REWARD_MODE != "legacy_tolerance_band":
    raise SystemExit("[preflight] 報酬モードが違う。中止。")
for tag, d in (("train", TR), ("test", TE)):
    ok = os.path.isdir(os.path.join(d, "days"))
    print(f"[preflight] {tag} bank {'既存' if ok else '見つからない'}: {d}", flush=True)
    if not ok:
        raise SystemExit(f"[preflight] {tag} バンクがない。構築が走るので中止。")
PY
if [ $? -ne 0 ]; then echo "=== preflight 失敗、中止 $(date '+%F %T') ===" >> "$LOG"; exit 1; fi

python pre_train.py --episodes "${EPISODES}" --model-name "prod_${TAG}_7station" >> "$LOG" 2>&1
echo "=== tag=${TAG} exit=$? $(date '+%F %T') ===" >> "$LOG"
