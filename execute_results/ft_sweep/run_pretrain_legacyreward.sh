#!/usr/bin/env bash
# run B と報酬モードだけ変えた pretrain。hobo100100 と同じ帯のステップ関数に戻す。
# 他の環境変数は run_pretrain_gamma.sh と同じ。揃えないと比較にならない。
set -u
TAG="$1"
EPISODES="${2:-150}"
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
# ここだけが run B との違い。hobo と同じ「帯の中なら +1、外は 2/150 /kW で線形に落ちる」。
export EVMA_GLOBAL_BALANCE_REWARD_MODE=legacy_tolerance_band

LOG="execute_results/ft_sweep/pretrain_${TAG}.log"
{
  echo "=== pretrain tag=${TAG} started $(date '+%F %T') ==="
  echo "balance reward mode = ${EVMA_GLOBAL_BALANCE_REWARD_MODE}"
  echo "central residual allocator = ${EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR}"
  echo "episodes = ${EPISODES}"
  echo "slope override = ${EVMA_BALANCE_REWARD_SLOPE:-default(0.01333)}"
  echo "scenario workers = ${EVMA_LOWER_TRAIN_SCENARIO_WORKERS}"
} > "$LOG"

python - >> "$LOG" 2>&1 <<'PY'
import sys; sys.path.insert(0, ".")
from EnvConfig import (GLOBAL_BALANCE_REWARD, GLOBAL_BALANCE_REWARD_MODE,
                       GLOBAL_BALANCE_REWARD_SLOPE, USE_CENTRAL_EV_RESIDUAL_ALLOCATOR)
from Config import GAMMA, GAMMA_GLOBAL, NUM_STATIONS
print(f"[preflight] mode={GLOBAL_BALANCE_REWARD_MODE} R={GLOBAL_BALANCE_REWARD} "
      f"slope={GLOBAL_BALANCE_REWARD_SLOPE:.5f}/kW", flush=True)
print(f"[preflight] gamma={GAMMA}/{GAMMA_GLOBAL} stations={NUM_STATIONS} "
      f"中央残差配分={'ON' if USE_CENTRAL_EV_RESIDUAL_ALLOCATOR else 'OFF'}", flush=True)
if GLOBAL_BALANCE_REWARD_MODE != "legacy_tolerance_band":
    raise SystemExit("[preflight] 報酬モードが切り替わっていない。中止。")
PY
if [ $? -ne 0 ]; then echo "=== preflight 失敗 ===" >> "$LOG"; exit 1; fi

python pre_train.py --episodes "${EPISODES}" --model-name "legacyrw_${TAG}_7station" >> "$LOG" 2>&1
echo "=== tag=${TAG} exit=$? $(date '+%F %T') ===" >> "$LOG"
