#!/usr/bin/env bash
# prod_hobomatch (ep2000) からの finetune。
#   usage: run_finetune_hm.sh <arm> <episodes> <allocator 0|1> [day] [eps_end]
# 報酬モード・傾き・クリップは pretrain と必ず同じ。中央残差分配だけ切り替える。
set -u
ARM="$1"
EPISODES="$2"
ALLOC="$3"
DAY="${4:-2024-12-04}"
EPS_END="${5:-0}"
cd "$(dirname "$0")/../.."

WARMSTART="${EVMA_FT_WARMSTART:-archive/prod_hobomatch_7station_20260918_040210/results/TEST2000}"

export EVMA_NUM_STATIONS=7
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_CONTEXT_OBS=1
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24

export EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR="${ALLOC}"
export EVMA_GLOBAL_BALANCE_REWARD_MODE=legacy_tolerance_band
export EVMA_BALANCE_REWARD_SLOPE=0.0272
export EVMA_GRAD_CLIP_MAX_GLOBAL=20.0

export EVMA_FINETUNE_EPSILON_END_EPISODE="${EPS_END}"
export EVMA_FINETUNE_SELECTION_SEEDS=1
export EVMA_LOWER_TRAIN_UPPER_BID_EVAL_SEEDS=1
export EVMA_TRAIN_INTERIM_INTERVAL=10

LOG="execute_results/ft_sweep/${ARM}.log"
{
  echo "=== finetune arm=${ARM} started $(date '+%F %T') ==="
  echo "warmstart  = ${WARMSTART}"
  echo "episodes   = ${EPISODES}"
  echo "day        = ${DAY}"
  echo "eps end ep = ${EPS_END}"
  echo "allocator  = ${ALLOC}  (学習中)"
  echo "reward     = ${EVMA_GLOBAL_BALANCE_REWARD_MODE} slope ${EVMA_BALANCE_REWARD_SLOPE}"
  echo "clip global= ${EVMA_GRAD_CLIP_MAX_GLOBAL}"
} > "$LOG"

python - "$WARMSTART" "$ALLOC" >> "$LOG" 2>&1 <<'PY'
import sys, glob, re, torch
from pathlib import Path
sys.path.insert(0, ".")
run, want_alloc = sys.argv[1], bool(int(sys.argv[2]))
pattern = (f"{run}/actor_0_ep*.pth" if Path(run).name.startswith("TEST")
           else f"{run}/results/TEST*/actor_0_ep*.pth")
cks = sorted(glob.glob(pattern),
             key=lambda p: int(re.search(r"ep(\d+)\.pth$", p).group(1)))
want = tuple(torch.load(cks[-1], map_location="cpu")["ev_action_head.0.weight"].shape)
from environment.EVEnv import EVEnv
from training.system_controller import build_agent
from EnvConfig import (GLOBAL_BALANCE_REWARD_MODE, GLOBAL_BALANCE_REWARD_SLOPE,
                       USE_CENTRAL_EV_RESIDUAL_ALLOCATOR)
from Config import GRAD_CLIP_MAX_GLOBAL
env = EVEnv()
got = tuple(build_agent(env).actors[0].state_dict()["ev_action_head.0.weight"].shape)
if got != want:
    raise SystemExit(f"[preflight] 観測レイアウト不一致: checkpoint={want} 構築={got}")
print(f"[preflight] 観測レイアウト一致 {got} ({cks[-1]})", flush=True)
print(f"[preflight] mode={GLOBAL_BALANCE_REWARD_MODE} slope={GLOBAL_BALANCE_REWARD_SLOPE:.5f} "
      f"clip_global={GRAD_CLIP_MAX_GLOBAL} allocator="
      f"{'ON' if USE_CENTRAL_EV_RESIDUAL_ALLOCATOR else 'OFF'}", flush=True)
bad = (GLOBAL_BALANCE_REWARD_MODE != "legacy_tolerance_band"
       or abs(GLOBAL_BALANCE_REWARD_SLOPE - 0.0272) > 1e-9
       or GRAD_CLIP_MAX_GLOBAL != 20.0
       or bool(USE_CENTRAL_EV_RESIDUAL_ALLOCATOR) != want_alloc)
if bad:
    raise SystemExit("[preflight] 条件が意図と違う。中止。")
PY
if [ $? -ne 0 ]; then
  echo "=== arm=${ARM} preflight 失敗、中止 $(date '+%F %T') ===" >> "$LOG"
  exit 1
fi

python fine_tune.py --day "${DAY}" --episodes "${EPISODES}" \
  --eval-command-scenarios 24 --selection-command-scenarios 24 \
  --warmstart "$WARMSTART" \
  --model-name "ft_${ARM}" >> "$LOG" 2>&1
echo "=== arm=${ARM} exit=$? $(date '+%F %T') ===" >> "$LOG"
