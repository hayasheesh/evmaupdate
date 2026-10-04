#!/usr/bin/env bash
# 1本の fine-tune アームを起動する。共通条件は固定、差分は呼び出し側の環境変数。
#   usage: run_arm.sh <arm_name>
set -u
ARM="$1"
cd "$(dirname "$0")/../.."

WARMSTART="archive/direct_bid_256cmd_25d_evcount_12hbid_7station_20260916_002104"

# pretrain の観測レイアウト。fine_tune.py は正規化プロファイルだけを引き継ぎ、
# レイアウトのフラグは引き継がないので、ここで揃える。揃っていないと
# 16分の入札solveを終えた後に actor の shape mismatch で落ちる。
export EVMA_NUM_STATIONS=7
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_CONTEXT_OBS=1
export EVMA_LOWER_BID_LOOKAHEAD_BLOCKS=24

# 評価は仮説を分けるための粗い設定。報告用の数字ではない。
export EVMA_FINETUNE_SELECTION_SEEDS=1
export EVMA_LOWER_TRAIN_UPPER_BID_EVAL_SEEDS=1
export EVMA_TRAIN_INTERIM_INTERVAL=10

LOG="execute_results/ft_sweep/${ARM}.log"
{
  echo "=== arm=${ARM} started $(date '+%F %T') ==="
  echo "LR actor/local/global = ${EVMA_LR_ACTOR:-default}/${EVMA_LR_CRITIC_LOCAL:-default}/${EVMA_LR_GLOBAL_CRITIC:-default}"
  echo "balanced replay final/decay = ${EVMA_BALANCED_REPLAY_RATIO_FINAL:-default}/${EVMA_BALANCED_REPLAY_DECAY_STEPS:-default}"
} > "$LOG"

# 入札を解く前に actor の入力次元を照合する。
python - "$WARMSTART" >> "$LOG" 2>&1 <<'PY'
import sys, torch
sys.path.insert(0, ".")
run = sys.argv[1]
import glob, re
cks = sorted(glob.glob(f"{run}/results/TEST*/actor_0_ep*.pth"),
             key=lambda p: int(re.search(r"ep(\d+)\.pth$", p).group(1)))
want = tuple(torch.load(cks[-1], map_location="cpu")["ev_action_head.0.weight"].shape)
from environment.EVEnv import EVEnv
from training.system_controller import build_agent
env = EVEnv()
got = tuple(build_agent(env).actors[0].state_dict()["ev_action_head.0.weight"].shape)
if got != want:
    raise SystemExit(f"[preflight] 観測レイアウト不一致: checkpoint={want} 構築={got}")
print(f"[preflight] 観測レイアウト一致 {got} ({cks[-1]})", flush=True)
PY
if [ $? -ne 0 ]; then
  echo "=== arm=${ARM} preflight 失敗、中止 $(date '+%F %T') ===" >> "$LOG"
  exit 1
fi

python fine_tune.py --day 2024-12-04 --episodes 100 \
  --eval-command-scenarios 24 --selection-command-scenarios 24 \
  --warmstart "$WARMSTART" \
  --model-name "ft_${ARM}" >> "$LOG" 2>&1
echo "=== arm=${ARM} exit=$? $(date '+%F %T') ===" >> "$LOG"
