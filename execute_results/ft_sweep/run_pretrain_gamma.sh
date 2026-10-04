#!/usr/bin/env bash
# 7station の pretrain を gamma を上げてやり直す。
#   usage: EVMA_GAMMA=0.99 [EVMA_GAMMA_GLOBAL=...] run_pretrain_gamma.sh <tag>
# バンクは既存の7station用（train_25_fixed_ev_256_all_commands /
# validation_5_fixed_ev_256_all_commands）をそのまま使う。構築は走らない。
set -u
TAG="$1"
cd "$(dirname "$0")/../.."

# 前回の pretrain と同じ観測レイアウト。揃えないと比較にならない。
export EVMA_NUM_STATIONS=7
export EVMA_ACTOR_EV_COUNT=1
export EVMA_LOWER_BID_CONTEXT_OBS=1
# 入札先読みは入れない。EnvConfig の既定 0 に任せる。24ブロック分の
# baseline/up/down を見せていたが、学習後の重みを測ると actor は7局すべてで
# 初期値より下げており（-6.8%、7/7）、この層の入力分散に占める取り分も
# 1.4% から 1.2% へ減っていた。残る instruction_scale_kw だけは大域批評家が
# 拾っている（+14.8%）ので LOWER_BID_CONTEXT_USE_OBS は既定の 1 のまま。
# 局所観測は 136 次元から 64 次元になり、replay の 1 遷移が約半分になる。
# 20station のバンク構築と同居させるための控えめな設定。単独なら 32 でよい。
export EVMA_LOWER_TRAIN_SCENARIO_WORKERS="${EVMA_LOWER_TRAIN_SCENARIO_WORKERS:-16}"

LOG="execute_results/ft_sweep/pretrain_${TAG}.log"
{
  echo "=== pretrain tag=${TAG} started $(date '+%F %T') ==="
  echo "GAMMA        = ${EVMA_GAMMA:-default(0.985)}"
  echo "GAMMA_GLOBAL = ${EVMA_GAMMA_GLOBAL:-default(0.95)}"
  echo "grad clip local/global = ${EVMA_GRAD_CLIP_MAX:-default(5.0)}/${EVMA_GRAD_CLIP_MAX_GLOBAL:-default(5.0)}"
  echo "central residual allocator = ${EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR:-default(1)}"
  echo "scenario workers = ${EVMA_LOWER_TRAIN_SCENARIO_WORKERS}"
} > "$LOG"

# 実際に効いた値と、使うバンクが既存かを起動前に確認する。
python - >> "$LOG" 2>&1 <<'PY'
import sys, os
sys.path.insert(0, ".")
from Config import GAMMA, GAMMA_GLOBAL, NUM_STATIONS
from EnvConfig import (LOWER_TRAIN_UPPER_BID_BANK_DIR as TR,
                       LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR as TE)
import math
for name, g in (("GAMMA", GAMMA), ("GAMMA_GLOBAL", GAMMA_GLOBAL)):
    eff = 1.0 / (1.0 - g)
    print(f"[preflight] {name}={g}  実効地平 {eff:.1f} step = {eff*5/60:.2f} h", flush=True)
from EnvConfig import USE_CENTRAL_EV_RESIDUAL_ALLOCATOR, TRAIN_USE_RESIDUAL_BESS
print(f"[preflight] stations={NUM_STATIONS}", flush=True)
# 学習中の環境。force は train ループから呼ばれないので常に無効。
print(f"[preflight] 学習環境: 中央残差配分={'ON' if USE_CENTRAL_EV_RESIDUAL_ALLOCATOR else 'OFF'} "
      f"BESS={'ON' if TRAIN_USE_RESIDUAL_BESS else 'OFF'} force=OFF(評価側のみ)", flush=True)
for tag, d in (("train", TR), ("test", TE)):
    ok = os.path.isdir(os.path.join(d, "days"))
    print(f"[preflight] {tag} bank {'既存' if ok else '見つからない'}: {d}", flush=True)
    if not ok:
        raise SystemExit(f"[preflight] {tag} バンクがない。構築が走るので中止。")
PY
if [ $? -ne 0 ]; then
  echo "=== tag=${TAG} preflight 失敗、中止 $(date '+%F %T') ===" >> "$LOG"
  exit 1
fi

python pre_train.py --episodes 2000 \
  --model-name "gamma_${TAG}_7station" \
  >> "$LOG" 2>&1
echo "=== tag=${TAG} exit=$? $(date '+%F %T') ===" >> "$LOG"
