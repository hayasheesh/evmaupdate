#!/usr/bin/env bash
# 20ステーションの本番 pretrain。7station 本番と報酬・クリップ・γを揃え、
# 変えるのは局数と入札バンクだけにする。
#   usage: run_pretrain_20station.sh <tag> [episodes]
set -u
TAG="$1"
EPISODES="${2:-3000}"
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

export EVMA_NUM_STATIONS=20
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
export EVMA_BALANCE_REWARD_SLOPE="${EVMA_BALANCE_REWARD_SLOPE:-0.0272}"
# 7station 本番はシェル側の export に依存していた。既定は 5.0/5.0 なので固定する。
export EVMA_GRAD_CLIP_MAX="${EVMA_GRAD_CLIP_MAX:-5.0}"
export EVMA_GRAD_CLIP_MAX_GLOBAL="${EVMA_GRAD_CLIP_MAX_GLOBAL:-20.0}"

# 20station バンクは最低入札 500 kW で構築済み。値を合わせないと
# manifest が不一致になり、再構築（4時間）が走る。
export EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW=500
export EVMA_LOWER_TRAIN_BID_BANK_DIR="${ROOT}/execute_results/bid_banks/train_25_20s_f500"
export EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR="${ROOT}/execute_results/bid_banks/validation_5_20s_f500"
# 不一致のとき黙って作り直させない。preflight で止める。
export EVMA_LOWER_TRAIN_BUILD_BID_BANK=0

LOG="execute_results/ft_sweep/pretrain_${TAG}.log"
{
  echo "=== 20station pretrain tag=${TAG} started $(date '+%F %T') ==="
  echo "stations            = ${EVMA_NUM_STATIONS}"
  echo "balance reward mode = ${EVMA_GLOBAL_BALANCE_REWARD_MODE}"
  echo "slope               = ${EVMA_BALANCE_REWARD_SLOPE}"
  echo "grad clip           = ${EVMA_GRAD_CLIP_MAX}/${EVMA_GRAD_CLIP_MAX_GLOBAL}"
  echo "central allocator   = ${EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR}"
  echo "min bid kW          = ${EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW}"
  echo "episodes            = ${EPISODES}"
  echo "scenario workers    = ${EVMA_LOWER_TRAIN_SCENARIO_WORKERS}"
} > "$LOG"

python - >> "$LOG" 2>&1 <<'PY'
import json, os, sys
sys.path.insert(0, ".")
from pathlib import Path
from EnvConfig import (GLOBAL_BALANCE_REWARD, GLOBAL_BALANCE_REWARD_MODE,
                       GLOBAL_BALANCE_REWARD_SLOPE, USE_CENTRAL_EV_RESIDUAL_ALLOCATOR,
                       LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW as MINBID,
                       LOWER_TRAIN_UPPER_BID_BANK_DIR as TR,
                       LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR as TE)
from Config import GAMMA, GAMMA_GLOBAL, NUM_STATIONS, GRAD_CLIP_MAX, GRAD_CLIP_MAX_GLOBAL
from training.bid_bank import manifest_settings_match
from training.lower_bid_training import upper_bid_bank_settings
from environment.arrival_context import ArrivalScenarioSampler

print(f"[preflight] stations={NUM_STATIONS} mode={GLOBAL_BALANCE_REWARD_MODE} "
      f"R={GLOBAL_BALANCE_REWARD} slope={GLOBAL_BALANCE_REWARD_SLOPE:.5f}/kW", flush=True)
print(f"[preflight] gamma={GAMMA}/{GAMMA_GLOBAL} clip={GRAD_CLIP_MAX}/{GRAD_CLIP_MAX_GLOBAL} "
      f"minbid={float(MINBID):.1f}kW "
      f"中央残差配分={'ON' if USE_CENTRAL_EV_RESIDUAL_ALLOCATOR else 'OFF'}", flush=True)

if int(NUM_STATIONS) != 20:
    raise SystemExit("[preflight] 局数が20でない。中止。")
if GLOBAL_BALANCE_REWARD_MODE != "legacy_tolerance_band":
    raise SystemExit("[preflight] 報酬モードが違う。中止。")
if abs(GRAD_CLIP_MAX - 5.0) > 1e-9 or abs(GRAD_CLIP_MAX_GLOBAL - 20.0) > 1e-9:
    raise SystemExit("[preflight] クリップが7station本番(5.0/20.0)と違う。中止。")
if USE_CENTRAL_EV_RESIDUAL_ALLOCATOR:
    raise SystemExit("[preflight] 学習中の中央残差配分がONになっている。中止。")

required = {"arrival_model": ArrivalScenarioSampler().settings_signature(),
            **upper_bid_bank_settings()}
ok = True
for tag, d in (("train", TR), ("test", TE)):
    path = Path(d) / "manifest.json"
    if not path.exists():
        print(f"[preflight] {tag} バンクの manifest がない: {path}", flush=True)
        ok = False
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    complete = bool(payload.get("complete", False))
    ignored = {"activation_source_dir", "activation_library_file_count",
               "activation_library_sha256"} if tag == "test" else ()
    match = manifest_settings_match(payload, required, ignored_keys=ignored)
    n = len(payload.get("days") or payload.get("entries") or [])
    print(f"[preflight] {tag} バンク complete={complete} 一致={match} 日数={n} {d}", flush=True)
    if not (complete and match):
        actual = payload.get("settings") or {}
        for k, v in required.items():
            if k in ignored:
                continue
            if actual.get(k) != v:
                print(f"[preflight]   不一致 {k}: bank={actual.get(k)!r} now={v!r}", flush=True)
        ok = False
if not ok:
    raise SystemExit("[preflight] バンクが使えない。再構築が走るので中止。")
print("[preflight] 通過", flush=True)
PY
if [ $? -ne 0 ]; then echo "=== preflight 失敗、中止 $(date '+%F %T') ===" >> "$LOG"; exit 1; fi

python pre_train.py --episodes "${EPISODES}" \
  --model-name "prod_${TAG}_20station" \
  --resume-checkpoint-interval 20 >> "$LOG" 2>&1
echo "=== tag=${TAG} exit=$? $(date '+%F %T') ===" >> "$LOG"
