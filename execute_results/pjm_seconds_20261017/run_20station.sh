#!/bin/bash
# 20 station の PJM の40組を、秒単位の層なし（off）・あり（plain、reach）で評価する（2026-10-17、研究室のコンテナで動かす）。
cd "$(dirname "$0")"
PY=${PY:-/workspace/.venvs/evma20/bin/python}
declare -A STAGE=([main20]=../resbid_20station_20261014 [pjmrule20]=../pjmrule_20station_20261015)
declare -A RUN=([main20]=../../archive/wr_cap77wd_pjmplan_20station_resbid_tanh_g04l15_20261015_025502
                [pjmrule20]=../../archive/wr_cap77wd_pjmplan_20station_pjmrule75_g04l15_20261015_135405)
declare -A FLAGS=([off]="--fast off" [plain]="--fast on --headroom plain" [reach]="--fast on --headroom reach")
for layer in off reach plain; do
  for bid in main20 pjmrule20; do
    for mode in marl marl_force rule; do
      name=${bid}_${mode}_${layer}
      [ -f results/$name.DONE ] && continue
      $PY -X utf8 eval_seconds.py --stage-dir ${STAGE[$bid]} --run-dir ${RUN[$bid]} --episode 2000 --mode $mode \
        ${FLAGS[$layer]} --out results/$name.jsonl --workers ${WORKERS:-20} > results/$name.log 2>&1
      echo "$name exit $? $(wc -l < results/$name.jsonl) sets"
    done
  done
done
echo ALL_DONE
